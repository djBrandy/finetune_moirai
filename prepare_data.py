import json
import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pathlib import Path

RESAMPLE_RULES = {"M5": "5min", "M1": "1min", "H1": "1h", "D1": "1D"}


def load_config():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    phase            = cfg["training"]["phase"]
    phase_cfg        = cfg["phases"][phase]
    cfg["_phase"]      = phase
    cfg["_primary_tf"] = phase_cfg["primary_tf"]
    cfg["_ctx"]        = phase_cfg["context_length"]
    cfg["_pred"]       = phase_cfg["prediction_length"]
    cfg["_best_model"] = phase_cfg["best_model"]
    return cfg


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["log_hl"]     = np.log(df["high"]  / df["low"])
    df["log_co"]     = np.log(df["close"] / df["open"])
    return df.dropna().reset_index(drop=True)


def resample_to_freq(df: pd.DataFrame, source_tf: str, target_rule: str) -> pd.DataFrame:
    df = df.set_index("timestamp")
    agg = {"log_return": "sum", "log_hl": "mean", "log_co": "last", "volume": "sum"}
    resampled = df.resample(target_rule).agg(agg).dropna().reset_index()
    resampled.columns = [f"{c}_{source_tf}" if c != "timestamp" else c
                         for c in resampled.columns]
    return resampled


def save_parquet(matrix: np.ndarray, start_ts: pd.Timestamp,
                 rule: str, path: str):
    schema = pa.schema([
        ("item_id",    pa.string()),
        ("start",      pa.timestamp("s")),
        ("target",     pa.list_(pa.list_(pa.float32()))),
        ("freq",       pa.string()),
        ("n_features", pa.int32()),
    ])
    table = pa.table({
        "item_id":    pa.array(["XBRUSD_multivariate"], type=pa.string()),
        "start":      pa.array([start_ts],              type=pa.timestamp("s")),
        "target":     pa.array([matrix.tolist()],       type=pa.list_(pa.list_(pa.float32()))),
        "freq":       pa.array([rule],                  type=pa.string()),
        "n_features": pa.array([matrix.shape[1]],       type=pa.int32()),
    }, schema=schema)
    pq.write_table(table, path)


def compute_return_stats(matrix: np.ndarray) -> dict:
    """Compute XBRUSD return distribution stats for signal calibration."""
    returns = matrix[:, 0]   # log_return of primary TF, channel 0
    return {
        "mean_abs_return": float(np.mean(np.abs(returns))),
        "std_return":      float(np.std(returns)),
        "median_abs_return": float(np.median(np.abs(returns))),
    }


def main():
    cfg        = load_config()
    raw        = cfg["data"]["raw_dir"]
    out        = cfg["data"]["processed_dir"]
    tfs        = cfg["data"]["timeframes"]
    train_end  = pd.Timestamp(cfg["data"]["train_end"])
    val_end    = pd.Timestamp(cfg["data"]["val_end"])
    primary_tf = cfg["_primary_tf"]
    rule       = RESAMPLE_RULES[primary_tf]
    phase      = cfg["_phase"]
    Path(out).mkdir(exist_ok=True)

    print(f"Phase {phase} | Primary TF: {primary_tf} | Rule: {rule}")
    print(f"  Train: up to {train_end.date()} | Val: {train_end.date()} → {val_end.date()}")

    frames = []
    for tf in tfs:
        path = os.path.join(raw, f"XBRUSD_{tf}.csv")
        if not os.path.exists(path):
            print(f"  Missing {path}, skipping.")
            continue
        df        = load_and_clean(path)
        df        = compute_features(df)
        resampled = resample_to_freq(df, tf, rule)
        frames.append(resampled)
        print(f"  {tf}: {len(resampled)} rows at {primary_tf} frequency")

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    feature_cols = [c for c in merged.columns if c != "timestamp"]
    n_features   = len(feature_cols)

    train = merged[merged["timestamp"] <= train_end].reset_index(drop=True)
    val   = merged[(merged["timestamp"] > train_end) &
                   (merged["timestamp"] <= val_end)].reset_index(drop=True)

    print(f"  Train rows: {len(train)} | Val rows: {len(val)} | Features: {n_features}")

    # Compute return stats from training data for signal calibration
    train_matrix = train[feature_cols].values.astype(np.float32)
    stats        = compute_return_stats(train_matrix)
    print(f"  Return stats → mean_abs: {stats['mean_abs_return']:.6f} | "
          f"std: {stats['std_return']:.6f}")

    # Phase-namespaced metadata — never overwritten by other phases
    meta = {
        "phase": phase, "primary_tf": primary_tf,
        "n_features": n_features, "rule": rule,
        "feature_cols": feature_cols,
        "return_stats": stats,
        "train_end": str(train_end.date()),
        "val_end":   str(val_end.date()),
    }
    meta_path = os.path.join(out, f"phase{phase}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {meta_path}")

    # Save train and val parquets (phase-namespaced)
    train_path = os.path.join(out, f"phase{phase}_train.parquet")
    val_path   = os.path.join(out, f"phase{phase}_val.parquet")

    save_parquet(train_matrix, train["timestamp"].iloc[0], rule, train_path)
    print(f"  Train parquet → {train_path}")

    if len(val) > 0:
        val_matrix = val[feature_cols].values.astype(np.float32)
        save_parquet(val_matrix, val["timestamp"].iloc[0], rule, val_path)
        print(f"  Val parquet   → {val_path}")
    else:
        print("  Warning: no validation rows in date range.")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
