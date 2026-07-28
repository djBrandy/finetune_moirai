import json
import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pathlib import Path

# Primary TF → pandas resample rule
RESAMPLE_RULES = {"M5": "5min", "M1": "1min", "H1": "1h", "D1": "1D"}


def load_config():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    phase     = cfg["training"]["phase"]
    phase_cfg = cfg["phases"][phase]
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


def main():
    cfg        = load_config()
    raw        = cfg["data"]["raw_dir"]
    out        = cfg["data"]["processed_dir"]
    tfs        = cfg["data"]["timeframes"]
    end        = pd.Timestamp(cfg["data"]["train_end"])
    primary_tf = cfg["_primary_tf"]
    rule       = RESAMPLE_RULES[primary_tf]
    phase      = cfg["_phase"]
    Path(out).mkdir(exist_ok=True)

    print(f"Phase {phase} | Primary TF: {primary_tf} | Resampling all TFs to {rule}")

    frames = []
    for tf in tfs:
        path = os.path.join(raw, f"XBRUSD_{tf}.csv")
        if not os.path.exists(path):
            print(f"  Missing {path}, skipping.")
            continue
        df      = load_and_clean(path)
        df      = compute_features(df)
        resampled = resample_to_freq(df, tf, rule)
        frames.append(resampled)
        print(f"  {tf}: {len(resampled)} rows at {primary_tf} frequency")

    # Inner join — only keep timestamps where all TFs have data
    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    train        = merged[merged["timestamp"] <= end].reset_index(drop=True)
    feature_cols = [c for c in train.columns if c != "timestamp"]
    n_features   = len(feature_cols)

    print(f"  Train rows: {len(train)} | Features: {n_features}")
    print(f"  Range: {train['timestamp'].iloc[0]} → {train['timestamp'].iloc[-1]}")

    # Save feature column order for inference
    with open(os.path.join(out, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save phase metadata for inference
    with open(os.path.join(out, "phase_meta.json"), "w") as f:
        json.dump({"phase": phase, "primary_tf": primary_tf,
                   "n_features": n_features, "rule": rule}, f, indent=2)

    target_matrix = train[feature_cols].values.astype(np.float32)

    schema = pa.schema([
        ("item_id",    pa.string()),
        ("start",      pa.timestamp("s")),
        ("target",     pa.list_(pa.list_(pa.float32()))),
        ("freq",       pa.string()),
        ("n_features", pa.int32()),
    ])
    table = pa.table({
        "item_id":    pa.array(["XBRUSD_multivariate"], type=pa.string()),
        "start":      pa.array([train["timestamp"].iloc[0]], type=pa.timestamp("s")),
        "target":     pa.array([target_matrix.tolist()], type=pa.list_(pa.list_(pa.float32()))),
        "freq":       pa.array([rule], type=pa.string()),
        "n_features": pa.array([n_features], type=pa.int32()),
    }, schema=schema)

    out_path = os.path.join(out, "train.parquet")
    pq.write_table(table, out_path)
    print(f"  Saved ({len(train)} × {n_features}) → {out_path}")
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
