import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import os
from pathlib import Path


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["log_hl"]     = np.log(df["high"]  / df["low"])
    df["log_co"]     = np.log(df["close"] / df["open"])
    return df.dropna().reset_index(drop=True)


def resample_to_daily(df: pd.DataFrame, source_tf: str) -> pd.DataFrame:
    df = df.set_index("timestamp")
    agg = {"log_return": "sum", "log_hl": "mean", "log_co": "last", "volume": "sum"}
    daily = df.resample("1D").agg(agg).dropna().reset_index()
    daily.columns = [f"{c}_{source_tf}" if c != "timestamp" else c for c in daily.columns]
    return daily


def main():
    cfg = load_config()
    raw = cfg["data"]["raw_dir"]
    out = cfg["data"]["processed_dir"]
    tfs = cfg["data"]["timeframes"]
    end = pd.Timestamp(cfg["data"]["train_end"])
    Path(out).mkdir(exist_ok=True)

    daily_frames = []
    for tf in tfs:
        path = os.path.join(raw, f"XBRUSD_{tf}.csv")
        if not os.path.exists(path):
            print(f"Missing {path}, skipping.")
            continue
        df = load_and_clean(path)
        df = compute_features(df)
        daily = resample_to_daily(df, tf)
        daily_frames.append(daily)
        print(f"{tf}: {len(daily)} daily rows")

    # Merge all TFs on timestamp — inner join keeps only days all TFs agree on
    merged = daily_frames[0]
    for frame in daily_frames[1:]:
        merged = pd.merge(merged, frame, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    train = merged[merged["timestamp"] <= end].reset_index(drop=True)
    feature_cols = [c for c in train.columns if c != "timestamp"]
    n_features   = len(feature_cols)

    print(f"Train rows: {len(train)} | Features: {n_features} | "
          f"{train['timestamp'].iloc[0]} → {train['timestamp'].iloc[-1]}")
    print(f"Features: {feature_cols}")

    # Save feature column names so inference.py knows the order
    import json
    with open(os.path.join(out, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    # Single multivariate entry: target shape (time, n_features) flattened row-major
    # We store as list-of-lists so each row is one timestep's feature vector
    target_matrix = train[feature_cols].values.astype(np.float32)  # (T, n_features)

    schema = pa.schema([
        ("item_id",    pa.string()),
        ("start",      pa.timestamp("s")),
        ("target",     pa.list_(pa.list_(pa.float32()))),  # (T, n_features)
        ("freq",       pa.string()),
        ("n_features", pa.int32()),
    ])
    table = pa.table({
        "item_id":    pa.array(["XBRUSD_multivariate"],  type=pa.string()),
        "start":      pa.array([train["timestamp"].iloc[0]], type=pa.timestamp("s")),
        "target":     pa.array([target_matrix.tolist()],     type=pa.list_(pa.list_(pa.float32()))),
        "freq":       pa.array(["1D"],                       type=pa.string()),
        "n_features": pa.array([n_features],                 type=pa.int32()),
    }, schema=schema)

    out_path = os.path.join(out, "train.parquet")
    pq.write_table(table, out_path)
    print(f"Saved multivariate series ({len(train)} timesteps × {n_features} features) → {out_path}")


if __name__ == "__main__":
    main()
