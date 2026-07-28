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


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["log_hl"]     = np.log(df["high"] / df["low"])          # volatility proxy
    df["log_co"]     = np.log(df["close"] / df["open"])        # intrabar direction
    return df.dropna().reset_index(drop=True)


def resample_to_daily(df: pd.DataFrame, source_tf: str) -> pd.DataFrame:
    """Resample sub-daily TFs to daily, aggregating into single feature columns."""
    df = df.set_index("timestamp")
    agg = {
        "log_return": "sum",
        "log_hl":     "mean",
        "log_co":     "last",
        "volume":     "sum",
    }
    daily = df.resample("1D").agg(agg).dropna().reset_index()
    daily.columns = [f"{c}_{source_tf}" if c != "timestamp" else c for c in daily.columns]
    return daily


def build_gluonts_entry(series: np.ndarray, start: pd.Timestamp, freq: str, item_id: str) -> dict:
    return {
        "item_id":   item_id,
        "start":     start,
        "target":    series.astype(np.float32),
        "freq":      freq,
    }


def save_arrow(entries: list, path: str):
    schema = pa.schema([
        ("item_id", pa.string()),
        ("start",   pa.timestamp("s")),
        ("target",  pa.list_(pa.float32())),
        ("freq",    pa.string()),
    ])
    arrays = {
        "item_id": pa.array([e["item_id"] for e in entries], type=pa.string()),
        "start":   pa.array([e["start"]   for e in entries], type=pa.timestamp("s")),
        "target":  pa.array([e["target"].tolist() for e in entries], type=pa.list_(pa.float32())),
        "freq":    pa.array([e["freq"]    for e in entries], type=pa.string()),
    }
    table = pa.table(arrays, schema=schema)
    pq.write_table(table, path)
    print(f"Saved {len(entries)} series → {path}")


def main():
    cfg  = load_config()
    raw  = cfg["data"]["raw_dir"]
    out  = cfg["data"]["processed_dir"]
    tfs  = cfg["data"]["timeframes"]
    end  = pd.Timestamp(cfg["data"]["train_end"])
    Path(out).mkdir(exist_ok=True)

    # Load all timeframes, compute features, resample to daily
    daily_frames = []
    for tf in tfs:
        path = os.path.join(raw, f"XBRUSD_{tf}.csv")
        if not os.path.exists(path):
            print(f"Missing {path}, skipping.")
            continue
        df = load_and_clean(path)
        df = compute_log_returns(df)
        daily = resample_to_daily(df, tf)
        daily_frames.append(daily)
        print(f"{tf}: {len(daily)} daily rows after resampling")

    # Merge all TFs on timestamp
    merged = daily_frames[0]
    for frame in daily_frames[1:]:
        merged = pd.merge(merged, frame, on="timestamp", how="inner")

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    train = merged[merged["timestamp"] <= end]
    print(f"Train rows: {len(train)} | from {train['timestamp'].iloc[0]} to {train['timestamp'].iloc[-1]}")

    # Build one GluonTS entry per feature column
    feature_cols = [c for c in train.columns if c != "timestamp"]
    entries = []
    for col in feature_cols:
        series = train[col].values
        entries.append(build_gluonts_entry(
            series=series,
            start=train["timestamp"].iloc[0],
            freq="1D",
            item_id=f"XBRUSD_{col}",
        ))

    save_arrow(entries, os.path.join(out, "train.parquet"))
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
