"""
inference.py
------------
Loads fine-tuned Moirai, runs forecast on latest XBRUSD data,
writes signal.json for the cTrader bot to consume.
"""

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from peft import PeftModel
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_features(cfg: dict) -> tuple[np.ndarray, list[str]]:
    raw = cfg["data"]["raw_dir"]
    out = cfg["data"]["processed_dir"]

    with open(os.path.join(out, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    frames = []
    for tf in cfg["data"]["timeframes"]:
        path = os.path.join(raw, f"XBRUSD_{tf}.csv")
        df   = pd.read_csv(path, parse_dates=["timestamp"])
        df   = df.dropna().sort_values("timestamp").reset_index(drop=True)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["log_hl"]     = np.log(df["high"]  / df["low"])
        df["log_co"]     = np.log(df["close"] / df["open"])
        df = df.set_index("timestamp")
        agg   = {"log_return": "sum", "log_hl": "mean", "log_co": "last", "volume": "sum"}
        daily = df.resample("1D").agg(agg).dropna().reset_index()
        daily.columns = [f"{c}_{tf}" if c != "timestamp" else c for c in daily.columns]
        frames.append(daily)

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    data = merged[feature_cols].values.astype(np.float32)
    return data, feature_cols


def compute_signal(forecasts: np.ndarray) -> dict:
    """
    forecasts: (num_samples, pred_len, n_features)
    Use the log_return_D1 channel (index 0) as the primary signal.
    """
    ret_forecasts = forecasts[:, :, 0]              # (samples, pred_len) — D1 log returns
    median    = np.median(ret_forecasts, axis=0)
    q10       = np.percentile(ret_forecasts, 10, axis=0)
    q90       = np.percentile(ret_forecasts, 90, axis=0)

    cum_return     = float(np.sum(median))
    interval_width = float(np.mean(q90 - q10))
    avg_abs_return = float(np.mean(np.abs(median)))

    confidence = min(1.0, avg_abs_return / interval_width) if interval_width > 1e-8 else 1.0
    direction  = float(np.tanh(cum_return * 10))
    score      = round(direction * confidence, 4)

    if   score >  0.3: action = "LONG"
    elif score < -0.3: action = "SHORT"
    else:              action = "FLAT"

    return {
        "score":           score,
        "action":          action,
        "cum_return":      round(cum_return, 6),
        "confidence":      round(confidence, 4),
        "interval_width":  round(interval_width, 6),
        "timestamp":       pd.Timestamp.utcnow().isoformat(),
    }


def main():
    cfg    = load_config()
    device = torch.device("cpu")
    torch.set_num_threads(cfg["training"]["cpu_threads"])

    best_path = cfg["paths"]["best_model"]
    if not os.path.exists(best_path):
        print("No fine-tuned model found. Run finetune.py first.")
        return

    ctx_len  = cfg["model"]["context_length"]
    pred_len = cfg["model"]["prediction_length"]
    patch    = cfg["model"]["patch_size"]
    n_feat   = cfg["model"]["target_dim"]

    module = MoiraiModule.from_pretrained(cfg["model"]["name"])
    base   = MoiraiForecast(
        module=module,
        prediction_length=pred_len,
        context_length=ctx_len,
        patch_size=patch,
        num_samples=200,
        target_dim=n_feat,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    model = PeftModel.from_pretrained(base, best_path).to(device)
    model.eval()

    data, feature_cols = load_features(cfg)

    if len(data) < ctx_len:
        print(f"Not enough data: need {ctx_len} bars, have {len(data)}")
        return

    context  = torch.tensor(data[-ctx_len:], dtype=torch.float32).unsqueeze(0)  # (1, ctx, n_feat)
    observed = torch.ones(1, ctx_len, n_feat, dtype=torch.bool)
    is_pad   = torch.zeros(1, ctx_len, dtype=torch.bool)

    with torch.no_grad():
        forecasts = model(
            past_target=context,
            past_observed_target=observed,
            past_is_pad=is_pad,
            num_samples=200,
        )  # (1, samples, pred_len, n_feat)

    forecasts = forecasts.squeeze(0).numpy()  # (samples, pred_len, n_feat)
    result    = compute_signal(forecasts)

    # Write signal file for C# bot
    signal_path = os.path.join(os.path.dirname(__file__), "signal.json")
    with open(signal_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*40}")
    print(f"  Score  : {result['score']:+.4f}")
    print(f"  Action : {result['action']}")
    print(f"  Confidence : {result['confidence']:.4f}")
    print(f"  Signal written → {signal_path}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
