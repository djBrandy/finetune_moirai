"""
inference.py
------------
Loads the fine-tuned Moirai-large LoRA model and outputs a signal score
in [-1, +1] based on the latest XBRUSD data.

Usage:
    python inference.py
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from peft import PeftModel
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_series(cfg: dict) -> np.ndarray:
    """Load D1 log-returns — the primary signal series."""
    path = os.path.join(cfg["data"]["raw_dir"], "XBRUSD_D1.csv")
    df   = pd.read_csv(path, parse_dates=["timestamp"])
    df   = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df["log_return"].dropna().values.astype(np.float32)


def compute_signal(median: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> float:
    """
    Derive a [-1, +1] signal from Moirai's forecast distribution.

    - Direction: sign of cumulative median return over prediction horizon
    - Confidence: inversely proportional to interval width (narrow = confident)
    - Wide interval → score shrinks toward 0 (stay flat)
    """
    cum_return    = float(np.sum(median))
    interval_width = float(np.mean(q90 - q10))
    avg_abs_return = float(np.mean(np.abs(median)))

    # Confidence: 1 when interval is tight relative to expected move
    if interval_width < 1e-8:
        confidence = 1.0
    else:
        confidence = min(1.0, avg_abs_return / interval_width)

    direction = np.tanh(cum_return * 10)   # squash to [-1, +1]
    score     = float(direction * confidence)
    return round(score, 4)


def main():
    cfg    = load_config()
    device = torch.device("cpu")
    torch.set_num_threads(cfg["training"]["cpu_threads"])

    ctx_len  = cfg["model"]["context_length"]
    pred_len = cfg["model"]["prediction_length"]
    patch    = cfg["model"]["patch_size"]

    best_path = cfg["paths"]["best_model"]
    if not os.path.exists(best_path):
        print("No fine-tuned model found. Run finetune.py first.")
        return

    # Load base + LoRA weights
    module = MoiraiModule.from_pretrained(cfg["model"]["name"])
    base   = MoiraiForecast(
        module=module,
        prediction_length=pred_len,
        context_length=ctx_len,
        patch_size=patch,
        num_samples=200,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    model = PeftModel.from_pretrained(base, best_path).to(device)
    model.eval()

    series = load_series(cfg)
    if len(series) < ctx_len:
        print(f"Not enough data: need {ctx_len} bars, have {len(series)}")
        return

    context = torch.tensor(series[-ctx_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        forecasts = model.predict(past_target=context)   # (num_samples, pred_len)

    forecasts = forecasts.squeeze().numpy()              # (200, 5)
    median    = np.median(forecasts, axis=0)
    q10       = np.percentile(forecasts, 10, axis=0)
    q90       = np.percentile(forecasts, 90, axis=0)

    score = compute_signal(median, q10, q90)

    print(f"\n{'='*40}")
    print(f"  Signal score : {score:+.4f}")
    print(f"  Median 5-day : {median}")
    print(f"  80% interval : [{q10}, {q90}]")
    print(f"  Interpretation:")
    if   score >  0.3: print("  → LONG  (confident upward regime)")
    elif score < -0.3: print("  → SHORT (confident downward regime)")
    else:              print("  → FLAT  (uncertain, stay out)")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
