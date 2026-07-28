"""
inference.py — Phase-aware signal generation.
Reads phase_meta.json to know which TF and features were used in training.
Writes signal.json for the cTrader bot.
"""

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from peft import PeftModel
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

RESAMPLE_RULES = {"M5": "5min", "M1": "1min", "H1": "1h", "D1": "1D"}


def load_config():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    phase     = cfg["training"]["phase"]
    phase_cfg = cfg["phases"][phase]
    cfg["_phase"]      = phase
    cfg["_ctx"]        = phase_cfg["context_length"]
    cfg["_pred"]       = phase_cfg["prediction_length"]
    cfg["_patch"]      = phase_cfg["patch_size"]
    cfg["_best_model"] = phase_cfg["best_model"]
    return cfg


def load_features(cfg: dict) -> np.ndarray:
    out = cfg["data"]["processed_dir"]
    raw = cfg["data"]["raw_dir"]

    with open(os.path.join(out, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(out, "phase_meta.json")) as f:
        meta = json.load(f)

    rule = meta["rule"]
    frames = []
    for tf in cfg["data"]["timeframes"]:
        path = os.path.join(raw, f"XBRUSD_{tf}.csv")
        df   = pd.read_csv(path, parse_dates=["timestamp"])
        df   = df.dropna().sort_values("timestamp").reset_index(drop=True)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["log_hl"]     = np.log(df["high"]  / df["low"])
        df["log_co"]     = np.log(df["close"] / df["open"])
        df = df.set_index("timestamp")
        resampled = df.resample(rule).agg(
            {"log_return": "sum", "log_hl": "mean", "log_co": "last", "volume": "sum"}
        ).dropna().reset_index()
        resampled.columns = [f"{c}_{tf}" if c != "timestamp" else c
                             for c in resampled.columns]
        frames.append(resampled)

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged[feature_cols].values.astype(np.float32), feature_cols


def compute_signal(forecasts: np.ndarray) -> dict:
    # forecasts: (samples, pred_len, n_feat) — channel 0 = log_return of primary TF
    ret  = forecasts[:, :, 0]
    med  = np.median(ret, axis=0)
    q10  = np.percentile(ret, 10, axis=0)
    q90  = np.percentile(ret, 90, axis=0)

    cum_ret    = float(np.sum(med))
    iw         = float(np.mean(q90 - q10))
    confidence = min(1.0, float(np.mean(np.abs(med))) / iw) if iw > 1e-8 else 1.0
    score      = round(float(np.tanh(cum_ret * 10)) * confidence, 4)

    return {
        "score":          score,
        "action":         "LONG" if score > 0.3 else "SHORT" if score < -0.3 else "FLAT",
        "cum_return":     round(cum_ret, 6),
        "confidence":     round(confidence, 4),
        "interval_width": round(iw, 6),
        "phase":          None,   # filled below
        "timestamp":      pd.Timestamp.utcnow().isoformat(),
    }


def main():
    cfg    = load_config()
    device = torch.device("cpu")
    torch.set_num_threads(cfg["training"]["cpu_threads"])

    best_path = cfg["_best_model"]
    if not os.path.exists(best_path):
        print(f"No fine-tuned model at {best_path}. Run finetune.py first.")
        return

    ctx_len = cfg["_ctx"]
    n_feat  = cfg["phases"][cfg["_phase"]]["best_model"]   # resolved below via meta
    data, feature_cols = load_features(cfg)
    n_feat  = data.shape[1]

    if len(data) < ctx_len:
        print(f"Not enough data: need {ctx_len}, have {len(data)}")
        return

    module = MoiraiModule.from_pretrained(cfg["paths"]["base_model"])
    base   = MoiraiForecast(
        module=module,
        prediction_length=cfg["_pred"],
        context_length=ctx_len,
        patch_size=cfg["_patch"],
        num_samples=200,
        target_dim=n_feat,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    model = PeftModel.from_pretrained(base, best_path).to(device)
    model.eval()

    context  = torch.tensor(data[-ctx_len:], dtype=torch.float32).unsqueeze(0)
    observed = torch.ones(1, ctx_len, n_feat, dtype=torch.bool)
    is_pad   = torch.zeros(1, ctx_len, dtype=torch.bool)

    with torch.no_grad():
        forecasts = model(
            past_target=context,
            past_observed_target=observed,
            past_is_pad=is_pad,
            num_samples=200,
        ).squeeze(0).numpy()   # (samples, pred_len, n_feat)

    result = compute_signal(forecasts)
    result["phase"] = cfg["_phase"]

    signal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal.json")
    with open(signal_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*42}")
    print(f"  Phase      : {result['phase']}")
    print(f"  Score      : {result['score']:+.4f}")
    print(f"  Action     : {result['action']}")
    print(f"  Confidence : {result['confidence']:.4f}")
    print(f"  Signal     → {signal_path}")
    print(f"{'='*42}\n")


if __name__ == "__main__":
    main()
