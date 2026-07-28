"""
inference.py — Phase-aware, calibrated signal generation.
Reads phase{N}_meta.json — never silently mismatches phases.
Writes signal.json with score, action, and conviction for C# position sizing.
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
    pc        = cfg["phases"][phase]
    cfg["_phase"]      = phase
    cfg["_ctx"]        = pc["context_length"]
    cfg["_pred"]       = pc["prediction_length"]
    cfg["_patch"]      = pc["patch_size"]
    cfg["_best_model"] = pc["best_model"]
    return cfg


def load_phase_meta(cfg: dict) -> dict:
    phase     = cfg["_phase"]
    meta_path = os.path.join(cfg["data"]["processed_dir"], f"phase{phase}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Phase metadata not found: {meta_path}\n"
            f"Run prepare_data.py for phase {phase} first."
        )
    with open(meta_path) as f:
        return json.load(f)


def load_features(cfg: dict, meta: dict) -> np.ndarray:
    raw          = cfg["data"]["raw_dir"]
    feature_cols = meta["feature_cols"]
    rule         = meta["rule"]

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

    # Strict column order check — prevents silent feature mismatch
    available = [c for c in merged.columns if c != "timestamp"]
    missing   = [c for c in feature_cols if c not in available]
    if missing:
        raise ValueError(f"Feature mismatch — missing columns: {missing}")

    return merged[feature_cols].values.astype(np.float32)


def compute_signal(forecasts: np.ndarray, meta: dict, sig_cfg: dict) -> dict:
    """
    forecasts: (samples, pred_len, n_feat)
    Channel 0 = log_return of primary TF.

    Calibrated scoring:
    - tanh_scale derived from actual XBRUSD return distribution (not arbitrary)
    - confidence accounts for both interval width AND minimum tradeable magnitude
    - conviction is a [0,1] value for C# position sizing
    """
    stats        = meta["return_stats"]
    mean_abs     = stats["mean_abs_return"]
    tanh_scale   = sig_cfg["tanh_scale"]
    min_trade    = sig_cfg["min_tradeable_return"]
    long_thr     = sig_cfg["long_threshold"]
    short_thr    = sig_cfg["short_threshold"]

    ret  = forecasts[:, :, 0]                          # (samples, pred_len)
    med  = np.median(ret, axis=0)                      # (pred_len,)
    q10  = np.percentile(ret, 10, axis=0)
    q90  = np.percentile(ret, 90, axis=0)

    cum_ret = float(np.sum(med))
    iw      = float(np.mean(q90 - q10))

    # Confidence: magnitude must exceed spread cost AND interval must be narrow
    # relative to the expected move — both conditions required
    magnitude_ok  = float(np.mean(np.abs(med))) > min_trade
    precision     = float(np.mean(np.abs(med))) / iw if iw > 1e-8 else 1.0
    confidence    = min(1.0, precision) * (1.0 if magnitude_ok else 0.3)

    # Direction: tanh scaled to actual return distribution
    direction = float(np.tanh(cum_ret * tanh_scale))
    score     = round(direction * confidence, 4)

    # Conviction: normalised [0,1] for C# position sizing
    # 0 = minimum lots, 1 = maximum lots
    conviction = round(min(1.0, abs(score) / max(abs(long_thr), abs(short_thr))), 4)

    if   score > long_thr:  action = "LONG"
    elif score < short_thr: action = "SHORT"
    else:                   action = "FLAT"

    return {
        "score":          score,
        "action":         action,
        "conviction":     conviction,        # C# uses this for position sizing
        "cum_return":     round(cum_ret, 6),
        "confidence":     round(confidence, 4),
        "interval_width": round(iw, 6),
        "magnitude_ok":   magnitude_ok,
        "phase":          meta["phase"],
        "primary_tf":     meta["primary_tf"],
        "timestamp":      pd.Timestamp.utcnow().isoformat(),
    }


def main():
    cfg  = load_config()
    meta = load_phase_meta(cfg)

    # Guard: ensure inference phase matches metadata phase
    if meta["phase"] != cfg["_phase"]:
        raise RuntimeError(
            f"Phase mismatch: config says phase {cfg['_phase']} "
            f"but metadata is phase {meta['phase']}. "
            f"Re-run prepare_data.py for phase {cfg['_phase']}."
        )

    device = torch.device("cpu")
    torch.set_num_threads(cfg["training"]["cpu_threads"])

    best_path = cfg["_best_model"]
    if not os.path.exists(best_path):
        print(f"No fine-tuned model at {best_path}. Run finetune.py first.")
        return

    data   = load_features(cfg, meta)
    n_feat = data.shape[1]
    ctx    = cfg["_ctx"]

    if len(data) < ctx:
        print(f"Not enough data: need {ctx}, have {len(data)}")
        return

    module = MoiraiModule.from_pretrained(cfg["paths"]["base_model"])
    base   = MoiraiForecast(
        module=module,
        prediction_length=cfg["_pred"],
        context_length=ctx,
        patch_size=cfg["_patch"],
        num_samples=200,
        target_dim=n_feat,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    model = PeftModel.from_pretrained(base, best_path).to(device)
    model.eval()

    context  = torch.tensor(data[-ctx:], dtype=torch.float32).unsqueeze(0)
    observed = torch.ones(1, ctx, n_feat, dtype=torch.bool)
    is_pad   = torch.zeros(1, ctx, dtype=torch.bool)

    with torch.no_grad():
        forecasts = model(
            past_target=context,
            past_observed_target=observed,
            past_is_pad=is_pad,
            num_samples=200,
        ).squeeze(0).numpy()   # (200, pred_len, n_feat)

    result = compute_signal(forecasts, meta, cfg["signal"])

    signal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal.json")
    with open(signal_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*44}")
    print(f"  Phase      : {result['phase']} ({result['primary_tf']})")
    print(f"  Score      : {result['score']:+.4f}")
    print(f"  Action     : {result['action']}")
    print(f"  Conviction : {result['conviction']:.4f}  ← C# uses for lot sizing")
    print(f"  Confidence : {result['confidence']:.4f}")
    print(f"  Mag OK     : {result['magnitude_ok']}")
    print(f"  Signal     → {signal_path}")
    print(f"{'='*44}\n")


if __name__ == "__main__":
    main()
