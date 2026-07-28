"""
test_pipeline.py
----------------
Tests every critical function in the pipeline.
Run with: conda run -n finetune_moirai_env python test_pipeline.py
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq

PASS = "  ✓"
FAIL = "  ✗"


def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))
    return condition


# ── 1. Config loading ─────────────────────────────────────────────────────────

def test_config():
    section("1. Config loading & phase resolution")
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    ok = True
    ok &= check("config loads",          cfg is not None)
    ok &= check("phase key exists",      "phase" in cfg["training"])
    ok &= check("phases 1-4 defined",    all(p in cfg["phases"] for p in [1,2,3,4]))

    phase = cfg["training"]["phase"]
    pc    = cfg["phases"][phase]
    ok &= check(f"current phase={phase} has primary_tf", "primary_tf" in pc)
    ok &= check("context_length > 0",    pc["context_length"] > 0)
    ok &= check("prediction_length > 0", pc["prediction_length"] > 0)
    ok &= check("best_model path set",   len(pc["best_model"]) > 0)
    return ok


# ── 2. Feature engineering ────────────────────────────────────────────────────

def test_feature_engineering():
    section("2. Feature engineering")
    from prepare_data import compute_features, load_and_clean

    # Build synthetic OHLCV
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open":   np.random.uniform(90, 100, n),
        "high":   np.random.uniform(100, 110, n),
        "low":    np.random.uniform(80, 90, n),
        "close":  np.random.uniform(90, 100, n),
        "volume": np.random.randint(100, 1000, n),
        "spread": np.full(n, 0.01),
    })

    ok = True
    df2 = compute_features(df)
    ok &= check("log_return computed",  "log_return" in df2.columns)
    ok &= check("log_hl computed",      "log_hl" in df2.columns)
    ok &= check("log_co computed",      "log_co" in df2.columns)
    ok &= check("no NaN after compute", df2[["log_return","log_hl","log_co"]].isna().sum().sum() == 0)
    ok &= check("log_hl >= 0",          (df2["log_hl"] >= 0).all())
    return ok


# ── 3. Resampling ─────────────────────────────────────────────────────────────

def test_resampling():
    section("3. Resampling to primary TF frequency")
    from prepare_data import compute_features, resample_to_freq

    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open":   np.random.uniform(90, 100, n),
        "high":   np.random.uniform(100, 110, n),
        "low":    np.random.uniform(80, 90, n),
        "close":  np.random.uniform(90, 100, n),
        "volume": np.random.randint(100, 1000, n),
        "spread": np.full(n, 0.01),
    })
    df = compute_features(df)

    ok = True
    resampled = resample_to_freq(df, "M1", "5min")
    ok &= check("resampled has fewer rows than input", len(resampled) < len(df))
    ok &= check("columns suffixed with _M1",
                all(c == "timestamp" or c.endswith("_M1") for c in resampled.columns))
    ok &= check("no NaN in resampled",
                resampled.drop(columns="timestamp").isna().sum().sum() == 0)
    return ok


# ── 4. Parquet save/load ──────────────────────────────────────────────────────

def test_parquet():
    section("4. Parquet save and load")
    import pyarrow as pa
    import pyarrow.parquet as pq

    T, F = 300, 16
    matrix = np.random.randn(T, F).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.parquet")
        schema = pa.schema([
            ("item_id",    pa.string()),
            ("start",      pa.timestamp("s")),
            ("target",     pa.list_(pa.list_(pa.float32()))),
            ("freq",       pa.string()),
            ("n_features", pa.int32()),
        ])
        table = pa.table({
            "item_id":    pa.array(["test"],                    type=pa.string()),
            "start":      pa.array([pd.Timestamp("2020-01-01")],type=pa.timestamp("s")),
            "target":     pa.array([matrix.tolist()],           type=pa.list_(pa.list_(pa.float32()))),
            "freq":       pa.array(["5min"],                    type=pa.string()),
            "n_features": pa.array([F],                         type=pa.int32()),
        }, schema=schema)
        pq.write_table(table, path)

        loaded = pq.read_table(path)
        raw    = np.array(loaded.column("target")[0].as_py(), dtype=np.float32)

    ok = True
    ok &= check("shape preserved",    raw.shape == (T, F), f"{raw.shape}")
    ok &= check("values preserved",   np.allclose(raw, matrix, atol=1e-5))
    ok &= check("n_features correct", int(loaded.column("n_features")[0].as_py()) == F)
    return ok


# ── 5. Dataset windowing ──────────────────────────────────────────────────────

def test_dataset():
    section("5. Dataset windowing")
    from finetune import MoiraiDataset

    T, F, CTX, PRED = 400, 16, 50, 5
    matrix = np.random.randn(T, F).astype(np.float32)

    import pyarrow as pa, pyarrow.parquet as pq
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "train.parquet")
        schema = pa.schema([
            ("item_id",    pa.string()),
            ("start",      pa.timestamp("s")),
            ("target",     pa.list_(pa.list_(pa.float32()))),
            ("freq",       pa.string()),
            ("n_features", pa.int32()),
        ])
        pq.write_table(pa.table({
            "item_id":    pa.array(["x"],                       type=pa.string()),
            "start":      pa.array([pd.Timestamp("2020-01-01")],type=pa.timestamp("s")),
            "target":     pa.array([matrix.tolist()],           type=pa.list_(pa.list_(pa.float32()))),
            "freq":       pa.array(["5min"],                    type=pa.string()),
            "n_features": pa.array([F],                         type=pa.int32()),
        }, schema=schema), path)

        ds = MoiraiDataset(path, CTX, PRED)

    ok = True
    expected_samples = T - (CTX + PRED) + 1
    ok &= check("sample count correct",  len(ds) == expected_samples, f"{len(ds)}")
    ok &= check("n_feat correct",        ds.n_feat == F)

    target, observed, is_pad = ds[0]
    ok &= check("target shape",   tuple(target.shape)   == (CTX + PRED, F))
    ok &= check("observed shape", tuple(observed.shape) == (CTX + PRED, F))
    ok &= check("is_pad shape",   tuple(is_pad.shape)   == (CTX + PRED,))
    ok &= check("observed all True",  observed.all().item())
    ok &= check("is_pad all False",   (~is_pad).all().item())
    return ok


# ── 6. Resume state ───────────────────────────────────────────────────────────

def test_resume_state():
    section("6. Resume state save/load")
    from finetune import load_resume_state, save_resume_state

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "resume.json")

        # Fresh state
        state = load_resume_state(path)
        ok &= check("fresh state epoch=0",     state["epoch"] == 0)
        ok &= check("fresh state best=inf",    state["best_loss"] == float("inf"))

        # Save and reload
        save_resume_state(path, {"phase": 1, "epoch": 42, "best_loss": 0.1234})
        state2 = load_resume_state(path)
        ok &= check("epoch persists",      state2["epoch"] == 42)
        ok &= check("best_loss persists",  abs(state2["best_loss"] - 0.1234) < 1e-6)
        ok &= check("phase persists",      state2["phase"] == 1)
    return ok


# ── 7. Signal computation ─────────────────────────────────────────────────────

def test_signal():
    section("7. Signal computation")
    from inference import compute_signal

    ok = True

    # Strong upward signal — narrow interval
    forecasts_up = np.random.normal(loc=0.01, scale=0.001, size=(200, 5, 16)).astype(np.float32)
    sig_up = compute_signal(forecasts_up)
    ok &= check("strong up → LONG",       sig_up["action"] == "LONG",  f"score={sig_up['score']}")
    ok &= check("score in [-1,1]",        -1 <= sig_up["score"] <= 1)
    ok &= check("confidence in [0,1]",    0 <= sig_up["confidence"] <= 1)

    # Strong downward signal
    forecasts_dn = np.random.normal(loc=-0.01, scale=0.001, size=(200, 5, 16)).astype(np.float32)
    sig_dn = compute_signal(forecasts_dn)
    ok &= check("strong down → SHORT",    sig_dn["action"] == "SHORT", f"score={sig_dn['score']}")

    # Wide interval → flat
    forecasts_flat = np.random.normal(loc=0.001, scale=0.05, size=(200, 5, 16)).astype(np.float32)
    sig_flat = compute_signal(forecasts_flat)
    ok &= check("wide interval → FLAT",   sig_flat["action"] == "FLAT", f"score={sig_flat['score']}")

    # Required keys present
    required = {"score", "action", "cum_return", "confidence", "interval_width", "timestamp"}
    ok &= check("all keys present",       required.issubset(sig_up.keys()))
    return ok


# ── 8. Phase chaining logic ───────────────────────────────────────────────────

def test_phase_chaining():
    section("8. Phase chaining — base model selection")
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    ok = True
    for phase in [1, 2, 3, 4]:
        prev = phase - 1
        prev_best = cfg["phases"].get(prev, {}).get("best_model", "")
        # Phase 1 should fall back to model_cache (prev_best doesn't exist on disk yet)
        if phase == 1:
            expected_base = cfg["paths"]["base_model"]
            actual_base   = prev_best if prev_best and os.path.exists(prev_best) \
                            else cfg["paths"]["base_model"]
            ok &= check(f"phase 1 base = model_cache", actual_base == expected_base)
        else:
            ok &= check(f"phase {phase} prev_best path defined",
                        len(prev_best) > 0, prev_best)
    return ok


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("\n" + "="*50)
    print("  Moirai Pipeline Test Suite")
    print("="*50)

    results = {
        "Config":              test_config(),
        "Feature engineering": test_feature_engineering(),
        "Resampling":          test_resampling(),
        "Parquet I/O":         test_parquet(),
        "Dataset windowing":   test_dataset(),
        "Resume state":        test_resume_state(),
        "Signal computation":  test_signal(),
        "Phase chaining":      test_phase_chaining(),
    }

    print(f"\n{'='*50}")
    print("  Summary")
    print(f"{'='*50}")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_pass &= passed

    print(f"{'='*50}")
    print(f"  {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print(f"{'='*50}\n")
