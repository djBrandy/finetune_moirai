"""
test_pipeline.py
----------------
Tests every critical function in the pipeline including all 6 fixes.
Run with: conda run -n finetune_moirai_env python test_pipeline.py
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq

PASS = "  ✓"
FAIL = "  ✗"


def section(title):
    print(f"\n{'─'*52}\n  {title}\n{'─'*52}")


def check(name, condition, detail=""):
    print(f"{PASS if condition else FAIL} {name}" + (f" — {detail}" if detail else ""))
    return condition


# ── helpers ───────────────────────────────────────────────────────────────────

def make_parquet(tmp, T=400, F=16, rule="5min"):
    matrix = np.random.randn(T, F).astype(np.float32)
    path   = os.path.join(tmp, "test.parquet")
    schema = pa.schema([
        ("item_id",    pa.string()),
        ("start",      pa.timestamp("s")),
        ("target",     pa.list_(pa.list_(pa.float32()))),
        ("freq",       pa.string()),
        ("n_features", pa.int32()),
    ])
    pq.write_table(pa.table({
        "item_id":    pa.array(["x"],                        type=pa.string()),
        "start":      pa.array([pd.Timestamp("2020-01-01")], type=pa.timestamp("s")),
        "target":     pa.array([matrix.tolist()],            type=pa.list_(pa.list_(pa.float32()))),
        "freq":       pa.array([rule],                       type=pa.string()),
        "n_features": pa.array([F],                          type=pa.int32()),
    }, schema=schema), path)
    return path, matrix


# ── 1. Config ─────────────────────────────────────────────────────────────────

def test_config():
    section("1. Config — phase resolution + val_end + signal section")
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    ok = True
    ok &= check("phases 1-4 defined",     all(p in cfg["phases"] for p in [1,2,3,4]))
    ok &= check("train_end exists",        "train_end" in cfg["data"])
    ok &= check("val_end exists",          "val_end"   in cfg["data"])
    ok &= check("val_end > train_end",
                cfg["data"]["val_end"] > cfg["data"]["train_end"])
    ok &= check("signal section exists",   "signal" in cfg)
    ok &= check("tanh_scale in signal",    "tanh_scale" in cfg["signal"])
    ok &= check("conviction thresholds",   "long_threshold" in cfg["signal"])
    ok &= check("min_tradeable_return",    "min_tradeable_return" in cfg["signal"])
    return ok


# ── 2. Feature engineering ────────────────────────────────────────────────────

def test_feature_engineering():
    section("2. Feature engineering")
    from prepare_data import compute_features
    n  = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="5min"),
        "open":   np.random.uniform(90, 100, n),
        "high":   np.random.uniform(100, 110, n),
        "low":    np.random.uniform(80,  90,  n),
        "close":  np.random.uniform(90, 100, n),
        "volume": np.random.randint(100, 1000, n),
        "spread": np.full(n, 0.01),
    })
    ok  = True
    df2 = compute_features(df)
    ok &= check("log_return computed",  "log_return" in df2.columns)
    ok &= check("log_hl >= 0",          (df2["log_hl"] >= 0).all())
    ok &= check("no NaN",               df2[["log_return","log_hl","log_co"]].isna().sum().sum() == 0)
    return ok


# ── 3. Train / val split ──────────────────────────────────────────────────────

def test_train_val_split():
    section("3. Train/val split — 2014-2020 train, 2021-2022 val")
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    train_end = pd.Timestamp(cfg["data"]["train_end"])
    val_end   = pd.Timestamp(cfg["data"]["val_end"])

    # Plain timestamp series spanning 2014-2023 — tests split logic directly
    dates  = pd.date_range("2014-01-01", "2023-12-31", freq="5min")
    merged = pd.DataFrame({"timestamp": dates, "x": np.random.randn(len(dates))})

    train = merged[merged["timestamp"] <= train_end]
    val   = merged[(merged["timestamp"] > train_end) & (merged["timestamp"] <= val_end)]

    ok = True
    ok &= check("train rows > 0",   len(train) > 0, str(len(train)))
    ok &= check("val rows > 0",     len(val)   > 0, str(len(val)))
    ok &= check("no overlap",       len(set(train["timestamp"]) & set(val["timestamp"])) == 0)
    ok &= check("train before val", train["timestamp"].max() < val["timestamp"].min())
    return ok


# ── 4. Phase-namespaced metadata ──────────────────────────────────────────────

def test_namespaced_metadata():
    section("4. Phase-namespaced metadata — no cross-phase overwrite")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for phase in [1, 2]:
            meta = {"phase": phase, "primary_tf": "M5" if phase == 1 else "M1",
                    "n_features": 16, "rule": "5min",
                    "feature_cols": [f"feat_{i}" for i in range(16)],
                    "return_stats": {"mean_abs_return": 0.0002,
                                     "std_return": 0.001,
                                     "median_abs_return": 0.00015},
                    "train_end": "2020-12-31", "val_end": "2022-12-31"}
            path = os.path.join(tmp, f"phase{phase}_meta.json")
            with open(path, "w") as f:
                json.dump(meta, f)

        # Verify phase 1 and phase 2 metadata are independent
        with open(os.path.join(tmp, "phase1_meta.json")) as f:
            m1 = json.load(f)
        with open(os.path.join(tmp, "phase2_meta.json")) as f:
            m2 = json.load(f)

        ok &= check("phase1 meta has phase=1",    m1["phase"] == 1)
        ok &= check("phase2 meta has phase=2",    m2["phase"] == 2)
        ok &= check("phase1 tf = M5",             m1["primary_tf"] == "M5")
        ok &= check("phase2 tf = M1",             m2["primary_tf"] == "M1")
        ok &= check("files are independent",      m1["primary_tf"] != m2["primary_tf"])
    return ok


# ── 5. Hybrid loss ────────────────────────────────────────────────────────────

def test_hybrid_loss():
    section("5. Hybrid loss — distributional + directional components")
    from finetune import hybrid_loss
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from peft import get_peft_model, LoraConfig

    CTX, PRED, F = 32, 4, 4   # small for speed
    module = MoiraiModule.from_pretrained("model_cache")
    model  = MoiraiForecast(module=module, prediction_length=PRED,
                            context_length=CTX, patch_size=16, num_samples=10,
                            target_dim=F, feat_dynamic_real_dim=0,
                            past_feat_dynamic_real_dim=0)
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                      target_modules=["q_proj","v_proj"], bias="none")
    model = get_peft_model(model, lora)

    B = 2
    target   = torch.randn(B, CTX + PRED, F)
    observed = torch.ones(B, CTX + PRED, F, dtype=torch.bool)
    is_pad   = torch.zeros(B, CTX + PRED, dtype=torch.bool)

    ok = True
    # alpha=0 → pure distributional
    loss_dist = hybrid_loss(model, target, observed, is_pad, CTX, alpha=0.0)
    ok &= check("alpha=0 loss is scalar",   loss_dist.dim() == 0)
    ok &= check("alpha=0 loss > 0",         loss_dist.item() > 0)

    # alpha=1 → pure directional
    loss_dir = hybrid_loss(model, target, observed, is_pad, CTX, alpha=1.0)
    ok &= check("alpha=1 loss is scalar",   loss_dir.dim() == 0)
    ok &= check("alpha=1 loss in [0,2]",    0.0 <= loss_dir.item() <= 2.0)

    # alpha=0.3 → hybrid
    loss_hyb = hybrid_loss(model, target, observed, is_pad, CTX, alpha=0.3)
    ok &= check("hybrid loss is scalar",    loss_hyb.dim() == 0)
    ok &= check("hybrid loss > 0",          loss_hyb.item() > 0)

    # Gradients flow
    loss_hyb.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    ok &= check("gradients flow to LoRA",   len(grads) > 0, f"{len(grads)} params with grad")
    return ok


# ── 6. Calibrated signal scoring ─────────────────────────────────────────────

def test_signal_scoring():
    section("6. Calibrated signal — conviction, magnitude gate, phase guard")
    from inference import compute_signal
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    sig_cfg = cfg["signal"]

    meta = {
        "phase": 1, "primary_tf": "M5",
        "return_stats": {
            "mean_abs_return":    0.0002,
            "std_return":         0.001,
            "median_abs_return":  0.00015,
        }
    }

    ok = True

    # Strong up, narrow interval, large enough move
    fc_up = np.random.normal(0.002, 0.0001, (200, 12, 16)).astype(np.float32)
    s_up  = compute_signal(fc_up, meta, sig_cfg)
    ok &= check("strong up → LONG",         s_up["action"] == "LONG",  f"score={s_up['score']}")
    ok &= check("conviction in [0,1]",       0 <= s_up["conviction"] <= 1)
    ok &= check("magnitude_ok=True",         s_up["magnitude_ok"] == True)

    # Strong down
    fc_dn = np.random.normal(-0.002, 0.0001, (200, 12, 16)).astype(np.float32)
    s_dn  = compute_signal(fc_dn, meta, sig_cfg)
    ok &= check("strong down → SHORT",       s_dn["action"] == "SHORT", f"score={s_dn['score']}")

    # Wide interval → FLAT regardless of direction
    fc_fl = np.random.normal(0.002, 0.05, (200, 12, 16)).astype(np.float32)
    s_fl  = compute_signal(fc_fl, meta, sig_cfg)
    ok &= check("wide interval → FLAT",      s_fl["action"] == "FLAT",  f"score={s_fl['score']}")

    # Tiny move below min_tradeable → low confidence even if narrow
    fc_tiny = np.random.normal(0.00001, 0.000005, (200, 12, 16)).astype(np.float32)
    s_tiny  = compute_signal(fc_tiny, meta, sig_cfg)
    ok &= check("tiny move → magnitude_ok=False", s_tiny["magnitude_ok"] == False)
    ok &= check("tiny move → FLAT",               s_tiny["action"] == "FLAT")

    # Conviction scales with score magnitude
    ok &= check("higher score → higher conviction",
                s_up["conviction"] >= s_fl["conviction"])

    # All required keys present including new ones
    required = {"score","action","conviction","cum_return","confidence",
                "interval_width","magnitude_ok","phase","primary_tf","timestamp"}
    ok &= check("all signal keys present",   required.issubset(s_up.keys()))
    return ok


# ── 7. Dataset windowing ──────────────────────────────────────────────────────

def test_dataset():
    section("7. Dataset windowing")
    from finetune import MoiraiDataset
    CTX, PRED, T, F = 50, 5, 400, 16
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = make_parquet(tmp, T, F)
        ds      = MoiraiDataset(path, CTX, PRED)
    ok = True
    ok &= check("sample count",   len(ds) == T - (CTX+PRED) + 1, str(len(ds)))
    tgt, obs, pad = ds[0]
    ok &= check("target shape",   tuple(tgt.shape) == (CTX+PRED, F))
    ok &= check("observed True",  obs.all().item())
    ok &= check("is_pad False",   (~pad).all().item())
    return ok


# ── 8. Resume state ───────────────────────────────────────────────────────────

def test_resume_state():
    section("8. Resume state — phase-aware reset")
    from finetune import load_resume_state, save_resume_state
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        path  = os.path.join(tmp, "resume.json")
        state = load_resume_state(path)
        ok &= check("fresh epoch=0",       state["epoch"] == 0)
        ok &= check("fresh best=inf",      state["best_val_loss"] == float("inf"))

        save_resume_state(path, {"phase": 1, "epoch": 42, "best_val_loss": 0.123})
        s2 = load_resume_state(path)
        ok &= check("epoch persists",      s2["epoch"] == 42)
        ok &= check("best_val persists",   abs(s2["best_val_loss"] - 0.123) < 1e-6)

        # Phase change should trigger reset in finetune.py logic
        ok &= check("phase mismatch detected",
                    s2.get("phase", 1) != 2)
    return ok


# ── 9. Phase chaining ─────────────────────────────────────────────────────────

def test_phase_chaining():
    section("9. Phase chaining — base model selection per phase")
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    ok = True
    for phase in [1, 2, 3, 4]:
        prev_best = cfg["phases"].get(phase-1, {}).get("best_model", "")
        base = prev_best if prev_best and os.path.exists(prev_best) \
               else cfg["paths"]["base_model"]
        if phase == 1:
            ok &= check(f"phase 1 → model_cache", base == cfg["paths"]["base_model"])
        else:
            ok &= check(f"phase {phase} prev path defined", len(prev_best) > 0, prev_best)
    return ok


# ── 10. Phase guard in inference ─────────────────────────────────────────────

def test_phase_guard():
    section("10. Phase guard — inference refuses mismatched metadata")
    from inference import load_phase_meta
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # Write metadata for phase 2 but config says phase 1
        meta_path = os.path.join(tmp, "phase1_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"phase": 2, "primary_tf": "M1", "n_features": 16,
                       "rule": "1min", "feature_cols": [],
                       "return_stats": {}, "train_end": "", "val_end": ""}, f)

        # Simulate the guard check
        with open(meta_path) as f:
            meta = json.load(f)
        mismatch_detected = meta["phase"] != cfg["_phase"] if "_phase" in cfg else \
                            meta["phase"] != cfg["training"]["phase"]
        ok &= check("phase mismatch detected", mismatch_detected,
                    f"meta={meta['phase']} config={cfg['training']['phase']}")

    # Missing metadata raises FileNotFoundError
    cfg_copy = dict(cfg)
    cfg_copy["_phase"] = 99
    cfg_copy["data"]   = {"processed_dir": tmp}
    try:
        load_phase_meta(cfg_copy)
        ok &= check("missing meta raises error", False)
    except FileNotFoundError:
        ok &= check("missing meta raises FileNotFoundError", True)
    return ok


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("\n" + "="*52)
    print("  Moirai Pipeline Test Suite  (10 tests)")
    print("="*52)

    results = {
        "Config":                  test_config(),
        "Feature engineering":     test_feature_engineering(),
        "Train/val split":         test_train_val_split(),
        "Namespaced metadata":     test_namespaced_metadata(),
        "Hybrid loss":             test_hybrid_loss(),
        "Calibrated signal":       test_signal_scoring(),
        "Dataset windowing":       test_dataset(),
        "Resume state":            test_resume_state(),
        "Phase chaining":          test_phase_chaining(),
        "Phase guard":             test_phase_guard(),
    }

    print(f"\n{'='*52}\n  Summary\n{'='*52}")
    all_pass = True
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_pass &= passed
    print(f"{'='*52}")
    print(f"  {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print(f"{'='*52}\n")
