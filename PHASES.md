# Progressive Fine-Tuning Strategy — XBRUSD Moirai-Large

## Philosophy
Each phase fine-tunes from the *previous phase's best model*, not from scratch.
The model accumulates knowledge bottom-up: micro-structure → macro regime.
At the end of every phase the model is fully deployable for trading.

## Phase Schedule

| Phase | Primary TF | Windows    | Est. Duration | Prediction Length | Context Length |
|-------|-----------|------------|---------------|-------------------|----------------|
| 1     | M5        | ~470,000   | 3–5 weeks     | 12 bars (1hr)     | 288 bars (1day)|
| 2     | M1        | ~2,160,000 | 2–4 months    | 60 bars (1hr)     | 1440 bars(1day)|
| 3     | H1        | ~39,000    | 2–4 days      | 5 bars            | 120 bars       |
| 4     | D1        | ~1,607     | 6–14 hours    | 5 bars            | 256 bars       |

## What Each Phase Teaches

- **Phase 1 (M5):** Intraday volatility structure, session opens/closes,
  momentum bursts, mean-reversion at 5-minute resolution.
  Model becomes deployable here.

- **Phase 2 (M1):** Micro-structure — tick-level order flow patterns,
  precise entry/exit timing, fine-grained volatility clustering.
  Deepens Phase 1 without forgetting it.

- **Phase 3 (H1):** Session-level context — London/NY overlap dynamics,
  multi-hour trend structure, daily range formation.

- **Phase 4 (D1):** Macro regime awareness — weekly/monthly cycles,
  fundamental-driven multi-day trends, long-term mean reversion.

## How Phases Chain

Each phase reads `base_model` from config:
- Phase 1: base = `model_cache`  (raw Salesforce weights)
- Phase 2: base = `checkpoints/phase1_best`
- Phase 3: base = `checkpoints/phase2_best`
- Phase 4: base = `checkpoints/phase3_best`

The LoRA adapter is re-initialized each phase but the base module
carries all previously learned representations forward.

## Feature Engineering (all phases)

Each timeframe contributes 4 features resampled to the primary TF frequency:
- `log_return_{TF}` — directional momentum
- `log_hl_{TF}`     — volatility (high-low range)
- `log_co_{TF}`     — intrabar direction (close vs open)
- `volume_{TF}`     — participation / conviction

Total features: 4 TFs × 4 features = **16 channels** (target_dim=16)

## Signal Logic (inference)

Primary signal channel: `log_return_{primary_TF}` (index 0)
- Forecast distribution → median + 10th/90th percentile
- Score = tanh(cumulative_median_return × 10) × confidence
- Confidence = avg_abs_return / interval_width  (narrow = confident)
- Score > +0.3  → LONG
- Score < -0.3  → SHORT
- Otherwise     → FLAT (stay out)

## Resuming Within a Phase

Any interruption (power loss, Ctrl+C) saves:
- `checkpoints/latest_lora.pt`     — weights after last completed epoch
- `checkpoints/resume_state.json`  — epoch number + best loss

Re-running `start_finetune.bat` resumes from exact epoch automatically.

## Advancing to Next Phase

When you are ready to advance:
1. Confirm `checkpoints/best_model` exists for current phase
2. Edit `config.yaml`:
   - Set `training.phase` to next phase number
   - The system auto-configures all other params from the phase table
3. Run `start_finetune.bat` — it detects the phase and loads the correct base
