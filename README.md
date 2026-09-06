# Moirai Progressive Fine-Tuning Pipeline

A phased fine-tuning pipeline for Salesforce's Moirai-large time-series foundation model, applied to forex/CFD instruments (current target: XBRUSD, Brent Crude).

## Approach
Training proceeds in four sequential phases, each fine-tuning from the previous phase's best checkpoint rather than from scratch — moving bottom-up from microstructure to macro regime:

| Phase | Timeframe | Training windows | Focus |
|---|---|---|---|
| 1 | M5 | ~470K | Intraday volatility, session structure |
| 2 | M1 | ~2.16M | Microstructure, order-flow timing |
| 3 | H1 | ~39K  | Broader trend context |
| 4 | D1 | ~1.6K | Macro regime |

Each phase is designed to produce a fully deployable checkpoint before advancing to the next.

## Status: in progress
Phase 1 (M5) is the active phase. Later phases are configured but gated on Phase 1 validation results.

## Stack
Python, PyTorch, gluonts/uni2ts, YAML-driven phase configuration
