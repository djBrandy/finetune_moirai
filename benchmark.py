"""
benchmark.py
------------
Runs a small timing test using dummy data to estimate total finetuning time.
No model download needed — uses random tensors to simulate the training loop.
"""

import time
import math
import torch
import yaml
import numpy as np

WARMUP_STEPS  = 3
MEASURE_STEPS = 10


def load_config():
    with open("config.yaml") as f:
        import yaml
        return yaml.safe_load(f)


def format_duration(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    if seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    days  = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    return f"{days} days {hours} hours"


def main():
    cfg = load_config()
    torch.set_num_threads(cfg["training"]["cpu_threads"])

    phase    = cfg["training"]["phase"]
    pc       = cfg["phases"][phase]
    ctx      = pc["context_length"]
    pred     = pc["prediction_length"]
    patch    = pc["patch_size"]
    batch    = cfg["training"]["batch_size"]
    epochs   = cfg["training"]["epochs"]

    # Estimate dataset size from actual parquet if available, else use known row count
    import os, pyarrow.parquet as pq
    parquet_path = os.path.join(cfg["data"]["processed_dir"], "train.parquet")
    if os.path.exists(parquet_path):
        table       = pq.read_table(parquet_path)
        n_series    = len(table)
        series_lens = [len(row.as_py()) for row in table.column("target")]
        window      = ctx + pred
        n_samples   = sum(max(0, l - window + 1) for l in series_lens)
    else:
        # Fallback: D1 has ~1943 rows, 12 feature columns after resampling
        n_samples = 12 * max(0, 1943 - (ctx + pred) + 1)

    steps_per_epoch = math.ceil(n_samples / batch)

    print(f"\n{'='*50}")
    print(f"  Config summary")
    print(f"{'='*50}")
    print(f"  CPU threads      : {cfg['training']['cpu_threads']}")
    print(f"  Context length   : {ctx}")
    print(f"  Prediction length: {pred}")
    print(f"  Batch size       : {batch}")
    print(f"  Estimated samples: {n_samples:,}")
    print(f"  Steps per epoch  : {steps_per_epoch:,}")
    print(f"  Total epochs     : {epochs}")
    print(f"{'='*50}\n")

    # Build a minimal transformer-like dummy model that approximates Moirai's compute
    # ctx/patch = number of patches fed into attention
    n_patches   = ctx // patch          # 8 patches
    d_model     = 512                   # moirai-large hidden dim
    n_heads     = 8
    n_layers    = 6

    class DummyMoirai(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model*4, batch_first=True
            )
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head     = torch.nn.Linear(d_model, pred)

        def forward(self, x):
            out = self.encoder(x)
            return self.head(out[:, -1, :])

    model     = DummyMoirai()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn   = torch.nn.MSELoss()

    def one_step():
        x      = torch.randn(batch, n_patches, d_model)
        target = torch.randn(batch, pred)
        optimizer.zero_grad()
        out  = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

    print("Warming up...")
    for _ in range(WARMUP_STEPS):
        one_step()

    print(f"Measuring {MEASURE_STEPS} steps...\n")
    start = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        one_step()
    elapsed = time.perf_counter() - start

    secs_per_step  = elapsed / MEASURE_STEPS
    secs_per_epoch = secs_per_step * steps_per_epoch
    total_secs     = secs_per_epoch * epochs

    print(f"{'='*50}")
    print(f"  Benchmark results")
    print(f"{'='*50}")
    print(f"  Time per step    : {secs_per_step*1000:.1f} ms")
    print(f"  Time per epoch   : {format_duration(secs_per_epoch)}")
    print(f"  TOTAL (estimate) : {format_duration(total_secs)}")
    print(f"{'='*50}")
    print()
    print("  NOTE: Actual time will be ~2-4x longer than this estimate")
    print("  because the real Moirai-large model is significantly larger")
    print("  than the dummy used here. Use this as a lower-bound estimate.")
    print()


if __name__ == "__main__":
    main()
