import os
import json
import math
import yaml
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import snapshot_download
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

import logging
logging.basicConfig(
    filename="logs/finetune.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ── Dataset ──────────────────────────────────────────────────────────────────

class MoiraiTimeSeriesDataset(Dataset):
    def __init__(self, parquet_path: str, context_len: int, pred_len: int):
        table  = pq.read_table(parquet_path)
        self.series = [np.array(row.as_py(), dtype=np.float32)
                       for row in table.column("target")]
        self.ctx    = context_len
        self.pred   = pred_len
        self.window = context_len + pred_len

        self.samples = []
        for s_idx, s in enumerate(self.series):
            for start in range(0, len(s) - self.window + 1):
                self.samples.append((s_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_idx, start = self.samples[idx]
        window = self.series[s_idx][start: start + self.window]  # (ctx+pred,)
        # MoiraiForecast._val_loss expects (batch, time, tgt_dim=1)
        target   = torch.tensor(window, dtype=torch.float32).unsqueeze(-1)          # (T, 1)
        observed = torch.ones(self.window, 1, dtype=torch.bool)                     # all observed
        is_pad   = torch.zeros(self.window, dtype=torch.bool)                       # none padded
        return target, observed, is_pad


# ── Resume state ─────────────────────────────────────────────────────────────

def load_resume_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"epoch": 0, "best_loss": float("inf")}


def save_resume_state(path: str, state: dict):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict):
    model_name = cfg["model"]["name"]
    ctx        = cfg["model"]["context_length"]
    pred       = cfg["model"]["prediction_length"]
    patch      = cfg["model"]["patch_size"]

    module = MoiraiModule.from_pretrained(model_name)
    model  = MoiraiForecast(
        module=module,
        prediction_length=pred,
        context_length=ctx,
        patch_size=patch,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, grad_clip, device, log_every):
    model.train()
    total_loss = 0.0
    for step, (target, observed, is_pad) in enumerate(tqdm(loader, leave=False)):
        target, observed, is_pad = target.to(device), observed.to(device), is_pad.to(device)

        loss = model._val_loss(
            patch_size=model.hparams.patch_size,
            target=target,
            observed_target=observed,
            is_pad=is_pad,
        ).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if (step + 1) % log_every == 0:
            log.info(f"  step {step+1} loss={total_loss/(step+1):.6f}")

    return total_loss / len(loader)


def main():
    cfg = load_config()

    # Pin CPU threads
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    torch.set_num_interop_threads(cfg["training"]["cpu_threads"])
    device = torch.device("cpu")

    Path(cfg["paths"]["checkpoints"]).mkdir(exist_ok=True)
    Path(cfg["paths"]["logs"]).mkdir(exist_ok=True)

    # Dataset
    dataset = MoiraiTimeSeriesDataset(
        parquet_path=os.path.join(cfg["data"]["processed_dir"], "train.parquet"),
        context_len=cfg["model"]["context_length"],
        pred_len=cfg["model"]["prediction_length"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,   # 0 on Windows to avoid multiprocessing issues
        pin_memory=False,
    )
    print(f"Dataset: {len(dataset)} samples")

    # Model
    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Resume
    resume_path = cfg["paths"]["resume_state"]
    state       = load_resume_state(resume_path)
    start_epoch = state["epoch"]
    best_loss   = state["best_loss"]

    ckpt_dir    = cfg["paths"]["checkpoints"]
    best_path   = cfg["paths"]["best_model"]
    resume_ckpt = os.path.join(ckpt_dir, "latest_lora.pt")

    if os.path.exists(resume_ckpt):
        model.load_state_dict(torch.load(resume_ckpt, map_location=device), strict=False)
        print(f"Resumed from epoch {start_epoch}, best loss {best_loss:.6f}")
        log.info(f"Resumed from epoch {start_epoch}")

    total_epochs = cfg["training"]["epochs"]
    save_every   = cfg["training"]["save_every_n_epochs"]
    log_every    = cfg["training"]["log_every_n_steps"]

    print(f"Starting from epoch {start_epoch + 1} / {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        avg_loss = train_epoch(model, loader, optimizer,
                               cfg["training"]["grad_clip"], device, log_every)

        log.info(f"Epoch {epoch+1}/{total_epochs} loss={avg_loss:.6f}")
        print(f"Epoch {epoch+1}/{total_epochs} | loss={avg_loss:.6f}")

        # Always save latest so any interruption is resumable
        torch.save(model.state_dict(), resume_ckpt)
        save_resume_state(resume_path, {"epoch": epoch + 1, "best_loss": best_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(best_path)
            save_resume_state(resume_path, {"epoch": epoch + 1, "best_loss": best_loss})
            print(f"  ✓ New best model saved (loss={best_loss:.6f})")
            log.info(f"  New best saved at epoch {epoch+1}")

        if (epoch + 1) % save_every == 0:
            ckpt = os.path.join(ckpt_dir, f"lora_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt)

    print("Fine-tuning complete.")
    log.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
