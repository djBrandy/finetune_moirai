import os
import json
import yaml
import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, PeftModel
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
        cfg = yaml.safe_load(f)
    phase     = cfg["training"]["phase"]
    phase_cfg = cfg["phases"][phase]
    cfg["_phase"]      = phase
    cfg["_primary_tf"] = phase_cfg["primary_tf"]
    cfg["_ctx"]        = phase_cfg["context_length"]
    cfg["_pred"]       = phase_cfg["prediction_length"]
    cfg["_patch"]      = phase_cfg["patch_size"]
    cfg["_best_model"] = phase_cfg["best_model"]
    # Base model: previous phase best if exists, else raw model_cache
    prev_best = cfg["phases"].get(phase - 1, {}).get("best_model", "")
    cfg["_base_model"] = prev_best if prev_best and os.path.exists(prev_best) \
                         else cfg["paths"]["base_model"]
    return cfg


# ── Dataset ───────────────────────────────────────────────────────────────────

class MoiraiDataset(Dataset):
    def __init__(self, parquet_path: str, context_len: int, pred_len: int):
        table       = pq.read_table(parquet_path)
        raw         = table.column("target")[0].as_py()
        self.data   = np.array(raw, dtype=np.float32)   # (T, n_feat)
        self.n_feat = self.data.shape[1]
        self.window = context_len + pred_len
        self.samples = list(range(len(self.data) - self.window + 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        w        = self.data[self.samples[idx]: self.samples[idx] + self.window]
        target   = torch.tensor(w, dtype=torch.float32)
        observed = torch.ones(self.window, self.n_feat, dtype=torch.bool)
        is_pad   = torch.zeros(self.window, dtype=torch.bool)
        return target, observed, is_pad


# ── Resume state ──────────────────────────────────────────────────────────────

def load_resume_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"epoch": 0, "best_loss": float("inf")}


def save_resume_state(path: str, state: dict):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, n_feat: int):
    """
    Phase chaining:
    - Phase 1: load raw MoiraiModule from model_cache, wrap with fresh LoRA
    - Phase N: load MoiraiModule from phase N-1 best_model (PeftModel merged),
               wrap with fresh LoRA for continued adaptation
    """
    base_path = cfg["_base_model"]
    phase     = cfg["_phase"]

    print(f"Phase {phase} | Base model: {base_path}")
    log.info(f"Phase {phase} | Base model: {base_path}")

    module = MoiraiModule.from_pretrained(base_path)
    model  = MoiraiForecast(
        module=module,
        prediction_length=cfg["_pred"],
        context_length=cfg["_ctx"],
        patch_size=cfg["_patch"],
        num_samples=100,
        target_dim=n_feat,
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    torch.set_num_interop_threads(cfg["training"]["cpu_threads"])
    device = torch.device("cpu")

    Path(cfg["paths"]["checkpoints"]).mkdir(exist_ok=True)
    Path(cfg["paths"]["logs"]).mkdir(exist_ok=True)

    dataset = MoiraiDataset(
        parquet_path=os.path.join(cfg["data"]["processed_dir"], "train.parquet"),
        context_len=cfg["_ctx"],
        pred_len=cfg["_pred"],
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                        shuffle=True, num_workers=0, pin_memory=False)

    print(f"Phase {cfg['_phase']} ({cfg['_primary_tf']}) | "
          f"{len(dataset)} samples | {dataset.n_feat} features")

    model     = build_model(cfg, dataset.n_feat).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    resume_path = cfg["paths"]["resume_state"]
    state       = load_resume_state(resume_path)
    # Reset resume state if phase changed
    if state.get("phase", cfg["_phase"]) != cfg["_phase"]:
        state = {"epoch": 0, "best_loss": float("inf")}

    start_epoch = state["epoch"]
    best_loss   = state["best_loss"]
    best_path   = cfg["_best_model"]
    resume_ckpt = os.path.join(cfg["paths"]["checkpoints"], "latest_lora.pt")

    if os.path.exists(resume_ckpt) and state.get("phase") == cfg["_phase"]:
        model.load_state_dict(torch.load(resume_ckpt, map_location=device), strict=False)
        print(f"Resumed phase {cfg['_phase']} from epoch {start_epoch} | best loss {best_loss:.6f}")

    total_epochs = cfg["training"]["epochs"]
    save_every   = cfg["training"]["save_every_n_epochs"]
    log_every    = cfg["training"]["log_every_n_steps"]

    print(f"Starting epoch {start_epoch + 1} / {total_epochs}")
    log.info(f"Phase {cfg['_phase']} start epoch {start_epoch + 1}")

    for epoch in range(start_epoch, total_epochs):
        avg_loss = train_epoch(model, loader, optimizer,
                               cfg["training"]["grad_clip"], device, log_every)

        print(f"Epoch {epoch+1}/{total_epochs} | loss={avg_loss:.6f}")
        log.info(f"Epoch {epoch+1}/{total_epochs} loss={avg_loss:.6f}")

        torch.save(model.state_dict(), resume_ckpt)
        save_resume_state(resume_path, {
            "phase": cfg["_phase"], "epoch": epoch + 1, "best_loss": best_loss
        })

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(best_path)
            save_resume_state(resume_path, {
                "phase": cfg["_phase"], "epoch": epoch + 1, "best_loss": best_loss
            })
            print(f"  ✓ Best model saved (loss={best_loss:.6f}) → {best_path}")
            log.info(f"  Best saved epoch {epoch+1} loss={best_loss:.6f}")

        if (epoch + 1) % save_every == 0:
            ckpt = os.path.join(cfg["paths"]["checkpoints"],
                                f"phase{cfg['_phase']}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt)

    print(f"Phase {cfg['_phase']} complete.")
    log.info(f"Phase {cfg['_phase']} complete.")


if __name__ == "__main__":
    main()
