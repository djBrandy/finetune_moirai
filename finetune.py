import os
import json
import yaml
import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig
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
    pc        = cfg["phases"][phase]
    cfg["_phase"]      = phase
    cfg["_primary_tf"] = pc["primary_tf"]
    cfg["_ctx"]        = pc["context_length"]
    cfg["_pred"]       = pc["prediction_length"]
    cfg["_patch"]      = pc["patch_size"]
    cfg["_best_model"] = pc["best_model"]
    prev_best = cfg["phases"].get(phase - 1, {}).get("best_model", "")
    cfg["_base_model"] = prev_best if prev_best and os.path.exists(prev_best) \
                         else cfg["paths"]["base_model"]
    return cfg


# ── Dataset ───────────────────────────────────────────────────────────────────

class MoiraiDataset(Dataset):
    def __init__(self, parquet_path: str, context_len: int, pred_len: int):
        table       = pq.read_table(parquet_path)
        raw         = table.column("target")[0].as_py()
        self.data   = np.array(raw, dtype=np.float32)
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
    return {"epoch": 0, "best_val_loss": float("inf")}


def save_resume_state(path: str, state: dict):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ── Hybrid loss ───────────────────────────────────────────────────────────────

def hybrid_loss(model, target: torch.Tensor, observed: torch.Tensor,
                is_pad: torch.Tensor, ctx: int, alpha: float = 0.3) -> torch.Tensor:
    """
    Combines:
    - distributional loss (_val_loss): calibrates the forecast distribution
    - directional loss: penalizes wrong-direction forecasts on channel 0 (primary return)

    alpha controls directional weight (0=pure distributional, 1=pure directional).
    0.3 means 70% distribution calibration + 30% directional penalty.
    """
    dist_loss = model._val_loss(
        patch_size=model.hparams.patch_size,
        target=target,
        observed_target=observed,
        is_pad=is_pad,
    ).mean()

    # Directional component — channel 0 is log_return of primary TF
    actual_future   = target[:, ctx:, 0]                    # (batch, pred_len)
    actual_dir      = torch.sign(actual_future.sum(dim=1))  # (batch,) — actual direction

    # Get median forecast direction via sampling
    with torch.no_grad():
        ctx_target   = target[:, :ctx, :]
        ctx_observed = observed[:, :ctx, :]
        ctx_is_pad   = is_pad[:, :ctx]
        samples = model(
            past_target=ctx_target,
            past_observed_target=ctx_observed,
            past_is_pad=ctx_is_pad,
            num_samples=20,                                  # low samples for speed
        )  # (batch, 20, pred_len, n_feat)

    pred_return  = samples[:, :, :, 0].sum(dim=2).median(dim=1).values  # (batch,)
    pred_dir     = torch.tanh(pred_return * 50)                          # soft direction

    # Penalize when predicted direction opposes actual direction
    # Loss = 0 when aligned, up to 2 when fully opposed
    dir_loss = torch.mean(torch.clamp(1.0 - actual_dir * pred_dir, min=0.0))

    return (1.0 - alpha) * dist_loss + alpha * dir_loss


# ── Train / eval epochs ───────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, cfg, device, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx        = cfg["_ctx"]
    alpha      = cfg["training"].get("directional_alpha", 0.3)
    log_every  = cfg["training"]["log_every_n_steps"]

    ctx_mgr = torch.enable_grad() if train else torch.no_grad()
    with ctx_mgr:
        for step, (target, observed, is_pad) in enumerate(tqdm(loader, leave=False)):
            target, observed, is_pad = (target.to(device),
                                        observed.to(device),
                                        is_pad.to(device))
            loss = hybrid_loss(model, target, observed, is_pad, ctx, alpha)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg["training"]["grad_clip"])
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if train and (step + 1) % log_every == 0:
                log.info(f"  step {step+1} loss={total_loss/(step+1):.6f}")

    return total_loss / len(loader)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, n_feat: int):
    print(f"Phase {cfg['_phase']} | Base: {cfg['_base_model']}")
    log.info(f"Phase {cfg['_phase']} | Base: {cfg['_base_model']}")

    module = MoiraiModule.from_pretrained(cfg["_base_model"])
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    torch.set_num_interop_threads(cfg["training"]["cpu_threads"])
    device = torch.device("cpu")

    Path(cfg["paths"]["checkpoints"]).mkdir(exist_ok=True)
    Path(cfg["paths"]["logs"]).mkdir(exist_ok=True)

    phase      = cfg["_phase"]
    data_dir   = cfg["data"]["processed_dir"]
    train_path = os.path.join(data_dir, f"phase{phase}_train.parquet")
    val_path   = os.path.join(data_dir, f"phase{phase}_val.parquet")

    train_ds = MoiraiDataset(train_path, cfg["_ctx"], cfg["_pred"])
    val_ds   = MoiraiDataset(val_path,   cfg["_ctx"], cfg["_pred"])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"Phase {phase} ({cfg['_primary_tf']}) | "
          f"Train: {len(train_ds)} | Val: {len(val_ds)} | Feat: {train_ds.n_feat}")

    model     = build_model(cfg, train_ds.n_feat).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    resume_path = cfg["paths"]["resume_state"]
    state       = load_resume_state(resume_path)
    if state.get("phase", phase) != phase:
        state = {"epoch": 0, "best_val_loss": float("inf")}

    start_epoch   = state["epoch"]
    best_val_loss = state["best_val_loss"]
    best_path     = cfg["_best_model"]
    resume_ckpt   = os.path.join(cfg["paths"]["checkpoints"], "latest_lora.pt")

    if os.path.exists(resume_ckpt) and state.get("phase") == phase:
        model.load_state_dict(torch.load(resume_ckpt, map_location=device), strict=False)
        print(f"Resumed phase {phase} epoch {start_epoch} | best val loss {best_val_loss:.6f}")

    total_epochs = cfg["training"]["epochs"]
    save_every   = cfg["training"]["save_every_n_epochs"]

    print(f"Starting epoch {start_epoch + 1} / {total_epochs}")
    log.info(f"Phase {phase} start epoch {start_epoch + 1}")

    for epoch in range(start_epoch, total_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, cfg, device, train=False)

        print(f"Epoch {epoch+1}/{total_epochs} | train={train_loss:.6f} | val={val_loss:.6f}")
        log.info(f"Epoch {epoch+1} train={train_loss:.6f} val={val_loss:.6f}")

        # Always save latest for resume
        torch.save(model.state_dict(), resume_ckpt)
        save_resume_state(resume_path, {
            "phase": phase, "epoch": epoch + 1, "best_val_loss": best_val_loss
        })

        # Best model saved on VALIDATION loss — not training loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(best_path)
            save_resume_state(resume_path, {
                "phase": phase, "epoch": epoch + 1, "best_val_loss": best_val_loss
            })
            print(f"  ✓ Best model saved (val={best_val_loss:.6f}) → {best_path}")
            log.info(f"  Best saved epoch {epoch+1} val={best_val_loss:.6f}")

        if (epoch + 1) % save_every == 0:
            ckpt = os.path.join(cfg["paths"]["checkpoints"],
                                f"phase{phase}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt)

    print(f"Phase {phase} complete.")
    log.info(f"Phase {phase} complete.")


if __name__ == "__main__":
    main()
