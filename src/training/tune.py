"""
Hyper‑parameter optimisation for UNet2D spinal‑CSF segmentation.

* One fixed 80 / 20 train‑validation split (same random seed in `CSFVolumeDataset`).
* Search space (8 configurations):
    – learning‑rate   ∈ {3e‑4, 1e‑3, 3e‑3}
    – base_channels   ∈ {16, 32}
    – LR scheduler    ∈ {"constant", "onecycle"}
* Early stopping: patience = 5 epochs, minimum ΔDice >= 0.003 (≈ +0.3 pp Dice).
* Optuna + Hyperband (ASHA) pruner keeps budget ≤ 2 GPU‑hours.

Usage – from project root:
    python -m src.training.tune --config config.yaml --trials 8 --timeout 7200

Outputs:
    ./outputs/tuning/
        trial_#/best.pt   – best checkpoint per trial
        best_overall.pt   – checkpoint from best trial (lowest val Dice loss)
        study.db          – Optuna SQLite study for reproducibility
"""
from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler
import optuna
from optuna.pruners import HyperbandPruner
from optuna.trial import Trial

# local imports ------------------------------------------------------------
from src import CSFVolumeDataset, UNet2D
from src.utils.misc import load_yaml, seed_everything, save_ckpt
from src.utils.losses import DiceLoss, FlowDiceLoss


def _get_loss_fn(cfg: Dict) -> torch.nn.Module:
    name = cfg["train"].get("loss", "dice").lower()
    if name == "dice":
        return DiceLoss()
    if name == "flow_dice":
        lam = cfg["train"].get("flow_lambda", 0.1)
        return FlowDiceLoss(lambda_flow=lam)
    raise ValueError(f"Unknown loss '{name}'")

# -------------------------------------------------------------------------
# Helper: single epoch run (adapted from train.py)
# -------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler | None,
    train: bool = True,
) -> float:
    model.train(train)
    running: float = 0.0
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        extra = {}
        if "phase" in batch:
            extra["phase"] = batch["phase"].to(device, non_blocking=True)
        if "v_enc" in batch:
            extra["v_enc"] = batch["v_enc"].to(device, non_blocking=True)
        if "pixel_size" in batch:
            extra["pixel_size"] = batch["pixel_size"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, masks, **extra)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        running += loss.item()
    return running / len(loader)

# -------------------------------------------------------------------------
# Objective for Optuna
# -------------------------------------------------------------------------

def objective(trial: Trial, cfg: Dict, device: torch.device, out_dir: Path) -> float:
    start_time = time.time()

    # ---------------- search space ---------------------------------------
    lr = trial.suggest_categorical("lr", [3e-4, 1e-3, 3e-3])
    base_ch = trial.suggest_categorical("base_channels", [16, 32])
    scheduler_name = trial.suggest_categorical("scheduler", ["constant", "onecycle"])

    # ---------------- dataset -------------------------------------------
    use_flow = cfg["train"].get("loss") == "flow_dice"
    metadata_csv = cfg["data"].get("metadata_csv")
    train_ds = CSFVolumeDataset(
        root_dir=Path(cfg["data"]["root"]) / cfg["data"]["train_dir"],
        split="train",
        crop_size=cfg["data"]["crop_size"],
        val_split=cfg["data"]["val_split"],
        augment_cfg=cfg["augment"],
        return_phase=use_flow,
        metadata_csv=metadata_csv,
    )
    val_ds = CSFVolumeDataset(
        root_dir=Path(cfg["data"]["root"]) / cfg["data"]["train_dir"],
        split="val",
        crop_size=cfg["data"]["crop_size"],
        val_split=cfg["data"]["val_split"],
        return_phase=use_flow,
        metadata_csv=metadata_csv,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              sampler=RandomSampler(train_ds), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=SequentialSampler(val_ds),
                            num_workers=2, pin_memory=True)

    # ---------------- model & optimiser ----------------------------------
    model = UNet2D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=base_ch,
    ).to(device)
    criterion = _get_loss_fn(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # LR scheduler ---------------------------------------------------------
    if scheduler_name == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=cfg["train"]["epochs"],
            steps_per_epoch=steps_per_epoch, pct_start=0.3)
    else:
        scheduler = None

    scaler = GradScaler() if cfg["train"].get("mixed_precision", True) else None

    # Early stopping params
    patience = 5
    min_delta = 0.003  # Dice loss (≈ 0.3 pp Dice)

    best_val = math.inf
    epochs_no_improve = 0
    total_epochs = cfg["train"]["epochs"]

    for epoch in range(1, total_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        if scheduler is not None:
            scheduler.step()