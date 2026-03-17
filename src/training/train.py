# src/training/train.py
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

# direct submodule imports (avoid src.__init__ re-exports)
from src.datasets.csf_volume_dataset import CSFVolumeDataset
from src.models.unet2d import UNet2D
from src.utils.misc import load_yaml, seed_everything, save_ckpt
from src.utils.losses import DiceLoss, FlowDiceLoss

def _get_loss_fn(cfg: Dict) -> torch.nn.Module:
    """Factory to create the loss function from config."""
    name = cfg["train"].get("loss", "dice").lower()

    if name == "dice":
        eps = float(cfg["train"].get("dice_eps", 1e-5))
        from src.utils.losses import DiceLoss
        return DiceLoss(eps=eps)

    if name == "tversky":
        from src.utils.losses import TverskyLoss
        alpha = float(cfg["train"].get("tversky_alpha", 0.5))
        beta  = float(cfg["train"].get("tversky_beta", 0.5))
        eps   = float(cfg["train"].get("tversky_eps", 1e-5))
        return TverskyLoss(alpha=alpha, beta=beta, eps=eps)

    if name == "focal_dice":
        from src.utils.losses import FocalDiceLoss
        gamma = float(cfg["train"].get("focal_gamma", 2.0))
        eps   = float(cfg["train"].get("focal_eps", 1e-5))
        return FocalDiceLoss(gamma=gamma, eps=eps)

    if name == "flow_dice":
        from src.utils.losses import FlowDiceLoss
        lam         = float(cfg["train"].get("flow_lambda", 0.1))
        eps         = float(cfg["train"].get("flow_eps", 1e-5))
        use_full_res= bool(cfg["train"].get("flow_use_full_res", False))
        dice_w      = float(cfg["train"].get("dice_weight", 1.0))          # <<< NEW
        bce_alpha   = float(cfg["train"].get("bce_alpha", 0.2))            # <<< NEW (was fixed)
        mse_alpha   = float(cfg["train"].get("flow_mse_alpha", 0.25))      # <<< NEW (optional)
        return FlowDiceLoss(lambda_flow=lam, eps=eps, use_full_res=use_full_res,
                            dice_weight=dice_w, bce_alpha=bce_alpha, mse_alpha=mse_alpha)

    raise ValueError(f"Unknown loss '{name}'")


def _in_channels_for_mode(mode: str) -> int:
    mode = (mode or "full").lower()
    if mode == "full":
        return 64
    if mode in ("pca", "dft_power", "tvt", "std"):
        return 1
    if mode == "dft_k123":
        return 3
    raise ValueError(f"Unknown input_mode '{mode}'")

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    train: bool = True,
    log_every: int = 10,
) -> float:
    model.train(train)
    running_loss = 0.0
    for step, batch in enumerate(loader, 1):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        extra = {}
        if "phase" in batch:
            extra["phase"] = batch["phase"].to(device, non_blocking=True)
        if "phase_full" in batch:  # NEW
            extra["phase_full"] = batch["phase_full"].to(device, non_blocking=True)
        if "v_enc" in batch:
            extra["v_enc"] = batch["v_enc"].to(device, non_blocking=True)
        if "pixel_size" in batch:
            extra["pixel_size"] = batch["pixel_size"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            if isinstance(criterion, FlowDiceLoss):
                # Fail loudly if anything is missing
                needed = ("phase", "v_enc", "pixel_size")
                missing = [k for k in needed if k not in extra]
                if missing:
                    raise RuntimeError(
                        f"FlowDiceLoss selected but missing batch keys: {missing}. "
                        "Ensure return_phase=True and metadata_csv is set."
                    )
                # (Optional) you can inspect ranges once:
                # pmin, pmax = extra['phase'].amin().item(), extra['phase'].amax().item()
                # print(f"[DBG] phase range: [{pmin:.3f}, {pmax:.3f}]")
                loss = criterion(logits, masks, **extra)
            else:
                loss = criterion(logits, masks)


        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item())
        if train and step % log_every == 0:
            print(f"Epoch {epoch} | step {step}/{len(loader)} | loss {loss.item():.4f}")

    return running_loss / max(len(loader), 1)

def get_model_name(cfg: Dict, custom_name: str | None = None) -> str:
    if custom_name:
        return custom_name
    input_mode    = cfg["data"].get("input_mode", "full")
    crop_size     = cfg["data"]["crop_size"]
    base_channels = cfg["model"]["base_channels"]
    loss_name     = cfg["train"].get("loss", "dice")
    return f"unet2d_{input_mode}_c{crop_size}_b{base_channels}_{loss_name}"

def main(cfg: Dict, model_name: str | None = None) -> None:
    seed_everything()
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model_name = get_model_name(cfg, model_name)
    print(f"Training model: {model_name}")

    input_mode = cfg["data"].get("input_mode", "full")

    # datasets & loaders --------------------------------------------------
    use_flow = cfg["train"].get("loss") == "flow_dice"
    use_full_res = bool(cfg["train"].get("flow_use_full_res", False))
    metadata_csv = cfg["data"].get("metadata_csv")

    train_ds = CSFVolumeDataset(
        root_dir=Path(cfg["data"]["root"]) / cfg["data"]["train_dir"],
        split="train",
        crop_size=int(cfg["data"]["crop_size"]),
        val_split=float(cfg["data"]["val_split"]),
        input_mode=input_mode,
        augment_cfg=cfg.get("augment", {}),
        return_phase=use_flow,
        return_phase_full=use_flow and use_full_res,  # NEW
        metadata_csv=metadata_csv,
    )
    val_ds = CSFVolumeDataset(
        root_dir=Path(cfg["data"]["root"]) / cfg["data"]["train_dir"],
        split="val",
        crop_size=int(cfg["data"]["crop_size"]),
        val_split=float(cfg["data"]["val_split"]),
        input_mode=input_mode,
        return_phase=use_flow,
        return_phase_full=use_flow and use_full_res,  # NEW
        metadata_csv=metadata_csv,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=RandomSampler(train_ds),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=SequentialSampler(val_ds),
        num_workers=2,
        pin_memory=True,
    )
    print(f"Train subjects: {len(train_ds)} | Val subjects: {len(val_ds)}")

    # model, loss, opt ----------------------------------------------------
    in_ch = _in_channels_for_mode(input_mode)
    model = UNet2D(
        in_channels=in_ch,
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
    ).to(device)
    # Optional: small positive prior ~2% foreground
    with torch.no_grad():
        prior_p = 0.02
        bias = math.log(prior_p / (1.0 - prior_p))
        if hasattr(model, "outc") and hasattr(model.outc, "bias") and model.outc.bias is not None:
            model.outc.bias.fill_(bias)

    criterion = _get_loss_fn(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    # AMP scaler ----------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler() if cfg["train"].get("mixed_precision", True) else None

    # logging -------------------------------------------------------------
    out_dir = Path("outputs") / model_name
    tb_writer = SummaryWriter(out_dir / "logs")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # early stopping ------------------------------------------------------
    best_val = math.inf
    patience = 20
    epochs_no_improve = 0
    best_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        # ----------------------------
        # Set flow weight for THIS epoch (curriculum)
        # ----------------------------
        if isinstance(criterion, FlowDiceLoss):
            warmup = int(cfg["train"].get("flow_warmup", 0))
            base = float(cfg["train"].get("flow_lambda", 0.0))
            if warmup > 0:
                # epoch=1 -> factor=0.0  (no flow yet)
                # epoch=warmup+1 -> factor=1.0 (full weight from then on)
                factor = min(1.0, (epoch - 1) / warmup)
            else:
                factor = 1.0
            criterion.lambda_flow = base * factor
            if epoch == 1 or epoch % 5 == 0:  # optional console log
                print(f"[Epoch {epoch}] λ_flow = {criterion.lambda_flow:.6f} (base={base}, warmup={warmup})")
            # log to TensorBoard
            tb_writer.add_scalar("Hyper/flow_lambda", criterion.lambda_flow, epoch)

        # ----------------------------
        # Train / Val
        # ----------------------------
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, train=True
        )
        val_loss = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device, scaler=None, epoch=epoch, train=False
        )

        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            save_ckpt(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": cfg,
                },
                path=best_path,
            )
            epochs_no_improve = 0
            print(f"  ↳ New best val loss {val_loss:.4f} (epoch {epoch}) — saved {best_path}")
        else:
            epochs_no_improve += 1
            if cfg["train"].get("early_stopping", True) and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"Training finished. Best val loss: {best_val:.4f}. Checkpoints in: {ckpt_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None, help="Override run name (optional)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    main(cfg, model_name=args.name)
