# src/utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (expects raw logits)."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **_) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # flatten per-sample
        dims = tuple(range(1, probs.ndim))
        num = 2.0 * torch.sum(probs * targets, dim=dims)
        den = torch.sum(probs + targets, dim=dims)
        dice = (num + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss (expects raw logits).
    L = 1 - T, with T = TP / (TP + α FN + β FP)

    Parameters
    ----------
    alpha : weight for FN
    beta  : weight for FP
    eps   : numerical stability
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **_) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))

        tp = torch.sum(probs * targets, dim=dims)
        fn = torch.sum((1.0 - probs) * targets, dim=dims)
        fp = torch.sum(probs * (1.0 - targets), dim=dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        return 1.0 - tversky.mean()


class FocalDiceLoss(nn.Module):
    """
    Focal-Dice loss (expects raw logits):
      L = (1 - Dice)^γ

    This is a simple focalisation of the soft Dice; γ > 1 focuses on hard samples.
    """
    def __init__(self, gamma: float = 2.0, eps: float = 1e-5):
        super().__init__()
        self.gamma = float(gamma)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **_) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        num = 2.0 * torch.sum(probs * targets, dim=dims)
        den = torch.sum(probs + targets, dim=dims)
        dice = (num + self.eps) / (den + self.eps)  # in [0,1]
        loss = torch.pow(1.0 - dice, self.gamma)
        return loss.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowDiceLoss(DiceLoss):
    """
    Dice + BCE (small) + robust flow-consistency loss.
    Flow loss is sign/offset/scale-invariant:
        L_flow = 1 - max(corr(f_pred, f_true), corr(-f_pred, f_true)) + α * MSE(zpred, ztrue)
    where curves are per-sample de-meaned and unit-variance normalized.
    """

    def __init__(self,
                 lambda_flow: float = 0.05,
                 eps: float = 1e-6,
                 bce_alpha: float = 0.2,
                 mse_alpha: float = 0.25,
                 use_full_res: bool = False,
                 dice_weight: float = 1.0):
        super().__init__(eps)
        self.lambda_flow = float(lambda_flow)
        self.bce_alpha   = float(bce_alpha)
        self.mse_alpha   = float(mse_alpha)
        self.use_full_res = bool(use_full_res)
        self.dice_weight = float(dice_weight)   # <<< NEW
        self.bce = nn.BCEWithLogitsLoss()
        self.last_parts = None  # if you added parts logging earlier


    @staticmethod
    def _center_unitvar(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """Per-sample de-mean + unit variance along time dim."""
        mu = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True).clamp_min(eps)
        return (x - mu) / std

    @staticmethod
    def _corr(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """Pearson correlation along *dim* for already standardized a,b."""
        # a and b should already be standardized, but we guard anyway
        a = FlowDiceLoss._center_unitvar(a, dim=dim, eps=eps)
        b = FlowDiceLoss._center_unitvar(b, dim=dim, eps=eps)
        return (a * b).mean(dim=dim)

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                *,
                phase: torch.Tensor | None = None,   # (B,T,H,W) crop or full
                v_enc: torch.Tensor | float | None = None,
                pixel_size: torch.Tensor | float | None = None,
                phase_full: torch.Tensor | None = None,   # (B,T,240,240) optional
                **_) -> torch.Tensor:

        dice_loss = super().forward(logits, targets)
        bce_loss  = self.bce(logits, targets)
        total = self.dice_weight * dice_loss + self.bce_alpha * bce_loss   # <<< weighted Dice

        # 3) Robust flow term
        if (phase is None and phase_full is None) or v_enc is None or pixel_size is None:
            # If metadata missing, just return supervised part
            return total

        device, dtype = logits.device, logits.dtype
        # Choose phase source: full-res preferred for SNR if provided
        if self.use_full_res and (phase_full is not None):
            ph = phase_full.to(device=device, dtype=dtype)        # (B,T,240,240)
            # build full-res masks by padding the (B,1,h,w) masks
            B, _, h, w = logits.shape
            full = 240
            pad_y = (full - h) // 2
            pad_x = (full - w) // 2
            probs_full = torch.sigmoid(logits)
            probs_full = F.pad(probs_full, (pad_x, full - w - pad_x, pad_y, full - h - pad_y))
            targets_full = F.pad(targets, (pad_x, full - w - pad_x, pad_y, full - h - pad_y))
            pred_mask = probs_full
            gt_mask   = targets_full
        else:
            # crop-level flow (what you had)
            ph = phase.to(device=device, dtype=dtype)             # (B,T,h,w)
            pred_mask = torch.sigmoid(logits)                     # (B,1,h,w)
            gt_mask   = targets                                   # (B,1,h,w)

        v_enc = torch.as_tensor(v_enc, device=device, dtype=dtype).view(-1, 1)        # (B,1)
        pix   = torch.as_tensor(pixel_size, device=device, dtype=dtype).view(-1, 1)   # (B,1)
        pix_area = pix ** 2

        # flow_t = v_enc * pix_area * sum_tissue( phase[t] * mask )
        # use soft mask for pred
        flow_pred = v_enc * pix_area * torch.sum(ph * pred_mask, dim=(2, 3))  # (B,T)
        flow_true = v_enc * pix_area * torch.sum(ph * gt_mask,   dim=(2, 3))  # (B,T)

        # De-mean + unit variance along time (remove DC and scale)
        zpred = self._center_unitvar(flow_pred, dim=1, eps=self.eps)
        ztrue = self._center_unitvar(flow_true, dim=1, eps=self.eps)

        # Sign-invariant correlation (maximize corr up to a sign)
        corr_pos = self._corr(zpred, ztrue, dim=1, eps=self.eps)          # (B,)
        corr_neg = self._corr(-zpred, ztrue, dim=1, eps=self.eps)         # (B,)
        corr = torch.maximum(corr_pos, corr_neg)                          # best sign
        corr_loss = 1.0 - corr.mean()                                     # (scalar)

        # Small MSE between normalized curves to help alignment
        mse_loss = F.mse_loss(zpred, ztrue)

        flow_loss = corr_loss + self.mse_alpha * mse_loss
        total = total + self.lambda_flow * flow_loss
        return total
