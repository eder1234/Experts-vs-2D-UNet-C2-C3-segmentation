# src/datasets/csf_volume_dataset.py
from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# ── relative imports to avoid circular package re-entry ────────────────
from ..utils.temporal import reorder_temporal_images
from ..utils.temporal_features import (
    temporal_std,
    temporal_tv,
    dft_bandpower_excl_dc,
    dft_magnitudes_bins,
)

# ── small helpers ──────────────────────────────────────────────────────
def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Crop a square patch of `size×size` centred in `arr` (H and W dims)."""
    h, w = arr.shape[-2:]
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top : top + size, left : left + size]

def _first_pc(vol: np.ndarray) -> np.ndarray:
    """
    Return the first principal component (PC1) of a (C,H,W) volume.
    Uses SVD on channel×pixel matrix; returns (H,W) scaled to original mean/std.
    """
    assert vol.ndim == 3, "Expected (C,H,W)"
    C, H, W = vol.shape
    X = vol.reshape(C, H * W).astype(np.float32)
    X = X - X.mean(axis=1, keepdims=True)  # centre channels
    # SVD: X ≈ U S Vt, PC scores along pixels are in (S * Vt[0]) for first PC
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc_scores = (S[0] * Vt[0]).reshape(H, W)
    # normalise to zero-mean/unit-var like other features; model sees z-scored later anyway
    pc_scores = (pc_scores - pc_scores.mean()) / (pc_scores.std() + 1e-8)
    return pc_scores

def _feature_mode_to_channels(mode: str) -> int:
    """
    Map input_mode → number of channels.
      - 'full'       : 64  (32 phase + 32 mag)
      - 'pca'        : 1
      - 'dft_power'  : 1
      - 'tvt'        : 1
      - 'std'        : 1
      - 'dft_k123'   : 3
    """
    mode = (mode or "full").lower()
    if mode == "full":
        return 64
    if mode in ("pca", "dft_power", "tvt", "std"):
        return 1
    if mode == "dft_k123":
        return 3
    raise ValueError(f"Unknown input_mode '{mode}'")

# ── dataset ────────────────────────────────────────────────────────────
class CSFVolumeDataset(Dataset):
    """
    Subject-level dataset. Each subject folder must contain:
        phase.npy  (32, 240, 240)
        mag.npy    (32, 240, 240)
        mask.npy   (240, 240)

    Output (depends on input_mode):
        - 'full'      : x  (64, crop, crop) = [32 phase + 32 mag]
        - 'pca'       : x  (1,  crop, crop) = PC1 of (64-stack)
        - 'dft_power' : x  (1,  crop, crop) = DFT band power of phase (excl. DC)
        - 'tvt'       : x  (1,  crop, crop) = temporal total variation of phase
        - 'std'       : x  (1,  crop, crop) = temporal std of phase
        - 'dft_k123'  : x  (3,  crop, crop) = |DFT| at k={1,2,3} for phase

        y  : float32 tensor (1,  crop, crop)     single 2-D mask
        id : folder name (string)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        crop_size: int = 80,
        val_split: float = 0.2,
        seed: int = 42,
        input_mode: str = "full",
        augment_cfg: Optional[dict] = None,
        return_phase: bool = False,
        return_phase_full: bool = False,   # NEW
        metadata_csv: str | Path | None = None,
    ):
        super().__init__()

        root_dir = Path(root_dir)
        all_subjects: List[Path] = sorted([p for p in root_dir.iterdir() if p.is_dir()])

        # reproducible subject-level split
        rng = random.Random(seed)
        rng.shuffle(all_subjects)
        val_count = math.ceil(len(all_subjects) * val_split)

        if split == "train":
            self.subjects = all_subjects[val_count:]
        elif split == "val":
            self.subjects = all_subjects[:val_count]
        elif split == "test":
            self.subjects = all_subjects
        else:
            raise ValueError(f"Unknown split '{split}'. Use train / val / test.")

        self.input_mode = input_mode
        self.crop_size = int(crop_size)
        self.split = split
        self.augment_cfg = augment_cfg or {}
        self.return_phase = return_phase
        self.return_phase_full = return_phase_full  # NEW
        self.temporal_shift = bool(self.augment_cfg.get("temporal_shift", True))

        # Load per-subject metadata (v_enc, pixel_size) if provided
        self.metadata: dict[str, dict[str, float]] = {}
        if metadata_csv is not None and Path(metadata_csv).exists():
            with open(metadata_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get("sample") or row.get("id") or row.get("subject")
                    if not sid:
                        continue
                    entry = {}
                    if "v_enc" in row and row["v_enc"] != "":
                        entry["v_enc"] = float(row["v_enc"])
                    if "pixel_size" in row and row["pixel_size"] != "":
                        entry["pixel_size"] = float(row["pixel_size"])
                    if entry:
                        self.metadata[sid] = entry

        # quick sanity on channels (raises if unknown mode)
        _ = _feature_mode_to_channels(self.input_mode)

    def __len__(self) -> int:
        return len(self.subjects)

    def _augment(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        phase: torch.Tensor | None = None,
        phase_full: torch.Tensor | None = None,   # NEW
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Apply identical spatial transforms to img, mask and optionally phase/phase_full."""
        cfg = self.augment_cfg
        if not cfg or self.split != "train":
            return img, mask, phase, phase_full

        # Random flip
        if cfg.get("flip", False):
            if random.random() < 0.5:  # horizontal
                img = torch.flip(img, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
                if phase is not None:
                    phase = torch.flip(phase, dims=[-1])
                if phase_full is not None:
                    phase_full = torch.flip(phase_full, dims=[-1])
            if random.random() < 0.5:  # vertical
                img = torch.flip(img, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
                if phase is not None:
                    phase = torch.flip(phase, dims=[-2])
                if phase_full is not None:
                    phase_full = torch.flip(phase_full, dims=[-2])

        # Random affine (rotation + translation)
        rot_deg = float(cfg.get("rotation", 0))
        trans_frac = float(cfg.get("translate", 0.0))
        if rot_deg > 0 or trans_frac > 0:
            angle = random.uniform(-rot_deg, rot_deg)
            max_trans = trans_frac * self.crop_size
            translate = (
                float(random.uniform(-max_trans, max_trans)),
                float(random.uniform(-max_trans, max_trans)),
            )
            img = F.affine(img, angle=angle, translate=translate, scale=1.0, shear=0)
            mask = F.affine(mask, angle=angle, translate=translate, scale=1.0, shear=0)
            if phase is not None:
                phase = F.affine(phase, angle=angle, translate=translate, scale=1.0, shear=0)
            if phase_full is not None:
                # same transform on full-res stack (B/T,H,W) → apply per-frame
                # (torchvision F.affine supports (C,H,W); our phase_full is (T,H,W), treat T as C)
                phase_full = F.affine(phase_full, angle=angle, translate=translate, scale=1.0, shear=0)

        # Gaussian noise (image only)
        noise_std = float(cfg.get("gaussian_noise", 0))
        if noise_std > 0:
            img = img + noise_std * torch.randn_like(img)

        return img, mask, phase, phase_full

    def _build_input(self, phase: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Build input tensor (C,H,W) based on input_mode. Feature modes use PHASE ONLY.
        """
        mode = (self.input_mode or "full").lower()
        if mode == "full":
            return np.concatenate([phase, mag], axis=0)  # (64,H,W)

        if mode == "pca":
            vol = np.concatenate([phase, mag], axis=0)  # (64,H,W)
            return _first_pc(vol)[None, ...]            # (1,H,W)

        # Shift-invariant temporal features on PHASE only (T=32,H,W)
        if mode == "dft_power":
            feat = dft_bandpower_excl_dc(phase)         # (H,W)
            return feat[None, ...]                      # (1,H,W)

        if mode == "tvt":
            feat = temporal_tv(phase)                   # (H,W)
            return feat[None, ...]                      # (1,H,W)

        if mode == "std":
            feat = temporal_std(phase)                  # (H,W)
            return feat[None, ...]                      # (1,H,W)

        if mode == "dft_k123":
            feats = dft_magnitudes_bins(phase, bins=(1, 2, 3))  # (3,H,W)
            return feats

        raise ValueError(f"Unknown input_mode '{mode}'")

    def __getitem__(self, idx: int) -> dict:
        subj_dir: Path = self.subjects[idx]

        phase = np.load(subj_dir / "phase.npy")  # (32, 240, 240)
        mag   = np.load(subj_dir / "mag.npy")    # (32, 240, 240)
        mask  = np.load(subj_dir / "mask.npy")   # (240, 240)

        # optional temporal cyclic shift augmentation (training only)
        if self.split == "train" and self.temporal_shift:
            shift = random.randint(0, 31)  # cyclic shift in [0, 31]
            phase, mag, _ = reorder_temporal_images(phase, mag, shift=shift)

        # Build input & center-crop
        img = self._build_input(phase, mag)
        img = _center_crop(img, self.crop_size)
        mask = _center_crop(mask, self.crop_size)

        # Prepare phase tensors
        phase_crop = _center_crop(phase, self.crop_size) if self.return_phase else None
        phase_full_np = phase if self.return_phase_full else None  # (T,240,240) or None

        img_t  = torch.from_numpy(img).float()                # (C, crop, crop)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()  # (1, crop, crop)
        phase_t = torch.from_numpy(np.ascontiguousarray(phase_crop.astype(np.float32))) if phase_crop is not None else None
        phase_full_t = torch.from_numpy(np.ascontiguousarray(phase_full_np.astype(np.float32))) if phase_full_np is not None else None

        # Apply identical augmentation
        img_t, mask_t, phase_t, phase_full_t = self._augment(img_t, mask_t, phase_t, phase_full_t)

        # Normalise input image (per-sample)
        img_t = (img_t - img_t.mean()) / (img_t.std() + 1e-8)

        sample = {
            "image": img_t,
            "mask":  mask_t,
            "id":    subj_dir.name,
        }

        if self.return_phase and phase_t is not None:
            sample["phase"] = phase_t.float()
        if self.return_phase_full and phase_full_t is not None:
            sample["phase_full"] = phase_full_t.float()  # (T,240,240)

        if self.return_phase and (self.metadata is not None):
            meta = self.metadata.get(subj_dir.name, {})
            if "v_enc" in meta:
                sample["v_enc"] = torch.tensor(meta["v_enc"], dtype=torch.float32)
            if "pixel_size" in meta:
                sample["pixel_size"] = torch.tensor(meta["pixel_size"], dtype=torch.float32)

        return sample


