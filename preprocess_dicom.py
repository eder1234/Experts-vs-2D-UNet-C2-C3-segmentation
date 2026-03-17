#!/usr/bin/env python3
# preprocess_dicom.py
#
# Unified helpers to:
#   • parse manual ROI .txt -> binary mask (optionally 1-based indices)
#   • crop/embed using mask center (accounting for Y-axis origin flip)
#   • load a full DICOM *series* from a single seed DICOM filepath
#   • split series into PHASE vs MAG/FFE using ImageType
#   • normalise and stack as (D,H,W)
#
# Dependencies: pydicom, numpy, opencv-python

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset

# --- ADD THIS in preprocess_dicom.py ---

from typing import DefaultDict
from collections import defaultdict

from collections import defaultdict
from typing import Optional, Dict, List, Tuple
import pydicom
from pathlib import Path
import numpy as np

def inventory_dicomdir(dicomdir_path: Path) -> List[dict]:
    """
    Return a list of per-series dicts with:
      {
        "series_number": int | None,
        "series_description": str | None,
        "series_uid": str | None,
        "files": [Path, ...],
        "n_total": int,
        "n_phase": int,
        "n_mag": int,
      }

    NOTE: DICOMDIR stores relative paths → we resolve against DICOMDIR.parent.
    We read *minimal* tags from the first file to get SeriesInstanceUID; for counts
    (phase vs mag) we scan ImageType of each referenced file robustly.
    """
    ddir = pydicom.dcmread(str(dicomdir_path))
    base = dicomdir_path.parent

    # 1) Collect series-level buckets from DICOMDIR records
    buckets: Dict[int, dict] = defaultdict(lambda: {
        "series_number": None,
        "series_description": None,
        "files": [],
        "types": []
    })
    current_sn: Optional[int] = None

    for rec in ddir.DirectoryRecordSequence:
        rtype = getattr(rec, "DirectoryRecordType", None)
        if rtype == "SERIES":
            sn = getattr(rec, "SeriesNumber", None)
            sd = getattr(rec, "SeriesDescription", None)
            current_sn = int(sn) if sn is not None else None
            if current_sn is None:
                # Bucket under -1 if SeriesNumber missing; still keep SeriesDescription
                current_sn = -1 * (len(buckets) + 1)
            b = buckets[current_sn]
            b["series_number"] = current_sn if current_sn >= 0 else None
            b["series_description"] = str(sd) if sd is not None else None

        elif rtype == "IMAGE" and current_sn is not None:
            ref = getattr(rec, "ReferencedFileID", None)
            if ref:
                path = base / Path(*ref)
                if path.exists():
                    buckets[current_sn]["files"].append(path)

    # 2) For each series, read files to compute counts and SeriesInstanceUID
    results: List[dict] = []
    for sn, b in buckets.items():
        files: List[Path] = b["files"]
        if not files:
            continue

        n_phase = n_mag = 0
        series_uid = None

        # Read first file robustly to get SeriesInstanceUID (if possible)
        try:
            ds0 = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
            series_uid = str(getattr(ds0, "SeriesInstanceUID", None))
        except Exception:
            series_uid = None

        # Count PHASE vs MAG via ImageType
        for p in files:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                img_type = [str(t).upper() for t in getattr(ds, "ImageType", [])]
                if any("PHASE" in t or "PCA" in t for t in img_type):
                    n_phase += 1
                else:
                    n_mag += 1
            except Exception:
                # If unreadable, ignore that file
                pass

        results.append({
            "series_number": b["series_number"],
            "series_description": b["series_description"],
            "series_uid": series_uid,
            "files": files,
            "n_total": n_phase + n_mag,
            "n_phase": n_phase,
            "n_mag": n_mag,
        })

    # sort by series_number if available; otherwise by description then count
    results.sort(key=lambda r: (999999 if r["series_number"] is None else r["series_number"], r["series_description"] or "", -r["n_total"]))
    return results


def choose_series_by_rules(series_list: List[dict],
                           descriptions: Tuple[str, ...] = ("PCV 5CervLCS", "PCV 10CervLCS"),
                           phase_mag_target: Tuple[int, int] = (32, 32)) -> Optional[dict]:
    """
    Pick the *best* matching series from an inventory:
      - SeriesDescription ∈ descriptions
      - n_phase == 32 and n_mag == 32  (default)
    Returns the chosen dict or None if ambiguous / none matched.
    If multiple match, returns the first (you can add tie-breakers if needed).
    """
    want_phase, want_mag = phase_mag_target
    cand = [
        s for s in series_list
        if (s["series_description"] in descriptions)
        and (s["n_phase"] == want_phase and s["n_mag"] == want_mag)
    ]
    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        # heuristic: prefer the one with non-null series_number; then earliest number
        cand.sort(key=lambda r: (r["series_number"] is None, 999999 if r["series_number"] is None else r["series_number"]))
        return cand[0]
    return None


def scan_dicomdir(dicomdir_path: Path) -> Dict[int, Dict[str, List[Path]]]:
    """
    Parse a DICOMDIR and return:
      { series_number: { "files": [Path,...], "types": [[ImageType], ...] } }
    Where each path is absolute (resolved relative to DICOMDIR's parent).
    """
    ddir = pydicom.dcmread(str(dicomdir_path))
    base = dicomdir_path.parent

    series_map: Dict[int, Dict[str, List]] = defaultdict(lambda: {"files": [], "types": []})
    current_sn: Optional[int] = None

    for rec in ddir.DirectoryRecordSequence:
        rtype = getattr(rec, "DirectoryRecordType", None)
        if rtype == "SERIES":
            sn = getattr(rec, "SeriesNumber", None)
            current_sn = int(sn) if sn is not None else None
        elif rtype == "IMAGE" and current_sn is not None:
            # ReferencedFileID is a multi-valued DICOM element → join as a relative path
            ref = getattr(rec, "ReferencedFileID", None)
            if ref:
                rel = Path(*ref)
                series_map[current_sn]["files"].append(base / rel)
                series_map[current_sn]["types"].append([str(t) for t in getattr(rec, "ImageType", [])])

    # ensure file paths exist (robustness on odd exports)
    for sn in list(series_map.keys()):
        files = series_map[sn]["files"]
        types = series_map[sn]["types"]
        keep = [(p, t) for p, t in zip(files, types) if p.exists()]
        if not keep:
            del series_map[sn]
            continue
        series_map[sn]["files"] = [p for p, _ in keep]
        series_map[sn]["types"] = [t for _, t in keep]
    return series_map


def load_series_from_dicomdir(dicomdir_path: Path, series_number: int) -> SeriesSplit:
    """
    Load one series (by SeriesNumber) from a DICOMDIR and split into PHASE vs MAG/FFE.
    Sorts by InstanceNumber if available.
    """
    series_map = scan_dicomdir(dicomdir_path)
    if series_number not in series_map:
        raise ValueError(f"Series {series_number} not found in {dicomdir_path}")

    files = series_map[series_number]["files"]
    dsets: List[FileDataset] = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False, force=True)
            # ensure pixel data is readable
            _ = ds.pixel_array
            dsets.append(ds)
        except Exception:
            continue
    if not dsets:
        raise RuntimeError(f"No readable images for series {series_number} in {dicomdir_path}")

    dsets.sort(key=_instance_number)

    phase, mag = [], []
    for ds in dsets:
        it = [str(t).upper() for t in getattr(ds, "ImageType", [])]
        if _is_phase(it) or _is_pca(it):
            phase.append(ds)
        else:
            mag.append(ds)

    return SeriesSplit(phase=phase, mag=mag)


def _parse_mask_header(line: str) -> Tuple[int, int, int, int, int, int]:
    """Return (ROI_width, ROI_height, mid_side, origin_x, origin_y, num_pix)."""
    parts = [int(p) for p in line.split()]
    if len(parts) < 6:
        raise ValueError("Mask header must contain at least 6 integers.")
    return tuple(parts[:6])  # type: ignore[return-value]


def _read_indices(lines: Iterable[str]) -> Iterable[int]:
    """Yield integer tokens until first non-integer token appears."""
    for ln in lines:
        for tok in ln.split():
            if tok.isdigit():
                yield int(tok)
            else:
                return


def make_mask_from_txt(mask_txt: Path, *, one_based: bool = False) -> Tuple[np.ndarray, int, int]:
    """
    Build binary mask from manual .txt and return (mask, cx, cy) in *mask coords*.

    NOTE on coordinates:
    - The mask coordinates in the .txt use bottom-left origin (historical convention).
    - We build the mask bitmap as (row, col) with NumPy top-left origin,
      then *flipud* so that bitmap rows match the DICOM image matrix after conversion.
    - The reported (cx, cy) are taken from the header BEFORE flip, so they are
      in the mask-indexing convention and must be converted when mapping to image.
    """
    with Path(mask_txt).open() as f:
        roi_w, roi_h, mid_side, cx, cy, num_pix = _parse_mask_header(next(f))
        if one_based:
            cx -= 1
            cy -= 1
        res = 2 * mid_side
        idx = np.fromiter(_read_indices(f), dtype=np.int64, count=num_pix)

    if idx.size != num_pix:
        raise ValueError(f"{mask_txt}: expected {num_pix} indices, got {idx.size}")

    mask = np.zeros(res * res, np.uint8)
    mask[idx] = 1
    mask = mask.reshape(res, res)
    mask = np.flipud(mask)  # align with image rows (important downstream)

    return mask, cx, cy


def _mask_to_image_y(cy_mask: int, img_h: int) -> int:
    """Convert mask Y (bottom-left origin) to image Y (top-left origin)."""
    return img_h - 1 - cy_mask


def crop_around_center(img: np.ndarray, mask_shape: Tuple[int, int], cx: int, cy_mask: int) -> np.ndarray:
    """Return a crop (Hmask, Wmask) centered at (cx, cy_mask). Error if OOB."""
    h_mask, w_mask = mask_shape
    h_img, w_img = img.shape
    cy_img = _mask_to_image_y(cy_mask, h_img)

    x0 = cx - w_mask // 2
    y0 = cy_img - h_mask // 2
    x1 = x0 + w_mask
    y1 = y0 + h_mask

    if x0 < 0 or y0 < 0 or x1 > w_img or y1 > h_img:
        raise ValueError("Crop window out of bounds (check mask center).")

    patch = img[y0:y1, x0:x1]
    if patch.shape != (h_mask, w_mask):
        raise ValueError(f"Patch shape {patch.shape} != mask shape {(h_mask, w_mask)}.")
    return patch


def embed_mask_in_image(image_shape: Tuple[int, int], mask_small: np.ndarray, cx: int, cy_mask: int) -> np.ndarray:
    """Return a full-size (Himg,Wimg) binary mask placed at the same center."""
    h_img, w_img = image_shape
    h_mask, w_mask = mask_small.shape
    cy_img = _mask_to_image_y(cy_mask, h_img)

    x0 = cx - w_mask // 2
    y0 = cy_img - h_mask // 2
    x1 = x0 + w_mask
    y1 = y0 + h_mask

    if x0 < 0 or y0 < 0 or x1 > w_img or y1 > h_img:
        raise ValueError("Embedding window out of bounds (check mask center).")

    full = np.zeros((h_img, w_img), np.uint8)
    full[y0:y1, x0:x1] = mask_small
    return full


def _norm01(img: np.ndarray) -> np.ndarray:
    """Normalise 2D image to [0,1] float32."""
    out = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    return out.astype(np.float32)


@dataclass
class SeriesSplit:
    """Holds per-class sorted (by InstanceNumber) FileDataset objects."""
    phase: List[FileDataset]
    mag: List[FileDataset]  # (a.k.a. FFE / magnitude)


def _is_phase(img_type: List[str]) -> bool:
    return any("PHASE" in t.upper() for t in img_type)


def _is_pca(img_type: List[str]) -> bool:
    return any("PCA" in t.upper() for t in img_type)


def _instance_number(ds: FileDataset) -> int:
    val = getattr(ds, "InstanceNumber", None)
    try:
        return int(val)
    except Exception:
        return 0


def load_series_from_seed(seed_dicom: Path) -> SeriesSplit:
    """
    From a single DICOM file path, load *all files in the same folder* that share
    the same SeriesInstanceUID. Split into PHASE vs MAG/FFE using ImageType.
    """
    seed = pydicom.dcmread(str(seed_dicom))
    siuid = getattr(seed, "SeriesInstanceUID", None)
    if siuid is None:
        raise ValueError("Seed DICOM has no SeriesInstanceUID.")

    folder = Path(seed_dicom).parent
    ds_list: List[FileDataset] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False, force=True)
            if getattr(ds, "SeriesInstanceUID", None) == siuid and hasattr(ds, "pixel_array"):
                ds_list.append(ds)
        except Exception:
            continue

    if not ds_list:
        raise ValueError("No DICOM siblings found with same SeriesInstanceUID.")

    ds_list.sort(key=_instance_number)

    phase, mag = [], []
    for ds in ds_list:
        img_type = [str(t) for t in getattr(ds, "ImageType", [])]
        if _is_phase(img_type) or _is_pca(img_type):
            phase.append(ds)
        else:
            mag.append(ds)

    return SeriesSplit(phase=phase, mag=mag)


def stack_large_and_cropped(
    dsets: List[FileDataset],
    mask_small: np.ndarray,
    cx: int,
    cy_mask: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (large_stack, small_stack) each (D,H,W).
    large_stack: full-slice normalised; small_stack: crop-to-mask normalised.
    """
    large_list, small_list = [], []
    for ds in dsets:
        img = ds.pixel_array.astype(np.float32)
        large_list.append(_norm01(img))
        small_list.append(_norm01(crop_around_center(img, mask_small.shape, cx, cy_mask)))
    large = np.stack(large_list, axis=0) if large_list else np.zeros((0, 1, 1), np.float32)
    small = np.stack(small_list, axis=0) if small_list else np.zeros((0, *mask_small.shape), np.float32)
    return large, small


def basic_meta(dsets: List[FileDataset]) -> Dict[str, Optional[float]]:
    if not dsets:
        return {"slice_count": 0}
    ds0 = dsets[-1]  # often last PHASE slice has v_enc/pixel spacing
    # pixel spacing
    px = getattr(ds0, "PixelSpacing", None)
    px_mm = None
    if px is not None:
        try:
            px_mm = float(np.mean([float(v) for v in px]))
        except Exception:
            pass
    # velocity encoding – tag varies by vendor; try common ones
    v_enc = None
    for tag in [("VelocityEncoding",), (0x0028, 0x1052), (0x0018, 0x9199)]:
        try:
            val = ds0[tag] if isinstance(tag, tuple) and len(tag) == 2 else getattr(ds0, tag[0])  # type: ignore[index]
            v_enc = float(val.value if hasattr(val, "value") else val)
            break
        except Exception:
            continue
    return {
        "slice_count": len(dsets),
        "pixel_size_mm": px_mm,
        "v_enc": v_enc,
        "rows": int(getattr(ds0, "Rows", 0)),
        "cols": int(getattr(ds0, "Columns", 0)),
    }
