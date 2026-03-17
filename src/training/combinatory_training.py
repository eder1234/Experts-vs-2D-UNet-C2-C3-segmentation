# src/training/combinatory_training.py
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import List

from src.utils.misc import load_yaml
from .train import main as train_one_combo

# Default grids
DEFAULT_INPUT_MODES: List[str] = ["tvt", "std", "dft_k123", "pca", "full", "dft_power"]
DEFAULT_LOSSES: List[str] = ["dice", "tversky", "focal_dice", "flow_dice"]


def parse_list_arg(arg_val: str | None, default: List[str]) -> List[str]:
    """
    Parse a comma-separated CLI list (e.g. "full,pca") or return default if None/empty.
    """
    if not arg_val:
        return default
    items = [x.strip() for x in arg_val.split(",") if x.strip()]
    return items or default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a full grid of (input_mode × loss) trainings using the base config."
    )
    parser.add_argument("--config", type=str, required=True, help="Base YAML config path.")
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help=f"Comma-separated input modes to run. Default: {','.join(DEFAULT_INPUT_MODES)}",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default=None,
        help=f"Comma-separated losses to run. Default: {','.join(DEFAULT_LOSSES)}",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="outputs_params",
        help="Root folder for outputs (each run still gets its own subfolder).",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Optional integer added to the base seed for each run index (to vary seeds across combos).",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)

    input_modes = parse_list_arg(args.modes, DEFAULT_INPUT_MODES)
    losses = parse_list_arg(args.losses, DEFAULT_LOSSES)

    # sanity on dataset-supported modes (kept here to fail fast)
    supported_modes = set(["tvt", "std", "dft_k123", "pca", "full", "dft_power"])
    for m in input_modes:
        if m not in supported_modes:
            raise ValueError(f"Unsupported input_mode '{m}'. Supported: {sorted(supported_modes)}")

    run_idx = 0
    for mode in input_modes:
        for loss in losses:
            run_idx += 1
            cfg = copy.deepcopy(base_cfg)

            # override fields for this combo
            cfg.setdefault("data", {})
            cfg["data"]["input_mode"] = mode

            cfg.setdefault("train", {})
            cfg["train"]["loss"] = loss

            # Optional: vary seed per run to avoid identical splits/augment RNG
            if "seed" in cfg:
                try:
                    cfg["seed"] = int(cfg["seed"]) + args.seed_offset + run_idx
                except Exception:
                    cfg["seed"] = args.seed_offset + run_idx

            # Friendly hint if flow-aware loss is requested without metadata
            if loss == "flow_dice":
                meta = cfg["data"].get("metadata_csv")
                if not meta:
                    print(
                        f"[WARN] flow_dice selected but data.metadata_csv is not set. "
                        "Flow term will only work if your dataset/loader provides v_enc/pixel_size."
                    )

            # Train this combo (train.main builds a name like unet2d_{mode}_c{crop}_b{base}_{loss})
            print("\n" + "=" * 80)
            print(f"Starting run {run_idx}: input_mode={mode} | loss={loss}")
            print("=" * 80)

            train_one_combo(cfg, model_name=None)

    print("\nAll combinations finished.")


if __name__ == "__main__":
    main()
