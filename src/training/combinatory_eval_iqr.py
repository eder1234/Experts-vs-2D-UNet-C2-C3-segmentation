import sys
import argparse
import torch
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path FIRST
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from src
from src.datasets.csf_volume_dataset import CSFVolumeDataset
from src.models.unet2d import UNet2D
from src.utils.metrics import compute_all
from src.utils.misc import load_ckpt, load_yaml

def extract_run_info(run_dir_name: str) -> Dict[str, Any]:
    """Extract params from run directory name to override config defaults."""
    match = re.search(
        r"unet2d_(?P<mode>\w+)_c(?P<crop>\d+)_b(?P<base>\d+)_(?P<loss>[\w_]+)",
        run_dir_name
    )
    if match:
        return {
            "mode": match.group("mode"),
            "loss": match.group("loss"),
            "crop_size": int(match.group("crop")),
            "base_channels": int(match.group("base"))
        }
    return {"mode": "unknown", "loss": "unknown", "crop_size": 80, "base_channels": 32}

def compute_iqr_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean and IQR statistics from a list of values."""
    values = np.array(values)
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "count": len(values)
    }

def evaluate_single_run(
    run_dir: Path,
    run_info: Dict[str, Any],
    split: str = "test",
    thresh: float = 0.5
) -> List[float]:
    """Evaluate a single run and return a list of Dice scores."""
    
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    
    # 1. Strategy for Config: Run-specific -> Root -> Fail
    run_cfg_path = run_dir / "config.yaml"
    root_cfg_path = project_root / "config.yaml"
    
    if run_cfg_path.exists():
        print(f"Loading config from: {run_cfg_path}")
        cfg = load_yaml(run_cfg_path)
    elif root_cfg_path.exists():
        print(f"Warning: Missing run config. Fallback to root config: {root_cfg_path}")
        cfg = load_yaml(root_cfg_path)
    else:
        print("Error: No config.yaml found in run dir or project root.")
        return []

    if not ckpt_path.exists():
        print(f"Error: Missing checkpoint in {run_dir}")
        return []
    
    # 2. Extract parameters & Determine Correct Data Path
    try:
        data_root_base = Path(cfg["data"]["root"])
        
        # Append the specific subdirectory for the requested split
        if split == "test":
            subdir = cfg["data"].get("test_dir", "test")
            data_root = data_root_base / subdir
        else:
            subdir = cfg["data"].get("train_dir", "train")
            data_root = data_root_base / subdir

        # Robust check for data path
        if not data_root.exists():
            resolved_root = (project_root / data_root).resolve()
            if resolved_root.exists():
                data_root = resolved_root
            else:
                print(f"Error: Constructed data path does not exist: {data_root}")
                return []
        
        # Priority: Folder Name > Config File > Default
        crop_size = run_info.get("crop_size", cfg["data"].get("crop_size", 80))
        input_mode = run_info.get("mode", cfg["data"].get("input_mode", "full"))
        
        # Model Params
        mode_map = {
            "full": 64, "pca": 1, "std": 1, "tvt": 1, "dft_power": 1, "dft_k123": 3
        }
        
        if input_mode in mode_map:
            in_channels = mode_map[input_mode]
        else:
            in_channels = cfg["model"]["in_channels"]
            
        out_channels = cfg["model"]["out_channels"]
        base_channels = run_info.get("base_channels", cfg["model"].get("base_channels", 32))
        
    except KeyError as e:
        print(f"Error parsing config: Missing key {e}")
        return []

    # 3. Load Dataset
    print(f"Loading dataset from: {data_root} (Split: {split})")
    dataset = CSFVolumeDataset(
        root_dir=data_root,
        split=split,
        crop_size=crop_size,
        input_mode=input_mode
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(dataset),
        num_workers=0
    )
    
    # 4. Load Model
    model = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels
    )
    
    try:
        # Load checkpoint
        checkpoint = load_ckpt(ckpt_path)
        
        # Handle state dict wrapping
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load weights.")
        print(f"Model expects {in_channels} input channels for mode '{input_mode}'.")
        print(f"Details: {e}")
        return []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    dice_scores = []
    
    print(f"Running inference on {len(dataset)} subjects (Mode={input_mode}, Crop={crop_size}, Base={base_channels})...")
    
    with torch.no_grad():
        for i, sample in enumerate(loader):
            inp = sample["image"].to(device)
            gt_mask = sample["mask"].to(device)
            
            # Predict
            pred_logits = model(inp)
            pred_prob = torch.sigmoid(pred_logits)
            pred_mask = (pred_prob > thresh).float()
            
            # Compute Dice
            # FIX: removed 'thresh=thresh' argument
            m = compute_all(pred_mask.cpu().numpy(), gt_mask.cpu().numpy())
            dice_scores.append(m["dice"])
            
    return dice_scores

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-subject IQR stats for a specific model run"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the specific run directory"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test", 
        help="Dataset split to evaluate (default: test)"
    )

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}")
        return

    # Extract info 
    run_info = extract_run_info(args.run_dir.name)
    print(f"Evaluating Run: {args.run_dir.name}")
    print(f"Parsed Info: Mode={run_info['mode']}, Loss={run_info['loss']}, Crop={run_info['crop_size']}, Base={run_info['base_channels']}")
    print("-" * 50)

    # Calculate
    dice_scores = evaluate_single_run(args.run_dir, run_info, split=args.split)

    if not dice_scores:
        print("No scores computed.")
        return

    # Stats
    stats = compute_iqr_stats(dice_scores)

    print(f"\nFinal Stats over {stats['count']} subjects:")
    print(f"==========================================")
    print(f"Mean Dice: {stats['mean']:.4f}")
    print(f"Median   : {stats['median']:.4f}")
    print(f"IQR      : [{stats['q1']:.4f}, {stats['q3']:.4f}]")
    print(f"Std Dev  : {stats['std']:.4f}")
    print(f"==========================================")
    
    print(f"\nCopy this for your paper:")
    print(f"Dice {stats['mean']:.3f} [{stats['q1']:.3f}--{stats['q3']:.3f}]")

if __name__ == "__main__":
    main()