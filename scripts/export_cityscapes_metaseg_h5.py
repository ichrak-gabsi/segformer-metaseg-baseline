"""
Export Cityscapes val predictions in MetaSeg HDF5 format.

Each .h5 file contains:
- probs (H, W, C) softmax probabilities
- gt (H, W) ground truth labels
- filename (str)

Output:
- outputs/cityscapes_val_metaseg_h5/*.h5
"""

import os
import argparse
from glob import glob

import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from mmseg.apis import init_model, inference_model


def find_pairs(root: str):
    img_pat = os.path.join(root, "leftImg8bit", "val", "*", "*_leftImg8bit.png")
    imgs = sorted(glob(img_pat))
    if not imgs:
        raise FileNotFoundError(f"No images found: {img_pat}")

    pairs = []
    for img_path in imgs:
        city = os.path.basename(os.path.dirname(img_path))
        stem = os.path.basename(img_path).replace("_leftImg8bit.png", "")
        gt_path = os.path.join(root, "gtFine", "val", city, f"{stem}_gtFine_labelIds.png")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing GT: {gt_path}")
        pairs.append((img_path, gt_path))
    return pairs


def read_gt(path: str) -> np.ndarray:
    gt = np.array(Image.open(path))
    if gt.ndim != 2:
        raise ValueError(f"GT must be single-channel, got {gt.shape} for {path}")
    return gt.astype(np.int32)


def get_probs(result) -> np.ndarray:
    # Need logits to compute softmax probabilities (MetaSeg needs probs, not only argmax labels)
    if not hasattr(result, "seg_logits") or result.seg_logits is None:
        raise RuntimeError("No seg_logits found. Cannot compute softmax probs for MetaSeg.")

    logits = result.seg_logits.data  # (C,H,W)
    probs = F.softmax(logits, dim=0).permute(1, 2, 0).contiguous()  # (H,W,C)
    return probs.detach().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cityscapes-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    model = init_model(args.config, args.checkpoint, device=device)

    pairs = find_pairs(args.cityscapes_root)
    print(f"Found {len(pairs)} Cityscapes val pairs")

    for img_path, gt_path in tqdm(pairs, desc="Export MetaSeg H5"):
        city = os.path.basename(os.path.dirname(img_path))
        stem = os.path.basename(img_path).replace("_leftImg8bit.png", "")
        out_path = os.path.join(args.out_dir, f"{city}__{stem}.h5")

        result = inference_model(model, img_path)
        probs = get_probs(result)       # (H,W,C)
        gt = read_gt(gt_path)           # (H,W)

        if gt.shape != probs.shape[:2]:
            raise ValueError(f"Shape mismatch: gt {gt.shape} vs probs {probs.shape} for {img_path}")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("probs", data=probs, compression="gzip", compression_opts=4)
            f.create_dataset("gt", data=gt, compression="gzip", compression_opts=4)
            f.create_dataset("filename", data=np.string_(os.path.basename(img_path)))

    print("Done.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available.")
    main()
