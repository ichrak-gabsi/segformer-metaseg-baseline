"""
Export SegFormer predictions on Cityscapes val split (argmax labels).

- Iterates over leftImg8bit/val/.
- Saves uint8 class index PNGs.

Output:
- outputs/cityscapes_val_labels/*.png
"""

import os
import argparse
from glob import glob
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
from mmseg.apis import init_model, inference_model

def find_cityscapes_images(root: str):
    pattern = os.path.join(root, "leftImg8bit", "val", "*", "*_leftImg8bit.png")
    imgs = sorted(glob(pattern))
    if not imgs:
        raise FileNotFoundError(f"No images found: {pattern}")
    return imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cityscapes-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = init_model(args.config, args.checkpoint, device=args.device)

    imgs = find_cityscapes_images(args.cityscapes_root)
    print(f"Found {len(imgs)} Cityscapes val images")

    for img_path in tqdm(imgs, desc="Export labels"):
        city = os.path.basename(os.path.dirname(img_path))
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{city}__{stem}.png"
        out_path = os.path.join(args.out_dir, out_name)

        result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        imageio.imwrite(out_path, pred)

    print("Done.")

if __name__ == "__main__":
    main()

