"""
Run SegFormer inference on a single image.

Purpose:
- Quick smoke test for model + checkpoint.
- Verifies config, weights, and inference pipeline.

Output:
- Predicted label map saved to outputs/one_image/.
"""

import os
import numpy as np
import torch

from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules

register_all_modules()

CONFIG = "configs/segformer_mit-b0_cityscapes.py"
CKPT   = "checkpoints/segformer_mit-b0_cityscapes.pth"
IMG    = "inputs/test.jpg"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = init_model(CONFIG, CKPT, device=device)
result = inference_model(model, IMG)

# predicted class per pixel
pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)

os.makedirs("outputs/one_image", exist_ok=True)
np.save("outputs/one_image/pred.npy", pred)

print("Saved:", "outputs/one_image/pred.npy", "shape:", pred.shape, "dtype:", pred.dtype)
