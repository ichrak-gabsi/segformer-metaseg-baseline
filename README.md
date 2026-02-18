# SegFormer (Cityscapes) Baseline – MetaSeg Export

SegFormer (MIT-B0) semantic segmentation baseline via MMSegmentation.
Exports:
- Cityscapes val predicted labels (PNG)
- MetaSeg-compatible outputs (HDF5: softmax probs + GT + filename)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

