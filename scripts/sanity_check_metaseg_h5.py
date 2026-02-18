"""
Sanity check for MetaSeg HDF5 exports.

Verifies:
- Probability shape and normalization
- GT shape
- Filename integrity
"""

import os
import argparse
import h5py
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    files = sorted([f for f in os.listdir(args.h5_dir) if f.endswith(".h5")])
    print("Found", len(files), "files")

    for f in files[:args.n]:
        path = os.path.join(args.h5_dir, f)
        with h5py.File(path, "r") as h:
            probs = h["probs"][:]
            gt = h["gt"][:]
            fn = h["filename"][()].decode("utf-8")
        print("\n", f)
        print(" filename:", fn)
        print(" probs:", probs.shape, probs.dtype, "sum(first pixel)=", float(probs[0,0,:].sum()))
        print(" gt:", gt.shape, gt.dtype, "unique:", len(np.unique(gt)))

if __name__ == "__main__":
    main()
