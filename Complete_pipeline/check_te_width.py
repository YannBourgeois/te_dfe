#!/usr/bin/env python3
"""Report padded TE-tensor width and realized valid-TE counts for stored sims.

Usage:
    python check_te_width.py 'test_simulations/test_*.npz'
    python check_te_width.py 'results/dataset.feats.npy'   # collated (+ its .te_mask.npy)
    python check_te_width.py 'test_shards/scenario_*.npz'  # RAW, uncapped

The padded width == N_TE_TRAIN_MAX at the time the file was written. If the max
realized valid count is well BELOW the width, the cap was not binding (genomes
just don't have that many TEs); if it pins AT the width, the cap truncated.
"""
import sys, glob
import numpy as np

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    for pat in sys.argv[1:]:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"(no match: {pat})"); continue
        for fn in files:
            if fn.endswith(".npy"):
                a = np.load(fn, mmap_mode="r")
                if "te_mask" in fn or a.ndim == 2 and a.shape == a.shape:  # mask-like
                    # try as mask: (n_sims, width)
                    pass
                if "feats" in fn:
                    width = a.shape[-2] if a.ndim >= 2 else a.shape[-1]
                    print(f"{fn}: n_sims={a.shape[0]}  padded_width={width}")
                    mfn = fn.replace(".feats.npy", ".te_mask.npy")
                    try:
                        m = np.load(mfn, mmap_mode="r"); s = m.sum(-1)
                        print(f"    valid/sim  min/med/max = "
                              f"{int(s.min())}/{int(np.median(s))}/{int(s.max())}"
                              f"   (cap binding if max=={width})")
                    except FileNotFoundError:
                        print(f"    (no {mfn} to count valid TEs)")
                else:
                    print(f"{fn}: shape={a.shape}")
            else:  # .npz
                d = np.load(fn)
                keys = list(d.files)
                if "te_mask" in keys:
                    w = d["te_mask"].shape[-1]; v = int(d["te_mask"].sum())
                    print(f"{fn}: padded_width={w}  valid_TEs={v}"
                          f"   (capped here if v=={w})")
                elif "feats_raw" in keys:
                    print(f"{fn}: RAW uncapped n_te={d['feats_raw'].shape[0]}")
                else:
                    print(f"{fn}: keys={keys}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
