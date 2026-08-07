#!/usr/bin/env python3
"""04_infer.py — Apply trained DFE + burst posterior to data with variable W."""
import argparse
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parent))
# Model selection before any te_dfe import (config dispatches at import time).
try:
    from model_select import prescan_model, add_model_arg
    prescan_model()
except Exception:
    add_model_arg = None


from te_dfe.config import (
    N_HAPS, N_SITES, N_COVARIATES, PARAM_NAMES, PARAM_LABELS,
    N_STATE_CHANNELS, N_CONTINUOUS_CHANNELS,
    MAF_MIN, SIM_LENGTH,
    MSMC_PATH, MU, RECOMB_MAP_PATH, ANNOTATION_PATH, ANNOTATION_FMT, CHROMOSOMES,
)
from te_dfe.demography import (
    MSMCDemography, GenomeAnnotation, WindowSampler,
)
from te_dfe.network   import TECovariateAttentionNet
from te_dfe.processor import TEDFEBurstProcessor
from te_dfe.simulator import DFEBurstSimulator


@contextmanager
def bypass_embedding_net(posterior, embedding_dim: int):
    de = posterior.posterior_estimator
    flow = de.net if hasattr(de, "net") else de
    orig_emb   = getattr(flow, "_embedding_net", None)
    orig_shape = getattr(de,   "_condition_shape", None)
    if orig_emb is not None:
        flow._embedding_net = nn.Identity()
    if orig_shape is not None:
        de._condition_shape = torch.Size([embedding_dim])
    try:
        yield
    finally:
        if orig_emb is not None:
            flow._embedding_net = orig_emb
        if orig_shape is not None:
            de._condition_shape = orig_shape


class DFEBurstInference:
    def __init__(self, network, posterior, device="cpu"):
        self.network   = network.to(device).eval()
        self.posterior = posterior
        self.device    = device
        self.processor = TEDFEBurstProcessor(maf_min=MAF_MIN)

    @classmethod
    def load(cls, results_dir, device="cpu"):
        d = Path(results_dir)
        # Optionally read training metadata (aggregator kind etc.) — if
        # missing, fall back to "attention" defaults for backward compat.
        meta_path = d / "training_meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            aggregator_kind  = meta.get("aggregator_kind", "attention")
            n_attention_heads = meta.get("n_attention_heads", 1)
            use_embedding_norm = meta.get("use_embedding_norm", False)
        else:
            aggregator_kind, n_attention_heads, use_embedding_norm = "attention", 1, False
        net = TECovariateAttentionNet(
            n_individuals=N_HAPS, n_sites=N_SITES, n_covariates=N_COVARIATES,
            n_state_channels=N_STATE_CHANNELS, n_cont_channels=N_CONTINUOUS_CHANNELS,
            aggregator_kind=aggregator_kind,
            n_attention_heads=n_attention_heads,
            use_embedding_norm=use_embedding_norm,
        )
        net.load_state_dict(torch.load(d / "network.pt", map_location=device))
        with open(d / "posterior.pkl", "rb") as f:
            posterior = pickle.load(f)
        return cls(net, posterior, device)

    def get_embedding(self, ts_list, covariates, recomb_maps=None):
        # Processor now returns (geno, pos). recomb_maps optional; if None,
        # pos channels will be zeros (acceptable degradation for inference
        # on real data without a recomb map).
        if recomb_maps is None:
            recomb_maps = [None] * len(ts_list)
        results = [self.processor(ts, recomb_map=rm)
                    for ts, rm in zip(ts_list, recomb_maps)]
        geno_list = [r[0] for r in results]
        pos_list  = [r[1] for r in results]
        genotypes = torch.tensor(np.stack(geno_list), dtype=torch.float32, device=self.device)
        positions = torch.tensor(np.stack(pos_list),  dtype=torch.float32, device=self.device)
        covs      = torch.tensor(covariates,           dtype=torch.float32, device=self.device)
        with torch.no_grad():
            emb, weights = self.network(
                genotypes.unsqueeze(0), positions.unsqueeze(0), covs.unsqueeze(0),
            )
        return emb.squeeze(0).cpu(), weights.squeeze(0).cpu()

    def run(self, ts_list, covariates, n_samples=2000):
        print(f"Embedding from {len(ts_list)} windows...")
        emb, attn = self.get_embedding(ts_list, covariates)
        print(f"  Embedding: {emb.shape}")
        print(f"Sampling {n_samples} posterior samples...")
        with bypass_embedding_net(self.posterior, embedding_dim=emb.shape[0]):
            samples = self.posterior.sample(
                (n_samples,), x=emb.unsqueeze(0),
            ).numpy()
        return {
            "samples":      samples,
            "attn_weights": attn.numpy(),
            "param_names":  PARAM_NAMES,
            "param_labels": PARAM_LABELS,
            "n_windows":    len(ts_list),
        }

    @staticmethod
    def summary(results, true_theta=None, present_day_ne=None):
        samples = results["samples"]
        print(f"\nPosterior summary ({results['n_windows']} windows)")
        print("=" * 78)
        hdr = f"{'Parameter':<40}{'Mean':>8}{'5%':>8}{'Median':>8}{'95%':>8}"
        if true_theta is not None: hdr += f"{'True':>8}"
        print(hdr); print("-" * 78)
        for i, lbl in enumerate(results["param_labels"]):
            s = samples[:, i].copy()
            t = true_theta[i] if true_theta is not None else None
            name = results["param_names"][i]
            # v4: all params are log-scale or log10 transforms
            if name in ("log_lambda", "log_gamma_shape", "log_mean_s",
                          "log_duration", "log_multiplier"):
                s = np.exp(s); t = float(np.exp(t)) if t is not None else None
            elif name == "log10_t_start":
                s = 10.0 ** s; t = float(10.0 ** t) if t is not None else None
            row = (f"{lbl:<40}{s.mean():>8.3g}{np.percentile(s,5):>8.3g}"
                   f"{np.median(s):>8.3g}{np.percentile(s,95):>8.3g}")
            if t is not None:
                row += f"{t:>8.3g}"
            print(row)
        print("=" * 78)

        # ---- Derived: binned DFE proportions via gamma CDF ----
        # Push posterior samples on (log_gamma_shape, log_mean_s) through
        # the gamma CDF to derive (p_nn, p_w, p_strong) posteriors.
        from te_dfe.prior import compute_binned_props
        idx_log_shape = results["param_names"].index("log_gamma_shape")
        idx_log_ms    = results["param_names"].index("log_mean_s")
        log_shape_post = samples[:, idx_log_shape]
        log_ms_post    = samples[:, idx_log_ms]
        # Use harmonic Ne ~500 by default (can also take from results)
        ne_h = float(results.get("ne_harmonic", 500))
        p_nn_post, p_w_post, p_strong_post = compute_binned_props(
            log_shape_post, log_ms_post, ne_h,
        )

        labels = [f"p_nn      (|2Ne.s| < 1, Ne_h={ne_h:.0f})",
                  f"p_w       (1 <= |2Ne.s| < 10)",
                  f"p_strong  (|2Ne.s| >= 10)"]
        posts = [p_nn_post, p_w_post, p_strong_post]
        print(f"\nDerived DFE category proportions (post-hoc via gamma CDF):")
        print("=" * 78)
        print(f"{'bin':<30}{'mean':>10}{'5%':>10}{'median':>10}{'95%':>10}")
        print("-" * 78)
        for lbl, col in zip(labels, posts):
            print(f"{lbl:<30}{col.mean():>10.3f}"
                  f"{np.percentile(col, 5):>10.3f}"
                  f"{np.median(col):>10.3f}"
                  f"{np.percentile(col, 95):>10.3f}")
        print("=" * 78)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results",   type=str, required=True)
    p.add_argument("--test-seed", type=int, default=None)
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--out",       type=str, default=None)
    p.add_argument("--present-day-ne", type=float, default=None,
                   help="Present-day Ne, used to compute discretized "
                        "|2Ne.s| DFE summary. If not provided, this "
                        "section is skipped.")
    # --- Real-data (VCF) inference, sumstat representation ---
    p.add_argument("--vcf-snp", type=str, default=None,
                   help="SNP VCF for real-data inference (triggers the VCF path; "
                        "requires REPRESENTATION=sumstat).")
    p.add_argument("--vcf-te", type=str, default=None,
                   help="TE VCF (omit + use --joint if TEs are flagged in the SNP VCF).")
    p.add_argument("--windows-bed", type=str, default=None,
                   help="One 'chrom start size' per line: the windows to pool TEs over.")
    p.add_argument("--annotation", type=str, default=None,
                   help="GTF/GFF for dist_to_gene on real data (else dist_to_gene=0).")
    p.add_argument("--joint", action="store_true",
                   help="TEs flagged inside the SNP VCF (no separate TE VCF).")
    p.add_argument("--te-info-field", type=str, default=None)
    p.add_argument("--te-info-value", type=str, default="TE")
    # Register --model so argparse accepts it (prescan_model already
    # consumed its VALUE into TE_DFE_MODEL; this makes the flag legal).
    if add_model_arg is not None:
        add_model_arg(p)
    return p.parse_args()


def run_vcf_sumstat(args):
    """
    Real-data inference (sumstat): pool per-TE features over the genome's
    windows, build the flat observation (+ optional seq-length covariate), and
    sample the trained posterior. The pickled posterior carries its own
    SumstatSBIWrapper, so we sample it directly (no matrix net needed).
    """
    import pickle
    from te_dfe.config import REPRESENTATION, SUMSTAT_USE_SEQLEN_COV, INFER_NAMES
    from te_dfe.processor import TEDFEBurstProcessor
    from vcf_to_tensors import (vcf_to_sumstat_observation, sumstat_flat_observation)
    if REPRESENTATION != "sumstat":
        print("ERROR: --vcf-snp requires REPRESENTATION=sumstat (set via --model).",
              file=sys.stderr); return 1
    if args.windows_bed is None:
        print("ERROR: --windows-bed is required for the VCF path.", file=sys.stderr)
        return 1
    windows = []
    with open(args.windows_bed) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chrom, start, size = line.split()[:3]
            windows.append((chrom, int(start), int(size)))
    print(f"Pooling TEs over {len(windows)} windows...")

    annotation = None
    if args.annotation is not None:
        annotation = (GenomeAnnotation.from_gtf(args.annotation)
                      if str(args.annotation).lower().endswith((".gtf", ".gtf.gz"))
                      else GenomeAnnotation.from_gff(args.annotation))

    proc = TEDFEBurstProcessor(maf_min=MAF_MIN)
    feats, te_mask, seq_len_bp = vcf_to_sumstat_observation(
        proc, args.vcf_snp, args.vcf_te, windows,
        te_info_field=args.te_info_field, te_info_value=args.te_info_value,
        joint=args.joint, annotation=annotation)
    n_te = int(te_mask.sum())
    print(f"  pooled {n_te} TEs; total sequence = {seq_len_bp/1e6:.1f} Mb"
          + ("; seqlen covariate ON" if SUMSTAT_USE_SEQLEN_COV else "")
          + ("" if annotation is not None else "; dist_to_gene=0 (no --annotation)"))
    if n_te == 0:
        print("ERROR: no TEs found in the given windows.", file=sys.stderr); return 1

    flat = sumstat_flat_observation(feats, te_mask, seq_len_bp)
    with open(Path(args.results) / "posterior.pkl", "rb") as f:
        posterior = pickle.load(f)
    x = torch.from_numpy(flat).float().unsqueeze(0)
    print(f"Sampling {args.n_samples} posterior samples...")
    samples = np.asarray(posterior.sample((args.n_samples,), x=x,
                                          show_progress_bars=False))
    print("\nPosterior (real data):")
    for j, nm in enumerate(INFER_NAMES):
        q = np.percentile(samples[:, j], [5, 50, 95])
        print(f"  {nm:24s} {q[1]:+.4f}  [90% CI {q[0]:+.4f}, {q[2]:+.4f}]")
    if args.out:
        np.savez_compressed(args.out, samples=samples, n_te=n_te,
                            seq_len_bp=seq_len_bp)
        print(f"\nSaved -> {args.out}")
    return 0


def main():
    args = parse_args()
    if args.vcf_snp is not None:
        return run_vcf_sumstat(args)
    infer = DFEBurstInference.load(args.results)
    if args.test_seed is None:
        print("ERROR: provide --test-seed", file=sys.stderr)
        return 1

    print(f"Loading genomic resources for synthetic test...")
    demog = MSMCDemography.from_file(MSMC_PATH, mu=MU, sim_length=SIM_LENGTH)
    annot = (GenomeAnnotation.from_gtf(ANNOTATION_PATH)
             if ANNOTATION_FMT == "gtf"
             else GenomeAnnotation.from_gff(ANNOTATION_PATH))
    sampler = WindowSampler(
        recomb_map_path=RECOMB_MAP_PATH, annotation=annot,
        chromosomes=CHROMOSOMES,
    )
    sim = DFEBurstSimulator(demog, sampler, n_windows=args.n_windows)

    print(f"Simulating test rep (seed={args.test_seed}, W={args.n_windows})...")
    windows, true_theta = sim(seed=args.test_seed)
    ts_list = [ts for ts, *_ in windows]
    covariates = np.stack([cov for _, cov, *_ in windows])

    results = infer.run(ts_list, covariates, n_samples=args.n_samples)
    infer.summary(results, true_theta=true_theta,
                  present_day_ne=args.present_day_ne)

    if args.out:
        np.savez_compressed(
            args.out, samples=results["samples"],
            attn_weights=results["attn_weights"], true_theta=true_theta,
        )
        print(f"\nSaved → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
