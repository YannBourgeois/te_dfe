#!/usr/bin/env python3
"""
03_train.py — Stream-train SNPE on the memmap dataset.

Adds support for K-flexible aggregators (deepsets, sigmoid_attn) with
variable-K training. Existing collated datasets are reused — the mask
column is appended to each observation at read time (all-ones), then
randomized per-batch during training to expose the network to variable K.

CLI additions:
  --aggregator-kind  {attention, deepsets, sigmoid_attn}    default: attention
  --n-attention-heads N                                      default: 1
  --variable-k                                               flag to enable
  --k-min N                                                  default: 5
  --k-max N                                                  default: N_TRAIN_WINDOWS

For backward compatibility, when --aggregator-kind=attention and no
--variable-k, the behavior is identical to the previous version:
  - SBIWrapper is constructed with include_mask=False
  - The flat tensor layout is unchanged
  - Existing checkpoints continue to work
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from sbi.neural_nets import posterior_nn
from sbi.inference.posteriors.direct_posterior import DirectPosterior

sys.path.insert(0, str(Path(__file__).parent))

# Model selection: te_dfe.config dispatches simple/context at IMPORT time from
# TE_DFE_MODEL, so we must set it before importing te_dfe. A tiny pre-parse of
# sys.argv for --model does that here; the CLI value overrides any env default.
from model_select import add_model_arg, prescan_model
prescan_model()                       # sets os.environ["TE_DFE_MODEL"] if --model given

from te_dfe.config import (
    N_HAPS, N_SITES, N_COVARIATES, N_TRAIN_WINDOWS, N_STATE_CHANNELS,
    N_CONTINUOUS_CHANNELS,
)
from te_dfe.network import (
    TECovariateAttentionNet, SBIWrapper,
)
from te_dfe.prior import DFEBurstPrior


def parse_args():
    p = argparse.ArgumentParser()
    add_model_arg(p)
    p.add_argument("--dataset",    type=str, default="results/dataset")
    p.add_argument("--out",        type=str, default="results/")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--patience",   type=int, default=30)
    p.add_argument("--val-frac",   type=float, default=0.10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--density-estimator", type=str, default="nsf",
                   choices=["maf", "nsf", "mdn"])
    p.add_argument("--symmetric-func", type=str, default="max",
                   choices=["max", "mean", "sum"])
    p.add_argument("--seed", type=int, default=0)

    # ---- New flags for K-flexible training ----
    p.add_argument("--aggregator-kind", type=str, default="attention",
                   choices=["attention", "deepsets", "sigmoid_attn"],
                   help="Cross-window aggregator. attention=K-fixed (legacy); "
                        "deepsets and sigmoid_attn are K-invariant.")
    p.add_argument("--n-attention-heads", type=int, default=1,
                   help="For aggregator-kind=attention only.")
    p.add_argument("--embedding-norm", dest="embedding_norm",
                   action="store_true", default=True,
                   help="LayerNorm the per-window embeddings + pooled context "
                        "(default ON; prevents trunk saturation / K-blindness).")
    p.add_argument("--no-embedding-norm", dest="embedding_norm",
                   action="store_false",
                   help="Disable embedding LayerNorm (legacy v5 behavior; "
                        "use to reproduce pre-normalization checkpoints).")
    p.add_argument("--variable-k", action="store_true",
                   help="Train with variable K (random K per batch sample). "
                        "Recommended for K-invariant aggregators.")
    p.add_argument("--k-min", type=int, default=5,
                   help="Min K when variable-K is enabled.")
    p.add_argument("--k-max", type=int, default=None,
                   help="Max K when variable-K is enabled. Default: N_TRAIN_WINDOWS.")
    p.add_argument("--n-min-train", type=int, default=20,
                   help="(sumstat) Min INDIVIDUAL TEs/sample under variable-N "
                        "(per-TE subsampling; matrix uses per-window --k-min).")
    p.add_argument("--n-max-train", type=int, default=None,
                   help="(sumstat) Max INDIVIDUAL TEs under variable-N (per-TE, "
                        "not per-window). Default N_TE_TRAIN_MAX.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset — includes mask column when include_mask=True
# ---------------------------------------------------------------------------
class MemmapSimDataset(Dataset):
    """
    Lazy dataset reading observations from numpy memmap(s).

    When include_mask=True, each observation has an additional `n_windows`
    elements at the end of its flat tensor, set to all-ones by default.
    The training loop modifies this slice on-the-fly to apply variable-K
    masking.
    """
    def __init__(self, theta_meta: dict, dataset_dir: Path,
                 n_windows: int, include_mask: bool = False):
        self.format       = theta_meta.get("format", "legacy")
        self.thetas       = theta_meta["thetas"]
        self.n_windows    = n_windows
        self.include_mask = include_mask

        # Inferred-parameter subset (marginalize-out). None => keep all columns
        # (no-op slice). Otherwise a list of kept column indices into theta.
        from te_dfe.config import INFER_INDICES, PARAM_NAMES
        self._infer_indices = (None
                               if list(INFER_INDICES) == list(range(len(PARAM_NAMES)))
                               else list(INFER_INDICES))
        if self._infer_indices is not None:
            from te_dfe.config import INFER_NAMES
            print(f"[infer-subset] training to infer {len(INFER_NAMES)} of "
                  f"{len(PARAM_NAMES)} params: {INFER_NAMES} "
                  f"(others marginalized)")

        # Training-stage feature filtering (train-without, for fair comparison).
        from te_dfe.config import (
            TRAIN_ZERO_CHANNELS, TRAIN_ZERO_REGIONS, TRAIN_ZERO_STATES,
            N_TE_SITES, N_TE_FLANK_SITES,
        )
        self._zero_channels = tuple(TRAIN_ZERO_CHANNELS)
        self._zero_regions  = tuple(TRAIN_ZERO_REGIONS)
        self._zero_states   = tuple(TRAIN_ZERO_STATES)
        self._n_te_sites    = N_TE_SITES
        self._n_flank_sites = N_TE_FLANK_SITES
        if self._zero_channels or self._zero_regions or self._zero_states:
            print(f"[train-filter] zeroing channels={self._zero_channels} "
                  f"regions={self._zero_regions} states={self._zero_states} "
                  f"during training")

        if self.format == "sumstat":
            self.feats = np.load(
                dataset_dir / theta_meta["feats_path"], mmap_mode="r",
            )
            self.te_mask = np.load(
                dataset_dir / theta_meta["te_mask_path"], mmap_mode="r",
            )
            assert len(self.feats) == len(self.thetas)
            assert len(self.te_mask) == len(self.thetas)
            from te_dfe.config import SUMSTAT_USE_SEQLEN_COV
            self._use_seqlen_cov = SUMSTAT_USE_SEQLEN_COV
            if self._use_seqlen_cov:
                if "seq_len_bp" not in theta_meta:
                    raise ValueError(
                        "SUMSTAT_USE_SEQLEN_COV is on but the collated dataset "
                        "has no 'seq_len_bp' — re-run 02_collate with sims that "
                        "recorded it (and vary the window count across sims!).")
                self.seq_len_bp = np.asarray(theta_meta["seq_len_bp"])
        elif self.format in ("newer", "new"):
            self.geno = np.load(
                dataset_dir / theta_meta["geno_path"], mmap_mode="r",
            )
            self.cov  = np.load(
                dataset_dir / theta_meta["cov_path"],  mmap_mode="r",
            )
            assert len(self.geno) == len(self.thetas)
            assert len(self.cov)  == len(self.thetas)
            self.has_pos = (self.format == "newer")
            if self.has_pos:
                self.pos = np.load(
                    dataset_dir / theta_meta["pos_path"], mmap_mode="r",
                )
                assert len(self.pos) == len(self.thetas)
        elif self.format == "legacy":
            self.obs = np.load(
                dataset_dir / theta_meta["obs_memmap_path"], mmap_mode="r",
            )
            assert len(self.obs) == len(self.thetas)
        else:
            raise ValueError(f"Unknown format '{self.format}'")

    def __len__(self):
        return len(self.thetas)

    def _maybe_filter(self, geno, pos):
        """Apply the shared train-stage feature filter (no-op if nothing set)."""
        if not (self._zero_channels or self._zero_regions or self._zero_states):
            return geno, pos
        from te_dfe.filtering import apply_feature_filter
        return apply_feature_filter(
            geno, pos,
            zero_channels=self._zero_channels,
            zero_regions=self._zero_regions,
            zero_states=self._zero_states,
            n_te_sites=self._n_te_sites,
            n_flank_sites=self._n_flank_sites,
        )

    def __getitem__(self, idx: int):
        if self.format == "sumstat":
            # Per-TE feature matrix + validity mask. Flat layout consumed by
            # SumstatSBIWrapper: [feats_flat | mask_flat]. No geno/pos/cov.
            feats = np.array(self.feats[idx], dtype=np.float32, copy=True)
            tmask = np.array(self.te_mask[idx], dtype=np.float32, copy=True)
            parts = [feats.flatten(), tmask.flatten()]
            if self._use_seqlen_cov:
                from te_dfe.config import seqlen_covariate
                parts.append(np.array(
                    [seqlen_covariate(float(self.seq_len_bp[idx]))],
                    dtype=np.float32))
            x = np.concatenate(parts)
            theta = self.thetas[idx]
            if self._infer_indices is not None:
                theta = theta[..., self._infer_indices]
            return theta, torch.from_numpy(x)
        if self.format == "newer":
            geno = np.array(self.geno[idx], dtype=np.int8,   copy=True)
            pos  = np.array(self.pos[idx],  dtype=np.float32, copy=True)
            cov  = np.array(self.cov[idx],  dtype=np.float32, copy=True)
            geno, pos = self._maybe_filter(geno, pos)
            parts = [
                geno.astype(np.float32).flatten(),
                pos.flatten(),
                cov.flatten(),
            ]
        elif self.format == "new":
            geno = np.array(self.geno[idx], dtype=np.int8,   copy=True)
            cov  = np.array(self.cov[idx],  dtype=np.float32, copy=True)
            geno, _ = self._maybe_filter(geno, None)
            parts = [
                geno.astype(np.float32).flatten(),
                cov.flatten(),
            ]
        else:  # legacy
            parts = [np.array(self.obs[idx], dtype=np.float32, copy=True)]

        if self.include_mask:
            parts.append(np.ones(self.n_windows, dtype=np.float32))
        x = np.concatenate(parts)
        theta = self.thetas[idx]
        # Marginalize-out: keep only the inferred-parameter columns (the dropped
        # ones still varied in the sims; they are integrated out by training the
        # flow on the reduced vector). self._infer_indices is the full column
        # set when INFER_PARAMS is None, so this is a no-op in the default case.
        if self._infer_indices is not None:
            theta = theta[..., self._infer_indices]
        return theta, torch.from_numpy(x)


# ---------------------------------------------------------------------------
# Variable-K masking — applied to a batch in-place
# ---------------------------------------------------------------------------
def apply_variable_k_mask(
    x: torch.Tensor,
    mask_len: int,
    k_min: int,
    k_max: int,
    mask_offset: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Subsample cardinality by editing each sample's mask block (the `mask_len`
    entries at `mask_offset`): pick K_true ∈ [k_min, k_max], keep a random
    K_true of the CURRENTLY-real (mask==1) entries, zero the rest. Padding
    (mask==0) is never turned on. Matrix: all windows real -> unchanged.
    Sumstat: variable real-TE count -> subsample only real TEs (K_true capped
    at the number that exist). One masker, both representations.

    IMPORTANT (current behavior): for the SUMSTAT representation this subsamples
    INDIVIDUAL TEs, NOT windows. TEs are pooled across windows before training
    and carry no window-of-origin tag, so there is no block structure to mask by
    window the way the MATRIX path does. Window-level (block-bootstrap) masking
    for sumstats would require tagging each TE with its source window and is not
    implemented. The sigmoid-attention sumstat model therefore sees a random
    subset of individual TEs per batch.
    """
    B = x.shape[0]
    out = x.clone()
    for b in range(B):
        cur = out[b, mask_offset:mask_offset + mask_len].detach().cpu().numpy()
        real = np.where(cur > 0.5)[0]
        n_real = int(real.size)
        if n_real == 0:
            continue
        k_true = min(int(rng.integers(k_min, k_max + 1)), n_real)
        keep = rng.permutation(real)[:k_true]
        newm = np.zeros(mask_len, dtype=np.float32)
        newm[keep] = 1.0
        out[b, mask_offset:mask_offset + mask_len] = torch.from_numpy(newm)
    return out


def train_one_epoch(de, loader, optimizer, device,
                    masker_args=None):
    de.train()
    total_loss = 0.0
    n_samples = 0
    for theta, x in loader:
        if masker_args is not None:
            x = apply_variable_k_mask(x, **masker_args)
        theta = theta.to(device, non_blocking=True)
        x     = x.to(device, non_blocking=True)
        log_prob = de.log_prob(theta, x).squeeze(0)
        loss = -log_prob.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = theta.size(0)
        total_loss += float(loss.detach()) * bs
        n_samples  += bs
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate(de, loader, device, masker_args=None):
    """
    During validation we DON'T randomize the mask — we want a consistent
    measure of val log-prob across epochs. The mask stays at all-ones
    (full K). This means val log-prob tracks the network's ability to
    handle the full training distribution, not the variable-K distribution.
    """
    de.eval()
    total = 0.0
    n = 0
    for theta, x in loader:
        theta = theta.to(device, non_blocking=True)
        x     = x.to(device, non_blocking=True)
        log_prob = de.log_prob(theta, x).squeeze(0)
        total += float(log_prob.sum().detach())
        n     += theta.size(0)
    return total / max(n, 1)


def main():
    args = parse_args()
    # REPRESENTATION is model-dependent and only resolved after parse_args()
    # (which runs prescan_model -> TE_DFE_MODEL). Bind it here, before its first
    # use in the variable-k checks below, so it isn't an unbound function-local.
    from te_dfe.config import REPRESENTATION
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset}...")

    dataset_path = Path(args.dataset)
    theta_path = dataset_path.with_suffix(".theta.pt")
    if not theta_path.exists():
        raise FileNotFoundError(
            f"Missing dataset metadata: {theta_path}\n"
            f"Run 02_collate.py first to produce the memmap format."
        )

    meta = torch.load(theta_path, weights_only=False)
    fmt = meta.get("format", "legacy")
    print(f"  format:       {fmt}")
    thetas = meta["thetas"]
    print(f"  thetas:       {thetas.shape}")

    # ---- Decide whether to include the mask column ----
    # Always include the mask when using a K-invariant aggregator (so we
    # can plug in different K at inference time). Otherwise include only
    # when --variable-k is set.
    include_mask = (args.aggregator_kind != "attention") or args.variable_k
    if include_mask:
        print(f"  mask column:  YES (n_windows={N_TRAIN_WINDOWS})")
    else:
        print(f"  mask column:  no")

    full_dataset = MemmapSimDataset(
        meta, dataset_path.parent,
        n_windows=N_TRAIN_WINDOWS,
        include_mask=include_mask,
    )
    print(f"  total sims:   {len(full_dataset)}")

    # ---- Validate K range ----
    k_max = args.k_max if args.k_max is not None else N_TRAIN_WINDOWS
    if args.variable_k and REPRESENTATION == "sumstat":
        from te_dfe.config import SUMSTAT_USE_SEQLEN_COV
        if SUMSTAT_USE_SEQLEN_COV:
            print("  WARNING: variable-N and the seq-length covariate are both "
                  "on; not jointly scaled. Prefer ONE (variable-N with cov OFF, "
                  "or cov + variable window count at sim time).")
    if args.variable_k and REPRESENTATION != "sumstat":
        if k_max > N_TRAIN_WINDOWS:
            raise ValueError(
                f"--k-max ({k_max}) > N_TRAIN_WINDOWS ({N_TRAIN_WINDOWS})"
            )
        if args.k_min < 1 or args.k_min > k_max:
            raise ValueError(
                f"--k-min ({args.k_min}) must be in [1, k_max={k_max}]"
            )
        print(f"  variable-K:   K ~ Uniform[{args.k_min}, {k_max}]")

    # ---- Train/val split ----
    n_val   = max(1, int(round(args.val_frac * len(full_dataset))))
    n_train = len(full_dataset) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val], generator=generator,
    )
    print(f"  train / val:  {n_train} / {n_val}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ---- Build the embedding network ----
    print(f"  aggregator:   {args.aggregator_kind}", end="")
    if args.aggregator_kind == "attention":
        print(f"  (heads={args.n_attention_heads})")
    else:
        print()
    print(f"  embedding-norm: {args.embedding_norm}")

    if REPRESENTATION == "sumstat":
        from te_dfe.config import (N_SUMSTATS_PER_TE, N_TE_TRAIN_MAX,
                                   N_SUMSTAT_GLOBAL_COV)
        from te_dfe.network import TESumstatNet, SumstatSBIWrapper
        core = TESumstatNet(
            n_features        = N_SUMSTATS_PER_TE,
            aggregator_kind   = args.aggregator_kind,
            n_attention_heads = args.n_attention_heads,
            use_embedding_norm = args.embedding_norm,
            n_global_cov      = N_SUMSTAT_GLOBAL_COV,
        )
        if N_SUMSTAT_GLOBAL_COV:
            print(f"  global covariate: seq-length ({N_SUMSTAT_GLOBAL_COV}-dim)")
        n_params = sum(p.numel() for p in core.parameters())
        print(f"  sumstat network params: {n_params:,}")
        emb_net = SumstatSBIWrapper(
            network=core, n_te_max=N_TE_TRAIN_MAX,
            n_features=N_SUMSTATS_PER_TE, include_mask=True,
            n_global_cov=N_SUMSTAT_GLOBAL_COV,
        )
    else:
        core = TECovariateAttentionNet(
            n_individuals    = N_HAPS,
            n_sites          = N_SITES,
            n_covariates     = N_COVARIATES,
            n_state_channels = N_STATE_CHANNELS,
            n_cont_channels  = N_CONTINUOUS_CHANNELS,
            symmetric_func   = args.symmetric_func,
            aggregator_kind  = args.aggregator_kind,
            n_attention_heads= args.n_attention_heads,
            use_embedding_norm = args.embedding_norm,
        )
        n_params = sum(p.numel() for p in core.parameters())
        print(f"  network params: {n_params:,}")

        emb_net = SBIWrapper(
            network=core, n_windows=N_TRAIN_WINDOWS,
            n_channels=N_STATE_CHANNELS, n_individuals=N_HAPS,
            n_sites=N_SITES, n_cov=N_COVARIATES,
            n_position_channels=N_CONTINUOUS_CHANNELS,
            include_mask=include_mask,
        )

    from te_dfe.prior import sbi_prior_subset
    prior = sbi_prior_subset()       # marginal over inferred subset (full if all)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build density estimator
    de_builder = posterior_nn(
        model=args.density_estimator, embedding_net=emb_net,
        z_score_x="none", z_score_theta="independent",
    )
    theta_init, x_init = next(iter(train_loader))
    de = de_builder(theta_init, x_init).to(device)
    print(f"Density estimator: {type(de).__name__}")

    optimizer = torch.optim.Adam(de.parameters(), lr=args.lr)

    # ---- Configure masker args (passed to train_one_epoch each call) ----
    if args.variable_k:
        if REPRESENTATION == "sumstat":
            from te_dfe.config import N_SUMSTATS_PER_TE, N_TE_TRAIN_MAX
            n_min = args.n_min_train
            n_max = args.n_max_train if args.n_max_train is not None else N_TE_TRAIN_MAX
            if not (1 <= n_min <= n_max <= N_TE_TRAIN_MAX):
                raise ValueError(f"variable-N bounds [{n_min},{n_max}] need "
                                 f"1<=n_min<=n_max<=N_TE_TRAIN_MAX({N_TE_TRAIN_MAX}).")
            masker_args = dict(mask_len=N_TE_TRAIN_MAX, k_min=n_min, k_max=n_max,
                               mask_offset=N_TE_TRAIN_MAX * N_SUMSTATS_PER_TE, rng=rng)
            print(f"  variable-N (sumstat): subsample INDIVIDUAL TEs "
                  f"~ U[{n_min},{n_max}] per sample (per-TE, NOT per-window)")
        else:
            mask_offset = (
                N_TRAIN_WINDOWS * N_HAPS * N_SITES
                + N_TRAIN_WINDOWS * N_CONTINUOUS_CHANNELS * N_SITES
                + N_TRAIN_WINDOWS * N_COVARIATES
            )
            masker_args = dict(mask_len=N_TRAIN_WINDOWS, k_min=args.k_min,
                               k_max=k_max, mask_offset=mask_offset, rng=rng)
    else:
        masker_args = None

    # ---- Training loop ----
    print("Training (streaming)...")
    t0 = time.time()
    train_log_probs = []
    val_log_probs   = []
    best_val = -float("inf")
    best_state = None
    epochs_since_improve = 0
    n_epochs_actual = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            de, train_loader, optimizer, device,
            masker_args=masker_args,
        )
        val_lp = validate(de, val_loader, device)
        train_log_probs.append(-train_loss)
        val_log_probs.append(val_lp)
        n_epochs_actual = epoch

        improved = val_lp > best_val
        if improved:
            best_val = val_lp
            best_state = {k: v.detach().cpu().clone()
                          for k, v in de.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        elapsed = time.time() - t0
        marker = "*" if improved else " "
        print(f"  [{epoch:3d}] train log p={-train_loss:+.4f}  "
              f"val log p={val_lp:+.4f}{marker}  "
              f"(best={best_val:+.4f}, {elapsed:.0f}s)")

        if epochs_since_improve >= args.patience:
            print(f"  Early stop: no improvement for {args.patience} epochs")
            break

    if best_state is not None:
        de.load_state_dict(best_state)

    print(f"Training done in {time.time()-t0:.1f}s ({n_epochs_actual} epochs)")

    # ---- Save ----
    np.savez(
        out_dir / "training_curves.npz",
        train_log_probs=np.array(train_log_probs, dtype=np.float32),
        val_log_probs=np.array(val_log_probs,     dtype=np.float32),
        n_epochs=int(n_epochs_actual),
    )
    print(f"Final train log-prob: {train_log_probs[-1]:+.4f}")
    print(f"Final val   log-prob: {val_log_probs[-1]:+.4f}")
    print(f"Best  val   log-prob: {best_val:+.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        epochs_x = np.arange(1, len(train_log_probs)+1)
        ax.plot(epochs_x, train_log_probs, label="train", lw=1.5)
        ax.plot(epochs_x, val_log_probs,   label="validation", lw=1.5, ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("log p(theta | x)  (higher = better)")
        ax.set_title(f"Training curves ({n_epochs_actual} epochs)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.pdf", dpi=140)
        plt.close(fig)
        print(f"  Saved → {out_dir}/training_curves.pdf")
    except Exception as e:
        print(f"  (Loss plot skipped: {e})")

    # ---- Save model artifacts. We also save metadata about whether
    # the SBIWrapper expects a mask, so 04_infer.py and 05_benchmark.py
    # can adapt their observation construction.
    posterior = DirectPosterior(
        posterior_estimator=de.cpu(),
        prior=prior, device="cpu",
    )

    torch.save(core.state_dict(), out_dir / "network.pt")
    with open(out_dir / "posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)
    with open(out_dir / "inference_engine.pkl", "wb") as f:
        pickle.dump({"prior": prior, "posterior": posterior}, f)

    # Metadata for downstream scripts
    from te_dfe.config import (INFER_NAMES, INFER_INDICES, PARAM_NAMES,
                               MODEL_NAME, FIX_PARAMS as _FIXP)
    with open(out_dir / "training_meta.pkl", "wb") as f:
        pickle.dump({
            "aggregator_kind":    args.aggregator_kind,
            "n_attention_heads":  args.n_attention_heads,
            "include_mask":       include_mask,
            "n_train_windows":    N_TRAIN_WINDOWS,
            "variable_k":         args.variable_k,
            "k_min":              args.k_min if args.variable_k else None,
            "k_max":              k_max      if args.variable_k else None,
            "use_embedding_norm": args.embedding_norm,
            "representation":     REPRESENTATION,
            "sumstat_seqlen_cov": bool(__import__("te_dfe.config",
                                      fromlist=["SUMSTAT_USE_SEQLEN_COV"]
                                      ).SUMSTAT_USE_SEQLEN_COV),
            # Model + inferred-parameter subset, so inference/benchmarking slice
            # theta to the SAME columns this checkpoint was trained on.
            "model_name":         MODEL_NAME,
            "param_names_full":   list(PARAM_NAMES),
            "infer_names":        list(INFER_NAMES),
            "infer_indices":      list(INFER_INDICES),
            "fix_params":         dict(_FIXP or {}),
        }, f)

    print(f"Saved → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
