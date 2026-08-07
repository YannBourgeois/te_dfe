"""
network.py — two-branch per-window encoder + flexible cross-window aggregator.

Per-window encoder (TwoBranchEncoder):
  STATE BRANCH (one-hot genotype, (H, S)):
      Conv2d → ELU → FrozenLayerNorm
      Conv2d → ELU → FrozenLayerNorm
      SymmetricLayer (max over H — popgen-npe pattern)
      squeeze H → (B, 160, S')

  CONTINUOUS BRANCH (position + density, (S) only):
      Conv1d → ELU → FrozenLayerNorm1d
      Conv1d → ELU → FrozenLayerNorm1d
                                → (B, 64, S')

  Concat on channel dim → (B, 224, S') → AdaptiveAvgPool1d(1) → Flatten
                                       → Linear → ELU → Linear → per_window_dim

Cross-window aggregator: selectable via `aggregator_kind`:
  "attention"   — softmax attention (K-fixed at train+infer time)
  "deepsets"    — mean+max+std pool (K-invariant)
  "sigmoid_attn"— per-window sigmoid gate * value, weighted mean (K-invariant)

For variable-K training, the aggregator and the std-features computation
both respect a (B, W) mask where 1 = real window, 0 = padding.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .aggregators import make_aggregator


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class FrozenLayerNorm(nn.Module):
    """LayerNorm without learnable affine. Works for Conv1d and Conv2d shapes."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[1:])


class SymmetricLayer(nn.Module):
    """Permutation-invariant pool over `axis` (max | mean | sum)."""
    def __init__(self, axis: int, func: str = "max"):
        super().__init__()
        self.axis = axis
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.func == "max":
            return torch.max(x, dim=self.axis, keepdim=True)[0]
        elif self.func == "mean":
            return torch.mean(x, dim=self.axis, keepdim=True)
        elif self.func == "sum":
            return torch.sum(x, dim=self.axis, keepdim=True)
        raise ValueError(f"unknown symmetric func: {self.func}")


# ---------------------------------------------------------------------------
# State branch — Conv2d over (H, S) with haplotype-symmetric pooling
# ---------------------------------------------------------------------------
class StateBranch(nn.Module):
    """One-hot state channels (B, n_state, H, S) → (B, n_filters_out, S')."""
    def __init__(
        self,
        n_state:        int = 4,
        n_filters_1:    int = 32,
        n_filters_2:    int = 160,
        kernel_size:    Tuple[int, int] = (1, 5),
        stride:         Tuple[int, int] = (1, 2),
        symmetric_func: str = "max",
    ):
        super().__init__()
        pad = (0, kernel_size[1] // 2)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_state, n_filters_1, kernel_size, stride=stride, padding=pad),
            nn.ELU(),
            FrozenLayerNorm(),
            nn.Conv2d(n_filters_1, n_filters_2, kernel_size, stride=stride, padding=pad),
            nn.ELU(),
            FrozenLayerNorm(),
        )
        self.symmetric = SymmetricLayer(axis=2, func=symmetric_func)
        self.out_channels = n_filters_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                # (B, C', H, S')
        x = self.symmetric(x)          # (B, C', 1, S')  keepdim
        return x.squeeze(2)            # (B, C', S')


# ---------------------------------------------------------------------------
# Continuous branch — Conv1d over (S) only
# ---------------------------------------------------------------------------
class ContinuousBranch(nn.Module):
    """Continuous channels (B, n_cont, S) → (B, n_filters_out, S')."""
    def __init__(
        self,
        n_cont:      int,
        n_filters_1: int = 32,
        n_filters_2: int = 64,
        kernel_size: int = 5,
        stride:      int = 2,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(n_cont, n_filters_1, kernel_size, stride=stride, padding=pad),
            nn.ELU(),
            FrozenLayerNorm(),
            nn.Conv1d(n_filters_1, n_filters_2, kernel_size, stride=stride, padding=pad),
            nn.ELU(),
            FrozenLayerNorm(),
        )
        self.out_channels = n_filters_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)             # (B, C', S')


# ---------------------------------------------------------------------------
# Two-branch per-window encoder
# ---------------------------------------------------------------------------
class TwoBranchEncoder(nn.Module):
    """Per-window encoder with separate state/continuous branches."""
    def __init__(
        self,
        n_state:        int,
        n_cont:         int,
        output_dim:     int = 64,
        state_filters_1: int = 32,
        state_filters_2: int = 160,
        cont_filters_1: int = 32,
        cont_filters_2: int = 64,
        kernel_size:    int = 5,
        stride:         int = 2,
        symmetric_func: str = "max",
        mlp_hidden:     int = 128,
        use_embedding_norm: bool = False,
    ):
        super().__init__()
        self.n_cont = n_cont
        self.use_embedding_norm = use_embedding_norm
        self.state_branch = StateBranch(
            n_state=n_state, n_filters_1=state_filters_1,
            n_filters_2=state_filters_2,
            kernel_size=(1, kernel_size), stride=(1, stride),
            symmetric_func=symmetric_func,
        )
        if n_cont > 0:
            self.cont_branch = ContinuousBranch(
                n_cont=n_cont, n_filters_1=cont_filters_1,
                n_filters_2=cont_filters_2,
                kernel_size=kernel_size, stride=stride,
            )
            merged_channels = state_filters_2 + cont_filters_2
        else:
            self.cont_branch = None
            merged_channels = state_filters_2

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        mlp_layers = [
            nn.Linear(merged_channels, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, output_dim),
        ]
        if use_embedding_norm:
            # Keep per-window embeddings ~ std 1 so the downstream trunk
            # doesn't saturate (the cause of K-blindness + signal loss in
            # the unnormalized variant). LayerNorm over the feature dim is
            # batch-size- and K-independent, so it's safe for variable-K.
            mlp_layers.append(nn.LayerNorm(output_dim))
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_dim = output_dim

    def forward(self, state_oh: torch.Tensor,
                cont: torch.Tensor = None) -> torch.Tensor:
        s = self.state_branch(state_oh)              # (B, C_s, S')
        if self.cont_branch is not None:
            assert cont is not None, "n_cont > 0 but cont tensor is None"
            c = self.cont_branch(cont)               # (B, C_c, S')
            if s.shape[-1] != c.shape[-1]:
                c = F.adaptive_avg_pool1d(c, s.shape[-1])
            x = torch.cat([s, c], dim=1)
        else:
            x = s
        x = self.global_pool(x).squeeze(-1)          # (B, C)
        return self.mlp(x)                           # (B, output_dim)


# ---------------------------------------------------------------------------
# Masked std helper — for use_std_features under variable-K
# ---------------------------------------------------------------------------
def _masked_std_per_dim(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Std over dim=1 of x (B, W, D), respecting mask (B, W). Returns (B, D).
    Equivalent to unbiased=False std over the masked-in subset.
    """
    m = mask.unsqueeze(-1).float()                                  # (B, W, 1)
    counts = m.sum(dim=1).clamp(min=1)                              # (B, 1)
    mean = (x * m).sum(dim=1) / counts                              # (B, D)
    diff_sq = ((x - mean.unsqueeze(1)) ** 2) * m                    # (B, W, D)
    var = diff_sq.sum(dim=1) / counts                               # (B, D)
    return torch.sqrt(var.clamp(min=1e-12))


# ===========================================================================
# ADDED: post-mask recomb summary helpers (TESumstatNet trunk features).
# All three operate over the KEPT (mask==1) TEs only, so they are computed on
# the same variable-K subset the aggregator pools -> N-honest (noisier at small
# K, which is correct). Used to hand gamma_r / alpha_r signal to the trunk that
# the mean-pooling aggregator otherwise saturates on.
# ===========================================================================
def _masked_quantile(x: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    """q-quantile of x (B, W) over the real (mask==1) entries, per row -> (B,).
    Padded / masked-out TEs are pushed to -inf so they never enter an upper
    quantile; rank is taken per row from K = mask.sum. Reads the upper tail of
    local_recomb (tracks gamma_r where the per-TE mean saturates)."""
    xm = x.masked_fill(mask <= 0, float("-inf"))
    xs, _ = torch.sort(xm, dim=1, descending=True)        # real at top, -inf at bottom
    K = mask.sum(dim=1)                                    # (B,)
    rank = torch.floor((1.0 - q) * (K - 1).clamp(min=0)).long()
    return xs.gather(1, rank.unsqueeze(1)).squeeze(1)      # (B,)


def _masked_bin_proportions(x: torch.Tensor, mask: torch.Tensor,
                            lo: float, hi: float) -> torch.Tensor:
    """Fraction of kept (mask==1) entries with x < lo / in [lo, hi] / > hi.
    Returns (B, 3), rows summing to 1 over the kept set. For local_recomb with
    edges +/- 1/RECOMB_LOG_SCALE these are the TE fractions in low / typical /
    high recombination -> a direct read on gamma_r."""
    m = (mask > 0).float()
    K = m.sum(dim=1).clamp(min=1.0)
    low  = ((x < lo).float()                 * m).sum(dim=1) / K
    mid  = (((x >= lo) & (x <= hi)).float()  * m).sum(dim=1) / K
    high = ((x > hi).float()                 * m).sum(dim=1) / K
    return torch.stack([low, mid, high], dim=-1)          # (B, 3)


def _masked_correlation(x: torch.Tensor, y: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of x and y (B, W) over the kept (mask==1) entries,
    per row -> (B,). Centering and moments use the kept set only. cor(te_freq,
    local_recomb) reads how TE frequency tracks recombination -> alpha_r."""
    m = (mask > 0).float()
    K = m.sum(dim=1).clamp(min=2.0)
    mx = (x * m).sum(dim=1) / K
    my = (y * m).sum(dim=1) / K
    xc = (x - mx.unsqueeze(1)) * m                        # padded -> 0, excluded
    yc = (y - my.unsqueeze(1)) * m
    cov = (xc * yc).sum(dim=1) / K
    sx = torch.sqrt(((xc ** 2).sum(dim=1) / K).clamp(min=1e-12))
    sy = torch.sqrt(((yc ** 2).sum(dim=1) / K).clamp(min=1e-12))
    return cov / (sx * sy)                                # (B,)


def _masked_quantile_per_dim(x: torch.Tensor, mask: torch.Tensor,
                             q: float) -> torch.Tensor:
    """Per-dimension q-quantile of x (B, W, D) over the kept (mask==1) TEs,
    -> (B, D). The POOLING PATH: a tail readout of the per-TE embedding that
    runs alongside the aggregator's weighted MEAN, so an extreme TE survives
    the pool instead of being averaged away (the mean-saturation fix). Padded /
    masked TEs are pushed to -inf; rank per row from K = mask.sum, so it is a
    function of the kept set only -> N-honest (noisier at small K)."""
    B, W, D = x.shape
    keep = mask > 0
    xm = x.masked_fill(~keep.unsqueeze(-1), float("-inf"))
    xs, _ = torch.sort(xm, dim=1, descending=True)        # descending over TEs
    K = keep.sum(dim=1)                                   # (B,)
    rank = torch.floor((1.0 - q) * (K - 1).clamp(min=0)).long()   # (B,)
    idx = rank.view(B, 1, 1).expand(B, 1, D)
    return xs.gather(1, idx).squeeze(1)                   # (B, D)


def _masked_mean_per_dim(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-dimension plain (unweighted) mean of x (B, W, D) over kept TEs -> (B, D).
    Complements the aggregator's ATTENTION-weighted mean and the pooled median."""
    m = (mask > 0).float().unsqueeze(-1)                  # (B, W, 1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
# ===========================================================================
# END ADDED
# ===========================================================================


# ---------------------------------------------------------------------------
# Legacy single-head attention class (kept for backward compatibility,
# but the canonical aggregator is now built via make_aggregator).
# ---------------------------------------------------------------------------
class CovariateAttention(nn.Module):
    """Covariate-conditioned attention over W windows."""
    def __init__(self, embed_dim: int, n_covariates: int):
        super().__init__()
        self.cov_proj = nn.Linear(n_covariates, embed_dim)
        self.score = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )
        self.output_dim = embed_dim

    def forward(self, emb: torch.Tensor, cov: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        cov_e    = self.cov_proj(cov)
        combined = torch.cat([emb, cov_e], dim=-1)
        scores   = self.score(combined)
        if mask is not None:
            scores = scores + (mask.unsqueeze(-1).float() - 1) * 1e9
        weights  = torch.softmax(scores, dim=1)
        context  = (weights * emb).sum(dim=1)
        return context, weights.squeeze(-1)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class TECovariateAttentionNet(nn.Module):
    """
    Two-branch per-window encoder + cross-window aggregator (selectable).
    """
    def __init__(
        self,
        n_individuals:    int,
        n_sites:          int,
        n_covariates:     int,
        n_state_channels: int = 4,
        n_cont_channels:  int = 0,
        per_window_dim:   int = 64,
        output_dim:       int = 128,
        symmetric_func:   str = "max",
        use_std_features: bool = True,
        aggregator_kind:  str = "attention",
        n_attention_heads: int = 1,
        use_embedding_norm: bool = False,
    ):
        super().__init__()
        self.n_state_channels = n_state_channels
        self.n_cont_channels  = n_cont_channels
        self.aggregator_kind  = aggregator_kind
        self.use_embedding_norm = use_embedding_norm

        self.per_window = TwoBranchEncoder(
            n_state=n_state_channels,
            n_cont=n_cont_channels,
            output_dim=per_window_dim,
            symmetric_func=symmetric_func,
            use_embedding_norm=use_embedding_norm,
        )

        self.aggregator = make_aggregator(
            kind=aggregator_kind,
            embed_dim=per_window_dim,
            n_covariates=n_covariates,
            n_heads=n_attention_heads,
        )

        self.use_std_features = use_std_features
        trunk_in = self.aggregator.output_dim + (per_window_dim if use_std_features else 0)
        # Optional LayerNorm on the pooled context, so the trunk input is
        # well-scaled regardless of per-window embedding magnitude. Prevents
        # the trunk's ELU from saturating (the cause of signal loss and
        # K-blindness in the unnormalized variant).
        self.context_norm = nn.LayerNorm(trunk_in) if use_embedding_norm else None
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, output_dim), nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim), nn.ELU(),
        )
        self.output_dim = output_dim

    def forward(self, state_oh: torch.Tensor,
                cont: torch.Tensor,
                covariates: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        state_oh   : (B, W, n_state, H, S)   one-hot
        cont       : (B, W, n_cont, S)        continuous channels (or None)
        covariates : (B, W, n_cov)
        mask       : (B, W) of 0/1 — 1 = real window, 0 = padding.
                     If None, all windows are treated as real.

        Returns: (B, output_dim), (B, W) per-window weights (or gates)
        """
        B, W, Cs, H, S = state_oh.shape
        s_bw = state_oh.reshape(B * W, Cs, H, S)
        if cont is not None and self.n_cont_channels > 0:
            Cc = cont.shape[2]
            c_bw = cont.reshape(B * W, Cc, S)
        else:
            c_bw = None

        per_win = self.per_window(s_bw, c_bw)        # (B*W, D)
        per_win = per_win.reshape(B, W, -1)           # (B, W, D)
        context, weights = self.aggregator(per_win, covariates, mask)

        if self.use_std_features:
            if mask is not None:
                std_per_win = _masked_std_per_dim(per_win, mask)
            elif W > 1:
                std_per_win = per_win.std(dim=1, unbiased=False)
            else:
                std_per_win = torch.zeros_like(context)
            context = torch.cat([context, std_per_win], dim=-1)

        if self.context_norm is not None:
            context = self.context_norm(context)

        return self.trunk(context), weights

    def embedding(self, state_oh, cont, covariates, mask=None):
        emb, _ = self.forward(state_oh, cont, covariates, mask=mask)
        return emb

    def per_window_embeddings(self, state_oh, cont, window_chunk=None):
        """
        Encode each window independently, returning (B, W, D). When
        window_chunk is set, windows are encoded in batches of that many at a
        time to bound peak memory — essential for whole-genome K (e.g. 200
        windows would otherwise build a B*W one-hot tensor that OOMs).
        """
        B, W, Cs, H, S = state_oh.shape
        if window_chunk is None or window_chunk >= W:
            s_bw = state_oh.reshape(B * W, Cs, H, S)
            c_bw = (cont.reshape(B * W, cont.shape[2], S)
                    if (cont is not None and self.n_cont_channels > 0) else None)
            return self.per_window(s_bw, c_bw).reshape(B, W, -1)

        outs = []
        for w0 in range(0, W, window_chunk):
            w1 = min(w0 + window_chunk, W)
            s_chunk = state_oh[:, w0:w1].reshape(B * (w1 - w0), Cs, H, S)
            if cont is not None and self.n_cont_channels > 0:
                c_chunk = cont[:, w0:w1].reshape(B * (w1 - w0), cont.shape[2], S)
            else:
                c_chunk = None
            outs.append(self.per_window(s_chunk, c_chunk).reshape(B, w1 - w0, -1))
        return torch.cat(outs, dim=1)                  # (B, W, D)

    def aggregate_from_per_window(self, per_win, covariates, mask=None):
        """
        Run the aggregator + trunk on precomputed per-window embeddings
        (B, W, D). Mirrors the tail of forward(); lets callers encode windows
        in memory-bounded chunks first (see per_window_embeddings).
        """
        context, weights = self.aggregator(per_win, covariates, mask)
        if self.use_std_features:
            if mask is not None:
                std_per_win = _masked_std_per_dim(per_win, mask)
            elif per_win.shape[1] > 1:
                std_per_win = per_win.std(dim=1, unbiased=False)
            else:
                std_per_win = torch.zeros_like(context)
            context = torch.cat([context, std_per_win], dim=-1)
        if self.context_norm is not None:
            context = self.context_norm(context)
        return self.trunk(context), weights

    def embedding_chunked(self, state_oh, cont, covariates, mask=None,
                          window_chunk=20):
        """
        Memory-bounded embedding for large W: encode windows in chunks, then
        aggregate once. Equivalent to embedding() but with peak memory set by
        window_chunk rather than W. Use for whole-genome deployment.
        """
        per_win = self.per_window_embeddings(state_oh, cont,
                                             window_chunk=window_chunk)
        emb, _ = self.aggregate_from_per_window(per_win, covariates, mask)
        return emb

    def embedding_chunked_with_weights(self, state_oh, cont, covariates,
                                       mask=None, window_chunk=20):
        """
        Like embedding_chunked but also returns the per-window aggregator
        weights (B, W) — the attention diagnostic. Lets the K-extrapolation
        path emit attention_per_sim.csv at whole-genome K, where the legacy
        forward-based extraction can't run.
        """
        per_win = self.per_window_embeddings(state_oh, cont,
                                             window_chunk=window_chunk)
        emb, weights = self.aggregate_from_per_window(per_win, covariates, mask)
        return emb, weights


# ---------------------------------------------------------------------------
# SBI wrapper — flat-tensor I/O with optional mask
# ---------------------------------------------------------------------------
class TESumstatNet(nn.Module):
    """
    Summary-statistic counterpart of TECovariateAttentionNet. Each TE is a
    fixed-length feature vector (te_dfe/sumstats.py); a shared per-TE MLP encodes
    it, then the SAME permutation-invariant aggregator pools the variable-size
    set of TEs, then the trunk. Because the encoder is per-element and the
    aggregator is cardinality-agnostic, this generalizes to any number of TEs
    (train on N, deploy on more) exactly like window K-extrapolation.

    Exposes the same .embedding()/.output_dim interface as the matrix net so the
    flow, SBI wrapper, and training loop are reused unchanged.
    """
    def __init__(self, n_features: int, per_te_dim: int = 192,
                 output_dim: int = 128, aggregator_kind: str = "sigmoid_attn",
                 n_attention_heads: int = 1, use_embedding_norm: bool = False,
                 n_global_cov: int = 0,
                 recomb_tail_quantiles: tuple = (),     # ADDED: e.g. (0.9,)
                 recomb_bin_edges: tuple = (),          # ADDED: e.g. (-1.0, 1.0)
                 corr_feature_pairs: tuple = (),        # ADDED: e.g. (("te_freq","local_recomb"),)
                 pool_quantiles: tuple = (),   # ADDED: quantile sketch of the embedding over TEs (0.1, 0.25, 0.5, 0.75, 0.9)
                 pool_mean: bool = False,                # ADDED: plain (unweighted) mean pool
                 pool_proj_dim: int = None):            # ADDED: project emb to this dim BEFORE pooling (caps params); None = full per_te_dim
        super().__init__()
        #Note that self, n_features: int, per_te_dim: int = 192 was self, n_features: int, per_te_dim: int = 64 before adding more heads to the context model
        self.n_features = n_features
        self.n_global_cov = n_global_cov
        self.use_embedding_norm = use_embedding_norm
        self.aggregator_kind = aggregator_kind
        # ADDED: post-mask recomb trunk features (off unless requested).
        self.recomb_tail_quantiles = tuple(recomb_tail_quantiles)
        self.recomb_bin_edges = tuple(recomb_bin_edges)
        self.corr_feature_pairs = tuple(corr_feature_pairs)
        self.pool_quantiles = tuple(pool_quantiles)        # ADDED: embedding tail-pool
        self.pool_mean = bool(pool_mean)                   # ADDED
        self.pool_proj = (nn.Linear(per_te_dim, pool_proj_dim)   # ADDED: optional pre-pool projection
                          if pool_proj_dim else None)
        if self.recomb_tail_quantiles or self.recomb_bin_edges:
            from .sumstats import FEATURE_NAMES
            self.recomb_idx = FEATURE_NAMES.index("local_recomb")
        if self.corr_feature_pairs:
            from .sumstats import FEATURE_NAMES
            self.corr_idx_pairs = [(FEATURE_NAMES.index(a), FEATURE_NAMES.index(b))
                                   for a, b in self.corr_feature_pairs]
        # Per-TE encoder (shared across TEs; order-free -> exchangeable).
        self.encoder = nn.Sequential(
            nn.Linear(n_features, per_te_dim), nn.ELU(),
            nn.Linear(per_te_dim, per_te_dim), nn.ELU(),
        )
        self.enc_norm = nn.LayerNorm(per_te_dim) if use_embedding_norm else None
        # Reuse the same aggregator; the per-TE feature vector serves as the
        # attention covariate, so the gate can up-weight informative TEs
        # (e.g. intermediate-frequency, long-carrier-haplotype — the burst
        # signature).
        self.aggregator = make_aggregator(
            kind=aggregator_kind, embed_dim=per_te_dim,
            n_covariates=n_features, n_heads=n_attention_heads,
        )
        agg_out = self.aggregator.output_dim
        self.context_norm = nn.LayerNorm(agg_out) if use_embedding_norm else None
        # A GLOBAL covariate (e.g. total sequence length) is concatenated to the
        # pooled, count-invariant context AFTER its norm, so abundance enters in
        # a genome-size-invariant way without disturbing the per-TE pooling.
        # ADDED: the recomb trunk features are appended on the same principle.
        # ADDED: pooling path width = (n quantiles + mean) x pooled_dim, where
        # pooled_dim is the projection dim if set, else the full per_te_dim.
        pooled_dim = pool_proj_dim if pool_proj_dim else per_te_dim
        n_pool_stats = len(self.pool_quantiles) + (1 if self.pool_mean else 0)
        trunk_in = (agg_out + n_global_cov
                    + len(self.recomb_tail_quantiles)
                    + (3 if self.recomb_bin_edges else 0)
                    + len(self.corr_feature_pairs)
                    + pooled_dim * n_pool_stats)               # ADDED: tail-pool path
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, output_dim), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim), nn.ELU(),
        )
        self.output_dim = output_dim

    def per_te_embeddings(self, feats):
        """feats: (B, K, n_features) -> (B, K, per_te_dim)."""
        emb = self.encoder(feats)
        if self.enc_norm is not None:
            emb = self.enc_norm(emb)
        return emb

    def aggregate(self, emb, feats, mask=None, global_cov=None):
        ctx, weights = self.aggregator(emb, feats, mask)   # weighted MEAN over TEs
        if self.context_norm is not None:
            ctx = self.context_norm(ctx)
        # ADDED: pooling path — mean + quantile sketch of the per-TE embedding
        # over kept TEs, concatenated to the attention-weighted mean context so
        # the trunk sees the full distribution of TEs (centre, spread, tails),
        # not just the average. Optional projection (self.pool_proj) caps the
        # width. Post-mask; the trunk learns which statistics matter.
        if self.pool_quantiles or self.pool_mean:
            m = mask if mask is not None else torch.ones_like(emb[:, :, 0])
            pe = self.pool_proj(emb) if self.pool_proj is not None else emb
            parts = []
            if self.pool_mean:
                parts.append(_masked_mean_per_dim(pe, m))
            parts += [_masked_quantile_per_dim(pe, m, q) for q in self.pool_quantiles]
            ctx = torch.cat([ctx] + parts, dim=-1)         # (B, +pooled_dim*n_stats)
        if self.n_global_cov:
            if global_cov is None:
                raise ValueError(
                    f"TESumstatNet built with n_global_cov={self.n_global_cov} "
                    f"but no global_cov passed to aggregate().")
            ctx = torch.cat([ctx, global_cov], dim=-1)
        # ADDED: post-mask recomb trunk features, concatenated like global_cov so
        # the trunk learns whether to use them (down-weights if unhelpful).
        if self.recomb_tail_quantiles:
            r = feats[:, :, self.recomb_idx]                  # (B, K) local_recomb
            m = mask if mask is not None else torch.ones_like(r)
            tails = torch.stack(
                [_masked_quantile(r, m, q) for q in self.recomb_tail_quantiles],
                dim=-1)                                       # (B, n_q)
            ctx = torch.cat([ctx, tails], dim=-1)
        if self.recomb_bin_edges:
            lo, hi = self.recomb_bin_edges
            r = feats[:, :, self.recomb_idx]
            m = mask if mask is not None else torch.ones_like(r)
            props = _masked_bin_proportions(r, m, lo, hi)     # (B, 3)
            ctx = torch.cat([ctx, props], dim=-1)
        if self.corr_feature_pairs:
            m = mask if mask is not None else torch.ones_like(feats[:, :, 0])
            corrs = torch.stack(
                [_masked_correlation(feats[:, :, i], feats[:, :, j], m)
                 for (i, j) in self.corr_idx_pairs], dim=-1)  # (B, n_pairs)
            ctx = torch.cat([ctx, corrs], dim=-1)
        return self.trunk(ctx), weights

    def embedding(self, feats, mask=None, global_cov=None):
        emb = self.per_te_embeddings(feats)
        out, _ = self.aggregate(emb, feats, mask, global_cov)
        return out

    def embedding_with_weights(self, feats, mask=None, global_cov=None):
        emb = self.per_te_embeddings(feats)
        out, w = self.aggregate(emb, feats, mask, global_cov)
        return out, w

    def forward(self, feats, mask=None, global_cov=None):
        return self.embedding(feats, mask, global_cov)


class SumstatSBIWrapper(nn.Module):
    """
    Flatten wrapper for the sumstat representation. Flat layout:
        [feats_flat | mask_flat?]
    feats_flat : (K * n_features,) float32, K = N_TE_TRAIN_MAX padded
    mask_flat  : (K,) float32 in {0,1} marking real (vs padded) TEs.
    global_cov : (n_global_cov,) float32 appended last, when n_global_cov>0
                 (e.g. the normalized total sequence length).
    Unpacks to (B, K, n_features) + (B, K) mask + (B, n_global_cov) and calls
    TESumstatNet.
    """
    def __init__(self, network: "TESumstatNet", n_te_max: int,
                 n_features: int, include_mask: bool = True,
                 n_global_cov: int = 0):
        super().__init__()
        self.network = network
        self.n_te_max = n_te_max
        self.n_features = n_features
        self.include_mask = include_mask
        self.n_global_cov = n_global_cov
        self.feats_flat = n_te_max * n_features
        self.mask_flat = n_te_max if include_mask else 0
        self.output_dim = network.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        expected = self.feats_flat + self.mask_flat + self.n_global_cov
        if x.shape[1] != expected:
            raise ValueError(
                f"Unrecognized sumstat obs size {x.shape[1]}, expected {expected}.")
        feats = x[:, :self.feats_flat].view(B, self.n_te_max, self.n_features)
        off = self.feats_flat
        mask = None
        if self.include_mask:
            mask = x[:, off:off + self.mask_flat]
            off += self.mask_flat
        global_cov = None
        if self.n_global_cov:
            global_cov = x[:, off:off + self.n_global_cov]
        return self.network.embedding(feats, mask, global_cov)


class SBIWrapper(nn.Module):
    """
    Flatten storage format for sbi.SNPE; unpack inside forward().

    Flat tensor layout:
        [geno_flat | pos_flat | cov_flat | mask_flat?]

    where:
        geno_flat   : (W * n_ind * n_sites,)  int8 cast to float32 in {0..3}
        pos_flat    : (W * n_position_channels * n_sites,) float32  (omitted if 0)
        cov_flat    : (W * n_cov,) float32
        mask_flat   : (W,) float32 in {0, 1}  (only if include_mask=True)

    Geno is expanded to one-hot (B, W, n_channels, n_ind, n_sites) inside.
    The continuous tensor stays at (B, W, n_position_channels, n_sites).
    Mask (B, W) is passed to network.embedding when present.
    """
    def __init__(self, network: TECovariateAttentionNet,
                 n_windows: int, n_channels: int, n_individuals: int,
                 n_sites: int, n_cov: int, n_position_channels: int = 0,
                 include_mask: bool = False):
        super().__init__()
        self.network             = network
        self.n_windows           = n_windows
        self.n_channels          = n_channels
        self.n_position_channels = n_position_channels
        self.n_individuals       = n_individuals
        self.n_sites             = n_sites
        self.n_cov               = n_cov
        self.include_mask        = include_mask

        self.geno_flat = n_windows * n_individuals * n_sites
        self.pos_flat  = n_windows * n_position_channels * n_sites
        self.cov_flat  = n_windows * n_cov
        self.mask_flat = n_windows if include_mask else 0
        self.output_dim = network.output_dim

    def _expand_one_hot(self, geno_int):
        return torch.nn.functional.one_hot(
            geno_int, num_classes=self.n_channels,
        ).permute(0, 1, 4, 2, 3).contiguous().float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        total = x.shape[1]
        expected = self.geno_flat + self.pos_flat + self.cov_flat + self.mask_flat
        if total != expected:
            raise ValueError(
                f"Unrecognized observation size {total}. Expected {expected} "
                f"(geno_flat={self.geno_flat}, pos_flat={self.pos_flat}, "
                f"cov_flat={self.cov_flat}, mask_flat={self.mask_flat})."
            )

        offset = 0
        geno_int = x[:, offset:offset + self.geno_flat].reshape(
            B, self.n_windows, self.n_individuals, self.n_sites,
        ).long()
        state_oh = self._expand_one_hot(geno_int)
        offset += self.geno_flat

        if self.n_position_channels > 0:
            cont = x[:, offset:offset + self.pos_flat].reshape(
                B, self.n_windows, self.n_position_channels, self.n_sites,
            )
            offset += self.pos_flat
        else:
            cont = None

        cov = x[:, offset:offset + self.cov_flat].reshape(
            B, self.n_windows, self.n_cov,
        )
        offset += self.cov_flat

        if self.include_mask:
            mask = x[:, offset:offset + self.mask_flat]   # (B, W)
        else:
            mask = None

        return self.network.embedding(state_oh, cont, cov, mask=mask)


def pack_observation(genotypes: torch.Tensor,
                     covariates: torch.Tensor) -> torch.Tensor:
    """Flatten one replicate's tensors. Kept for compatibility."""
    return torch.cat([genotypes.flatten(), covariates.flatten()])
