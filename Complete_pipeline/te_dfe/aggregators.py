"""
aggregators.py — K-flexible cross-window aggregators.

Three modes selectable by name:

  "attention"     — softmax-attention with covariate conditioning.
                    K-FIXED at train+infer time. Same behavior as legacy
                    CovariateAttention but supports a (B, W) mask for
                    masked-out (padding) windows.

  "deepsets"      — mean + max + std pool over W, then optional linear
                    projection back to embed_dim. No learned per-window
                    weighting. K-INVARIANT under masking and replication.

  "sigmoid_attn"  — per-window sigmoid gate × value-projected embedding,
                    weighted mean. Content-dependent gating WITHOUT softmax
                    coupling. K-INVARIANT.

All three accept the (emb, cov, mask) interface, so they're drop-in
interchangeable inside TECovariateAttentionNet via `make_aggregator`.
"""
from typing import Optional
import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over W (dim=1) of x (B, W, D), respecting mask (B, W)."""
    m = mask.unsqueeze(-1).float()
    summed = (x * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1)
    return summed / counts


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max over W of x (B, W, D), respecting mask. Masked positions get -inf."""
    m = mask.unsqueeze(-1).float()
    very_neg = torch.tensor(-1e9, dtype=x.dtype, device=x.device)
    return torch.where(m > 0, x, very_neg).max(dim=1).values


def masked_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Std over W of x (B, W, D), respecting mask. Returns 0 where count<2."""
    mean = masked_mean(x, mask)
    m = mask.unsqueeze(-1).float()
    diff_sq = ((x - mean.unsqueeze(1)) ** 2) * m
    summed = diff_sq.sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1)
    var = summed / counts
    return torch.sqrt(var.clamp(min=1e-12))


class AttentionAggregator(nn.Module):
    """
    Softmax attention with covariate conditioning. K-FIXED (best results
    when train K == infer K). Optionally accepts a mask: padding positions
    get score -1e9 → ≈0 softmax weight.

    Forward signature: (emb: (B,W,D), cov: (B,W,C), mask: Optional[(B,W)])
    Returns: (context: (B, D), weights: (B, W))
    """
    def __init__(self, embed_dim: int, n_covariates: int, n_heads: int = 1):
        super().__init__()
        assert embed_dim % n_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        self.embed_dim = embed_dim
        self.output_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.cov_proj = nn.Linear(n_covariates, embed_dim)
        self.score = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, n_heads),
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # Small head-bias diversification when multi-head
        if n_heads > 1:
            with torch.no_grad():
                self.score[-1].bias.data += torch.linspace(-0.1, 0.1, n_heads)

    def forward(self, emb: torch.Tensor, cov: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        B, W, D = emb.shape
        K = self.n_heads
        Dh = self.head_dim
        cov_e = self.cov_proj(cov)
        combined = torch.cat([emb, cov_e], dim=-1)
        scores = self.score(combined)                           # (B, W, K)
        if mask is not None:
            mask_e = mask.unsqueeze(-1).float()                 # (B, W, 1)
            scores = scores + (mask_e - 1) * 1e9
        weights = torch.softmax(scores, dim=1)                  # (B, W, K)
        vals = self.value_proj(emb).view(B, W, K, Dh)
        context = (weights.unsqueeze(-1) * vals).sum(dim=1)     # (B, K, Dh)
        context = context.reshape(B, D)
        mean_weights = weights.mean(dim=-1)                     # (B, W)
        return context, mean_weights


class DeepSetsAggregator(nn.Module):
    """
    Mean + max + std pool over W. Covariates are additively projected into
    each window's embedding so per-window covariate context still informs
    the pool. K-INVARIANT.

    Output dim = embed_dim if collapse=True (project 3D→D), else 3*embed_dim.

    Forward: (emb, cov, mask) -> (context, attn_diag)
    `attn_diag` here is mask-normalized (so visualization code still gets
    a (B, W) weight-like array summing to 1).
    """
    def __init__(self, embed_dim: int, n_covariates: int,
                 collapse: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.cov_proj = nn.Linear(n_covariates, embed_dim)
        self.collapse = collapse
        if collapse:
            self.collapse_layer = nn.Linear(3 * embed_dim, embed_dim)
            self.output_dim = embed_dim
        else:
            self.output_dim = 3 * embed_dim

    def forward(self, emb: torch.Tensor, cov: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        B, W, D = emb.shape
        cov_e = self.cov_proj(cov)
        combined = emb + cov_e
        if mask is None:
            mask = torch.ones(B, W, device=emb.device, dtype=emb.dtype)
        m  = masked_mean(combined, mask)
        mx = masked_max(combined, mask)
        sd = masked_std(combined, mask)
        pooled = torch.cat([m, mx, sd], dim=-1)
        if self.collapse:
            pooled = self.collapse_layer(pooled)
        # Diagnostic "weights" — uniform over kept windows
        attn_diag = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled, attn_diag


class SigmoidAttentionAggregator(nn.Module):
    """
    Per-window sigmoid gate × value-projected embedding, then weighted mean
    normalized by the sum of gates. Content-dependent gating without
    softmax coupling, so K-INVARIANT.

    MULTI-HEAD (n_heads > 1) mirrors AttentionAggregator: each head gates the
    TEs independently and pools its own embed_dim // n_heads slice, and the
    slices are concatenated back to embed_dim. Unlike softmax attention this
    stays EXACTLY K-invariant per head, so it is the only option that is both
    multi-view and free of the exp(-sigma^2) drift in effective sample size
    that softmax shows when deployment K far exceeds training K.

    Output dim = embed_dim.

    Forward: (emb, cov, mask) -> (context, gates)
    `gates` is (B, W) of per-window sigmoid values in [0, 1] — useful as
    a diagnostic of which windows the network attends to.
    """
    def __init__(self, embed_dim: int, n_covariates: int, n_heads: int = 1):
        super().__init__()
        assert embed_dim % n_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        self.embed_dim = embed_dim
        self.output_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.cov_proj = nn.Linear(n_covariates, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, n_heads),
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # Same head-bias diversification as AttentionAggregator, so the heads
        # do not start identical and collapse onto one another.
        if n_heads > 1:
            with torch.no_grad():
                self.gate[-1].bias.data += torch.linspace(-0.1, 0.1, n_heads)

    def forward(self, emb: torch.Tensor, cov: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        B, W, D = emb.shape
        K, Dh = self.n_heads, self.head_dim
        cov_e = self.cov_proj(cov)
        combined = torch.cat([emb, cov_e], dim=-1)
        gate = torch.sigmoid(self.gate(combined))               # (B, W, K)
        if mask is not None:
            gate = gate * mask.unsqueeze(-1)                    # (B, W, K)
        vals = self.value_proj(emb).view(B, W, K, Dh)           # (B, W, K, Dh)
        weighted = (gate.unsqueeze(-1) * vals).sum(dim=1)       # (B, K, Dh)
        # Each head normalises by its OWN gate sum, so K-invariance is
        # preserved per head: the relative weighting within a head does not
        # change when more TEs are added.
        norm = gate.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)    # (B, K, 1)
        context = (weighted / norm).reshape(B, D)               # (B, D)
        mean_gate = gate.mean(dim=-1)                           # (B, W) diagnostic
        return context, mean_gate


def make_aggregator(
    kind: str,
    embed_dim: int,
    n_covariates: int,
    n_heads: int = 1,
    deepsets_collapse: bool = True,
) -> nn.Module:
    """Build an aggregator by name."""
    if kind == "attention":
        return AttentionAggregator(embed_dim, n_covariates, n_heads=n_heads)
    elif kind == "deepsets":
        if n_heads > 1:
            import warnings
            warnings.warn(
                f"aggregator_kind='deepsets' ignores n_heads (got {n_heads}); "
                f"deepsets uses fixed mean/max/std statistics with no learned "
                f"weights. Set n_attention_heads=1 to silence this.",
                stacklevel=2)
        return DeepSetsAggregator(embed_dim, n_covariates,
                                   collapse=deepsets_collapse)
    elif kind == "sigmoid_attn":
        return SigmoidAttentionAggregator(embed_dim, n_covariates,
                                          n_heads=n_heads)
    else:
        raise ValueError(
            f"Unknown aggregator kind: {kind!r}. "
            f"Choose from: attention, deepsets, sigmoid_attn"
        )
