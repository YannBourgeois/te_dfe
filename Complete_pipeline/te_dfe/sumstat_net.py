#!/usr/bin/env python3
"""
sumstat_net.py — embedding network for the per-TE summary-statistic
representation.

Mirrors the matrix encoder's structure (per-element encoder -> exchangeable
aggregator -> trunk -> embedding) but the per-element encoder is a small MLP
over each TE's fixed-length statistic vector instead of a CNN over a genotype
matrix. The TEs are an UNORDERED set, so:

  * the per-TE MLP (shared weights) is applied independently to each TE — an
    RNN would be wrong here (order-sensitive, breaks permutation invariance,
    does not extrapolate cleanly to large N);
  * the exchangeable aggregator (the same te_dfe.aggregators used for windows)
    pools the per-TE embeddings permutation-invariantly. sigmoid_attn is the
    production choice: it is count-SENSITIVE, so more TEs sharpen the posterior
    (a pure mean pool would give 500 and 5000 TEs the same embedding).

Variable-N training + N-extrapolation work exactly like the window
K-extrapolation: train with up to N_train TEs (masking the padding), deploy at
any N via embedding_chunked. The interface (embedding / embedding_chunked /
embedding_chunked_with_weights) matches the matrix encoder so SBIWrapper and the
training/inference scripts are reused unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregators import make_aggregator


class TEDFESumstatNet(nn.Module):
    """
    Per-TE summary-statistic encoder. Input per replicate: a set of TE feature
    vectors (B, N, F) plus a mask (B, N) marking real vs padded TEs. Produces a
    fixed embedding (B, output_dim).
    """
    def __init__(self, n_features, output_dim=128, per_te_dim=64,
                 hidden=128, aggregator_kind="sigmoid_attn",
                 n_attention_heads=4, use_embedding_norm=True):
        super().__init__()
        self.n_features = n_features
        self.output_dim = output_dim
        self.per_te_dim = per_te_dim
        self.aggregator_kind = aggregator_kind

        # Per-TE MLP encoder phi (shared across TEs, applied independently).
        self.phi = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, per_te_dim),
        )
        self.use_embedding_norm = use_embedding_norm
        if use_embedding_norm:
            self.emb_norm = nn.LayerNorm(per_te_dim)

        # Exchangeable aggregator over the TE set. There are no separate set-
        # level covariates (context is already per-TE in the features), so a
        # zero covariate vector of width 1 is passed.
        self._n_cov = 1
        self.aggregator = make_aggregator(
            aggregator_kind, embed_dim=per_te_dim, n_covariates=self._n_cov,
            n_heads=n_attention_heads)

        # Trunk: aggregated set embedding -> output embedding.
        self.trunk = nn.Sequential(
            nn.Linear(per_te_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def per_te_embeddings(self, feats, te_chunk=None):
        """feats: (B, N, F) -> per-TE embeddings (B, N, per_te_dim)."""
        B, N, Fdim = feats.shape
        if te_chunk is None or te_chunk >= N:
            e = self.phi(feats)
        else:
            outs = []
            for s in range(0, N, te_chunk):
                outs.append(self.phi(feats[:, s:s + te_chunk, :]))
            e = torch.cat(outs, dim=1)
        if self.use_embedding_norm:
            e = self.emb_norm(e)
        return e

    def aggregate(self, per_te, mask=None):
        B, N, _ = per_te.shape
        # The window aggregators take per-element covariates (B, N, n_cov);
        # context is already inside the per-TE features, so pass a per-TE zero
        # covariate of width 1.
        cov = torch.zeros(B, N, self._n_cov, device=per_te.device,
                          dtype=per_te.dtype)
        ctx, weights = self.aggregator(per_te, cov, mask)
        return ctx, weights

    def embedding(self, feats, mask=None):
        per_te = self.per_te_embeddings(feats)
        ctx, _ = self.aggregate(per_te, mask)
        return self.trunk(ctx)

    def embedding_chunked(self, feats, mask=None, te_chunk=500):
        per_te = self.per_te_embeddings(feats, te_chunk=te_chunk)
        ctx, _ = self.aggregate(per_te, mask)
        return self.trunk(ctx)

    def embedding_chunked_with_weights(self, feats, mask=None, te_chunk=500):
        per_te = self.per_te_embeddings(feats, te_chunk=te_chunk)
        ctx, weights = self.aggregate(per_te, mask)
        return self.trunk(ctx), weights

    def forward(self, feats, mask=None):
        return self.embedding(feats, mask)
