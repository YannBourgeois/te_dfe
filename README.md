# te_dfe, NPE package for TE DFE + burst inference
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FYannBourgeois%2Fte_dfe.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FYannBourgeois%2Fte_dfe?ref=badge_shield)


Neural Posterior Estimation for transposable-element evolution from genotype
data. Two models can be selected at runtime:

  * **DFE, simple**: 6-parameter gamma DFE over |s| + a single rectangular burst.
  * **context**: context-dependent fitness effects + insertion preference (7 params in the
                  recombination-only default): selection and insertion rate vary
                  with local recombination.

The user can download this git repository, modify the contents of data/ so it matches the demography and the genome organization they wish, and run from the Complete_pipeline folder. More information is also available here: https://yannbourgeois.github.io/projects/1_project/
This is still work in progress, and a more detailed documentation should follow. At the moment, the pipeline has not been tested on actual biological data, although the required elements should be in place (for example for masking uncallable regions, and dealing with unphased or missing data).

## Selecting the model

Every pipeline script takes `--model {simple,context}`. The choice is locked
into the `TE_DFE_MODEL` environment variable BEFORE `te_dfe.config` is imported
(config dispatches to the right overlay at import time), so `--model` on the CLI
overrides the config default. If omitted, `TE_DFE_MODEL` (or the default,
`simple`) is used.

```bash
python 03_train.py --model context --aggregator-kind sigmoid_attn --embedding-norm
TE_DFE_MODEL=context python 04_infer.py        # env var also works
```

To read the active model in code: `te_dfe.config.MODEL_NAME`.

## Configuration layout (shared base + model overlays)

```
te_dfe/config_base.py     all SHARED settings: paths, simulation, window layout,
                          base channels, SLiM Q-rescaling, and the realism layer
                          (background selection, coverage mask, dead-pixel
                          missingness, SFS folding, phasing, training-stage
                          feature filtering). One source of truth so the two
                          models never drift on these.
te_dfe/config_simple.py   imports base; adds the 6-param gamma-DFE prior + the
                          optional local-recomb channel (5 continuous channels).
te_dfe/config_context.py  imports base; adds the context prior (alpha_*/gamma_*
                          slopes, configurable via CONTEXT_PARAMS) + the
                          dist-to-gene and log-recomb channels (6 channels).
te_dfe/config.py          dispatcher: re-exports the chosen overlay based on
                          TE_DFE_MODEL. `from te_dfe.config import N_PARAMS, ...`
                          works unchanged everywhere.
```

The dispatch pattern (one dispatcher + two overlays, selected by TE_DFE_MODEL)
is also used for the prior (`prior_simple` / `prior_context`) and the simulator
(`simulator_simple` / `simulator_context`).

## Package modules

```
te_dfe/
  config*.py        configuration (above)
  prior*.py         DFEBurstPrior + DFE-bin helpers, dispatched by model
  simulator*.py     SLiM driver, dispatched by model. Both return a UNIFORM
                    4-tuple per window: (ts, covariate, recomb_map,
                    gene_intervals), so pipeline scripts unpack identically.
  processor.py      tree-seq/VCF -> (geno, pos). ONE model-aware processor:
                    the three-region layout, masking, missingness, folding,
                    unphased collapse, and the VCF-shared build_from_variant_table
                    are common; only the continuous-channel assembly branches by
                    model (simple: +recomb; context: +dist-gene +recomb).
  network.py        two-branch encoder + aggregator + flow; reads channel counts
                    from config, so it adapts to either model automatically.
  aggregators.py    attention / deepsets / sigmoid_attn.
  demography.py     MSMCDemography, WindowSampler (+ window_from_coords for the
                    stratified manifest), CoverageMask.
  filtering.py      shared channel/region/state zeroing, used by BOTH training
                    (train-without) and the 05b ablation (inference reliance) so
                    the masking is provably identical.
  unphased.py       collapse_to_unphased (diploid 6-state) + shuffle_phase.
  diagnostics.py, sim_introspect.py, snpe_refine.py
model_select.py     prescan_model() / add_model_arg() — the --model plumbing.
```

## Pipeline

```
00_sample_windows.py    stratified window MANIFEST (covariate-space coverage),
                        inspectable before launching sims.
01_simulate.py          per-replicate SLiM sims.
01_simulate_shard.py    SHARDED sims (window-chunk per task) + 01_collate_shard.py.
                        Reads the manifest for stratified per-replicate windows.
02_collate.py           shards -> memmap dataset.
03_train.py             stream-train NPE -> network.pt, posterior.pkl, meta.
04_infer.py             posterior sampling on observed data.
05_benchmark.py         quick SBC/recovery.
05a_benchmark_simulate.py + 05a_collate.py   sharded benchmark sims.
05b_benchmark_analyze.py recovery / SBC / coverage / tail-shape CIs / DFE bins /
                        K-extrapolation / ablation / attention.
05c_benchmark_K_sweep.py / 05d_posterior_shapes.py   K sweep, posterior shapes.
06_snpe_refine.py       (optional) sequential refinement.
vcf_to_tensors.py       real-VCF -> (geno, pos) via the processor's shared
                        builder, honoring folding / phasing / masking. Closes
                        the train/inference representation loop.
audit_priors.py         prior-consistency gate.
```


## Summary-statistic representation (REPRESENTATION = "sumstat")

An alternative to the per-window genotype matrix: each TE becomes a fixed-length
per-TE summary-statistic vector, and the variable-size SET of TEs is pooled by
the same permutation-invariant aggregator, so the representation generalizes to
any number of TEs (train on N, deploy on more) the way window K-extrapolation
does. Set `REPRESENTATION = "sumstat"` in config.

Per-TE features (te_dfe/sumstats.py, all carrier-vs-non-carrier and
polarization-free, since TE presence is the known derived/core allele):
TE frequency; flanking-SNP expected heterozygosity (mean+SD, per group);
flanking-SNP Tajima's D (single pooled value per group); a haplotype-extension
contrast (nSL-style identity-tract length when phased, Kelly's ZnS when not);
distance-to-gene and local recombination; and reliability features (flank-SNP
count, validity flags). Flanking SNPs are pooled across both sides. The
statistics module is shared by the tree-seq and VCF paths — the same discipline
as build_from_variant_table — so train and inference see identical statistics.

## Observation representations: matrix vs summary statistics

Set by `REPRESENTATION` in config_base (applies to both models):

  * **matrix** (not optimal) : the per-window genotype matrix (state one-hot +
    continuous channels), encoded by the exchangeable CNN. Fixed site layout.
  * **sumstat** : Each TE gets a fixed-length
    feature vector (te_dfe/sumstats.py): TE frequency; carrier vs non-carrier
    flanking expected-heterozygosity (mean + SD); carrier vs non-carrier
    flanking Tajima's D (single pooled value per group); a haplotype-extension
    contrast (nSL-style identity-tract length when PHASED, ZnS LD when not);
    distance-to-gene and local recombination; and reliability features
    (n_flank, valid flags). All statistics are computed over all pooled
    flanking SNPs (left + right), split by carrier status.

Phasing: nSL needs phased haplotypes and a defined carrier-haplotype set, so it
is used only when phased; unphased splits carriers at the individual level
(dosage >= 1) and the haplotype slot falls back to ZnS. The feature-vector
length is fixed across modes (the slot is reused), so the network shape is
phasing-stable.



## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FYannBourgeois%2Fte_dfe.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2FYannBourgeois%2Fte_dfe?ref=badge_large)