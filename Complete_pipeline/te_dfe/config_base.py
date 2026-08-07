"""
config_base.py — configuration shared by BOTH the simple and context models.

This holds everything that does not depend on which model is selected: input
paths, the demographic/simulation setup, the per-window genotype-matrix layout,
the base continuous channels, the SLiM rescaling factor, the realism options
(background selection, coverage mask, dead-pixel missingness, SFS folding,
phasing), training-stage feature filtering, and the network/training defaults.

The model overlays (config_simple.py, config_context.py) import everything from
here with `from .config_base import *` and then add or override the
model-specific pieces (parameter set, prior ranges, continuous-channel count).
config.py dispatches to one overlay based on the TE_DFE_MODEL environment
variable (set by each pipeline script from its --model flag; CLI overrides the
default). See config.py for the dispatch.

The realism knobs live here because they apply identically to both models — a
single source of truth so the two never drift.
"""
import numpy as np

# ---------------------------------------------------------------------------
# External input files
# ---------------------------------------------------------------------------
MSMC_PATH       = "data/msmc_output.txt"
MU              = 1.25e-7       # per-bp per-gen mutation rate; must match MSMC --mu
RECOMB_MAP_PATH = "data/recomb_map.txt"
ANNOTATION_PATH = "data/genes.gtf"
ANNOTATION_FMT  = "gtf"
CHROMOSOMES     = None

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SIM_LENGTH      = 5000
NEUTRAL_MU      = MU
N_INDIVIDUALS   = 50
N_HAPS          = N_INDIVIDUALS * 2
WINDOW_SIZE     = 5_000_000
MAX_INSERTIONS_PER_WINDOW = 5_000

# Burst timing prior (shared by both models).
LOG10_T_START_RANGE  = (1.5, 3.5)
LOG_DURATION_RANGE   = (np.log(3.0), np.log(100.0))
LOG_MULTIPLIER_RANGE = (np.log(1.0), np.log(1000.0))

# ---------------------------------------------------------------------------
# SLiM rescaling factor Q (see realism notes). Q=1 is exact (no-op).
# ---------------------------------------------------------------------------
SLIM_Q          = 1.0

# ---------------------------------------------------------------------------
# Phasing mode (shared). PHASED=True -> one row per haplotype, 4 state channels.
# PHASED=False -> collapse to per-individual diploid dosage, 6 state channels.
# Switching requires retraining (input shape changes).
# ---------------------------------------------------------------------------
PHASED           = True
N_INPUT_ROWS     = N_HAPS if PHASED else N_INDIVIDUALS
N_STATE_CHANNELS = 4 if PHASED else 6

# ---------------------------------------------------------------------------
# Individual subsampling (robustness testing: simulate a large panel once, then
# process/train at a smaller sample size without re-simulating).
# ---------------------------------------------------------------------------
# Default OFF. When OFF, the tree's sample count MUST equal N_HAPS or the
# processor raises (catches config/tree mismatches early). When ON, the
# processor draws N_INDIVIDUALS individuals from a tree that has at least that
# many, KEEPING each individual's two haplotype nodes together (via the tskit
# individuals table that SLiM/pyslim populate) — so the unphased pairing is
# correct and any within-individual structure (e.g. selfing) is preserved.
# Selection is the first N_INDIVIDUALS by default (unbiased for unstructured
# sims, no seed needed); set SUBSAMPLE_SEED to an int for a seeded RANDOM draw
# (use this if the sim carries population/spatial structure in sample order).
SUBSAMPLE_INDIVIDUALS = False
SUBSAMPLE_SEED        = None

# ---------------------------------------------------------------------------
# SNP polarization (SFS folding). False=polarized (0=anc,1=der),
# True=folded (0=major,1=minor). SNPs only; TEs never folded.
# ---------------------------------------------------------------------------
FOLD_SFS        = False

# ---------------------------------------------------------------------------
# Background selection (optional nuisance; not inferred). See config docs.
# ---------------------------------------------------------------------------
BG_SELECTION        = True
BG_USE_GTF          = True
BG_DEL_SHAPE        = 0.3
BG_DEL_MEAN_S       = 0.01
BG_DEL_MU           = 0.85e-7 ##say about 2/3 of new mutations are deleterious. Approx.
BG_DEL_DOMINANCE    = 0.5
BG_MEAN_EXON_LEN    = 150
BG_SD_EXON_LEN      = 60
BG_MEAN_INTEREXON   = 1500
BG_SD_INTEREXON     = 1000

# ---------------------------------------------------------------------------
# Coverage mask (optional, BED-driven). MASK_AS_COVARIATE changes N_COVARIATES.
# ---------------------------------------------------------------------------
COVERAGE_MASK_BED         = None
MASK_AS_COVARIATE         = False
DROP_FULLY_MASKED_WINDOWS = False
MIN_CALLABLE_FRACTION     = 0.5

# ---------------------------------------------------------------------------
# Dead-pixel masking (per-individual missingness; training augmentation).
# None | scalar | list[len N_INDIVIDUALS] | path. For real-data inference: None.
# ---------------------------------------------------------------------------
MISSINGNESS = None

# ---------------------------------------------------------------------------
# Training-stage feature filtering (train-without, for fair comparison).
# Mirrors the 05b ablation vocabulary; applied via te_dfe/filtering.py so train
# and benchmark masking are identical.
# ---------------------------------------------------------------------------
TRAIN_ZERO_CHANNELS = ()
TRAIN_ZERO_REGIONS  = ()
TRAIN_ZERO_STATES   = ()

# ---------------------------------------------------------------------------
# Covariates: [gene_density, recomb_scaled, te_density, snp_density] (+masked
# fraction when MASK_AS_COVARIATE). Window-level summary covariates.
# ---------------------------------------------------------------------------
N_COVARIATES    = 5 if MASK_AS_COVARIATE else 4
RECOMB_SCALE    = 1e-7

# ---------------------------------------------------------------------------
# Per-window genotype-matrix layout (three regions on the site axis).
# ---------------------------------------------------------------------------
# THREE PRIMITIVES drive the whole layout; every other size is DERIVED from
# them with the same formulas the processor uses, so config and processor can
# never disagree (a past source of drift):
#   N_TE_SITES        number of TE anchors = max TEs represented per window
#   N_RANDOM_BLOCKS   number of random background blocks
#   N_FLANK_PER_SIDE  flanking SNPs taken on EACH side of a TE (and of each
#                     random-block anchor); 12/side -> 24 flank slots per TE.
N_TE_SITES        = 60          # TE anchors (max TEs/window; >this -> subsample)
N_RANDOM_BLOCKS   = 60
N_FLANK_PER_SIDE  = 12           # 3 flanking SNPs on EACH side (6 SNPs + 1 TE/block)

# Derived (do not hand-edit): a TE's flank region is 2*N_FLANK_PER_SIDE slots,
# a random block is its anchor + 2*N_FLANK_PER_SIDE flanks.
BLOCK_SIZE           = 1 + 2 * N_FLANK_PER_SIDE                 # random-block width
N_TE_FLANK_SITES     = N_TE_SITES * 2 * N_FLANK_PER_SIDE        # = anchors * 24
N_RANDOM_BLOCK_SITES = N_RANDOM_BLOCKS * BLOCK_SIZE
N_SITES = N_TE_SITES + N_TE_FLANK_SITES + N_RANDOM_BLOCK_SITES
MAF_MIN = 0.0

SNP_SAMPLING_STRATEGY = "random"
FLANK_WINDOW_BP       = 15_000
MAX_FLANK_DISTANCE_BP = 15_000 #vestigial TODO: remove
MIN_TE_FLANK_DIST_BP  = 50
MIN_RANDOM_ANCHOR_DIST_BP = 15_000

# State codes in the genotype matrix.
ANCESTRAL_STATE = 0
SNP_STATE       = 1
TE_STATE        = 2
PADDING_STATE   = 3

# ---------------------------------------------------------------------------
# Base continuous channels (per site): phys pos, gen pos, TE density, SNP
# density. Models extend this: the simple model adds a recomb channel; the
# context model adds dist-to-gene + log-recomb. N_CONTINUOUS_CHANNELS is set in
# the overlay, since it differs by model.
# ---------------------------------------------------------------------------
N_POSITION_CHANNELS = 2
N_DENSITY_CHANNELS  = 2
CM_NORMALIZER       = 0.5
LOCAL_DENSITY_BP    = 250_000
DENSITY_LOG_NORMALIZER = 50.0

# Drop TEs within this many bp of a window edge from the FOCAL (emitted) set,
# where the local-density estimate is noisiest even after length-normalization.
# Excluded TEs STILL count as neighbours for interior TEs' density, so this only
# trims which TEs are analysed, not the counts. 0 = keep all TEs (rely on the
# length-normalized rescale alone). Set to LOCAL_DENSITY_BP for the fully
# edge-clean set (drops ~2*EDGE_EXCLUDE_BP/window_size of TEs, e.g. ~10% at 5 Mb).
EDGE_EXCLUDE_BP     = 250_000 #this is the safest way to avoid edge effects. to edit if TE density is too low, and to adjust depending on LOCAL_DENSITY_BP and window_size 

# Recomb-rate channel normalization (shared formula; used by simple ch4 and
# context ch5). log10(r / RECOMB_TYPICAL) / RECOMB_LOG_SCALE, clipped [-2,2].
RECOMB_TYPICAL     = 1e-7
RECOMB_LOG_SCALE   = 1.0
RECOMB_MIN_FOR_LOG = 1e-12

# Distance-to-gene channel normalization (context model).
DIST_TO_GENE_NORMALIZER = 1e6

# ---------------------------------------------------------------------------
# Network / training defaults (shared).
# ---------------------------------------------------------------------------
DROPOUT         = 0.1
N_TRAIN_WINDOWS = 20

# ---------------------------------------------------------------------------
# Inferred-parameter subset (marginalize-out unwanted outputs).
# ---------------------------------------------------------------------------
# By default the network infers ALL of the model's parameters (PARAM_NAMES in
# the overlay). To infer only a SUBSET, set INFER_PARAMS to a list of the names
# you want as outputs; the rest are MARGINALIZED — they still vary in the
# simulations (the sims and existing datasets are untouched), but they are
# dropped from the theta vector the flow sees, so the network learns the
# posterior over the kept parameters with the others integrated out.
#
#   INFER_PARAMS = None                 -> infer all (default)
#   INFER_PARAMS = ["log_multiplier"]   -> infer only the burst multiplier
#   INFER_PARAMS = ["log10_t_start", "log_duration", "log_multiplier"]  -> burst only
#
# Names must be a subset of the active model's PARAM_NAMES. The kept order
# follows PARAM_NAMES (not the order you list), so the column mapping is
# canonical and stable. training_meta records the kept names + indices so
# inference/benchmarking slice the same columns; a checkpoint trained with a
# given INFER_PARAMS must be used with the same subset.
#
# NOTE (statistics): marginalizing CANNOT narrow the kept parameters' credible
# intervals — it can only remove outputs you don't care about and fold the
# dropped parameters' variation into the noise. To actually tighten a
# confounded parameter you must FIX the other at a constant (a different,
# generative-model change), not marginalize. This option is the cheap "I just
# want fewer outputs" path.
INFER_PARAMS = None

# ---------------------------------------------------------------------------
# Fixed parameters (remove a degree of freedom from the generative model).
# ---------------------------------------------------------------------------
# Unlike INFER_PARAMS (which marginalizes — the parameter still varies in the
# sims, just isn't output), FIX_PARAMS HOLDS a parameter at a constant in the
# simulations, so it is removed from the generative model entirely. This is the
# lever that can genuinely CONCENTRATE the remaining parameters' posteriors when
# they are confounded with the fixed one — at the cost of asserting you know its
# value. A fixed parameter is automatically excluded from the inferred outputs
# (you cannot infer a constant).
#
# REQUIRES REGENERATING SIMS: the constant must actually be the value used in
# SLiM, so datasets simulated with a different (or no) fix are not valid here.
#
# Values are given in the parameter's NATIVE (transformed) units — the same
# units as the theta vector and the *_RANGE bounds. To fix burst duration at
# 30 generations, set log_duration to log(30):
#
#   import numpy as np
#   FIX_PARAMS = {"log_duration": np.log(30.0)}      # duration fixed at 30 gen
#   FIX_PARAMS = {"log10_t_start": 2.5}              # t_start fixed at 10^2.5
#
# Names must be in the active model's PARAM_NAMES and must not also appear in
# INFER_PARAMS. training_meta records the fix so inference/benchmarking use the
# identical generative model.
FIX_PARAMS = {"log_duration": np.log(10.0)}

# ---------------------------------------------------------------------------
# Observation representation: how a window/locus is turned into network input.
# ---------------------------------------------------------------------------
#   "matrix"  : the per-window genotype matrix (state one-hot + continuous
#               channels) consumed by the exchangeable CNN. Default.
#   "sumstat" : a per-TE summary-statistic vector (te_dfe/sumstats.py) consumed
#               by a per-TE MLP + the same permutation-invariant aggregator. The
#               set of per-TE vectors generalizes to any number of TEs (train on
#               N, deploy on more) the same way window K-extrapolation works.
# The two representations share the simulator, prior, flow, training loop, and
# the FIX/INFER machinery; they differ only in the processor output and the
# embedding net.
REPRESENTATION = "matrix"

# Per-TE summary-statistic vector length (sumstat representation). Derived from
# te_dfe.sumstats so the two stay in lockstep.
def _n_sumstats_per_te():
    try:
        from .sumstats import N_SUMSTATS_PER_TE
        return int(N_SUMSTATS_PER_TE)
    except Exception:
        return 14
N_SUMSTATS_PER_TE = _n_sumstats_per_te()

# Max TEs per locus used at TRAINING (the set is variable-size; this caps the
# padded tensor width, analogous to N_TRAIN_WINDOWS). Deployment can exceed it
# via the aggregator's cardinality extrapolation.
N_TE_TRAIN_MAX = 10000  #this one should be set to the maximum number of TEs that will be used (algorithm masks empty entries for training and benchmarking)

# ---------------------------------------------------------------------------
# ADDED (OFF by default): recombination-decile O/E enrichment of TEs.
# Availability-normalized replacement for the old 3-bin TE recomb proportions.
# O = TE proportion per recomb decile (numerator, where TEs ARE);
# E = genome-bp proportion per recomb decile (exposure, from the recomb map);
# feature = log((O+eps)/(E+eps)) per decile -> a per-sim GLOBAL covariate fed on
# the existing global_cov rail (te_dfe/enrichment.py). Supplies the empty-region
# exposure the per-TE view discards -> targets high gamma_r.
#
# Edges are FIXED genome-wide deciles (computed ONCE from the recomb map, shared
# by every sim and the observed data); the DENOMINATOR is each sim's own realized
# windows (so window-sampling variation divides out). See enrichment.py.
ENRICH_OE_DECILES     = False      # master switch. When True, wire the O/E vector
                                   # into global_cov (see the 4 integration blocks).
OE_N_BINS             = 10         # number of recombination deciles
OE_PSEUDOCOUNT        = 1e-3       # eps in log((O+eps)/(E+eps)); keeps thin bins finite
OE_EXPOSURE_COMPANION = False      # also append log10 bp/bin so the trunk can
                                   # down-weight thin deciles (adds OE_N_BINS more
                                   # global_cov dims)
# The decile edges are computed at startup, once, e.g.:
#   from te_dfe.enrichment import genome_decile_edges
#   RECOMB_DECILE_EDGES = genome_decile_edges(map_positions, map_rates, OE_N_BINS)
# and cached; they are data-derived (not a static constant) so they live wherever
# the recomb map is first loaded (WindowSampler / processor init).


# --- Global sequence-length covariate (sumstat representation) -------------
# When ON, the total sequence length the TEs were pooled from is fed to the
# network as a global covariate, so abundance is usable in a GENOME-SIZE-
# INVARIANT way (N per unit sequence = density) rather than via raw set size.
# REQUIRES training sims with a VARIABLE window count (so the covariate varies
# and is decorrelated from lambda); with a constant window count it is a
# constant and teaches nothing. Default OFF -> no behavior change.
SUMSTAT_USE_SEQLEN_COV = False
N_SUMSTAT_GLOBAL_COV   = 1 if SUMSTAT_USE_SEQLEN_COV else 0
# The covariate value fed to the net is (log10(total_bp) - REF) / SCALE, kept
# ~O(1). REF ~ log10(20 windows * 5 Mb = 1e8) = 8; SCALE spans the range of
# genome/window-count sizes seen in training.
SEQLEN_LOG_REF   = 8.0
SEQLEN_LOG_SCALE = 1.0

def seqlen_covariate(total_bp):
    """Normalized global covariate from a total sequence length in bp."""
    import numpy as _np
    total_bp = max(float(total_bp), 1.0)
    return (_np.log10(total_bp) - SEQLEN_LOG_REF) / SEQLEN_LOG_SCALE

# Max flanking SNPs per TE used for the per-TE statistics (upper bound; fewer is
# fine when missing data / low diversity reduce the count). Matches the matrix
# flank layout for consistency.
SUMSTAT_FLANK_MAX = N_TE_FLANK_SITES // 2

# Number of TEs sampled per replicate for the sumstat set during training
# (deploy/extrapolate to more at inference) If I get it right, max number of TEs that are sampled at inference/benchmarking.
SUMSTAT_N_TE_TRAIN = 10000 ##this one is dead


# DFE selection-class thresholds (|2 Ne_h s|): nearly-neutral / weak / strong.
SEL_THRESHOLD_NN     = 1.0

SEL_THRESHOLD_STRONG = 10.0

