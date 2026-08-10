# te_dfe, NPE package for TE DFE + burst inference

Neural Posterior Estimation for transposable-element evolution from genotype
data. Two models can be selected at runtime:

  * **DFE, simple**: 6-parameter gamma DFE over |s| + a single rectangular burst.
  * **context**: context-dependent fitness effects + insertion preference (7 params in the
                  recombination-only default): selection and insertion rate vary
                  with local recombination.

The user can download this git repository, modify the contents of data/ so it matches the demography and the genome organization they wish, and run from the Complete_pipeline folder. More information is also available here: https://yannbourgeois.github.io/projects/1_project/
This is still work in progress, and a more detailed documentation should follow. At the moment, the pipeline has not been tested on actual biological data, although the required elements should be in place (for example for masking uncallable regions, and dealing with unphased or missing data).

## A few commands

One can install locally on their own cluster using uv. I provide here an example:
```{bash}
#| eval: false
#| warning: false
# Install uv:
# Tell the installer where to place the binary if you don't have rights:

INSTALL_PATH="/local/install/path/"   ##replace local/install/path by your own personal path where you can install uv

export UV_INSTALL_DIR="${INSTALL_PATH}/uv-bin"
#Disable the installer's prompt to modify your shell rc files
export UV_NO_MODIFY_PATH=1
export XDG_CONFIG_HOME="${INSTALL_PATH}/.config" 

# Run the installer
curl -LsSf https://astral.sh/uv/install.sh | sh
# /shared/projects/datepalm/test_DFE_deeplearning/activate.sh
cat > ${INSTALL_PATH}/activate.sh << 'EOF' 
# Project environment for te_dfe_burst on the HPC cluster
# Source this from your shell or from SLURM scripts:
#   source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh

export UV_PROJECT_DIR="/shared/projects/datepalm/test_DFE_deeplearning"
export PATH="${UV_PROJECT_DIR}/uv-bin:$PATH"
export UV_CACHE_DIR="${UV_PROJECT_DIR}/.uv_cache"
export UV_PYTHON_INSTALL_DIR="${UV_PROJECT_DIR}/.uv_python"

# Redirect uv's config away from ~/.config (which is quota-limited)
export XDG_CONFIG_HOME="${UV_PROJECT_DIR}/.config"
mkdir -p "${XDG_CONFIG_HOME}"
EOF


source ${INSTALL_PATH}/activate.sh
uv venv --python 3.11
source .venv/bin/activate

uv pip install msprime pyslim tskit torch sbi joblib scipy matplotlib numpy
uv pip install moments-popgen ##latest addition

#Download popgen-npe:
cd ${INSTALL_PATH}
git clone https://github.com/kr-colab/popgen-npe.git


##then
module load slim/4.3 ##I have SLiM preinstalled on my cluster. You can also install it locally, in which case you will need to add it to your PATH. The scripts work with version 4.3 at the moment, but should be compatible with version 5.xx, the main change being in the SLiM scripts where sim.chromosome would become sim.chromosomes with an S at the end.

source ${INSTALL_PATH}/.venv/bin/activate
source ${INSTALL_PATH}/activate.sh
export PYTHONPATH="${INSTALL_PATH}/popgen-npe:$PYTHONPATH"

```

Here is how we can run the procedure step by step for the DFE model with summary statistics. For the context model, once can switch the environment variables `TE_DFE_MODEL="simple" MODEL="simple"` to `TE_DFE_MODEL="context" MODEL="context"`. Obviously, a clean version of the pipeline should use automated scripts, for example with snakemake. Since this project is still in development, I keep working sequentially.

```{bash}
#| eval: false
#| warning: false
source ${INSTALL_PATH}/.venv/bin/activate
source ${INSTALL_PATH}/activate.sh
export PYTHONPATH="${INSTALL_PATH}/popgen-npe:$PYTHONPATH"

###Simulate under the DFE (simple) model, keep the tree sequences. The number and length of genomic windows per simulation is controlled in config_base.py. In case you do not have enough tmp space, you can specify a path to a tmp directory (TE_DFE_TMPDIR)

TE_DFE_TMPDIR=/shared/projects/datepalm/tmp_scratch TE_DFE_MODEL="simple" MODEL="simple" KEEP_TREES=1 N_MIN=20 N_MAX=1000 SEEDS_PER_TASK=10 MAX_CONCURRENT=999 SIM_DIR=/shared/projects/datepalm/simulations_trees_sumstats_simple sbatch slurm/01_simulate.sh


###If you want to reprocess simulations stored as tree sequences (for example if we add new summary statistics and need to update), this can be done also:

TREES_DIR=/shared/projects/datepalm/simulations_trees_sumstats_simple/trees OUT_DIR=/shared/projects/datepalm/simulations_trees_sumstats_simple_renewed/ SEEDS_PER_TASK=5 \
    sbatch --array=0-995%990 slurm/01b_reprocess.sh

###Here we group all simulations into a single file that will be stored in results_sumstat_simple/ This allows faster access of the training algorithm to the training sims.

TE_DFE_MODEL="simple" MODEL="simple" KEEP_TREES=1 N_MIN=20 N_MAX=1000 SEEDS_PER_TASK=10 MAX_CONCURRENT=999 SIM_DIR=/shared/projects/datepalm/simulations_trees_sumstats_simple COLLATED_TRAINING_FOLDER="results_sumstat_simple" sbatch slurm/02_collate.sh
 
 
TE_DFE_MODEL="simple" MODEL="simple" KEEP_TREES=1 N_MIN=20 N_MAX=1000 SEEDS_PER_TASK=10 MAX_CONCURRENT=999 AGGREGATOR=attention COLLATED_TRAINING_FOLDER="results_sumstat_simple" RESULTS_DIR="results_sumstat_simple" N_HEADS=12 BATCH_SIZE=32  sbatch slurm/03_train.sh

###Model 200 pseudo-observed benchmark simulations, each made of 200 windows of 5Mb (1Gb in total).

MODEL="simple" TE_DFE_MODEL="simple" N_SCENARIOS=200 N_WINDOWS=200 CHUNK_SIZE=50 SHARD_DIR=shards_simple sbatch slurm/05a_benchmark.sh

###We parallelized before by splitting the same scenario across distinct SLURM jobs. Now we gather all windows from the same scenario together.

MODEL="simple" TE_DFE_MODEL="simple" N_SCENARIOS=200 N_WINDOWS=200 CHUNK_SIZE=50 SHARD_DIR=shards_simple TEST_DIR=benchmark_simple sbatch slurm/05a_collate.sh

###Fitting the trained model on benchmark
MODEL="simple" TE_DFE_MODEL="simple" N_SCENARIOS=200 TEST_DIR=benchmark_simple RESULTS_DIR=results_sumstat_simple/ sbatch slurm/05d_sweep_benchmark_sumstats.sh

###Benchmarking results and the trained model files are found in results_CNN_simple/

###reserve 05c_sweep_benchmark.sh for the matrix representation. For example:
MODEL="simple" TE_DFE_MODEL="simple" N_SCENARIOS=200 RESULTS_DIR=results_CNN_simple/ TEST_DIR=benchmark_CNN_simple/ sbatch slurm/05c_sweep_benchmark.sh

```

There are a few helper functions that can help checking the content of any given simulation/trained model:

```{python}
#| eval: false
#| warning: false
##checking whether a sim is under the expected model (DFE, context) and representation (sumstats, CNN...)
python inspect_dataset.py simulations/sim_000001.npz 


##check the properties of the model: execute this in python
import pickle; print(pickle.load(open("results_CNN_simple/training_meta.pkl","rb")))

```

We can also train using msprime simulations, then test whether adding an increasing number of SLiM simulations improves our algorithm. At the moment this is only implemented for the context model:

```{bash}
#| eval: false
#| warning: false
###For msprime we also parallelize heavily, like for benchmarking sims.
SHARD_DIR=sim_shards_msprime_sproxy/ REP_PER_TASK=10 sbatch --array=0-999%800 slurm/msprime_sproxy_sims.sh

SHARD_DIR=sim_shards_msprime_sproxy OUT=simulations_msprime/dataset sbatch slurm/msprime_sproxy_collate.sh


###BUDGETS specifies how many SLiM simulations (stored in this example in results_sumstat_context/) should be used to retrain the pure msprime model.

MSPRIME_DATASET=simulations_msprime/dataset SLIM_DATASET=results_sumstat_context/dataset \
BUDGETS="100 250 500 1000 2500" TE_DFE_MODEL="context" MODEL="context" AGGREGATOR=attention HEADS=12 OUT_ROOT=mfnpe bash ./run_mfnpe.sh

```

We also have wrapper scripts to fit the DFE model with the moments implementation.

```{bash}
#| eval: false
#| warning: false
###In this example we reextract TE frequencies and their age from the tree sequence files stored in /shared/projects/datepalm/simulations_wo_BGS/trees, alter the actual age with a jitter of 0.3, and use 13 bins of age, whose edges are entered as 0,10,25,50,75,100,150,200,250,300,350,400,500. We estimate uncertainties using block bootstrap over our simulated genomic windows and refit the model to each bootstrapped dataset. We try 100 independent combination of starting parameters, refine the best likelihood estimates starting from the previous estimate with the highest likelihood, over 5 rounds. We fix the duration of the burst at exp(2.3026) = 10 generations.

EXTRA="--bound log_mean_s=-12,-0.69 --bound log_gamma_shape=-2.3,2.3 --uncerts-bootstrap-refit 100" AGE_JITTER=0.3 TREES_ROOT=/shared/projects/datepalm/simulations_wo_BGS/trees AGE_EDGES=0,10,25,50,75,100,150,200,250,300,350,400,500 BOOT_BLOCK_COL=window FIX="log_duration=2.3026" N_STARTS=100 N_ROUNDS=5 sbatch --array=0-200%200 slurm/dfe_aged_sweep.sh


###This script will parse the individual reports for each of the 200 training simulations and produce a summary file we can examine in R for benchmarking:

python collect_fits.py dfe_aged_fits -o dfe_aged_fits/summary_final.txt

```

Summary statistics are stored for each benchmark dataset, with TEs being subsampled at the benchmark value. These datasets can be compared to training data as illustrated above. We can obtain these summary statistics when using the sumstats representation, like this:

```{bash}
#| eval: false
#| warning: false

module load slim/4.3 
source ${INSTALL_PATH}/.venv/bin/activate
source ${INSTALL_PATH}/activate.sh
export PYTHONPATH="${INSTALL_PATH}/popgen-npe:$PYTHONPATH"
###These four calls are needed if you used uv to install the required programs. 


###statistics for DFE training model

python 07_export_sumstat_summaries.py --model simple --dataset results_sumstat_simple/dataset --out summaries_simple.csv

###statistics for context training simulations

python 07_export_sumstat_summaries.py --model context --dataset results_sumstat_context/dataset --out summaries_context.csv


###statistics for benchmarking data (pseudo observed data for ABC benchmarking)

python 07b_collate_benchmark.py --in-dir benchmark_simple/ --out benchmark_simple/observations
python 07_export_sumstat_summaries.py --dataset benchmark_simple/observations --out observed_summaries.csv

```

Of course, this is still very dense and complex, I am working on making the whole pipeline more accessible, but hopefully this should already give an idea of the procedure.

To read the active model in code: `te_dfe.config.MODEL_NAME`.

## Configuration layout (shared base + model overlays)

```
te_dfe/config_base.py     all SHARED settings: paths, simulation, window layout,
                          base channels, and the realism layer
                          (background selection, coverage mask, missingness, SFS folding, phasing, training-stage
                          feature filtering). 
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
  unphased.py       collapse_to_unphased + shuffle_phase.
  diagnostics.py, sim_introspect.py, snpe_refine.py
model_select.py     prescan_model() / add_model_arg() — the --model plumbing.
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

