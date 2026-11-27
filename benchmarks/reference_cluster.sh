#!/bin/bash
# reference_cluster.sh -- Slurm batch job: the tightest reference bracket of S on BYU's Office of
# Research Computing (ORC) cluster, i.e.  python reference_bounds.py --cluster
#
# Levels g = 0.14 ... 0.025 (finest: 13.6 million nodes).  Expected bracket width about 4e-11
# relative (default levels on a PC: about 2e-10).
#
# One-time setup on a login node (module names below are PLACEHOLDERS: check `module avail python`):
#     module load python/3.12                 # any Python >= 3.10
#     python -m venv ~/venvs/ide-bench
#     source ~/venvs/ide-bench/bin/activate
#     pip install numpy scipy
# and copy this folder (bench_common.py, fem_tensor.py, reference_bounds.py, this script) to the
# cluster.  fem2d.py / ide_model.py are NOT needed for this job (bench_common only prints where
# they are; set DIELECTRIC_IDES if you want the banner to show them).
#
# Submit from this folder:      sbatch reference_cluster.sh
# Watch:                        squeue -u $USER      and      tail -f reference_<jobid>.log
# Results: results/reference.txt (table + lines for bench_common.py) and rows appended to
# results/benchmark_results.csv.
#
# Resources (from runs up to 1.6 million nodes in Claude's sandbox, extrapolated):
#   memory: about 1.8 GB per million nodes at the finest level -> ~34 GB at 13.6M nodes;
#           64 GB requested for safety (the extrapolation of SuperLU's fill is uncertain by ~30%)
#   time:   ~10 min for the finest level, ~25 min in all at the sandbox's speed (Xeon 2.1 GHz);
#           2 h requested
#   one core: SuperLU is single-threaded and bench_common pins numpy/scipy to one thread
#SBATCH --job-name=ide-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=reference_%j.log
##SBATCH --mail-type=END,FAIL                # optional: remove one '#' and set your address
##SBATCH --mail-user=YOUR_NETID@byu.edu

set -euo pipefail

module purge
module load python/3.12                      # PLACEHOLDER: a name from `module avail python`
source "$HOME/venvs/ide-bench/bin/activate"  # PLACEHOLDER: the venv made above

# export DIELECTRIC_IDES="$HOME/dielectric-IDEs"   # optional, see above

cd "$SLURM_SUBMIT_DIR"
echo "Host $(hostname), started $(date)"
python reference_bounds.py --cluster
# For an even finer level (more memory: about 2.6 GB per million nodes there; SuperLU's 32-bit
# indices allow about 2e9 nonzeros in each of L and U, roughly 30 million nodes), e.g.
#     python reference_bounds.py --levels 0.05,0.03,0.022
echo "Finished $(date)"
