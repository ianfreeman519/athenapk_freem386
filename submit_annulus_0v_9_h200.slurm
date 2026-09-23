#!/bin/bash
#SBATCH --job-name=marz-0v9
#SBATCH --account=general
#SBATCH --partition=general-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=h200:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/gs21/scratch/freem386/MARZ/uct/annulus_0v_9
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

# These are the files sourced by the interactive sourcepixi/sourceapk aliases.
# Load pixi first and the AthenaPK Spack environment last so its MPI, CUDA, and
# parallel-HDF5 libraries take precedence for the simulation executable.
source "${HOME}/pixi_init.bash"
source "${HOME}/apk_profile.sh"

# Enable strict unset-variable and pipeline handling only after sourcing the
# third-party environment initializers, which are not guaranteed to be nounset-safe.
set -uo pipefail

run_dir="/mnt/gs21/scratch/freem386/MARZ/uct/annulus_0v_9"
input_file="pulsed_reconnection_marz.in"
executable="athenaPK"

cd "${run_dir}"

if [[ ! -x "./${executable}" ]]; then
  echo "ERROR: ${run_dir}/${executable} is missing or not executable." >&2
  exit 1
fi
if [[ ! -r "${input_file}" ]]; then
  echo "ERROR: ${run_dir}/${input_file} is not readable." >&2
  exit 1
fi

# AthenaPK/Kokkos uses one host thread to launch work on each GPU. Extra CPUs are
# available to MPI and parallel HDF5, but OpenMP threading would only add contention.
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "Start: $(date --iso-8601=seconds)"
echo "Host(s): ${SLURM_JOB_NODELIST}"
echo "Run directory: ${run_dir}"
echo "MPI ranks: ${SLURM_NTASKS}; GPUs: ${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Use one MPI rank on one H200 so the job can occupy a free GPU fragment instead of
# waiting for an entire H200 node. All 152 mesh blocks remain available to the GPU.
srun --mpi=pmix_v4 \
  --cpu-bind=cores \
  --gpu-bind=single:1 \
  "./${executable}" -i "${input_file}"

echo "Finish: $(date --iso-8601=seconds)"
