#!/bin/bash
# Run inside an existing allocation with one V100 and the appropriate MPI/CUDA environment.
# Usage: bash run_pilot.sh /absolute/path/to/V100/athenaPK
set -euo pipefail
pilot_executable="${1:?Pass an absolute path to a V100-compatible AthenaPK executable}"
[[ "${pilot_executable}" = /* && -x "${pilot_executable}" ]] || { echo "Executable must be an absolute executable path" >&2; exit 1; }
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if compgen -G 'parthenon.out*.phdf' > /dev/null || compgen -G 'parthenon.out*.rhdf' > /dev/null; then
  echo "Pilot output already exists; choose a clean directory or restart explicitly." >&2
  exit 1
fi
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
srun --ntasks=1 --mpi=pmix_v4 --cpu-bind=cores --gpu-bind=single:1 \
  "${pilot_executable}" -i pulsed_reconnection_marz.in -t 00:45:00
