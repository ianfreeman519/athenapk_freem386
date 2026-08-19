#!/usr/bin/env bash
#SBATCH --job-name=snmake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-08:00:00
#SBATCH --output=snakemake-controller.%j.out
#SBATCH --error=snakemake-controller.%j.err

set -euo pipefail

repo_root="${ATHENAPK_ROOT:-${HOME}/uct-athenapk}"
workflow_dir="${repo_root}/workflow"
pixi_bin="${PIXI_BIN:-${HOME}/.pixi/bin/pixi}"
pixi_manifest="${PIXI_MANIFEST:-${HOME}/pixi-env/pixi.toml}"

mkdir -p "${workflow_dir}/logs/slurm"
cd "${repo_root}"

# dry run command for copy paste
# snakemake -s workflow/Snakefile   --configfile workflow/config.yaml -n
# snakemake -s workflow/Snakefile   --configfile workflow/config.yaml workflow/report_profiles/mhd_wave_static.yaml -n

"${pixi_bin}" run --manifest-path "${pixi_manifest}" snakemake \
  -s workflow/Snakefile \
  --configfile workflow/config.yaml \
  --executor slurm \
  --jobs 24 \
  --rerun-incomplete \
  --keep-going \
  --latency-wait 60 \
  --slurm-logdir workflow/logs/slurm


# "${pixi_bin}" run --manifest-path "${pixi_manifest}" snakemake \
#   -s workflow/Snakefile \
#   --configfile workflow/config.yaml \
#   --report test-suite-uct-ct-glm/athenapk_3D_all_tests.zip

# snakemake -s workflow/Snakefile \
#   --configfile workflow/config.yaml workflow/report_profiles/mhd_wave_static.yaml \
#   --executor slurm \
#   --jobs 20 \
#   --keep-going \
#   --rerun-incomplete \
#   --latency-wait 60 \
#   --slurm-logdir workflow/logs/slurm \
#   --consider-ancient \
#     run_linear_wave_static=executable \
#     make_linear_wave_static_mesh_plot=executable

# After the workflow finishes, generate its combined static-wave report with:
# workflow/make_mhd_wave_report.sh
