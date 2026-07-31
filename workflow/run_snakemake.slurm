#!/bin/bash --login
#SBATCH --job-name=snmake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-08:00:00
#SBATCH --output=/mnt/home/wysongj2/athenapk/workflow/logs/snakemake-controller.%j.out
#SBATCH --error=/mnt/home/wysongj2/athenapk/workflow/logs/snakemake-controller.%j.err

cd /mnt/home/wysongj2/athenapk

# dry run command for copy paste
# snakemake -s workflow/Snakefile   --configfile workflow/config.yaml -n
# snakemake -s workflow/Snakefile   --configfile workflow/config.yaml workflow/report_profiles/mhd_wave_static.yaml -n

# activate conda env here if needed
module purge
module load Miniforge3/25.11.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate snakemake

snakemake -s workflow/Snakefile \
  --configfile workflow/config.yaml \
  --executor slurm \
  --jobs 24 \
  --rerun-incomplete \
  --keep-going \
  --latency-wait 60 \
  --slurm-logdir workflow/logs/slurm


# /mnt/home/wysongj2/.conda/envs/snakemake/bin/snakemake \
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
