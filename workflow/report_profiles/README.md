# Wave-convergence report profiles

The profiles select which independent result namespace is included in a Snakemake
report:

- `linear_wave_static.yaml`: static-refinement convergence tables and mesh plot.
- `linear_wave_unrefined.yaml`: uniform-grid convergence tables.
- `linear_wave_combined.yaml`: both scenarios in one report.

Run the desired profile before creating its report. For example, the static study
can be submitted with:

```bash
snakemake -s workflow/Snakefile \
  --configfile workflow/config.yaml workflow/report_profiles/linear_wave_static.yaml \
  --executor slurm --jobs 2 --keep-going --latency-wait 60 \
  --slurm-logdir workflow/logs/slurm
```

After all selected outputs exist, create a self-contained report archive with:

```bash
workflow/make_linear_wave_report.sh static
workflow/make_linear_wave_report.sh unrefined
workflow/make_linear_wave_report.sh combined
```

The default archives are written below `test-suite-reports/linear-wave/`. A custom
archive path can be supplied as the second argument.

## Combined static wave report

`mhd_wave_static.yaml` selects both the statically refined linear MHD wave and CPAW
convergence studies while leaving their outputs in independent namespaces. Submit
that profile with:

```bash
snakemake -s workflow/Snakefile \
  --configfile workflow/config.yaml workflow/report_profiles/mhd_wave_static.yaml \
  --executor slurm --jobs 20 --keep-going --latency-wait 60 \
  --slurm-logdir workflow/logs/slurm
```

After all selected outputs exist, create the combined report with:

```bash
workflow/make_mhd_wave_report.sh
```

The default archive is
`test-suite-reports/mhd-wave/mhd_wave_static_3D_report.zip`. A custom archive path
can be supplied as the first argument.
