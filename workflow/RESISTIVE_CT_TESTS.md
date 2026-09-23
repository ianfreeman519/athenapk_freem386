# Resistive UCT validation workflows

The extended workflow compares `ucthlldmhd/hlld` and `ucthllemhd/hlle`. It runs:

- finite-amplitude force-free ABC decay, checking magnetic-to-thermal conversion,
  total-energy conservation, and the analytic Ohmic decay rate;
- a periodic antiparallel current sheet at three diffusivities, checking the analytic
  effective reconnection rate `R = sqrt(eta/(pi*t))` and fitted `R proportional to
  S_L^-1/2` scaling;
- finite primitive variables, the CT `maxFaceDivB` history diagnostic, and agreement
  between the two UCT implementations.

Enable only this test without editing `workflow/config.yaml`:

```bash
/mnt/ffs24/home/freem386/pixi-env/.pixi/envs/default/bin/snakemake \
  -s workflow/Snakefile \
  --configfile workflow/config.yaml workflow/resistive_ct_extended.yaml \
  --cores 4 --resources gpu=4 \
  --rerun-incomplete --printshellcmds
```

Use `--dry-run` first to inspect the planned commands. Add `--forceall` to replace an
existing result set. No Slurm executor is needed for the command above.

Each simulation is routed through `workflow/gpu_lock_launcher.py`. It atomically
claims one of the configured `gpu_ids`, sets `CUDA_VISIBLE_DEVICES`, and holds the
advisory lock until AthenaPK exits. This lets four rules execute concurrently without
selecting the same GPU. Change `gpu_ids` in the extended section of
`workflow/config.yaml` if the available physical device IDs are not `0,1,2,3`.

The workflow passes only if
`workflow/results/resistive_ct_extended/resistive_ct_extended.passed` exists and
contains `PASS`. Detailed metrics are written to `resistive_ct_extended_summary.txt`
and `resistive_ct_extended.csv`; simulation stderr files are under `runs/`.

Acceptance criteria are:

- positive magnetic-energy loss and positive thermal-energy gain;
- at least 90% of the magnetic loss appears as thermal energy;
- magnetic/thermal/kinetic energy closure within 5%;
- total-energy drift below `5e-5` and analytic magnetic-decay error below 8%;
- current-sheet rate within 25% of the analytic diffusion solution;
- fitted log-log rate exponent between 0.40 and 0.60;
- `maxFaceDivB <= 1e-10`;
- heating agreement within 5% and reconnection-rate agreement within 10% between
  the HLLD and HLLE UCT fluids.

This current-sheet problem is a controlled resistive reconnection benchmark: it tests
the same `S_L^-1/2` scaling as Sweet--Parker through the exact diffusion of two
antiparallel sheets. It is not a claim that the small problem reaches a nonlinear,
steady Sweet--Parker exhaust.
