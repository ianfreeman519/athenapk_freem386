# MARZ annulus_s5

Prepared from the actual annulus_s3 input. No executable was changed.
Annular initial/replenishment density: 5e-3 g/cm^3.
Inner initial/replenishment density: 1e-4 g/cm^3, radius 1.8 cm.
Initial and supported inner temperature increments: 1e5 K (background adds 1e5 K).
Initial B=v=0. Injected annular mass receives up to 100 km/s; existing mass is not reset.
Confined-current peak field 2e6 G at 1 cm; wire-current transition 0.0999999 cm.
Fixed annular temperature support 2.3e5 K; capped Spitzer diffusivity unchanged.
No cooling, conduction, or viscosity. Domain unchanged to isolate source changes.

Finest spacing 29.296875 microns (static level 2); target 350 ns.
PHDF every 10 ns, restart every 100 ns, history/log every 100 cycles.
One H200, 8 CPUs, general-short/general, 4-hour allocation.
AthenaPK wall timer 3h45m permits final output before scheduler cancellation.
Completion time is not guaranteed; inspect the final physical time and restart if needed.

Submit from this directory with `sbatch submit_h200.slurm`.
Fresh-run scripts refuse existing PHDF/restart outputs. Jobs have NOT been submitted.

## Suite order

1. Optional but recommended: S7 pilot_v100, separate directory, 58.6 microns to 160 ns.
   Use a V100-compatible executable with the same problem generator, not the H200 binary.
   In an existing one-V100 allocation (allow >=1 hour), load its MPI/CUDA environment,
   then run `bash run_pilot.sh /absolute/path/to/V100/athenaPK` from pilot_v100.
   The 45-minute internal timer is a budget, not an estimated completion time.
2. S7 full H200 run first; S5 control next (these may run concurrently).
3. S6 and S8 bracket S7's annular density; then S9 tests inner support at S7 density.

Compare at 100, 130, 160 ns and later 200, 250, 300, 350 ns.
Measure upstream Bx, density, normal velocity, temperature and Alfven speed together,
including a fixed +/-2 mm sample averaged along |x|<8.75 mm for comparison with S0-S4.
Also locate shocks and measure immediately upstream; a fixed sample can enter a shock.
Evaluate magnetic delivery relative to ram pressure, not just maximum B or a thin density feature.
The coarse V100 pilot screens delivery and stability, not sheet-width convergence.
Do not submit the rest blindly if the pilot has weak B at similar mass loading or floor failures.

S5 is a NEW matched-reservoir control, not an exact resolution-only repeat of S3:
initial inner rho/T changes from 1e-7/1e8 to 1e-4/1e5; resolution also changes.
S5-S8 isolate annular density; S7 versus S9 isolates inner density.

## Update after pilot_v100_2 (2026-09-22)

The main H200 input now uses magnetic wire inner/outer radii 2.0/2.2 cm,
with transition width 0.0999999 cm. Mass, temperature and velocity masks remain
1.9-2.1 cm. The source-field taper therefore lies immediately outside those masks.
The archived pilot_v100 input is unchanged; pilot_v100_2 tested the extended cutoff.
Pilot 2 completed 160 ns normally, in 513 s on H200 despite its directory name.
At x approximately zero, y approximately 2 mm, Bx at 150 ns increased from
1.80 T to 4.03 T, with nearly unchanged density and inflow velocity.
Proceed with S7 and S5, then S6/S8, then S9. No further pilot is required first.
This replaces the earlier pending-pilot recommendation above.
