# Small pulsed-power tests after the compatible-energy regression pass

Prepared inputs: `/mnt/ffs24/home/freem386/pulsed_power_tests`.
No build, simulation, or job submission was performed during preparation.
Use one rebuilt executable for every case, and start every case fresh.

## Matrix and GPU layout

| Case | Mesh | Energy scheme | Pulse | Purpose |
|---|---|---|---|---|
| screen_legacy_off | 512 x 512 | Legacy, correction off | On | Reproduce thermal failure at low cost |
| screen_legacy_on | 512 x 512 | Legacy, correction on | On | Expose the previous correction/temperature behavior |
| screen_compatible | 512 x 512 | Ideal-only correction + compatible resistive transport | On | Matched experimental case |
| screen_no_drive | 512 x 512 | Compatible | Off | Check the evolving fluid/EOS/floor baseline without magnetic driving |
| confirm_legacy_off | 1024 x 1024 | Legacy, correction off | On | Original-resolution matched baseline |
| confirm_compatible | 1024 x 1024 | Compatible | On | Check whether the improvement survives resolution change |

Run the four screening cases simultaneously, one MPI rank on each V100.
Review them before launching confirmation. Run the two confirmation cases
simultaneously, with two MPI ranks/GPUs each. If the analytic pass was serial-only,
run the MPI regression matrix before trusting the multi-GPU confirmation.
The launchers assume a single machine/allocation with four visible GPUs and
respect an existing CUDA_VISIBLE_DEVICES ordering. They neither build nor submit.

## Fixed physical setup

All driven cases retain the investigated geometry: arrays separated by 6 cm,
1.95--2.05 cm fluid annulus, magnetic conductor ramp 0.5--1 cm and return-current
transition 1.95--2.05 cm. Peak drive is 2e6 G with the original 300 ns peak time.
Do not shorten the pulse to make the test cheaper: that changes the physics.
Initial B and imposed initial velocity are zero. Reservoir rho remains .1 g/cm3;
it is not prescribed or replenished. These runs do not validate that reservoir
model or solve any later core implosion.

- Domain: [-6,6] squared cm, outflow in-plane, unchanged z extent [-1,1].
- Units: L=1 cm, M=1e-10 g, time=1e-7 s.
- Background rho=1e-6 g/cm3; code dfloor=1e4, exactly the background density.
- T_background=1000 K; Tfloor=1000 K; the existing additive reservoir T remains.
- Tabulated aluminum EOS, unchanged conversion factors.
- RK2, PLM, UCT-HLLE, hyperbolic CFL .4, diffusion CFL .15.
- Explicit unsplit Spitzer resistivity: log Lambda=10, eta bounds 1e3--1e7 cm2/s.
- No cooling, conduction, viscosity, reservoir mask, mass injection, or velocity injection.
- No imposed azimuthal density/temperature perturbation (N=0).
- Uniform mesh, 128-squared blocks; AMR is explicitly disabled to avoid the
  separate tagging bug and coarse/fine initialization differences.

512 spacing is 234.375 microns: only 2.13 cells across each .05 cm transition.
It is a screen, not a resolved production result. At 1024 spacing is 117.1875
microns, with 4.27 cells across the transition. The fine run is necessary even
if the coarse screen looks clean.

## Duration, cost controls, and output

End time is 80 ns, beyond the earlier 51--70 ns warning/stall interval.
Screening has a 100000-cycle cap; confirmation has a 250000-cycle cap.
A persistent dt <=1e-8 code (1e-6 ns) for 100 cycles aborts the run; this
is a stopping guard, not a timestep floor. No dt_force, dt_floor, or CFL
relaxation is applied. A cycle cap can stop the executable with exit code zero
before 80 ns: that is a censored/stalled run, not a pass.

The screen launcher defaults to a 45-minute per-case timeout; confirmation
uses 90 minutes. These are cost caps, not runtime predictions. Override with
CASE_TIMEOUT if the allocation permits. Timeout, abort, or early cycle-stop
is recorded and must remain visible in the comparison.

PHDF every 2 ns, history every 100 cycles, and restart every 20 ns.
PHDF includes prim, cons, Bface, divB, curlBz, eta, beta, EOS T and Zbar.
Compatible cases additionally write ct_energy_exchange and ct_energy_diagnostics.
Diagnostics remain enabled in those cases because thermal-energy accounting
is the acceptance target. All runs preserve their logs and exit status.

## Run commands

From the repository root, with DRIVER pointing to the rebuilt executable:

```bash
bash /mnt/ffs24/home/freem386/pulsed_power_tests/run_screen.sh "$DRIVER"
```

After reviewing screening results:

```bash
bash /mnt/ffs24/home/freem386/pulsed_power_tests/run_confirmation.sh "$DRIVER"
```

The launchers refuse to overwrite an existing run.log. Create a new bundle for
reruns rather than mixing old and new executable outputs. To relocate/regenerate:

```bash
python tst/regression/test_suites/compatible_ct_energy/prepare_pulsed_power_tests.py \
  --input /path/to/high_density_floor/pulsed_reconnection.in \
  --eos-table /path/to/aluminum_saha_eos_three_chart.h5 \
  --output /new/path/to/pulsed_power_tests
```

## Measurements and decision rules

Run the measurement script after screening and again after confirmation:

```bash
python tst/regression/test_suites/compatible_ct_energy/analyze_pulsed_power_tests.py \
  /mnt/ffs24/home/freem386/pulsed_power_tests
```

Use the Python environment with numpy/h5py. It produces per-case CSV time series
and `analysis/summary.json`; it does not launch simulations or assign a blanket
pass to a run. It explicitly marks incomplete/censored runs and missing common
physical times. It compares coarse/fine quantities at 40,60,80 ns using snapshots
within 1 ns of each target; inspect the reported actual sample times.

Regions are measured around both arrays: core r<1.05 cm, gap 1.5<r<1.85 cm,
and return layer 1.85<r<2.1 cm. Report temperature percentiles, minimum density,
thermal energy, eta, fraction at the density floor, raw-energy warning counts
per 1000 cycles, and compatible-mode ideal/EOS energy exchanges. Integrals
include the input z extent, like history. Inspect temperature/current maps at
40,60,80 ns as well as scalar summaries; watch for pixel-scale/spoke structure.
Do not demand an isothermal gap: compression and Ohmic heating can be physical.

Screening gates:

1. Compatible driven case must reach 80 ns without fatal EOS errors, nonpositive
   saved thermal states, persistent tiny-dt abort, timeout, or cycle exhaustion.
2. No-drive case must remain finite with zero magnetic field and no severe raw
   thermal-energy failures. It is an evolving fluid baseline, not a force-balanced
   equilibrium. If it fails, investigate the EOS/floor/initial fluid model first.
3. Compatible mode should reduce interior raw-energy warning frequency at common
   physical times relative to legacy-off. A disappearing warning is insufficient
   if cumulative EOS energy replacement is substantial.
4. Record cumulative gap EOS energy addition / current gap thermal energy. Use
   1% as an initial screening target; exceeding it flags floor-dependent thermal
   results requiring investigation, not an excuse to raise floors again.

Confirmation gates:

- Both compatible resolutions must reach the common end time. A stalled legacy
  run remains a measured baseline failure; compare earlier common times and do
  not extrapolate its 80 ns state.
- Check gap thermal energy, gap T99 and return-layer T99 at common times. Relative
  coarse/fine changes <=10% are a practical initial sensitivity target, not proof
  of asymptotic convergence. Larger changes require a further resolution study
  before production conclusions. Do not loosen the target to accommodate a run.
- Verify div B remains near roundoff in the uniform runs, and inspect normalized
  divergence (`max|div B|*dx/max|B|`) alongside the dimensional maximum.
- Confirm that the apparent improvement is not solely a larger dt or a larger
  ideal energy subtraction. Plot ideal/EOS source budgets with thermal energy.

Global energy need not be constant: the pulse injects magnetic energy and the
boundaries are outflow. The new ct_energy_exchange budget records only ideal CT
and EOS adjustments, not pulse work. Do not claim a closed global energy balance
from those counters alone; the periodic analytic tests establish that property
in their appropriate unforced setting. Density-floor mass additions are also
not directly tracked by the new energy counters.

If the compatible run remains cold-floor-dependent or grid-structured after
confirmation, retain the outputs and design the separately evolved thermal-energy
fallback next. If it passes, reintroduce the azimuthal perturbation as a separate
study with sufficient angular resolution; do not mix it into these diagnostic runs.
