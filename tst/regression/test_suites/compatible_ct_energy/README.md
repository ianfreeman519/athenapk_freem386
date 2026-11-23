# Compatible CT energy verification

The existing `check_compatible_energy.py` group checks accounting, restart,
refinement, EOS, and configuration restrictions. `--group analytic` adds exact
force-free diffusion benchmarks, independent of the numerical energy budget.

## References and fixtures

New resistive-diffusion problem selectors:

- `iprob=40`: B=(0, cos(2 pi x), sin(2 pi x)), on N x 4 cells.
- `iprob=41`: k=2 pi (1,2,0), e=(-2,1,0)/sqrt(5),
  B=e cos(k.x)+z_hat sin(k.x), on N x N cells. Set mode_x=1, mode_y=2.

Both use amplitude 1, density 1, pressure 0.1, gamma=5/3, eta=0.05, periodic
unit domains, zero velocity, compatible CT, RK2, unsplit diffusion, and PLM.
The in-plane field is initialized from differences of A_z=-sin(k.x)/|k|;
B_z is a cell average. Conserved energy uses the actual centered field.

The continuum solutions have B(t)=B(0) exp(-eta |k|^2 t) and uniform heating
Delta e_th=0.5 [1-exp(-2 eta |k|^2 t)]. Since curl B is parallel to B,
Lorentz force vanishes, velocity stays zero, and density stays one.
Pressure is (gamma-1)e_th. The analyzer uses continuum cell averages for B;
face-to-cell centering introduces an O(dx^2) representation error.
Discrete initialization is not exactly force-free, so spurious motion must
converge away. No velocity freezing or special evolution operator is used.

Runs end at eta |k|^2 t=0.2, with midpoint and final output. Heating errors
are normalized by the analytic heating increment, never by background energy.
Finest-grid B and heating errors must be below 2%, with order >=1.7 between
N=64 and 128. Density L1 error must be below 1%, RMS velocity below 0.01.
The budget subtracts recorded ideal and EOS exchanges; analytic heating is
checked separately. Existing absolute div B and budget checks also apply.

## Build and run

Rebuild your configured AthenaPK build first: selectors 40/41 require C++
compilation. Reconfigure CMake if needed to register the added CTests.
Run the simulations on a compute allocation with the matching runtime stack.
From the repository root, for example:

```bash
PY=/mnt/ffs24/home/freem386/pixi-env/.pixi/envs/default/bin/python
$PY tst/regression/test_suites/compatible_ct_energy/analytic_diffusion.py
$PY tst/regression/test_suites/compatible_ct_energy/check_compatible_energy.py \
  --driver build-v100/bin/athenaPK --group analytic --output-dir /tmp/ct-heating-serial
$PY tst/regression/test_suites/compatible_ct_energy/check_compatible_energy.py \
  --driver build-v100/bin/athenaPK --group analytic --launcher 'mpiexec -n 2' \
  --driver-arg=--kokkos-map-device-id-by=mpi_rank --output-dir /tmp/ct-heating-mpi
```

Choose fresh output directories, or pass `--resume` to reuse completed analytic
cases with identical inputs. Use the same executable and launcher when resuming;
the harness checks input text but does not fingerprint the binary. Incomplete
case directories are preserved with an `.incomplete_N` suffix. The MPI suite additionally runs serial reference
cases and compares conserved fields sorted by physical coordinates. Each process
must have access to a suitable device. Serial reference runs inherit driver args.

CTest names: `compatible_ct_energy_analytic`,
`compatible_ct_energy_analytic_mpi` (MPI builds), and
`compatible_ct_energy_analytic_reference` (no driver needed).
Simulation suites have a 30-minute CTest limit and are not assumed to finish
within 60 seconds; the reference self-test takes seconds.

Artifacts include per-case inputs, logs, PHDF/history files,
`analytic_results.json` (updated during execution), and
`analytic_convergence.png`. Outputs are retained even on failure. Default runs
create a persistent temporary results directory. The runner refuses to overwrite
an explicit existing output directory. JSON includes midpoint and final errors
and cumulative ideal/EOS exchanges. Legacy comparisons are diagnostic only.

Coverage includes N=32/64/128, half diffusion CFL at N=128, RK1/RK3 at N=128,
eta=0 controls, and a uniform-field control. The reference self-test verifies
that correct magnetic decay with absent heating fails acceptance. Thresholds
are provisional until measured; failures must be investigated, not silently
relaxed. Tabulated EOS, Spitzer diffusivity, AMR, and restart remain covered by
the existing group. The RKL2 extension below adds dedicated coverage;
3D compatible mode remains unsupported.

On the November 18 V100 environment the default `mpirun` resolved to an
incompatible PMIx installation. The matching OpenMPI 4.1.5 launcher worked with
`--mca pml ob1 --mca btl self,vader,tcp`; its default UCX transport crashed while
sending GPU buffers. This is a runtime workaround, not a change to the solver.
Legacy diagnostic crashes are recorded in JSON and do not stop compatible-mode
acceptance tests. All compatible-mode failures still stop the suite.

## RKL2 compatible-energy extension

Status: source and analysis checks only; a newly compiled executable is required.
No RKL2 simulation results are claimed before that build and the acceptance runs.

Use the existing `diffusion/integrator=rkl2` and
`hydro/ct_energy_correction_mode=compatible`, with
`hydro/ct_energy_correction=false`. Supported scope is 2D UCT Ohmic diffusion,
fixed or Spitzer coefficient, ideal-gas or tabulated EOS, without conduction,
viscosity, or reservoir masks.

The ideal RK step saves only ideal CT corrections. Each diffusion evaluation
initializes the Ohmic scratch storage and Bz flux, synchronizes Ohmic edge EMFs,
constructs energy fluxes using those synchronized edges, then synchronizes face
fluxes. Total energy, Bz, and the face field use the RKL recurrence; source budgets
use the same history weights with a zero diffusive source operator. Re-centering
face fields does not replace magnetic energy during diffusion.

The existing Ohmic heating limit remains a hard timestep cap. The explicit
parabolic timestep determines the RKL stage count, with a 10% coefficient-growth
margin. Each intermediate stage is checked collectively after boundary exchanges
and before primitive recovery, including raw thermal energy, EOS floors/ceilings,
EOS table domain, and the diffusivity bound. A failed trial restores conserved
and primitive states, face fields, and cumulative energy budgets, then retries
with half the diffusion substep. Accepted subdivisions cover the entire halfstep;
a maximum of eight halvings per split halfstep is allowed. Diagnostics are derived
again on the next accepted trial. A trial uses a private task pool, avoiding
re-entry into the outer driver's pool.

During RKL stages, diagnostic magnetic change and energy transport are measured
relative to the beginning of that superstep, and ideal correction is zero.
`ct_energy_exchange` is cumulative/restartable; stage diagnostics must not be
summed as if they were cumulative source budgets. Raw thermal energy and EOS
change are filled by primitive recovery as before.

### Cases and reference solutions

- Fixed helices 40/41: N=32,64,128; exact signed component decay and uniform heat
  gain, with sine/cosine Fourier coefficients and direction angles.
- New `iprob=42`: unit-amplitude axis helix plus half-amplitude oblique helix,
  using mode_x=1, mode_y=2. It drives fluid motion, so only full-MHD unsplit
  references are used. The reference timestep is halved until magnetic and
  thermal-increment changes are <0.1% and RMS direction changes <0.05 degrees.
  There are at most nine reference levels; failure to converge fails the test.
- Ideal-gas Spitzer helices at N=32,64,128: independently integrate da/dt=-K^2 eta(T)a with
  e_th=e_th0+(1-a^2)/2. The unclamped coefficient must change by at least 20%.
  Clamped coefficients must match the fixed-coefficient numerical solution.
- Tabulated Spitzer: helices and mixed field, with the same full-MHD reference
  refinement and comparisons of thermal energy, temperature, and ionization.
  The repository aluminum table is the default; override with `--eos-table`.
- STS ratios near 2,10,50: use amplitude 0.25 and density 100 so the hyperbolic
  and heating caps permit acceleration. Parse actual accepted tau/explicit_dt
  and RKL stage counts; the maximum accepted ratio must be within 20% of target.
- Temporal refinement: fixed mesh, 16/32/64 macrosteps or stricter physical
  limits, compared with a more finely stepped unsplit reference. Require order
  >=1.7 for magnetic and thermal errors. No thresholds are relaxed automatically.
- Uniform field, zero resistivity, static refinement, restart, and serial/MPI
  agreement controls. Existing unsupported-configuration tests remain; the old
  RKL2 rejection check is now an acceptance/budget smoke test.

Angular error uses atan2(|B cross B_ref|, B dot B_ref), preserving the distinction
between alignment and reversal. RMS angles are volume-weighted. Exclude only
reference vectors smaller than 1e-6 times initial RMS field; report excluded
volume and check signed components everywhere. A zero numerical vector against
a nonzero reference still fails signed-component accuracy.

Finest-grid and reference-comparison requirements: every signed component L1
error <2% of initial RMS field, thermal error <2% of reference heating, RMS angle
<0.5 degrees, maximum angle <2 degrees. Exact-reference spatial convergence must
have order >=1.7. Full-MHD thermal error is normalized by the volume mean of
|e_th_ref(t)-e_th_initial|. Temperature and ionization errors must also be <2%.
The mixed reference must show >0.05 degrees RMS direction change from its initial
field, so an effectively unchanged direction does not pass that coverage check.
Require negligible EOS adjustment, existing corrected budget/divergence checks,
and serial/MPI agreement at rtol=1e-10, atol=1e-11.

### Running after compilation

```bash
PY=/mnt/ffs24/home/freem386/pixi-env/.pixi/envs/default/bin/python
$PY -B tst/regression/test_suites/compatible_ct_energy/rkl2_diffusion.py
$PY -u -B tst/regression/test_suites/compatible_ct_energy/check_compatible_energy.py \
  --driver build-v100/bin/athenaPK --group rkl2 --rkl2-section fixed \
  --output-dir /tmp/ct-rkl2-fixed --resume
```

Manual RKL2 batches default to a 55-second budget. Exit code 75 means incomplete,
not passed: repeat the same command to reuse completed cases and continue.
An interrupted driver's entire process group is killed; its partial directory
is preserved before retry. Cached cases require matching input text, executable
SHA256, EOS table SHA256 when applicable, launcher, driver arguments,
successful return code, and final time.
Never substitute an old executable for the new source build.

Repeat with sections `spitzer`, `tabulated`, `temporal`, `controls`, or `all`.
Use separate output directories for different sections or launchers. Add
`--launcher '/path/to/matching/mpirun -n 2 --bind-to none --mca pml ob1 --mca btl self,vader,tcp'`
and `--driver-arg=--kokkos-map-device-id-by=mpi_rank` for the V100 MPI environment.

CTest registers `compatible_ct_rkl2_<section>` and MPI counterparts, plus the
fast `compatible_ct_rkl2_reference` self-test. Tabulated CTests are registered
when the repository table exists. CTests explicitly disable the batch budget
and have a 30-minute timeout; manual runs retain the 55-second default.

Artifacts: `rkl2_<section>_results.json`, component/direction/heating comparison
plots, convergence plots, and a `rkl2_<section>_PASS` file only after the selected
section completes. Per-case inputs, execution fingerprints, logs, PHDFs, and
histories are retained. Missing table data, unconverged references, insufficient
STS acceleration, and physical acceptance failures are failures, not skips.

## Small coarse-to-fine flow test

`--group refinement-flow` uses the existing compiled `field_loop` fixture, so no
C++ changes are needed. It advects a radius-0.15 magnetic loop and a density-2
material marker through a stationary coarse/fine interface at x=0.25. Background
density and pressure are 1, field amplitude is 0.1, and velocity is (1,0,0).
The periodic domain is [-1,1] x [-0.5,0.5], with one unit-depth z cell.

The root mesh is 64x32, blocks are 8x8, and a full-height strip 0.25<x<0.75 is
refined once, giving finest spacing 1/64. The uniform reference is 128x64 with 32x16 blocks to reduce launch overhead.
At t=0 the loop/marker is entirely coarse. Its nominal leading edge reaches the
fine strip at t=0.1, its center crosses at t=0.25, and its trailing edge crosses
at t=0.4. Runs end at t=0.55, when the marker should predominantly occupy fine
cells. The test explicitly requires fine-level excess-density mass to increase
from <5% to >65%; static refinement alone is not counted as crossing coverage.

There are ten small simulations: ideal advection on both meshes, then fixed
eta=0.02 and ideal-gas Spitzer diffusion with both unsplit and RKL2 on both
meshes. The Spitzer units are length=1 cm, mass=1e-10 g, time=1.5e-7 s, mu=1,
Zbar=3, log Lambda=10; diffusivity varies across the advected density marker.
All runs use compatible CT energy, RK2 and PLM, without dynamic remeshing.

Checks include finite/positive thermal energy, discrete divergence, periodic
mass conservation, and total-energy change minus cumulative ideal/EOS exchange
at every saved state. EOS adjustments must be negligible. Record both net and
absolute local ideal corrections normalized by initial magnetic energy, and the
pressure/thermal extrema in a band two coarse cells around the entry interface.
This prevents a small net correction from concealing large cancelling corrections.

At the final time, volume-average the uniform fine reference onto the actual
SMR leaf cells. Thermal energy is calculated on fine cells before averaging;
subtracting magnetic energy of an averaged field would create a false heating
signal. Require SMR magnetic L1 discrepancy <15% of initial RMS B, thermal L1
integral discrepancy <30% of initial magnetic energy, and entry-band maximum
thermal discrepancy <50% of initial magnetic energy density. These deliberately
broad smoke-test limits include coarse-grid transport error and are not a
production accuracy tolerance or an asymptotic convergence claim.

On each identical mesh, require RKL2/unsplit magnetic discrepancy <5%, thermal
integral discrepancy <10% of initial magnetic energy, and RMS direction error
<2 degrees for reference fields above 1% of initial loop amplitude. Record the
excluded direction volume. Discrepancies are retained as failures, not used to
adjust thresholds after execution.

```bash
PY=/mnt/ffs24/home/freem386/pixi-env/.pixi/envs/default/bin/python
$PY -B tst/regression/test_suites/compatible_ct_energy/refinement_flow.py
$PY -u -B tst/regression/test_suites/compatible_ct_energy/check_compatible_energy.py \
  --driver build-v100/bin/athenaPK --group refinement-flow --resume \
  --output-dir test_runs/refinement_flow_smoke
```

Each manual batch is capped at 55 seconds; repeat after exit 75. Existing
execution/EOS fingerprints and partial-output preservation apply. Outputs are
`refinement_flow_results.json`, `refinement_flow_failures.json`, per-case energy
correction/interface-pressure histories, and final pressure/density comparison
plots. A PASS marker appears only after all acceptance checks complete.
CTest names are `compatible_ct_refinement_flow` and the driver-free
`compatible_ct_refinement_flow_reference`.

This specifically exercises material advection into finer cells and static
coarse/fine flux/EMF synchronization. It does not test refinement triggered
on moving structures, creation/deletion of refined blocks, low-beta reconnection,
or a tabulated EOS.

## Smooth interface thermal diagnostic

`--group interface-thermal` removes the dense contact and nonsmooth loop from
`refinement-flow`. It uses the axis helix `B=(0,A cos(2 pi x),A sin(2 pi x))`,
A=0.1, rho=p=1, gamma=5/3, on a periodic unit square. The exact field translates
at vx=0 or 1 while decaying at rate eta*(2*pi)^2. Exact thermal energy is uniform:
`1.5 + A^2/2*(1-exp(-2*eta*(2*pi)^2*t))`. Magnetic comparisons use cell averages
(sinc factors); thermal comparisons use the continuum solution, so discrete
initialization errors are expected to converge rather than vanish identically.

The base matrix has 16 full runs to t=0.1: uniform field/fixed resistivity,
helix/no resistivity, helix/unsplit fixed eta=0.02, and helix/RKL2 fixed eta=0.02,
each stationary/moving on uniform 64^2 and static refinement (32^2 root,
0.25<x<0.75 refined once). Uniform fields are (0.03,0.08,0.06).
Blocks are 8^2 on the root mesh and 32^2 on the uniform reference.
Each full case has an additional cheap one-step run to capture the first
response. Initial state and first-step thermal profiles are retained separately.
Interface bands extend two root-grid cells on either side of both interfaces;
"bulk" is the remainder, including for the uniform-grid controls.

- `--interface-section base`: the matrix above, with same-mesh integrator comparisons.
- `--interface-section resolution`: moving ideal/unsplit/RKL2, uniform/refined,
  at root N=32 and 64 (finest 64 and 128), including their first steps.
- `--interface-section temporal`: moving refined N=32, successively halved
  unsplit reference steps (up to six halvings) until B error normalized by A,
  thermal error normalized by exact heat gain, and interface thermal error
  normalized by exact heat gain change by <0.1%. Compare RKL2 at dt and dt/2
  against that reference; require each temporal discrepancy to decrease unless
  both are below 1e-12. An accelerated run targets an actual diffusion
  half-step/explicit-step ratio near 3 (accepted maximum 2.4–3.6), with >1 stage.
- The default `all` runs these sections, reusing common cases.

For the base and resolution comparisons, dt is fixed to `0.1/ceil(0.1/(0.8*dt_exp))`,
where `dt_exp=0.25*0.1*dx_fine^2/0.02`. Both integrators receive the identical
`parthenon/time/dt_force`; full-precision PHDF timesteps are recorded. The accelerated
macrostep is approximately six explicit timesteps, producing diffusion halfsteps
of approximately three explicit timesteps. Its chosen parameters also keep the
macrostep below the initial hyperbolic CFL estimate for vx=1.

### Optional solver instrumentation

The new default-off `hydro/ct_interface_diagnostics=true` enables the cell field
`ct_interface_diagnostics` (26 labeled components). The helical problem generator
initializes this scratch field to zero; it is intended for these diagnostic runs.
The problem generator now accepts `problem/resistive_diffusion/vx` (default 0),
including the corresponding momentum and kinetic energy for both EOS paths.

The probes record current and resistive EMF before/after EMF synchronization,
current before/after boundary exchange, eta*J^2, update changes of E/M/K/thermal
energy, boundary changes of thermal and face magnetic energy, and the thermal
change from the actual centering operation. The pressure-work and synchronized
resistive-transport fields are **instantaneous rates**, not cumulative heating.
Current is a cell average of edge Jz; eta*J^2 additionally includes central Bz
derivatives. It is an independent stencil diagnostic, not an exact discrete heat
identity. No positivity repair or change to diffusivity-bound selection is added.

Components `rkl_transport_increment`, `rkl_previous_transport`, and
`rkl_initial_transport_rate` independently apply the RKL recurrence to the
synchronized resistive energy-flux divergence, starting from zero at each
trial substep. `rkl_thermal_increment` is measured from the actual state relative
to its diffusion-substep start; `rkl_balance_residual` subtracts the independently
predicted transport minus actual magnetic-energy change. This also exposes
unexpected kinetic-energy changes in pure diffusion. Require |residual|<1e-10.
These are **latest-stage/substep diagnostics**, not cumulative physical budgets;
only `ct_energy_exchange` carries cumulative ideal/EOS exchange. Rejected trial
scratch is replaced on retry, and output occurs only after accepted macrosteps.

RKL failures additionally report the first offending cell per partition, rank,
block, level, i/j/k, ghost status, rho, specific internal energy, total energy,
and diffusivity/bound values when available. `check=1` denotes nonfinite state
or density; 2 internal energy/velocity/EOS domain; 3 recovered EOS state; 4
invalid diffusivity or insufficient diffusion timestep. Inapplicable values are
printed as NaN. The acceptance/retry policy itself is unchanged.

Uniform controls must preserve conserved variables and zero current/EMF within
1e-11; all saved states retain existing energy/divergence/positivity checks,
relative mass tolerance 1e-10, and EOS exchange <1e-11. Interface thermal L1 error
must decrease with refinement, except when both errors are below 1e-12.
No post-run threshold tuning is performed. Large signed interface errors are
reported even if integrated thermal changes cancel.

JSON contains regional signed means, absolute L1, min/max/Linf, corrected
budgets and kinetic/magnetic/thermal energies, stage and cumulative corrections,
actual timestep ratios and stage counts. Per-case figures show signed thermal
profiles, currents, individual stage energy changes, and instantaneous rates.
A separate section-specific PASS marker is written only after all requested
checks pass. This suite diagnoses fixed-resistivity static-refinement behavior;
it does not certify Spitzer, dynamic refinement, or reconnection accuracy.

After compiling, run (repeat after exit 75; each simulation batch is capped at 55 s):

```bash
PY=/mnt/ffs24/home/freem386/pixi-env/.pixi/envs/default/bin/python
$PY -B tst/regression/test_suites/compatible_ct_energy/interface_thermal.py
$PY -u -B tst/regression/test_suites/compatible_ct_energy/check_compatible_energy.py \
  --driver build-v100/bin/athenaPK --group interface-thermal \
  --interface-section base --resume --output-dir test_runs/interface_thermal_base
```

Use analogous distinct output directories for `resolution` and `temporal`.
Timestep checks use full-precision PHDF attributes rather than rounded ASCII history values.
Binary/input fingerprints protect execution reuse; analyzer/input/execution hashes
protect cached analysis so resuming does not repeatedly spend the batch plotting
completed cases. Logs and failed/incomplete run directories remain available.
The N=64 and smaller-timestep interface cases write restart checkpoints every
0.0125 code-time units. On a batch timeout, `--resume` continues the latest
readable checkpoint only when binary, launcher, arguments, and input still match.
This preserves the macrostep settings and initial snapshots; logs are appended
with continuation markers. An actual nonzero solver exit is not automatically
continued from a checkpoint. Other incomplete runs retain the archive/retry behavior.
CTest entries are `compatible_ct_interface_thermal_{base,resolution,temporal}`
and the driver-free `compatible_ct_interface_thermal_reference`.

## Compatible RKL2 coefficient bounds and retries

The compatible RKL2 bound now uses centered conserved fields and the same
EOS-pressure-to-resistivity lookup as the Ohmic operator. Coefficient bounds
include the interior and first x/y ghost halo, including corners; the unused
second halo cannot reject a stage for exceeding its diffusivity bound.
Admissibility remains mandatory in **every** cell recovered by `FillDerived`,
including the second halo. No floors are applied by these checks.

Rejected trials restore conserved/primitive fields, face fields, and cumulative
energy exchanges on all ranks. Retries halve the substep and retain the smallest
valid observed stage bound, with a 10% reserve for selecting stages. Invalid
state/EOS, invalid coefficient, and bound failures have separate flags (1, 2, 4).
Eight halvings are allowed for each attempted substep; acceptance resets that
counter. An invalid restored base fails immediately. The heating-time cap remains
separate from the accelerated parabolic bound.

Compile both `athenaPK` and the new `compatible_ct_rkl_bounds` target. The latter
has serial and two-rank CTest entries (`compatible_ct_rkl_bounds_serial` and
`compatible_ct_rkl_bounds_mpi`), checking first-halo and corner coefficients,
second-halo exclusion, full-halo state validation, retry bounds/counts, and
rejection originating on only one rank. These are focused helper checks; the
simulation regression below exercises the complete solver.

After compilation, run the original Spitzer/SMR configuration and a sequence of
same-mesh unsplit references in batches below 60 seconds:

```bash
$PY -u -B tst/regression/test_suites/compatible_ct_energy/check_compatible_energy.py \
  --driver build-v100/bin/athenaPK --group refinement-flow \
  --refinement-section guard --batch-seconds 55 --resume \
  --output-dir test_runs/spitzer_smr_guard_serial
```

Repeat after exit 75. For two ranks, use a separate output directory and add
`--launcher 'mpiexec -n 2'` (plus the allocation's GPU arguments if required).
Checkpoint continuation retains the original initial snapshots and appends logs.
Only output/checkpoint settings are added to the failing RKL2 physical setup;
its final time remains 0.55. Unsplit references halve `hydro/max_dt` from
0.00025, requiring magnetic changes below 0.1% of initial RMS field, thermal
changes below 0.1% of mean absolute reference thermal increment, and RMS direction
changes below 0.05 degrees. Failure to converge within five caps is reported.
The existing integrator comparison limits remain B L1 <5%, thermal L1 <10% of
initial magnetic energy, and direction RMS <2 degrees. This guard regression
also requires no EOS intervention, the existing divergence/budget tolerances,
and measured super-time-stepping coverage. JSON/logs remain available on failure.
CTest simulation entries are `compatible_ct_spitzer_smr_guard{,_mpi}`; these run
without batch limits and are intended for a suitable compute allocation.
Existing fixed-resistivity interface fixtures and their thresholds are unchanged.
