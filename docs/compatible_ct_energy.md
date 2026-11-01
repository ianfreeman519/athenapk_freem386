# Compatible CT energy mode (experimental)

Enable the new path explicitly:

```ini
<hydro>
ct_energy_correction_mode = compatible
ct_energy_correction = false
ct_energy_diagnostics = true
```

The mode itself enables ideal-only CT correction. The old boolean must remain
false: setting both requests an ambiguous algorithm and is rejected.
`ct_energy_correction_mode = legacy` is the default and preserves the old
boolean behavior. Diagnostics default to false and require compatible mode.

Supported: unmasked two-dimensional UCT-HLLE or UCT-HLLD, RK1/2/3, ideal or
tabulated EOS, and no diffusion or explicit unsplit Ohmic resistivity with fixed
or Spitzer coefficients. Masking, RKL2, conduction, viscosity, 3D and other
integrators/fluids are rejected. Existing EOS/Riemann restrictions still apply;
in particular tabulated EOS supports UCT-HLLE, not UCT-HLLD. Cooling is unchanged.
This change does not repair density-AMR dispatch or introduce a thermal-energy
fallback. Static refinement is included in the deferred validation matrix.

## Stage algorithm

1. Save the ideal UCT edge electric field and the actual RK-weighted starting
   face field in `ct_ideal_reference`.
2. Compute the Ohmic electric field from the existing face-centered current
   and coefficient interpolation. Retain it in `ct_ohmic_reference`, and add
   it to the physical CT electric field. Do not obtain it by subtracting two
   large electric fields.
3. Synchronize all edge electric fields. Construct resistive Poynting fluxes
   from that synchronized Ohmic electric field and the stage-input face B:
   `Sx=-average(Ez*By)` and `Sy=average(Ez*Bx)`. At each edge, arithmetic-average
   the transverse B across its perpendicular direction; then arithmetic-average
   the products along the face. Synchronize the resulting energy fluxes before
   updating conserved variables. The second synchronization waits for the first
   send and receive/set to finish and reuses the existing communication path.
4. Update the conservative state and actual CT field. Reconstruct the ideal-only
   CT field with the stage's `gam0`, `gam1`, and `beta*dt`, then add
   `0.5*(Bideal_CT^2-Bconservative^2)` to total energy. Synchronize the
   cell-centered B to the actual CT field before driving and primitive recovery.
   The legacy centering correction remains off, preventing double correction.

The physical Ohmic change in magnetic energy remains in thermal recovery.
No extra `eta*J^2` source is added. The original timestep restrictions remain
active. The shared spatial discretization does not guarantee pointwise positive
finite-step heating: analytic energy-transfer and convergence tests are required.
The ideal correction is explicitly nonconservative and has a separate budget.

For 2.5D states, Bz is a cell variable. Compatible mode also evolves its resistive
face flux `-eta*grad(Bz)` and uses exactly that flux times face-average Bz for
its Poynting contribution. The in-plane ideal correction does not compensate
this physical Bz diffusion. This differs intentionally from the legacy UCT path,
which skips cell-centered resistive B fluxes.

Stage scratch fields are derived, rebuilt each stage, and are not restarted.
Their face metadata participate in the same synchronization/refinement machinery
as Bface. `ct_resistive_flux` carries resistive energy and Bz face fluxes; its
cell values are unused scratch (the generic flux updater may overwrite them).

## Accounting and outputs

Compatible mode always enrolls `ct_energy_exchange`, with cumulative signed
source densities `ideal_energy` and `eos_energy`. They use the same RK combinations
as the conserved state, are conservatively remapped, and survive same-mode restart.
History includes their volume integrals as `ct_ideal_energy` and `ct_eos_energy`.
The EOS budget observes the actual signed total-energy change during recovery,
including floors and ceilings; it is not limited to positive temperature-floor
additions. Ghost-cell adjustments are excluded. Initial recovery is included.
Neither physical Ohmic heating nor the problem's imposed magnetic pulse is an
external source in this ideal-correction budget; pulse accounting remains the
problem's responsibility.

With `ct_energy_diagnostics=true`, add `ct_energy_diagnostics` to PHDF variables.
Its five components are:

1. `ideal_correction`: latest stage energy-density correction.
2. `resistive_magnetic_change`: actual minus ideal-only magnetic energy density,
   including the Bz resistive increment; includes finite-step cross terms.
3. `resistive_energy_transport`: latest stage `-beta*dt*div(S_ohmic)`.
4. `raw_thermal_energy`: thermal energy density immediately before the latest
   EOS recovery, before its density/thermal floors.
5. `eos_energy_change`: signed change from that recovery.

Components 1-3 are stage increments, not rates or cumulative budgets. Components
4-5 reflect the latest recovery; another recovery during initialization/remeshing
can overwrite the earlier sample. Use cumulative exchange fields for accounting.
Enable PHDF output of `ct_energy_exchange` for restart/energy comparisons.
In periodic unforced tests, total-energy change should match the sum of the two
exchange-budget changes. The production pulse adds a separate physical source.

## Deferred validation (no build or simulations performed)

After building, run on a compute allocation using the specified Python environment:

```bash
/mnt/ffs24/home/freem386/pixi-env/.pixi/envs/default/bin/python \
  tst/regression/test_suites/compatible_ct_energy/check_compatible_energy.py \
  --driver /absolute/path/to/rebuilt/athenaPK \
  --eos-table /absolute/path/to/aluminum_saha_eos_three_chart.h5 \
  --output-dir /new/path/to/validation_outputs
```

Repeat with `--launcher 'mpiexec -n 2'` in a separate output directory. Optional
`--baseline /path/to/original/athenaPK` checks disabled-mode output bit for bit.
The driver path must be absolute or resolvable from the invocation directory.
CTest `compatible_ct_energy` runs the core matrix; the table-dependent cases
require the explicit command above. No test builds or submits jobs.

The harness checks legacy equivalence, ideal correction equivalence for RK1/2/3,
Fourier decay and energy transfer (including Bz), low-beta circular-field
convergence, EOS floor accounting, ideal/tabulated Spitzer paths, UCT-HLLD,
static refinement, restart, and unsupported configurations. The circular fixture
samples 0.14<r<0.28 in a unit box, away from its conductor and return-current
layers; it checks error convergence, not an exact discrete equilibrium.
The static-refinement case uses the two-dimensional Fourier vector potential
(iprob=10), whose discrete-curl face values telescope under area restriction.
The face-center sine samples of iprob=1 are only a uniform-grid fixture: normal
face restriction at a coarse/fine interface introduces initial divergence.
Energy checks reject initial divergence as well as final divergence, so a bad
initial state cannot be mistaken for divergence generated by CT evolution.

The circular equilibrium fixture sets `problem/resistive_diffusion/circular_force_balance=true`.
It uses quintic current transitions over 0.02<r<0.10 and 0.34<r<0.42, with
pressure satisfying `dp/dr=-B_phi*dB_phi/dr-B_phi^2/r` in the continuum.
The sampled gap retains the original `pressure=1e-8`; no numerical floor is
introduced. Pressure support and the vector potential are initialized by
piecewise Simpson quadrature in the problem generator. Finite-volume force
balance is approximate and must converge with resolution.

With `circular_force_balance=false` (default), iprob=30 retains the original
uniform-pressure, unbalanced current-layer fixture. That is a stress test, not
a stationary equilibrium: at amp=0.01 its first Euler stage can consume more
kinetic energy than the initial cold thermal budget, even with a CFL-stable
step. It previously caused a first-step negative-pressure abort at 64 squared.
Do not treat that abort as proof that the current-free gap failed, or hide it
by enabling pressure floors. Compatible-mode fatal pressure diagnostics now
include block/cell indices and the thermal, kinetic, and magnetic energy terms.

Do not use the production pilots to establish correctness before analytic tests
pass. Prepare three fresh, otherwise identical pilots with:

```bash
python tst/regression/test_suites/compatible_ct_energy/prepare_pilots.py \
  --input /path/to/high_density_floor/pulsed_reconnection.in \
  --eos-table /path/to/aluminum_saha_eos_three_chart.h5 \
  --output /new/path/to/pilots
```

The preparer creates `legacy_off`, `legacy_on`, and `compatible` directories,
with physical background 1e-6 g/cm3, code density floor 1e4, no density
perturbation, and end time 80 ns. It requires the original L=1 cm, M=1e-10 g,
t=1e-7 s normalization and preserves source mesh/AMR settings. It does not
build, submit, or run simulations. Compare common-time temperature/current
morphology, timestep, raw-energy warning frequency per cycle, and compatible-mode
source budgets. Legacy mode has no new counters; absence of warnings alone is
not proof of energy accuracy. Restore the mode-150 seed only after validation.
Start fresh when changing modes; same-mode compatible restart is the supported
restart test. Do not reinterpret legacy checkpoint data as a compatible checkpoint.
