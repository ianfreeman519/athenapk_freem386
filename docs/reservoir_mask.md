# Prescribed pulsed-reconnection reservoir

`inputs/ct_glm_compatible/pulsed_reconnection_reservoir_mask.in` is a separate
S9-based example. Existing inputs and staged runs are unchanged. It uses:

```ini
<problem/pulsed_reconnection>
reservoir_mask_enable = true
reservoir_mask_core_radius = 1.2
reservoir_mask_outer_radius = 1.6
reservoir_mask_relaxation_time_ns = 5.0
```

Lengths are physical cm, like the other problem geometry inputs; time is ns.
Each array has its own circular mask. Radii are required when enabled. The mask
must not overlap another mask or any fluid, velocity, or magnetic return-current
annulus. Initial/driven fluid profiles must be annular or disabled. Supported
configurations are 2D UCT-HLLE/UCT-HLLD, RK1/2/3, ideal/tabulated EOS, and no
diffusion or explicit unsplit fixed/Spitzer Ohmic resistivity. Free-free cooling
is supported with the tabulated EOS. Tabular cooling, RKL2, conduction, and
viscosity are rejected when the mask is enabled.

## Evolution

The mask weight is one in the core, zero outside the outer radius, and a quintic
smootherstep in between. Core density and thermal energy follow the unperturbed
initial reservoir profiles, with zero velocity. The buffer relaxes density,
velocity, and specific internal energy with
`exp(-dt_stage*w/(tau*(1-w)))`. Each stage uses `dt_stage=dt/nstages`, with exact
reset/exterior branches. RK mixing also mixes prior stage states: the full-step
buffer response is therefore the composition of these stage relaxations and RK
averages, not an exact full-step exponential. This is a prescribed source
splitting, not an assertion of the RK method's formal order for stiff damping.
Passive scalars retain their local mass fractions when density is changed.

The complete self-consistent CT electric field is multiplied by `1-w` before
flux synchronization. The analytic vector-potential pulse is still supplied
once by the existing driver. The core therefore follows its initialized field
plus the imposed pulse, with no self-consistent induction or diffusion. No face
field is reset or independently blended. An unmasked reference EMF participates
in the same AMR/MPI flux synchronization so the difference in CT magnetic energy
can be supplied externally without being mistaken for thermal energy. Resistive
energy fluxes are tapered at their faces; EMFs receive the taper only once.
No resistivity is evaluated at fully suppressed faces/edges. Timestep exclusions
require an entire one-cell stencil to be prescribed; all other cells retain
conservative unmodified diffusion/heating limits. Hyperbolic CFL stays active.

Core fluid recovery runs before primitive recovery, after each RK stage (after
the existing driver on the last stage), and during restart/remesh initialization.
The buffer receives no extra relaxation during output, restart, or remeshing.
Legacy reservoir floor top-ups are suppressed wherever the mask has support.
The prescription persists after the magnetic pulse ends.

## Free-free cooling

With `cooling/enable_cooling = free_free`, the radiative specific-energy loss
is multiplied by `1-w` at the cell center. The fixed core returns before any
cooling EOS/rate/floor evaluation. The buffer's adaptive RK12/RK45 integrator
uses the tapered rate, and the cooling CFL uses that same tapered rate. Ordinary
cooling and its temperature floor remain active outside the mask. Core cooling
is excluded as a local ODE; it does not require the resistivity stencil exclusion.

The existing core reset and buffer fluid relaxation still run after the source
update at each fluid RK stage. In the buffer this represents a thermostat acting
alongside tapered radiation: its thermal-energy replacement is recorded in the
fluid apparatus budget. The radiative loss itself is a separate physical sink
and is not included in that apparatus budget. Hydro/EOS floors remain in effect.
No new input flag is needed beyond enabling the mask and the existing free-free
cooling model. Tabular cooling remains explicitly rejected with the mask.

The cooling extension has been prepared for debug-node validation; it has not
been compiled or run on the login node. Regenerate the run bundle with:

```bash
python scripts/prepare_reservoir_cooling_test.py
```

Its default directory is `test_runs/reservoir_cooling`, already gitignored.
The prepared Lustre bundle is
`/p/lustre5/freeman48/al-eos/masked_reservoir_cooling_test`. Its
`run_on_debug_node.sh` accepts the path to your rebuilt executable and runs the
independent cooling checks plus four small actual-EOS pilots. It neither builds
nor submits jobs. See the bundle README for launcher overrides and output paths.

Use a fresh start to obtain the clean prescribed magnetic profile. Restart
continuation from files written with the mask enabled is supported and tested.
Enabling the mask on an older unmasked restart is currently blocked on some HDF5
versions by Parthenon's missing-dataset probe for `reservoir_mask_exchange`.
That upstream issue is being addressed separately; no Parthenon changes are
included here. If that probe is fixed separately, existing core magnetic
perturbations are frozen rather than erased when enabling the mask on an old
restart. Fresh simulations and continuation of masked simulations do not need
that fix.
The original ideal-gas `T` output uses `mbar_over_kb`, whereas this problem's
initialization uses `mbar/k_b`; the existing definitions differ by the hydrogen
to atomic-mass-unit ratio 1.007947. Mask targets deliberately match the existing
initialization; this change does not alter global temperature conventions.

## Diagnostics

Add `reservoir_mask_diagnostics` and `reservoir_mask_exchange` to PHDF variables.
They are present in the example. Diagnostic components are mask weight, local
resistive diffusion timestep, local Ohmic-heating timestep, and local cooling
timestep (all timesteps in code time). Excluded or inactive constraints use the
largest finite `Real` as a sentinel. The timestep fields provide the minimum's
physical location through the file's cell coordinates. History output adds all
three global timestep minima.

The seven exchange components are cumulative signed source densities for mass,
three momenta, fluid total energy, CT magnetic-energy correction, and imposed
pulse magnetic energy. History reports their volume integrals. Positive values
mean supplied to the simulation. They use the same RK combinations as the state,
are conservatively remapped, and survive restart. The pulse budget covers the
whole domain; the fluid budget covers prescribed core/buffer changes. These are
apparatus exchanges, not claims of a closed, conservative system. Density or
energy corrections applied independently by EOS floors are not included.

## Validation and production comparison

Run the black-box checks with:

```bash
python tst/regression/test_suites/reservoir_mask/check_reservoir_mask.py \
  --driver /path/to/athenaPK
```

Optional `--baseline /path/to/original/athenaPK` checks disabled-mask compatibility
bit for bit. `--launcher 'mpiexec -n 2'` runs the same fixtures across MPI ranks.
Tests use temporary directories and require NumPy, h5py, and the existing EOS
table. The test is also registered as CTest `reservoir_mask`.

Before drawing production conclusions, run the same annulus input without the
mask and with radii 1.1/1.5, 1.2/1.6, and 1.3/1.7 cm, all at 5 ns, through a
common end time beyond the previous stall. Keep output cadence, mesh, EOS,
resistivity, and other source settings identical. Resolve the taper with at least
four cells on the coarsest mesh admitted there. Check both convergence with mesh
spacing and sensitivity to these mask radii.

The example outputs the fields needed for:

```bash
python scripts/analyze_reservoir_mask.py \
  --run default=RUN_12_16 --run smaller=RUN_11_15 --run larger=RUN_13_17 \
  --run unmasked=RUN_UNMASKED --reference default \
  --sheet-current-threshold YOUR_COMMON_THRESHOLD > comparison.json
```

The report includes timestep minima and locations, annular mass flux and radial
MHD momentum flux through a 2.3 cm shell of width 0.2 cm, upstream field, central
sheet current, and a central physical `E_z` proxy. Fluxes are in code units per
unit z length; measurements sum both arrays. Upstream sampling uses
`|x|<0.5 cm, 0.2<|y|<0.4 cm`; sheet current uses
`|x|<1 cm, |y|<0.2 cm`. The central physical electric-field proxy is not the
numerical CT EMF or a topology-based reconnection-rate measurement.

Choose and report a common physical current threshold for sheet formation.
Formation times are first sampled threshold crossings, so use output cadence
fine enough to resolve a 5% difference. The script compares final annular fluxes
and formation times with a 5% criterion and refuses a verdict when end times
or threshold crossings are missing. Inspect the full time series, including
upstream field and electric field, rather than relying solely on that verdict.
Production runs require an appropriate compute allocation; the small regression
fixtures do not establish scientific independence from the reservoir model.
