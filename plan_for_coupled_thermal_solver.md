# Plan For Coupled Thermal Solver

## Summary

This plan implements a coupled internal-energy source solver for AthenaPK that evolves
tabular radiative cooling and Spitzer ohmic heating together during each hydro stage.
The coupled solve must iterate on temperature without mutating the conserved state until
convergence, using the resistive internal-energy source

`S_IEN = -div(F_eta,E)`

computed from the divergence of the resistive energy flux rather than the local
approximation `eta J^2`. After the thermal iteration converges, the code will:

1. apply the accepted volumetric internal-energy update exactly once, and
2. apply the resistive magnetic-field update exactly once, but without any resistive
   `IEN` flux contribution.

This preserves consistency with the existing conservative resistive operator while
avoiding the unstable operator splitting between cooling and temperature-dependent
resistivity.

The first implementation is intentionally limited to:

- `cooling/enable_cooling = tabular`
- `diffusion/resistivity = ohmic`
- `diffusion/resistivity_coeff = spitzer`
- `diffusion/integrator = unsplit`

No conduction, viscosity, or RKL2 coupling is implemented in this version.

---

## Design Goals

The implementation must satisfy all of the following:

1. The solver must iterate on a predicted thermodynamic state (`eint`, `T`, `p`) while
   holding `rho`, momentum, and magnetic field fixed during the source iteration.
2. Spitzer resistivity must be recomputed from the current iterate temperature on every
   fixed-point iteration.
3. The resistive heating term must come from the divergence of the resistive energy
   flux, using the same face-flux formula currently used in the resistive diffusion
   operator.
4. The coupled thermal solve must not directly mutate the real conserved state until the
   fixed-point loop has converged.
5. The final resistive magnetic update must not double count energy by also updating
   `cons.flux(..., IEN, ...)`.

---

## Step 1: Add runtime controls for the coupled internal-energy solver

### Objective

Introduce a dedicated runtime control block for the coupled internal-energy source solve
so that the feature is activated explicitly and does not appear to belong to the cooling
package.

### Exact changes

Add a new input block:

```ini
[thermal_source_solver]
enabled = false
integrator = rk12
max_iter = 8
temp_rtol = 1e-3
e_rtol = 1e-6
under_relaxation = 1.0
couple_cooling = true
couple_ohmic_heating = true
couple_conduction = false
couple_viscous_heating = false
```

### Implementation details

In `src/hydro/hydro.cpp`, inside `Hydro::Initialize(ParameterInput *pin)`:

- Parse and store the following `Hydro` package params:
  - `thermal_source_solver_enabled`
  - `thermal_source_integrator`
  - `thermal_source_max_iter`
  - `thermal_source_temp_rtol`
  - `thermal_source_e_rtol`
  - `thermal_source_under_relaxation`
  - `thermal_couple_cooling`
  - `thermal_couple_ohmic`
  - `thermal_couple_conduction`
  - `thermal_couple_viscous`

Use names with a consistent `thermal_` prefix so they are easy to discover and do not
collide with the existing `cooling` and `diffusion` namespaces.

### Validation rules

Still in `Hydro::Initialize()`, add startup checks:

- If `thermal_source_solver_enabled == true` and `thermal_couple_cooling == true`, then
  require `cooling/enable_cooling == tabular`.
- If `thermal_source_solver_enabled == true` and `thermal_couple_ohmic == true`, then
  require:
  - `diffusion/resistivity == ohmic`
  - `diffusion/resistivity_coeff == spitzer`
  - `diffusion/integrator == unsplit`
- If `thermal_source_solver_enabled == true` and no `thermal_couple_*` flags are true,
  fail immediately.
- If `thermal_source_integrator` is not `rk12`, fail for now. The existing code has
  `rk45` and `townsend`, but the first coupled implementation should support only the
  RK12 subcycling path.

### Why this matters

This keeps the coupled solve as a general internal-energy source framework rather than
embedding it under `TabularCooling`, which would make future coupling with conduction or
other physics awkward.

---

## Step 2: Add a dedicated internal-energy solver module

### Objective

Create a new module that owns the fixed-point iteration and source coupling logic.

### Exact changes

Create:

- `src/hydro/srcterms/internal_energy_solver.hpp`
- `src/hydro/srcterms/internal_energy_solver.cpp`

Register them in `src/CMakeLists.txt` by adding:

```cmake
hydro/srcterms/internal_energy_solver.hpp
hydro/srcterms/internal_energy_solver.cpp
```

near the existing `tabular_cooling` entries.

### Public API

In `internal_energy_solver.hpp`, define:

```cpp
#ifndef HYDRO_SRCTERMS_INTERNAL_ENERGY_SOLVER_HPP_
#define HYDRO_SRCTERMS_INTERNAL_ENERGY_SOLVER_HPP_

#include <parthenon/package.hpp>

#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro {

struct InternalEnergySolverConfig {
  int max_iter;
  Real temp_rtol;
  Real e_rtol;
  Real under_relaxation;
};

TaskStatus AddCoupledInternalEnergySources(MeshData<Real> *md,
                                           const SimTime &tm,
                                           const Real dt);

} // namespace Hydro

#endif
```

### Internal helpers

Keep all orchestration helpers in the `.cpp` file in an anonymous namespace. Suggested
helpers:

- `InternalEnergySolverConfig GetInternalEnergySolverConfig(StateDescriptor *hydro_pkg);`
- `void InitializeThermalScratch(MeshData<Real> *md);`
- `void UpdateThermalIterateFromAcceptedStep(MeshData<Real> *md, Real relax);`
- `bool CheckThermalConvergence(MeshData<Real> *md, const InternalEnergySolverConfig &cfg);`
- `void CommitCoupledInternalEnergyUpdate(MeshData<Real> *md);`
- `void ApplyMagneticOnlyResistiveUpdate(MeshData<Real> *md);`

### Why this matters

This prevents the cooling package from becoming the owner of an algorithm that is
actually a general internal-energy source solve.

---

## Step 3: Register thermal scratch fields on the Hydro package

### Objective

Store predicted thermodynamic states and lagged source terms in scratch fields so the
solver can iterate without mutating the real conserved state.

### Exact changes

In `Hydro::Initialize()` in `src/hydro/hydro.cpp`, add one-copy cell-centered fields:

- `eint_init`
- `eint_iter`
- `eint_next`
- `temp_iter`
- `temp_next`
- `s_ohm_iter`
- `s_ohm_prev`
- `coupled_iter_count`
- `coupled_temp_err`
- `coupled_e_err`

Use metadata consistent with other Hydro scratch fields:

```cpp
auto m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
pkg->AddField("eint_init", m);
...
```

### Field meanings

- `eint_init`: initial specific internal energy at the start of the coupled source solve
- `eint_iter`: current iterate of specific internal energy
- `eint_next`: next iterate after integrating the cooling ODE with lagged heating
- `temp_iter`: temperature implied by `eint_iter`
- `temp_next`: temperature implied by `eint_next`
- `s_ohm_iter`: current volumetric ohmic source derived from flux divergence
- `s_ohm_prev`: previous iteration source, useful for debug or under-relaxation
- `coupled_iter_count`: number of iterations actually taken
- `coupled_temp_err`: final relative temperature error
- `coupled_e_err`: final relative specific-energy error

### Why this matters

The algorithm explicitly requires a predicted thermal state that is iterated without
updating the actual conserved variables until the end of the source solve.

---

## Step 4: Dispatch the coupled solver from the Hydro source-term hook

### Objective

Replace the direct cooling call with a conditional dispatch to the new coupled internal
energy solver.

### Exact changes

In `src/hydro/hydro.cpp`, update `TaskStatus AddUnsplitSources(...)`.

Current logic:

```cpp
if (enable_cooling == Cooling::tabular) {
  const TabularCooling &tabular_cooling =
      hydro_pkg->Param<TabularCooling>("tabular_cooling");
  tabular_cooling.SrcTerm(md, beta_dt);
}
```

Replace with:

1. read `thermal_source_solver_enabled`
2. read `thermal_couple_cooling`
3. read `thermal_couple_ohmic`
4. determine whether the coupled internal-energy solver should be used for cooling
   specifically
5. branch

Pseudo-code:

```cpp
const bool thermal_solver_enabled =
    hydro_pkg->Param<bool>("thermal_source_solver_enabled");
const bool couple_cooling = hydro_pkg->Param<bool>("thermal_couple_cooling");
const bool couple_ohmic = hydro_pkg->Param<bool>("thermal_couple_ohmic");

const bool use_coupled_cooling =
    thermal_solver_enabled && couple_cooling && enable_cooling == Cooling::tabular;

if (use_coupled_cooling) {
  AddCoupledInternalEnergySources(md, tm, beta_dt);
} else if (enable_cooling == Cooling::tabular) {
  const TabularCooling &tabular_cooling =
      hydro_pkg->Param<TabularCooling>("tabular_cooling");
  tabular_cooling.SrcTerm(md, beta_dt);
}
```

Do not use `thermal_source_solver_enabled` by itself as the dispatch condition. The
global flag only means the general coupled solver exists for this run. The actual branch
into the coupled path must also check whether cooling is one of the source terms selected
to participate in that solve.

The same pattern must be used later for any other coupled thermal contributor:

- resistivity participates only if `thermal_solver_enabled && thermal_couple_ohmic`
- conduction participates only if `thermal_solver_enabled && thermal_couple_conduction`
- any future source must have both:
  - the global coupled-solver enable
  - its own explicit `thermal_couple_*` participation flag

### Include changes

At the top of `src/hydro/hydro.cpp`, add:

```cpp
#include "srcterms/internal_energy_solver.hpp"
```

### Why this matters

This keeps the existing thermal source integration point intact while redirecting the
physics to the coupled path only when explicitly requested.

---

## Step 5: Refactor resistivity into magnetic and energy-source components

### Objective

Separate the resistive magnetic-field update from the resistive internal-energy source
construction so that the thermal iteration can use the source without modifying real
fluxes.

### Exact changes

In `src/hydro/diffusion/diffusion.hpp`, add declarations:

```cpp
void OhmicDiffusionMagneticFlux(MeshData<Real> *md);
void ComputeOhmicHeatingSourceFromFluxDivergence(MeshData<Real> *md);
```

Do not expose an ambiguous function name like `CoupledSrcTerm`. The new names must
describe exactly what they do.

### Refactor target

The current functions:

- `OhmicDiffFluxIsoFixed(MeshData<Real> *md)`
- `OhmicDiffFluxGeneral(MeshData<Real> *md)`

write both:

- magnetic fluxes in `IB1`, `IB2`, `IB3`
- resistive energy flux in `IEN`

This must be split.

### New behavior

#### `OhmicDiffusionMagneticFlux(MeshData<Real> *md)`

- computes the same face-centered currents and resistive coefficients currently used in
  the resistivity operator
- writes only:
  - `cons.flux(X?DIR, IB1, ...)`
  - `cons.flux(X?DIR, IB2, ...)`
  - `cons.flux(X?DIR, IB3, ...)`
- does not write `cons.flux(..., IEN, ...)`

#### `ComputeOhmicHeatingSourceFromFluxDivergence(MeshData<Real> *md)`

- computes the same resistive energy face flux currently written to
  `cons.flux(..., IEN, ...)`
- reduces those face fluxes to a cell-centered volumetric source `s_ohm_iter`
- does not mutate any real flux arrays
- does not update `cons(IEN)`

### Why this matters

This is the most important physics refactor in the whole plan. Without it, the coupled
solver will either:

- double count resistive energy, or
- fail to use the same conservative operator as the current resistive implementation.

---

## Step 6: Compute `S_IEN` from the divergence of the resistive energy flux

### Objective

Implement the resistive heating term exactly as the divergence of the resistive energy
flux already present in the resistivity operator.

### Exact changes

In `src/hydro/diffusion/resistivity.cpp`:

1. Reuse the current face-flux formulas for the resistive energy flux.
2. For each cell, compute:

   ```text
   S_IEN = -div(F_eta,E)
   ```

3. Store the result in the scratch field `s_ohm_iter`.

### Required sign convention

Do not guess the sign. Match Parthenon’s update convention used by
`UpdateWithFluxDivergence<MeshData<Real>>`.

The source must be equivalent to the backend conservative update:

```text
delta U = -dt * div(F)
```

so the source field must be defined consistently as:

```text
s_ohm_iter = -(div F_eta,E)
```

### Geometry handling

Use the same coordinate-aware divergence pattern as Parthenon’s flux update helpers.
Since this branch appears to assume Cartesian geometry in the diffusion operators, keep
the same assumptions in the first version and do not introduce a new divergence
definition.

### Units

Store `s_ohm_iter` as a volumetric energy source with the same units as `cons(IEN)` per
unit time.

Later, inside the cooling integrator, convert to specific internal-energy source by
dividing by `rho`.

### Why this matters

This is the exact algorithmic requirement that replaces operator splitting. Using
`eta J^2` would be simpler, but it would no longer be the same resistive energy operator
implemented by the code today.

---

## Step 7: Make Spitzer resistivity depend on the current thermal iterate

### Objective

Ensure the resistive source uses the current predicted temperature rather than the stale
primitive pressure.

### Exact changes

In the new `ComputeOhmicHeatingSourceFromFluxDivergence(MeshData<Real> *md)` path:

- keep density fixed from the current state
- compute pressure from the thermal iterate:

  ```cpp
  p_iter = rho * eint_iter * (gamma - 1.0);
  ```

- compute temperature using the same relation used elsewhere:

  ```cpp
  T_iter = mbar / k_B * p_iter / rho;
  ```

- evaluate Spitzer `eta` using the `OhmicDiffusivity::Get(p_iter, rho)` interface

### Face values

Because the current resistive flux kernels evaluate `eta` at cell faces using face
averages of pressure and density, the coupled-source path must do the same:

- face density: average neighboring cell densities
- face pressure: average neighboring iterate pressures
- face temperature is implied from those quantities via `OhmicDiffusivity::Get()`

### Why this matters

The whole point of the iteration is to resolve the competition between temperature
dependent resistive heating and radiative cooling on the same timescale. Using stale
pressure would defeat the coupling.

---

## Step 8: Extend tabular cooling with a coupled-RHS RK12 path

### Objective

Allow the cooling integrator to evolve specific internal energy under both radiative
cooling and lagged ohmic heating.

### Exact changes

In `src/hydro/srcterms/tabular_cooling.hpp`, add:

```cpp
void CoupledRK12Step(MeshData<Real> *md, const Real dt) const;
```

Do not name it `CoupledSrcTerm`, because that name becomes ambiguous once more source
physics are added.

### New behavior

In `src/hydro/srcterms/tabular_cooling.cpp`, add a new path derived from the existing
`SubcyclingFixedIntSrcTerm<RK12Stepper>()` logic:

- input:
  - `eint_iter`
  - `s_ohm_iter`
- output:
  - `eint_next`

The RHS becomes:

```text
de/dt = Q_rad(e, rho) + s_ohm_iter / rho
```

where:

- `Q_rad(e, rho)` is the existing cooling table rate returned by `CoolingTableObj::DeDt`
- `s_ohm_iter / rho` converts the volumetric resistive source into specific internal
  energy rate

### Restrictions

The first version should:

- support only the RK12 path
- not support Townsend exact cooling
- not update `cons(IEN)` directly
- not update `prim(IPR)` directly

The coupled path is only a predictor step inside the new internal-energy solver.

### Why this matters

The current cooling code directly commits changes to the real conserved variables. The
coupled solve instead needs a predictor that writes to scratch state.

---

## Step 9: Initialize thermal scratch state from the real conserved state

### Objective

Set up the fixed-point iteration from the post-hydro-flux thermodynamic state.

### Exact changes

In `internal_energy_solver.cpp`, implement `InitializeThermalScratch(MeshData<Real> *md)`
to:

1. compute cell-centered specific internal energy from `cons`
2. store it in:
   - `eint_init`
   - `eint_iter`
3. compute the corresponding temperature and store it in:
   - `temp_iter`
4. zero:
   - `s_ohm_iter`
   - `s_ohm_prev`
   - `coupled_iter_count`
   - `coupled_temp_err`
   - `coupled_e_err`

### Internal-energy computation

Use the same formula used in `tabular_cooling.cpp`:

- start from total energy `cons(IEN)`
- subtract kinetic energy
- if MHD, subtract magnetic energy
- divide by density to obtain specific internal energy

### Why this matters

The solver must iterate on the actual post-flux thermodynamic state for the current
stage, not on stale primitive variables or on a pre-stage state.

---

## Step 10: Implement the fixed-point iteration

### Objective

Carry out the coupled solve:

1. recompute lagged resistive heating from the current thermal iterate
2. integrate cooling plus heating over the full stage `dt`
3. test convergence
4. repeat until converged or `max_iter`

### Exact changes

In `AddCoupledInternalEnergySources(...)` in `internal_energy_solver.cpp`:

1. call `InitializeThermalScratch(md)`
2. read the runtime config from `Hydro`
3. loop for `iter = 0 .. max_iter-1`

Within each iteration:

1. copy `s_ohm_iter -> s_ohm_prev` for diagnostics if desired
2. call `ComputeOhmicHeatingSourceFromFluxDivergence(md)`
3. call `tabular_cooling.CoupledRK12Step(md, dt)`
4. compute:
   - relative temperature change
   - relative specific internal-energy change
5. if converged:
   - record diagnostics
   - break
6. else:
   - update `eint_iter` from `eint_next`
   - apply under-relaxation if `under_relaxation < 1.0`
   - recompute `temp_iter`

### Recommended convergence criteria

Converged if either:

- `|T_next - T_iter| / max(T_iter, tiny) < temp_rtol`

or, if `temp_rtol <= 0`, use:

- `|e_next - e_iter| / max(|e_iter|, tiny) < e_rtol`

Store both diagnostic errors even if only one controls convergence.

### Under-relaxation update

Use:

```text
eint_iter <- (1 - a) * eint_iter + a * eint_next
```

with `a = under_relaxation`.

For the first version, default `a = 1.0` and use under-relaxation only if the iteration
proves noisy.

### Failure mode

If the solver does not converge by `max_iter`:

- keep the last iterate
- record the final diagnostic errors
- emit a warning on rank 0 in debug mode or once-per-run mode
- do not hard-fail unless you are intentionally running in a stricter debug build

### Why this matters

This loop is the actual implementation of the coupled algorithm. Without it, the feature
is still operator splitting.

---

## Step 11: Commit the accepted internal-energy update exactly once

### Objective

Convert the converged thermal iterate into a real conservative energy update after the
iteration finishes.

### Exact changes

In `internal_energy_solver.cpp`, implement `CommitCoupledInternalEnergyUpdate(md)`:

1. compute:

   ```text
   delta_eint = eint_iter - eint_init
   ```

2. update total energy:

   ```text
   cons(IEN) += rho * delta_eint
   ```

3. update `prim(IPR)` consistently for safety:

   ```text
   prim(IPR) = rho * eint_iter * (gamma - 1.0)
   ```

Do not change:

- density
- momentum
- magnetic field

### Why this matters

The proposed algorithm explicitly defers the actual conservative internal-energy update
until the coupled source solve has converged.

---

## Step 12: Apply the magnetic-only resistive update exactly once

### Objective

Update the magnetic field with the accepted resistive operator after the coupled thermal
solve has converged, but do not reapply any resistive internal-energy contribution.

### Exact changes

In `internal_energy_solver.cpp`, after the call to `CommitCoupledInternalEnergyUpdate(md)`:

1. call `ResetFluxes(md)` if necessary for the dedicated magnetic-only resistive update
   path, or ensure the flux state is in a known good state
2. call `OhmicDiffusionMagneticFlux(md)`
3. call the same Parthenon flux-divergence update path used elsewhere to apply the
   magnetic update

### Important constraint

This final update must not touch `cons.flux(..., IEN, ...)`.

The coupled solver has already incorporated resistive heating volumetrically.

### Why this matters

This prevents double counting while still evolving the magnetic field consistently with
the temperature-dependent resistivity implied by the converged thermal state.

---

## Step 13: Keep the legacy diffusion path intact for non-coupled runs

### Objective

Preserve current behavior when the coupled thermal solver is disabled.

### Exact changes

In `src/hydro/diffusion/diffusion.cpp`:

- keep `CalcDiffFluxes(...)` behavior unchanged for all legacy cases
- when the coupled thermal solver is enabled, avoid using the standard resistive path
  that writes `IEN` fluxes

There are two acceptable implementations:

#### Option A: Branch in `CalcDiffFluxes(...)`

- if `thermal_source_solver_enabled && thermal_couple_ohmic`:
  - call `OhmicDiffusionMagneticFlux(md)`
- else:
  - call the old full resistive operator

#### Option B: Bypass resistivity in `CalcDiffFluxes(...)` for coupled runs

- leave `CalcDiffFluxes(...)` alone for non-resistive physics
- handle resistive magnetic evolution entirely from `internal_energy_solver.cpp`

### Recommended choice

Use Option A. It is easier to keep the resistive evolution in one conceptual location and
avoids duplicating the decision logic between Hydro and diffusion code.

### Important dispatch rule

Do not branch on `thermal_source_solver_enabled` alone in the resistivity code either.
The resistive operator should switch to magnetic-only behavior only when ohmic heating is
explicitly configured to participate in the coupled internal-energy solve. Otherwise,
resistivity should remain on its legacy path even if the global coupled solver exists for
some other source term.

### Why this matters

The new feature must not regress existing cooling-only or resistivity-only runs.

---

## Step 14: Keep timestep estimation conservative in the first implementation

### Objective

Preserve robust timestep limiting without inventing a new coupled-source limiter before it
is needed.

### Exact changes

In `src/hydro/hydro.cpp`, keep the existing timestep estimates:

- cooling timestep from `TabularCooling::EstimateTimeStep(md)`
- resistive diffusion timestep from `EstimateResistivityTimestep(md)`
- ohmic heating timestep from `EstimateOhmicHeatingTimestep(md)`

Do not add a new coupled timestep estimator in the first implementation.

### Assumed behavior

For coupled runs, the global timestep is still limited by the minimum of:

- hyperbolic
- diffusion
- cooling
- ohmic heating
- problem-specific

This is conservative and acceptable for the first version.

### Why this matters

This avoids unnecessary risk while the coupled thermal algorithm is being validated.

---

## Step 15: Add diagnostics for debugging and physics validation

### Objective

Expose enough state to confirm that the coupled thermal iteration is converging and that
the resistive source has the expected sign and magnitude.

### Exact changes

In `pulsed_reconnection` or a more general Hydro diagnostic path, add output access for:

- `s_ohm_iter`
- `eint_iter`
- `temp_iter`
- `coupled_iter_count`
- `coupled_temp_err`
- `coupled_e_err`

If adding generic Hydro output fields is too invasive, at minimum expose:

- `s_ohm_iter`
- `coupled_iter_count`
- `coupled_temp_err`

through the existing problem-specific diagnostics in
`src/pgen/pulsed_reconnection.cpp`.

### Physics checks enabled by these diagnostics

- whether the coupled iteration converges quickly
- whether cooling dominates in cold zones
- whether ohmic heating dominates near strong current sheets
- whether `s_ohm_iter` matches the old resistive `IEN` flux divergence in frozen-state
  tests

### Why this matters

Without diagnostics, it will be difficult to distinguish a real physical balance from an
algorithmic failure or silent sign error.

---

## Step 16: Test and validate the implementation in a strict order

### Objective

Verify the feature incrementally so that build, regression, and physics errors can be
isolated.

### Test sequence

#### 1. Build-only validation

- confirm the new files are registered in `src/CMakeLists.txt`
- confirm all new headers are included correctly
- confirm no duplicate symbol or missing declaration errors

#### 2. Legacy regression checks

With `thermal_source_solver/enabled = false`:

- cooling-only run should match prior behavior
- resistivity-only run should match prior behavior
- pulsed reconnection without coupling should match prior branch behavior

#### 3. Frozen-state resistive source comparison

Create a diagnostic test where:

- `cons` and `prim` are fixed
- resistive `IEN` flux divergence is computed using the old operator
- `s_ohm_iter` is computed using the new helper

Confirm the two match to floating-point tolerance.

#### 4. Small coupled smoke test

Use a small mesh with:

- tabular cooling on
- Spitzer resistivity on
- coupled solver enabled

Confirm:

- iteration count is finite
- `cons(IEN)` changes once per stage
- magnetic field evolves
- no NaNs appear

#### 5. Convergence behavior test

Vary:

- `max_iter`
- `temp_rtol`
- `under_relaxation`

Confirm the solution is stable and that the iteration converges monotonically or at least
reliably.

#### 6. Full pulsed reconnection validation

Run the target `pulsed_reconnection.in` case and confirm:

- the simulation remains stable in strongly cooled / strongly heated regions
- the timestep is limited conservatively
- the solver converges in a reasonable number of iterations
- results differ from operator splitting in the expected direction

### Acceptance criteria

The implementation is acceptable when all of the following are true:

- non-coupled behavior is unchanged when the feature is off
- `s_ohm_iter` matches the old resistive energy-flux divergence operator
- the coupled solver converges on target runs
- there is no double counting of resistive internal-energy updates
- the final magnetic update uses magnetic-only resistive fluxes

---

## File-by-file edit map

### `src/CMakeLists.txt`

- register `internal_energy_solver.hpp`
- register `internal_energy_solver.cpp`

### `src/hydro/hydro.cpp`

- parse `thermal_source_solver` inputs
- add Hydro package params for coupled thermal solver
- add startup validation
- register scratch fields
- dispatch `AddUnsplitSources()` to the new solver when enabled

### `src/hydro/srcterms/internal_energy_solver.hpp`

- declare config struct
- declare `AddCoupledInternalEnergySources(...)`

### `src/hydro/srcterms/internal_energy_solver.cpp`

- implement scratch initialization
- implement fixed-point iteration
- call resistive source builder
- call coupled cooling RK12 step
- implement convergence logic
- commit accepted `cons(IEN)` update
- apply final magnetic-only resistive update

### `src/hydro/diffusion/diffusion.hpp`

- declare `OhmicDiffusionMagneticFlux(...)`
- declare `ComputeOhmicHeatingSourceFromFluxDivergence(...)`

### `src/hydro/diffusion/diffusion.cpp`

- branch resistive behavior for coupled vs legacy runs

### `src/hydro/diffusion/resistivity.cpp`

- refactor resistive operator into:
  - magnetic-only flux path
  - resistive energy-source-from-flux-divergence path
- use iterate pressure/temperature for Spitzer `eta`

### `src/hydro/srcterms/tabular_cooling.hpp`

- declare `CoupledRK12Step(...)`

### `src/hydro/srcterms/tabular_cooling.cpp`

- implement coupled RK12 predictor using:
  - `Q_rad(e, rho)`
  - `s_ohm_iter / rho`
- write to scratch state only

### `src/pgen/pulsed_reconnection.cpp`

- optionally expose new diagnostics for output

---

## Assumptions and defaults

- The first implementation is limited to tabular cooling plus Spitzer resistive heating.
- The resistive heating term is defined by the divergence of the resistive energy flux,
  not by `eta J^2`.
- The fixed-point iteration uses full stage `dt` each iteration.
- The first coupled implementation supports only the RK12 subcycling path.
- `diffusion/integrator = unsplit` is required.
- Under-relaxation defaults to `1.0`.
- If the iteration fails to converge by `max_iter`, the last iterate is accepted and a
  warning is preferred over a hard failure.

---

## Final implementation checklist

- [ ] Add `thermal_source_solver` runtime controls and Hydro package params
- [ ] Add startup validation for supported coupled physics combinations
- [ ] Create and register `internal_energy_solver` module
- [ ] Add Hydro scratch fields for thermal iteration
- [ ] Dispatch `AddUnsplitSources()` to the new solver when enabled
- [ ] Split resistivity into magnetic-only and energy-source components
- [ ] Compute `s_ohm_iter = -div(F_eta,E)` from resistive energy fluxes
- [ ] Use iterate pressure/temperature for Spitzer resistivity evaluation
- [ ] Add coupled RK12 cooling predictor that reads/writes scratch state
- [ ] Implement fixed-point iteration on `eint_iter` and `temp_iter`
- [ ] Commit `cons(IEN)` update exactly once after convergence
- [ ] Apply magnetic-only resistive update exactly once after convergence
- [ ] Preserve legacy paths when the coupled solver is disabled
- [ ] Add diagnostics for iteration and source validation
- [ ] Run build, regression, frozen-state, smoke, and target-problem validation tests


## Prompts for codex:

1. Scaffolding and activation
  Ask for:

  - add thermal_source_solver input parsing
  - add Hydro package params
  - add startup validation
  - add Hydro scratch fields
  - add the new internal_energy_solver.hpp/.cpp
  - register the new files in src/CMakeLists.txt
  - wire AddUnsplitSources() to dispatch based on the global solver flag plus per-physics participation flags

  This chunk should not change resistive physics yet. It just creates the framework.

  2. Resistivity refactor
  Ask for:

  - split resistivity into:
      - legacy full flux path
      - magnetic-only flux path
      - S_IEN = -div(F_eta,E) source-construction path

  - make the source-construction path use iterate pressure/temperature for Spitzer eta
  - preserve non-coupled behavior exactly

  This is the riskiest physics refactor. It should be done and reviewed in isolation before touching cooling.

  3. Coupled cooling predictor
  Ask for:

  - extend tabular cooling with a scratch-state RK12 predictor that evolves
    de/dt = Q_rad(e, rho) + s_ohm_iter / rho

  - keep the existing cooling path intact
  - do not update cons(IEN) in the new predictor path

  This isolates the cooling-side API and makes it testable before the fixed-point loop exists.

  4. Fixed-point solver and final commit path
  Ask for:

  - implement the full fixed-point iteration in internal_energy_solver.cpp
  - initialize scratch state from cons
  - iterate s_ohm_iter -> coupled cooling predictor -> convergence test
  - commit cons(IEN) once after convergence
  - apply the magnetic-only resistive update once after convergence
  - add iteration diagnostics

  This is where the actual coupled algorithm comes together.

  5. Validation and polish
  Ask for:

  - add or expose diagnostics in pulsed_reconnection
  - run/build targeted tests
  - verify frozen-state agreement between old resistive IEN flux divergence and new s_ohm_iter
  - verify non-coupled regressions
  - clean up any API naming or comments that are misleading

  That final pass is where you catch sign mistakes, double counting, or bad assumptions.
  If you want the exact prompts to give me for each chunk, use these:

  1. “Implement the scaffolding from plan_for_coupled_thermal_solver.md: add the thermal_source_solver runtime controls, Hydro package params, startup validation, scratch fields,
     new internal_energy_solver files, CMake registration, and the AddUnsplitSources() dispatch. Do not change resistive or cooling physics yet.”

  2. “Implement the resistivity refactor from plan_for_coupled_thermal_solver.md: split the resistive operator into legacy full behavior, magnetic-only flux updates, and S_IEN =
     -div(F_eta,E) source construction using iterate thermodynamics. Preserve legacy behavior when coupling is off.”

  3. “Implement the coupled cooling predictor from plan_for_coupled_thermal_solver.md: add the RK12 scratch-state cooling/heating predictor that reads eint_iter and s_ohm_iter, writes eint_next, and does not directly modify cons(IEN).”

  4. “Implement the fixed-point coupled internal-energy solver from plan_for_coupled_thermal_solver.md: initialize scratch state, iterate resistive source plus cooling predictor to convergence, commit the accepted internal-energy update once, and apply the final magnetic-only resistive update once.”

  5. “Finish validation and diagnostics from plan_for_coupled_thermal_solver.md: expose coupled-solver diagnostics, run the relevant checks, verify the new s_ohm_iter matches the
     old resistive IEN flux divergence in frozen-state comparisons, and confirm legacy behavior is unchanged when coupling is disabled.”


# Now those changes have been made: Validate.

1. Separate-source validation

  - You still need a clean test where the legacy resistive IEN update can be isolated from hyperbolic transport.
  - The hot-background comparison showed that using full-step Δcons(IEN) is not a valid check when advection/compression is active.
  - You want a frozen-state or diffusion-only test proving that ComputeOhmicHeatingSourceFromFluxDivergence() reproduces the intended resistive energy-flux divergence in the collapse-to-legacy limit.

2. Coupled fixed-point behavior in realistic regimes

  - Run short but physically relevant cases where cooling and ohmic heating truly compete on similar timescales.
  - Check iteration counts, coupled_temp_err, and coupled_e_err fields.
  - Make sure convergence is robust, not just “hits max_iter and moves on.”

3. Sensitivity studies

  - Vary thermal_source_solver/max_iter, temp_rtol, e_rtol, and under_relaxation.
  - Confirm the solution is stable with respect to those controls.
  - If the answer moves materially when those knobs change, the solver is not ready.

4. Timestep and stiffness stress tests

  - Probe regimes with:
      - strong cooling
      - strong ohmic heating
      - cells near the cooling cutoff
      - large Spitzer resistivity contrasts

  - Make sure the downward-only cooling floor logic behaves correctly and does not create hidden energy injection.

5. Conservation/accounting checks

  - Verify there is no double counting of ohmic heating over a full update.
  - Verify the final magnetic update is applied exactly once.
  - Verify the internal-energy commit is applied exactly once.

6. Legacy regression coverage

  - Confirm coupling disabled gives unchanged answers on existing resistive and cooling problems.
  - That should include more than the one-step spot checks.

7. Practical diagnostics for production

  - Right now the diagnostics are useful, but before long runs I would want a small routine summary per cycle or per output:
      - max coupled iterations used
      - fraction of cells near tolerance failure
      - extrema of s_ohm_iter