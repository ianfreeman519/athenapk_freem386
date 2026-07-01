# Plan For Thermal Simplified-SDC In AthenaPK

## Current Status After Picard Cleanup

This branch now treats the stage-local thermal corrector as the accepted implementation:

- one hydro build per explicit stage
- one stage-local pre-flux predictor fed by `thermal_src_lagged`
- a fixed-count thermal corrector after the stage flux divergence update
- no repeated hydro-plus-thermal SDC sweep around the full stage

Validation status carried forward explicitly:

- frozen-state ohmic-source comparison against the legacy resistive energy update has
  passed
- the pre-flux validation still shows a structural mismatch: `thermal_src_lagged` is
  initialized and pre-flux diagnostic snapshots exist, but `eint_adv` / `thermal_ae`
  still do not demonstrate that the predictor-applied source is consumed consistently by
  the single-stage hydro build

Technical debt carried forward explicitly:

- there is still no `cons_stage_start` snapshot
- `thermal_ae` remains the scalar approximation
  `A_e = (eint_adv - eint_stage_start) / dt_stage`
- `thermal_src_pre_flux` and `thermal_src_ohmic_pre_flux` are intentionally retained as
  diagnostics for the unresolved pre-flux mismatch
- if the mismatch is pursued next, the smallest extension is still a dedicated
  post-`ApplyPreFluxThermalSource()` snapshot such as `eint_after_pre_flux` or equivalent
  conserved-state bookkeeping

Current runtime interface after cleanup:

- `thermal_source_solver/enabled`
- `thermal_source_solver/iterations`
- `thermal_source_solver/source_predictor = end_of_interface_state`
- `thermal_source_solver/store_diagnostics`
- `thermal_source_solver/couple_cooling`
- `thermal_source_solver/couple_ohmic_heating`
- `thermal_source_solver/couple_conduction`
- `thermal_source_solver/couple_viscous_heating`

This document is a handoff for replacing the current coupled thermal fixed-point solve
with a Zingale-style simplified-SDC thermal coupling. It is written against the current
state of this branch.

Important architectural correction:

- AthenaPK applies unsplit sources after the stage flux divergence update, not before
  reconstruction / Riemann solves.
- The current coupled thermal path therefore does not influence the hydro build for the
  stage.
- A viable simplified-SDC implementation in this codebase must be designed around the
  explicit stage structure in the hydro driver, not as a post-hydro replacement local to
  `AddCoupledInternalEnergySources()`.

The end state should satisfy all of the following:

- no under-relaxed Picard iteration remains in the codebase
- no convergence logic based on `eint_iter -> eint_next` fixed-point residuals remains
- the coupled thermal update is driven by a lagged thermal source inside each explicit
  hydro stage
- ohmic heating enters as a cell-centered thermal source derived from the resistive
  energy-flux divergence
- cooling remains a volumetric stiff ODE update
- legacy behavior remains unchanged when the new thermal simplified-SDC mode is disabled

This plan uses the algorithmic pattern from `reference/Zingale_2022_ApJ_936_6.pdf`,
adapted to AthenaPK’s finite-volume MHD layout and to the narrower problem of thermal
source coupling rather than full hydro plus reactions.

## 1. What Was Implemented In The First Pass

The current branch already contains useful infrastructure. Some of it should be kept and
repurposed. Some of it exists only to support the Picard scheme and must be removed.

### 1.1 Runtime controls and Hydro package parameters

Implemented in [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp):

- `thermal_source_solver/enabled`
- `thermal_source_solver/integrator`
- `thermal_source_solver/max_iter`
- `thermal_source_solver/temp_rtol`
- `thermal_source_solver/e_rtol`
- `thermal_source_solver/under_relaxation`
- `thermal_source_solver/couple_cooling`
- `thermal_source_solver/couple_ohmic_heating`
- `thermal_source_solver/couple_conduction`
- `thermal_source_solver/couple_viscous_heating`

Hydro package params added:

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

What this does now:

- turns on the current coupled thermal solver path
- validates that coupling is only used with tabular cooling and unsplit Spitzer ohmic
  diffusion

What should survive:

- the existence of a dedicated `thermal_source_solver` input block
- the per-physics coupling booleans

What should be removed or changed:

- `under_relaxation`
- `temp_rtol`
- `e_rtol`
- any validation or runtime assumptions specific to Picard convergence

Suggested replacement controls for simplified-SDC:

- `thermal_source_solver/enabled`
- `thermal_source_solver/method = simplified_sdc`
- `thermal_source_solver/iterations = 2`
- `thermal_source_solver/couple_cooling = true`
- `thermal_source_solver/couple_ohmic_heating = true`
- optional later:
  - `thermal_source_solver/source_predictor = end_of_interface_state`
  - `thermal_source_solver/store_diagnostics = true`

### 1.2 Scratch fields added to the Hydro package

Implemented in [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp).

Fields added:

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

What these do now:

- hold the thermodynamic state during the Picard iteration
- store the lagged ohmic source between iterations
- expose residual diagnostics for the fixed-point iteration

What should survive:

- `s_ohm_iter` or a renamed equivalent for the lagged ohmic thermal source
- possibly a reduced set of thermal scratch state fields if the new SDC integration needs
  them
- diagnostics fields, but only if repurposed for SDC iteration diagnostics

What should be removed:

- `eint_iter`
- `eint_next`
- `temp_iter`
- `temp_next`
- `s_ohm_prev`
- `coupled_temp_err`
- `coupled_e_err`

Fields likely needed for thermal simplified-SDC instead:

- `thermal_src_lagged`
- `thermal_src_ohmic`
- `thermal_src_total`
- `eint_sdc_old`
- `eint_sdc_new`
- `thermal_sdc_iter_count`

Keep names flexible, but do not carry over Picard-specific names if the data semantics
change.

### 1.3 New internal-energy solver module

Implemented in:

- [src/hydro/srcterms/internal_energy_solver.hpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.hpp)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp)

What it does now:

- owns the fixed-point outer loop
- initializes thermal scratch state
- repeatedly rebuilds `s_ohm_iter`
- repeatedly calls the coupled RK12 predictor
- under-relaxes the iterate
- commits `cons(IEN)` once
- applies the magnetic-only resistive update once

What should survive:

- this module can remain the owner of the thermal coupling orchestration
- helper routines that initialize thermodynamic scratch state can be repurposed
- the final “apply magnetic-only resistive update” helper is still useful

What must be removed:

- `InternalEnergySolverConfig::temp_rtol`
- `InternalEnergySolverConfig::e_rtol`
- `InternalEnergySolverConfig::under_relaxation`
- `CopyCurrentOhmicSource()`
- `ComputeConvergenceStats()`
- `CheckThermalConvergence()`
- `AcceptNextIterate()`
- any outer loop of the form “repeat until converged or max_iter”

The replacement module should orchestrate a fixed number of simplified-SDC iterations,
normally `2`, not convergence-driven Picard iterations.

### 1.4 Resistivity refactor

Implemented in [src/hydro/diffusion/resistivity.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/resistivity.cpp).

Useful pieces already present:

- legacy full resistive behavior is preserved when coupling is off
- magnetic-only resistive update path exists
- `ComputeOhmicHeatingSourceFromFluxDivergence()` computes the cell-centered thermal
  source from the resistive energy-flux divergence
- the source uses `eint_iter` to evaluate Spitzer resistivity during the thermal loop

What should survive:

- the split between:
  - legacy full resistive update
  - magnetic-only resistive update
  - cell-centered ohmic thermal source construction
- the flux-divergence-based ohmic source construction

What must change:

- `ComputeOhmicHeatingSourceFromFluxDivergence()` should stop reading Picard-specific
  fields like `eint_iter`
- it should instead read the thermodynamic state used by the current SDC iteration
- if possible, rename the entry point to something more general, for example:
  - `BuildOhmicThermalSourceFromFluxDivergence()`

The important invariant is:

- this source represents the thermal source seen by the SDC thermodynamic update
- it is not a direct update to `cons(IEN)`

### 1.5 Cooling predictor extension

Implemented in [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp) and [src/hydro/srcterms/tabular_cooling.hpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.hpp).

Useful piece:

- `TabularCooling::CoupledRK12Step()` already integrates a cell-local ODE of the form
  `de/dt = DeDt_cool(e, rho) + heating_rate`

What should survive:

- the ability to integrate cooling plus an externally provided thermal source together
- the rule that the cooling floor only limits cooling downward, not heating

What must change:

- the current routine is hard-wired to Picard scratch fields:
  - reads `eint_iter`
  - reads `s_ohm_iter`
  - writes `eint_next`
  - writes `temp_next`
- it should be refactored into a more general routine that accepts:
  - an input thermodynamic state
  - a lagged advection-plus-ohmic thermal source
  - an output thermodynamic state

Recommended end-state API:

```cpp
void TabularCooling::IntegrateThermalODEWithSource(MeshData<Real> *md,
                                                   const std::string &eint_in,
                                                   const std::string &thermal_src,
                                                   const std::string &eint_out,
                                                   const Real dt) const;
```

or a tighter typed helper if string-based field naming is considered too loose.

### 1.6 AddUnsplitSources dispatch

Implemented in [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp).

What it does now:

- dispatches to `AddCoupledInternalEnergySources()` when the thermal solver is enabled

What should survive:

- the top-level dispatch point

What must change:

- the dispatched algorithm must become simplified-SDC orchestration instead of Picard

### 1.7 New test problem generator and test inputs

Implemented in:

- [src/pgen/current_sheet_thermal.cpp](/home/ianfr/athenapk_freem386/src/pgen/current_sheet_thermal.cpp)
- [src/pgen/pgen.hpp](/home/ianfr/athenapk_freem386/src/pgen/pgen.hpp)
- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp)
- [src/pgen/CMakeLists.txt](/home/ianfr/athenapk_freem386/src/pgen/CMakeLists.txt)

Validation inputs added:

- [test/coupled_thermal_validation/current_sheet_thermal_frozen.in](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/current_sheet_thermal_frozen.in)
- [test/coupled_thermal_validation/current_sheet_thermal_diffusing.in](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/current_sheet_thermal_diffusing.in)
- [test/coupled_thermal_validation/current_sheet_thermal_competing_frozen.in](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/current_sheet_thermal_competing_frozen.in)
- [test/coupled_thermal_validation/current_sheet_thermal_competing_diffusing.in](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/current_sheet_thermal_competing_diffusing.in)
- [test/coupled_thermal_validation/summarize_coupled_outputs.py](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/summarize_coupled_outputs.py)

What should survive:

- the `current_sheet_thermal` problem generator
- the validation inputs
- the summary script, after adapting it to new diagnostics

What should change:

- diagnostics currently named around Picard residuals should be replaced with SDC
  diagnostics
- the summary script should stop expecting `coupled_temp_err` and `coupled_e_err`

## 2. Minimum Viable Thermal Simplified-SDC In AthenaPK

This is the minimum viable algorithmic target. It is not the full Castro reacting
hydrodynamics machinery. It is the thermal-source analogue appropriate for AthenaPK.

### 2.1 Governing split to target

For the thermal update, define the specific internal energy equation as:

`de/dt = A_e + Q_rad(e, rho) + Q_ohm(e, rho, B)`

where:

- `A_e` is the hydro-provided advective/compressive internal-energy source seen over the
  full hydro step
- `Q_rad` is tabular cooling
- `Q_ohm` is the ohmic thermal source from `-div(F_eta,E))/rho`

The key simplified-SDC idea is:

- do not iterate the thermal source after a fully completed hydro step as an external
  correction
- instead, build a lagged thermal source from the previous SDC iterate
- include that source in the stage-centered hydro/source update
- then perform the stiff thermal ODE integration over the same stage using that lagged
  source
- reconstruct a new lagged thermal source for the next iterate

### 2.2 Stage-scoped implementation target

AthenaPK is organized around explicit integrator stages with stage-local `beta_dt`
weights. The simplified-SDC implementation must therefore define its scope explicitly.

The recommended minimum-viable target is:

- simplified-SDC correction local to each explicit hydro stage
- not an outer SDC loop wrapped around the entire multi-stage timestep

Reasons:

- the driver already computes fluxes, applies flux divergence, and then applies unsplit
  sources once per stage
- `AddUnsplitSources()` is too late to affect interface reconstruction or the Riemann
  solve
- a full-timestep outer SDC would require substantially deeper surgery to the task graph
  and stage storage

Any future document or implementation should use stage-accurate language:

- “current stage”
- “stage start state”
- “stage lagged source”
- “stage thermal ODE solve”

Do not describe the algorithm only as “per hydro step” unless the exact mapping from
stages to timestep-level SDC iterations is specified.

### 2.3 Minimum viable iteration count

Use a fixed iteration count:

- `kmax = 2` by default

This matches the role of simplified-SDC in the Zingale paper:

- second-order coupling from a fixed small number of correction passes
- no convergence-based Picard loop

Do not retain:

- “iterate until residual < tolerance”
- “under-relax accepted iterate”

### 2.4 Thermal simplified-SDC data model

At minimum, each hydro stage needs:

- stage-start state
- lagged thermal source from iteration `k-1`
- advective/compressive internal-energy contribution over the current stage
- thermally updated internal energy after integrating `A_e + Q_rad + Q_ohm`
- reconstructed lagged thermal source for the next iteration

Recommended scratch quantities:

- `eint_stage_start`
- `eint_adv`
- `eint_sdc`
- `thermal_src_lagged`
- `thermal_src_ohmic`
- `thermal_src_total`
- `thermal_sdc_iter_count`
- `cons_stage_start` or an equivalent way to reconstruct the stage-local advective change

If you need additional state for debugging:

- `thermal_src_rad`
- `thermal_src_reconstructed`

### 2.5 Minimum viable algorithm

For each explicit hydro stage:

1. Start from the accepted stage-start state.
2. Initialize the lagged thermal source.
   - For `k = 1`, use the source from the previous accepted stage or previous timestep if
     that bookkeeping is available and consistent.
   - If no previous source exists, initialize from the stage-start state:
     - build `Q_ohm^n` from the old thermodynamic state
     - optionally compute an old-state cooling source estimate
3. Build the hydro stage using the lagged thermal source in a pre-flux source hook.
   - This is the central structural change relative to the current branch.
   - The hydro predictor/interface-state construction must “see” the lagged thermal
     source before fluxes are computed.
   - `AddUnsplitSources()` by itself is not sufficient for this role.
4. From that stage update, construct the advective/compressive internal-energy source
   `A_e^(k+1/2)`.
   - You need a cell-centered representation of the hydro contribution to internal
     energy over the current stage.
   - This is the thermal analogue of the `A(U)` term in Zingale.
5. Integrate the thermal ODE over the stage time interval:
   - `de/dt = A_e^(k+1/2) + Q_rad(e, rho) + Q_ohm_lagged`
   - or, if you choose to include ohmic heating fully in the stiff solve, update
     `Q_ohm` using the current thermodynamic state during the ODE solve only if that can
     be done consistently without rebuilding the resistive flux source mid-stage
   - minimum viable version should keep `Q_ohm` lagged over a single SDC iteration
6. Reconstruct the thermal source for the next SDC iteration from the new thermodynamic
   state.
   - rebuild `Q_ohm` from the updated thermodynamic state
   - reconstruct the net thermal source as:
     - `thermal_src_reconstructed = (eint_sdc - eint_adv) / dt_stage`
     - or separately track `Q_rad + Q_ohm` if that is easier to expose cleanly
7. If `k < kmax`, repeat the stage-local hydro build and thermal integration using the
   newly lagged source.
8. After the last SDC iteration for the stage:
   - commit the accepted internal-energy update once for that stage
   - apply the magnetic-only resistive update once for that stage if ohmic coupling is
     enabled
   - do not apply a separate resistive `IEN` flux update anywhere else

### 2.6 Required structural change in AthenaPK

This is the main implementation hurdle.

The current branch performs:

- calculate fluxes for the stage
- apply flux divergence for the stage
- then apply unsplit source terms for the stage

The minimum viable thermal simplified-SDC requires:

- thermal lagged source available before the hydro predictor/update
- hydro stage construction using that lagged source
- thermal ODE solve over the same stage using the resulting advective contribution

That means the next implementation must add or identify a correct pre-flux insertion
point in AthenaPK for a stage-centered thermal source. The likely locations to inspect
are:

- the hydro driver task graph before flux calculation
- any primitive or interface-state predictor hooks
- the unsplit hydro driver task graph
- any source-term application path that feeds primitive prediction or conservative
  updates before the Riemann solve

The next Codex run should assume the following:

- `AddUnsplitSources()` alone is not sufficient for simplified-SDC coupling
- a new pre-flux hook or source-aware predictor path is required
- the implementation is not a local rewrite of `internal_energy_solver.cpp`

### 2.7 Defining `A_e` concretely

The plan must not treat `A_e` as an abstract quantity. AthenaPK evolves total energy,
not internal energy directly, so the implementation needs an explicit operational
definition.

Minimum viable definition:

- save the stage-start thermodynamic state before any stage-local coupled thermal update
- build the hydro stage using the lagged thermal source
- construct an advective/compressive internal-energy state `eint_adv` from the stage
  result before the stiff thermal ODE commit
- define:
  - `A_e = (eint_adv - eint_stage_start) / dt_stage`

The implementation should document exactly how `eint_adv` is obtained from the
conservative update so that the energy bookkeeping is auditable and reproducible.

This extraction should be validated first in tests without cooling and without
resistivity, using simple compressive / expansive flows.

### 2.8 Ohmic source construction requirements

The existing ohmic source builder is close to what is needed, but its current interface
is too Picard-specific.

Required changes:

- stop hard-wiring reads from `eint_iter`
- stop hard-wiring writes to `s_ohm_iter`
- accept the thermodynamic input state and output source field explicitly
- document what is lagged during one SDC iteration:
  - only `eta(T)` with `J` rebuilt from the magnetic state
  - or both `eta(T)` and the magnetic/current state

Minimum viable recommendation:

- lag temperature dependence through `eta(T)`
- rebuild the cell-centered ohmic source from the current magnetic field state available
  at the stage
- keep the source cell-centered and derived from the resistive energy-flux divergence

### 2.9 Recommended implementation phases

#### Phase A: Thermal ODE and source refactor

- refactor the thermal ODE routine to accept generic input and output fields
- refactor the ohmic source builder to accept generic thermodynamic/source fields
- preserve legacy and magnetic-only resistive paths
- keep the existing Picard path functional during this phase

#### Phase B: Stage-local advective source extraction

- add a way to build a cell-centered stage-local internal-energy source `A_e`
- verify this source in simple noncooling, nonresistive compression/expansion tests

#### Phase C: Pre-flux lagged-source hydro integration

- add a pre-flux source hook or source-aware predictor path
- ensure the lagged thermal source is seen by reconstruction / predictor logic
- run two SDC passes per stage

#### Phase D: SDC validation alongside legacy Picard

- validate against frozen-current and evolving-current thermal problems
- compare energy bookkeeping between legacy and new paths in regimes where both should
  agree

#### Phase E: Final cleanup

- delete all Picard-only code
- rename diagnostics and docs
- rerun validation

## 3. Concrete Removal Instructions For The First-Pass Picard Scheme

This section is intentionally deferred. Do not start here.

Picard-specific code should only be removed after:

- the new stage-level simplified-SDC path exists
- the pre-flux coupling hook is implemented
- the `A_e` extraction has been validated
- the new path has passed the thermal validation problems

Until then, the current Picard path remains a reference implementation and fallback.

Once those conditions are satisfied, the goal is that the branch no longer contains dead
or misleading Picard-specific code.

### 3.1 Remove Picard runtime controls

In [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp):

- delete parsing of:
  - `thermal_source_solver/temp_rtol`
  - `thermal_source_solver/e_rtol`
  - `thermal_source_solver/under_relaxation`
- delete package params:
  - `thermal_source_temp_rtol`
  - `thermal_source_e_rtol`
  - `thermal_source_under_relaxation`
- replace `thermal_source_solver/integrator` with either:
  - `thermal_source_solver/method`
  - or leave `integrator` only if it now means the thermal ODE method, not the outer
    coupling algorithm

### 3.2 Remove Picard scratch fields

In [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp):

- delete:
  - `eint_iter`
  - `eint_next`
  - `temp_iter`
  - `temp_next`
  - `s_ohm_prev`
  - `coupled_temp_err`
  - `coupled_e_err`

If the new SDC implementation introduces fields with similar roles, give them SDC-based
names so future readers do not mistake them for fixed-point iterates.

### 3.3 Remove Picard config type members

In [src/hydro/srcterms/internal_energy_solver.hpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.hpp):

- remove from `InternalEnergySolverConfig`:
  - `temp_rtol`
  - `e_rtol`
  - `under_relaxation`

Replace with something like:

- `iterations`
- `store_diagnostics`

### 3.4 Remove Picard helper functions

In [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp), delete:

- `CopyCurrentOhmicSource()`
- `ComputeConvergenceStats()`
- `CheckThermalConvergence()`
- `AcceptNextIterate()`

Then rewrite `AddCoupledInternalEnergySources()` so it is no longer a convergence loop.

### 3.5 Remove the convergence-driven outer loop

Delete the current loop structure:

```cpp
for (int iter = 0; iter < cfg.max_iter; ++iter) {
  ...
  converged = CheckThermalConvergence(...);
  AcceptNextIterate(...);
  if (converged) break;
}
```

Replace it with a fixed-count SDC pass loop, typically:

```cpp
for (int k = 0; k < cfg.iterations; ++k) {
  ...
}
```

### 3.6 Remove Picard-specific cooling entry point semantics

In [src/hydro/srcterms/tabular_cooling.hpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.hpp) and [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp):

- either remove `CoupledRK12Step()`
- or rename and refactor it so it no longer implies:
  - reading `eint_iter`
  - writing `eint_next`
  - participation in a Picard iteration

Do not keep the old name if the semantics have changed substantially.

### 3.7 Remove Picard diagnostics from scripts and docs

Delete or rename references to:

- `coupled_temp_err`
- `coupled_e_err`
- “under-relaxation”
- “fixed-point convergence”
- “max_iter ceiling hit fraction” if it refers to Picard iterations

Update:

- [test/coupled_thermal_validation/summarize_coupled_outputs.py](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/summarize_coupled_outputs.py)
- any markdown plans still describing the Picard solver

At the end, `plan_for_coupled_thermal_solver.md` should either be deleted or replaced
with a note that it is obsolete.

## 4. Detailed Testing Instructions

Testing must answer four separate questions:

1. legacy behavior unchanged when the thermal SDC solver is off
2. separate source pieces still behave correctly
3. the thermal simplified-SDC iterations are numerically stable and do what they claim
4. realistic competition cases are more robust than the current Picard branch

### 4.1 Build and execution notes

Use the existing local build tree:

- `build-mpi`

Run single-rank tests first:

```bash
./build-mpi/bin/athenaPK -i <input>
```

### 4.2 Legacy-off regression tests

These should be run before any SDC-specific validation.

#### Test A: Cooling-only legacy path

- disable `thermal_source_solver/enabled`
- enable tabular cooling
- verify results match pre-SDC baseline within expected roundoff/tolerance

#### Test B: Resistivity-only legacy path

- disable `thermal_source_solver/enabled`
- enable unsplit ohmic diffusion with Spitzer resistivity
- verify results match legacy branch behavior

#### Test C: Existing coupled-solver-off reconnection input

- run the usual `pulsed_reconnection.in` setup with thermal coupling disabled
- verify the change set did not alter non-SDC behavior

### 4.3 Separate-source validation

These tests isolate pieces needed by the SDC implementation.

#### Test D: Frozen-state ohmic thermal source reproduction

Goal:

- verify that the cell-centered ohmic thermal source still reproduces the intended
  resistive energy-flux divergence in a frozen thermodynamic state

Use the existing validation assets:

- [test/coupled_thermal_validation/validate_frozen_ohmic_source.py](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/validate_frozen_ohmic_source.py) if still present in the branch
- if the script no longer matches field names, update it

Required check:

- compare the source built by the SDC machinery against the flux-divergence operator in a
  diffusion-only or frozen-state run with hyperbolic transport absent

Do not use full-step `cons(IEN)` changes from a live hydro run as this comparison.

#### Test E: Cooling ODE with external source

Goal:

- verify the refactored cooling integrator correctly solves
  `de/dt = Q_rad(e, rho) + S_ext`

Construct a one-cell or uniform-box test where:

- `rho` is constant
- `S_ext` is prescribed
- compare against a high-resolution reference integration or a hand-check in a regime
  where the solution is monotone

### 4.4 Current-sheet thermal validation

Use the problem generator added on this branch:

- [src/pgen/current_sheet_thermal.cpp](/home/ianfr/athenapk_freem386/src/pgen/current_sheet_thermal.cpp)

What the generator provides:

- a static Harris-like current sheet
- adjustable:
  - `B0`
  - `delta`
  - `rho0`
  - `T0`
  - guide field
  - optional bulk velocity
  - optional pressure bump
- derived diagnostics:
  - `curlBz`
  - `T`
  - `eta`
  - `dt_heat_local`
  - `dt_cool_local`
  - `dt_diff_local`
  - `dt_hyp_fms`

These make it a good minimal problem for thermal competition without full reconnection
dynamics.

#### Test F: Mild diffusing current sheet

Input:

- [test/coupled_thermal_validation/current_sheet_thermal_diffusing.in](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/current_sheet_thermal_diffusing.in)

Purpose:

- confirm the new SDC implementation remains stable in a regime where the current branch
  already eventually behaved well

Checks:

- no crashes
- no negative pressure/internal energy
- SDC iteration count is the configured fixed count, not a convergence count
- temperature and eta evolve smoothly

#### Test G: Strongly competing frozen current sheet

Input:

- [test/coupled_thermal_validation/current_sheet_thermal_competing_frozen.in](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/current_sheet_thermal_competing_frozen.in)

Purpose:

- exercise the regime where cooling and ohmic heating are comparable near the current
  layer

Checks:

- current-sheet cells have `dt_heat_local / dt_cool_local ~ O(1)`
- no oscillatory divergence in temperature between the two SDC passes
- final thermal state is bounded and physically sensible

#### Test H: Strongly competing diffusing current sheet

Input:

- [test/coupled_thermal_validation/current_sheet_thermal_competing_diffusing.in](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/current_sheet_thermal_competing_diffusing.in)

Purpose:

- this is the main reduced test for the target regime

Checks:

- robust completion on one rank
- no runaway checkerboarding in temperature or eta
- no evidence of the “hits 32 iterations and moves on” failure mode, because that mode
  should no longer exist
- compare successive outputs with the summary script after it is updated for SDC

### 4.5 Realistic regime tests

After reduced tests pass, rerun short versions of the physically motivated setups:

- aluminum LTE cooling table
- hot background around `1e6 K`
- current-layer temperatures reaching `~100 eV`
- density around `1e20 cm^-3` in physical interpretation
- opposing magnetic fields of order `40 T`

The exact dimensional setup may still be easiest through your reconnection-style inputs,
but the key requirement is:

- cooling and ohmic heating must genuinely compete on similar timescales in active cells

Checks:

- thermal solution quality near the current layer
- no pathological sensitivity to CFL
- results should be more robust than the current Picard branch in the same regime

### 4.6 Diagnostics to expose for the SDC implementation

Do not keep Picard residual diagnostics. Replace them with diagnostics that actually
describe the new method.

Recommended fields or metadata:

- `thermal_sdc_iter_count`
- `thermal_src_ohmic`
- `thermal_src_total`
- optionally `thermal_src_adv`
- optionally `thermal_src_reconstructed`

Recommended summary quantities:

- min/max of `thermal_src_ohmic`
- min/max of `thermal_src_total`
- competition ratio `dt_heat_local / dt_cool_local`
- temperature range in current-sheet cells
- eta range in current-sheet cells

### 4.7 Pass criteria

The thermal simplified-SDC replacement should be considered ready for broader runs only
if all of the following are true:

- legacy-off behavior is unchanged
- frozen-state ohmic source validation passes
- reduced current-sheet tests run stably without convergence-loop artifacts
- strongly competing current-sheet tests do not show oscillatory or pathological thermal
  behavior
- short realistic reconnection-like tests are materially more robust than the Picard
  branch

## 5. Recommended File-Level Work Plan For The Next Codex Run

Use this as the implementation order.

1. Refactor runtime controls in [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp).
2. Refactor thermal scratch fields in [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp).
3. Replace the outer algorithm in [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp).
4. Refactor the cooling helper in [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp).
5. Refactor the ohmic-source builder in [src/hydro/diffusion/resistivity.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/resistivity.cpp).
6. Identify and modify the hydro predictor/update insertion point so the lagged thermal
   source is seen by the unsplit hydro step.
7. Update diagnostics and validation scripts under
   [test/coupled_thermal_validation](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation).
8. Delete obsolete Picard-only code and obsolete plan text.

## 6. Important Caveat

The largest unresolved implementation question is not in cooling or resistivity. It is
where AthenaPK’s unsplit hydro machinery should be modified so the lagged thermal source
enters the time-centered hydro update in the same spirit as simplified-SDC.

The next Codex conversation should start by tracing:

- how unsplit hydro source terms are fed into primitive prediction
- where a cell-centered thermal source can alter the internal-energy part of the update
- how to extract a consistent cell-centered advective internal-energy source `A_e`

Without that structural change, any replacement will risk becoming another post-hydro
thermal corrector rather than a true simplified-SDC coupling.

## 7. Copy-Pasteable Prompt Sequence For Future Codex Runs

Use the following prompts in order. They are intentionally narrow enough that each run
can complete a coherent implementation slice, verify it, and report blockers for the
next prompt.

### Prompt 1: Refactor thermal ODE and ohmic source interfaces

```text
Implement Phase A from plan_for_thermal_simplified_sdc.md.

Task:
- Refactor the thermal ODE helper and the ohmic source builder so they no longer depend
  on Picard-specific scratch field names.
- Keep the current Picard path working after the refactor.

Requirements:
- Inspect the existing coupled thermal, tabular cooling, and resistive source code paths
  first.
- Refactor src/hydro/srcterms/tabular_cooling.cpp and .hpp so the coupled thermal ODE
  routine accepts explicit input/output field names or another comparably clear
  field-agnostic interface.
- Refactor src/hydro/diffusion/resistivity.cpp and the relevant declarations so the
  ohmic heating source builder accepts explicit thermodynamic input and source output
  fields instead of hard-wiring eint_iter and s_ohm_iter.
- Preserve current behavior by adapting the existing Picard path to use the new
  interfaces.
- Make the smallest defensible API changes needed to support later SDC work.
- Run relevant build/tests if possible, or explain exactly why you could not.

Deliverables:
- Code changes
- A concise summary of the new APIs
- Any blockers or risks for the next phase
```

### Prompt 2: Add SDC-oriented controls and scratch fields without removing Picard

```text
Implement the next setup phase from plan_for_thermal_simplified_sdc.md.

Task:
- Add SDC-oriented runtime controls and scratch fields alongside the legacy Picard ones.
- Do not remove or break the Picard path yet.

Requirements:
- Inspect the current Hydro package parameter parsing and scratch field registration
  first.
- Update src/hydro/hydro.cpp to add new thermal_source_solver controls appropriate for
  the upcoming simplified-SDC path, while preserving the old controls for now.
- Add new SDC-oriented scratch fields with clear names and comments where needed.
- Avoid misleading reuse of Picard-specific names for new semantics.
- Keep legacy behavior unchanged unless the new mode is explicitly selected.
- Update any validation or input parsing checks needed to support both paths cleanly.
- Run relevant build/tests if possible, or explain exactly why you could not.

Deliverables:
- Code changes
- A short description of the new params and fields
- Notes on any naming or compatibility tradeoffs
```

### Prompt 3: Implement and validate stage-local internal-energy bookkeeping

```text
Implement the A_e bookkeeping phase from plan_for_thermal_simplified_sdc.md.

Task:
- Add stage-local bookkeeping for eint_stage_start, eint_adv, and A_e.
- Validate the bookkeeping in a non-cooling, non-resistive case before any SDC coupling
  is added.

Requirements:
- Inspect the hydro driver stage flow and current internal-energy commit logic first.
- Implement a concrete, auditable way to save the stage-start thermodynamic state and
  construct the stage-local advective/compressive internal-energy contribution.
- Use the plan’s operational definition:
  A_e = (eint_adv - eint_stage_start) / dt_stage
- Make the implementation explicit about how eint_adv is derived from the stage update.
- Add a focused validation test or diagnostic path for simple compression/expansion
  bookkeeping without cooling or resistivity.
- Do not yet add the full SDC iteration loop.
- Run the relevant tests if possible, or explain exactly why you could not.

Deliverables:
- Code changes
- A concise explanation of the energy bookkeeping
- Validation results and any remaining ambiguities
```

### Prompt 4: Identify and implement the pre-flux lagged-source insertion point

```text
Implement the pre-flux coupling phase from plan_for_thermal_simplified_sdc.md.

Task:
- Find and implement the correct insertion point so a lagged thermal source is seen by
  the hydro stage before flux calculation.

Requirements:
- Inspect the hydro driver task graph, any predictor/reconstruction hooks, and the
  current source application path first.
- Note that Phase B only stores scalar thermal bookkeeping (`eint_stage_start`,
  `eint_adv`, `thermal_ae`) and does not preserve a full `cons_stage_start` snapshot.
  Only add a dedicated stage-start conserved-state register if this phase is actually
  blocked without it, and explain why.
- Do not assume AddUnsplitSources() is sufficient.
- Implement the smallest defensible driver change that allows a stage-local lagged
  thermal source to influence the hydro build before the Riemann solve.
- Use the new field-agnostic thermal APIs introduced in Phase A, but keep resistive
  flux reconstruction based on the existing prim state unless this phase explicitly
  requires a broader state abstraction.
- Preserve existing behavior when the new path is disabled.
- Explain clearly where in the stage flow the new hook lives and why.
- Keep the current Picard path functional if practical; if not, explain the exact
  conflict.
- Run relevant build/tests if possible, or explain exactly why you could not.

Deliverables:
- Code changes
- A short explanation of the task-graph change
- Risks for later SDC iterations
```

### Prompt 5: Implement the first simplified-SDC stage-local path

```text
Implement the first working simplified-SDC thermal coupling path from
plan_for_thermal_simplified_sdc.md.

Task:
- Add a stage-local simplified-SDC path with a fixed iteration count, default 2.
- Use lagged ohmic heating plus tabular cooling.
- Keep the legacy Picard path available behind the old option until validation is done.

Requirements:
- Inspect the refactored thermal ODE path, ohmic source path, and new pre-flux hook
  first.
- Note that Prompt 4 introduced a real pre-flux driver consumer of
  `thermal_src_lagged`. Do not assume that field is self-managing: this phase must make
  its ownership, initialization time, stage-local validity window, and update cadence
  explicit.
- Note that Phase B reconstructs `A_e` from `eint_stage_start` plus the post-flux
  stage state and does not yet save a full `cons_stage_start` copy. If the SDC
  implementation needs more than that scalar bookkeeping, add the smallest justified
  extension and explain the dependency clearly.
- Do not assume the current `thermal_ae` extraction is already the exact SDC `A_e`
  term you ultimately want. If this phase relies on that equivalence, validate it or
  state the approximation explicitly.
- Implement the new stage-local SDC orchestration in the appropriate owner module.
- Use a fixed iteration count, not convergence-based iteration.
- Avoid under-relaxation and Picard-style residual acceptance logic.
- Use the new field-agnostic thermal APIs introduced in Phase A, but keep resistive
  flux reconstruction based on the existing prim state unless this phase explicitly
  requires a broader state abstraction.
- Do not redesign the Phase A string-based field interface unless the SDC
  implementation is actually blocked by it; prefer the smallest extension needed.
- Ensure the magnetic-only resistive update and thermal energy bookkeeping are applied
  exactly once per accepted stage.
- Preserve legacy behavior when simplified-SDC mode is not selected.
- Run relevant build/tests if possible, or explain exactly why you could not.

Deliverables:
- Code changes
- A concise explanation of the stage-local algorithm as implemented
- Any mismatches between the implementation and the plan that still need resolution
```

### Prompt 6: Add validation coverage for the new SDC path

```text
Implement the validation phase for the new simplified-SDC path from
plan_for_thermal_simplified_sdc.md.

Task:
- Add or update validation inputs, diagnostics, and summary scripts for the new SDC
  implementation.
- Compare behavior against the legacy path where that comparison is meaningful.

Requirements:
- Inspect the current thermal validation problem generator, test inputs, and summary
  tooling first.
- Treat the current simplified-SDC implementation as a stage-local thermal corrector
  with a single hydro build per stage unless a previous phase explicitly changed that.
  Do not describe or validate it as a full repeated hydro-plus-thermal SDC pass unless
  the code actually does that.
- Treat the new pre-flux path as part of what must be validated, not just the thermal
  ODE solve. Include checks or discussion that the lagged source is initialized and
  consumed consistently by the stage build when simplified-SDC is enabled.
- Keep in mind that no full `cons_stage_start` snapshot exists unless a previous phase
  added one. If validation depends on exact stage replay or rollback semantics, say so
  explicitly instead of assuming that state is available.
- Do not assume the current `thermal_ae` field is a proven exact representation of the
  advective/compressive source for SDC purposes. The current path approximates
  `A_e = (eint_adv - eint_stage_start) / dt_stage` from scalar bookkeeping only; if
  validation uses that as ground truth, justify that choice, and if validation shows it
  is inadequate, say what smallest extension is needed next.
- Update diagnostics to use SDC-relevant quantities instead of Picard residuals.
- Preserve or improve the existing current-sheet thermal validation problems.
- Add comparisons for frozen-current and evolving-current cases if feasible.
- Report clearly what was tested, what passed, and what remains untested.
- Run the relevant tests if possible, or explain exactly why you could not.

Deliverables:
- Code and test changes
- Validation summary
- Residual risks before cleanup
```

### Prompt 7: Remove Picard-only code after validation

```text
Implement the final cleanup phase from plan_for_thermal_simplified_sdc.md.

Task:
- Remove Picard-only runtime controls, scratch fields, convergence logic, diagnostics,
  and obsolete documentation now that the simplified-SDC path exists and has been
  validated.

Requirements:
- Inspect the current state of both the legacy and new paths first.
- Carry forward the current validation status explicitly before cleanup:
  - frozen-state ohmic-source comparison against the legacy resistive energy update has
    passed
  - the pre-flux validation still shows a structural mismatch: the lagged source is
    initialized and diagnostic snapshots exist, but `eint_adv` / `thermal_ae` do not
    yet demonstrate that the predictor-applied source is consumed consistently by the
    single stage hydro build
  Do not proceed as though validation is complete if that mismatch still exists.
- Before deleting compatibility code, confirm whether the accepted algorithm is still
  the current stage-local thermal corrector with one hydro build per stage, or whether a
  later phase upgraded it to a true repeated hydro-plus-thermal SDC pass. Do not clean
  up as though that architectural question was already settled if it was not.
- Before deleting compatibility code, confirm that the simplified-SDC path fully owns
  `thermal_src_lagged` lifecycle semantics and no remaining stage logic still depends on
  Picard-era initialization assumptions.
- Preserve any diagnostics needed to investigate the current pre-flux mismatch, such as
  stage-start source snapshots or equivalent bookkeeping, until that mismatch is either
  resolved or intentionally deferred with clear documentation.
- Before removing bookkeeping or shims, confirm whether the final SDC implementation
  still works without a `cons_stage_start` snapshot. If later phases introduced one,
  keep or remove it based on demonstrated need rather than on this plan’s earlier
  assumptions.
- Do not treat the current `thermal_ae` extraction as settled unless validation has
  shown it is the correct `A_e` for the accepted algorithm. If it remains the scalar
  approximation `A_e = (eint_adv - eint_stage_start) / dt_stage`, document that
  technical debt or replace it with the smallest validated extension before removing the
  evidence.
- If the pre-flux mismatch remains unresolved, assume the smallest next extension is a
  dedicated snapshot immediately after `ApplyPreFluxThermalSource()` such as
  `eint_after_pre_flux` or equivalent conserved-state bookkeeping. Do not remove the
  surrounding diagnostics until that question is settled.
- Remove Picard-specific parameters from src/hydro/hydro.cpp.
- Remove Picard-only scratch fields and diagnostics.
- Remove convergence-based outer-loop logic, under-relaxation, and residual-based
  acceptance code from the internal energy solver path.
- Update docs, tests, and validation scripts to match the final interface.
- Keep the final result internally consistent; do not leave dead compatibility shims
  unless there is a clear reason.
- Run relevant build/tests if possible, or explain exactly why you could not.

Deliverables:
- Code and doc changes
- A concise summary of what was removed
- Any remaining technical debt or follow-up work
```
