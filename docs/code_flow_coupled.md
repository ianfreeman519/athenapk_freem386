# Coupled Code Flow When the Thermal Solver Is Enabled

This note describes the execution path in AthenaPK when the coupled thermal solver is enabled. This is the newer path where thermal source terms are not applied as a simple legacy cooling update. Instead, AthenaPK runs a stage-local internal-energy solve and then commits the result back into the main hydro state.

This is written for a reader who is comfortable with Python and object-oriented design, but less familiar with C++.

## Core Idea

As in the legacy path, AthenaPK carries two main views of the fluid:

- `cons`: conserved variables. This is the state that the hydro solver advances.
- `prim`: primitive variables. This is the state used by reconstruction, Riemann solves, and many source terms.

The coupled thermal solver adds a third category:

- stage-local thermal scratch fields such as `eint_stage_start`, `eint_adv`, `eint_sdc`, `thermal_src_lagged`, and `thermal_src_total`

These scratch fields are not part of `cons` or `prim`. They are separate mesh fields used to store intermediate thermal state and source estimates:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1025)

The design is:

1. advance the usual hydro state
2. estimate what advection did to internal energy
3. solve a thermal correction problem on stage-local scratch fields
4. commit only the thermal correction back to `cons(IEN)` and `prim(IPR)`

## How the Coupled Path Is Chosen

The key branch is still in `AddUnsplitSources()`:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:358)

The coupled path is used when:

- `thermal_source_solver/enabled = true`
- and at least one of the `couple_*` flags is enabled

This is represented by:

- `thermal_source_solver_enabled`
- `thermal_couple_cooling`
- `thermal_couple_ohmic`
- `thermal_couple_conduction`
- `thermal_couple_viscous`

The branch that selects the coupled path is:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:371)

If that condition is true, `AddUnsplitSources()` calls:

- `AddCoupledInternalEnergySources()`

which then calls:

- `RunSimplifiedSDCThermalCoupling()`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:322)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:232)

## Why Input Flags Are Checked Up Front

The coupled mode has more constraints than the legacy path, so initialization validates the requested configuration early:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:917)

Checks performed when `thermal_source_solver_enabled` is true:

1. at least one `couple_*` flag must be true
2. coupled conduction is currently rejected
3. coupled viscous heating is currently rejected
4. iteration count must be positive
5. the only supported predictor is `"end_of_interface_state"`
6. coupled cooling requires `cooling/enable_cooling = tabular`
7. coupled ohmic heating requires:
   - `diffusion/resistivity = ohmic`
   - Spitzer resistivity
   - `diffusion/integrator = unsplit`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:943)

These flags are checked during initialization for two reasons:

1. to reject unsupported combinations before a run starts
2. to let later code take cheap branches without repeatedly re-validating the configuration

## Where Non-`cons` and Non-`prim` Quantities Are Stored

The coupled solver relies on stage-local scratch fields registered in the Hydro package:

- `eint_next`
- `temp_next`
- `eint_stage_start`
- `eint_adv`
- `eint_sdc`
- `thermal_ae`
- `thermal_src_lagged`
- `thermal_src_pre_flux`
- `thermal_src_ohmic_pre_flux`
- `thermal_src_ohmic`
- `thermal_src_total`
- `thermal_sdc_iter_count`

These are one-component cell fields:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1025)

High-level meanings:

- `eint_stage_start`: accepted specific internal energy at the start of the stage
- `eint_adv`: specific internal energy after the hydro flux update
- `thermal_ae`: advection-only internal-energy rate estimate
- `eint_sdc`: the current iterate of the thermal solve
- `eint_next`: the next iterate produced by one thermal integration pass
- `temp_next`: temperature corresponding to `eint_next`
- `thermal_src_total`: thermal source used in the current iteration
- `thermal_src_ohmic`: ohmic part of the thermal source
- `thermal_src_lagged`: accepted combined thermal source, used by the pre-flux predictor

## Initialization Flow

### Stage 0: Program startup

`main()` sets up Parthenon, installs package callbacks, selects the problem generator, and then runs the Hydro driver:

- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp:37)
- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp:196)

Problem hooks may still install additional unsplit or split source terms, just as in the legacy path.

### Stage 1: Hydro package construction

`Hydro::Initialize()` reads all hydro, cooling, diffusion, and thermal-solver options:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:406)

It registers:

- `cons` as the independent evolved field
- `prim` as a derived field
- thermal scratch fields for the coupled solver

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1014)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1025)

### Stage 2: Initial `cons -> prim` sync model

`FillDerived` calls the EOS conversion to rebuild `prim` from `cons`:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:215)
- [src/eos/adiabatic_hydro.cpp](/home/ianfr/athenapk_freem386/src/eos/adiabatic_hydro.cpp:33)

The EOS conversion may also repair `cons` using floors and ceilings:

- [src/eos/adiabatic_hydro.hpp](/home/ianfr/athenapk_freem386/src/eos/adiabatic_hydro.hpp:52)

## Main Loop Flow

The coupled path differs from the legacy path in two major places:

1. before fluxes, it applies a lagged thermal predictor to the state used for reconstruction
2. after fluxes, it runs a simplified-SDC internal-energy correction instead of direct legacy cooling

## Per-cycle pre-step work

`PreStepMeshUserWorkInLoop()` computes global helper quantities needed before the timestep task graph is executed:

- `mindx`
- `dt_hyp`
- `dt_diff`
- `c_h`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:103)

This matters for the coupled path because:

- GLM-MHD source terms need `c_h`
- resistive or diffusive pieces may participate in the thermal solve
- the driver wants these values ready before stage tasks execute

## Stage 1 setup before the main RK/VL update

On stage 1, the driver may:

1. allocate register `u1`
2. apply half-step RKL2 diffusion if needed
3. apply initial Strang-split problem sources

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:442)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:497)

## For each integration stage

The following steps run for each stage.

### 1. Save stage-start internal energy

The driver calls `SaveStageStartInternalEnergy()`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:562)

This computes specific internal energy from `cons` and stores:

- `eint_stage_start`
- `eint_adv`
- `thermal_ae = 0`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:251)

What it reads:

- `cons`

What it writes:

- thermal scratch fields only

### 2. Initialize lagged stage thermal source

If the pre-flux thermal predictor is enabled, the driver calls `InitializeStageLaggedThermalSource()`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:564)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:289)

This step:

1. zeros stage-local thermal source fields
2. copies `eint_stage_start -> eint_sdc`
3. builds a cooling source from `eint_stage_start` if cooling is coupled
4. builds an ohmic source from `eint_stage_start` if ohmic heating is coupled
5. combines them into `thermal_src_lagged`
6. snapshots that field into `thermal_src_pre_flux`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:299)

This is stage-local state preparation. The important idea is:

- the solver forms a lagged estimate of the thermal source using the accepted stage-start internal energy

### 3. Apply the pre-flux thermal predictor

The driver then calls `ApplyPreFluxThermalSource()`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:570)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:321)

This uses `thermal_src_lagged` to update the state **before** flux calculation:

- `cons(IEN) += dt * thermal_src_lagged`
- `prim(IPR) += (gamma - 1) * dt * thermal_src_lagged`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:343)

What it reads:

- `thermal_src_lagged`
- `cons`
- `prim`

What it writes:

- `cons(IEN)`
- `prim(IPR)`

Why this exists:

- the coupled solver wants the reconstruction/Riemann step to see a state that already reflects the lagged thermal source

### 4. Refresh boundaries and rebuild derived state

After the pre-flux thermal predictor, the driver does boundary exchange and `FillDerived` before calculating fluxes:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:573)

This is a major difference from the legacy path.

Why:

- the state used for reconstruction must be consistent after the pre-flux thermal update
- ghost cells and derived primitives need to match the predicted state

### 5. Compute fluxes from `prim`

The solver reconstructs and solves fluxes using `prim`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:578)

Then flux divergence updates `cons`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:599)

After this:

- `cons` is newly advanced by advection
- `prim` is not yet fully resynced to the newest `cons`

### 6. Record the advection-only internal-energy change

The driver calls `UpdateStageAdvectiveInternalEnergy()`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:605)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:286)

This computes:

- new specific internal energy from updated `cons`
- `eint_adv`
- `thermal_ae = (eint_adv - eint_stage_start) / dt_stage`

Interpretation:

- `thermal_ae` is the solver’s estimate of what advection alone did to internal energy during this stage

### 7. Apply unsplit sources

The driver calls `AddUnsplitSources()`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:613)

The order inside `AddUnsplitSources()` is still:

1. GLM-MHD source if `fluid == glmmhd`
2. coupled internal-energy source path if enabled
3. problem unsplit source hook

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:358)

## Source-term details in the coupled path

### A. GLM-MHD Dedner source

This runs before the thermal coupled solve when `fluid == glmmhd`:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:361)

It reads `prim` and package parameters like `c_h`, `mindx`, and `glmmhd_alpha`, and writes `cons`:

- always damps `cons(IPS)`
- in extended mode also updates momentum and total energy

Relevant code:

- [src/hydro/glmmhd/dedner_source.cpp](/home/ianfr/athenapk_freem386/src/hydro/glmmhd/dedner_source.cpp:17)

### B. Coupled thermal internal-energy solve

This is the main coupled-only step.

Dispatch:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:375)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:322)

The real work happens in `RunSimplifiedSDCThermalCoupling()`:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:232)

#### 7B-1. Initialize the SDC iteration state

At the start of the thermal solve:

1. `eint_stage_start -> eint_sdc`
2. zero `thermal_src_ohmic`
3. zero `thermal_src_total`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:250)

Interpretation:

- `eint_stage_start` is treated as the accepted stage-start initial condition
- `eint_sdc` is the current iterate that will be corrected

#### 7B-2. Fixed number of thermal iterations

The code runs `thermal_source_iterations` passes:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:255)

Each iteration does:

1. if ohmic coupling is enabled, build `thermal_src_ohmic` from magnetic flux divergence
2. otherwise zero `thermal_src_ohmic`
3. build `thermal_src_total = rho * thermal_ae + thermal_src_ohmic`
4. integrate the thermal ODE from `eint_stage_start` using `thermal_src_total`
5. write result into `eint_next`
6. copy `eint_next -> eint_sdc`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:256)

This is an important design point:

- the ODE always starts from `eint_stage_start`
- but the source being iterated depends on the current guess `eint_sdc`

That is the “simplified-SDC” style correction loop.

#### 7B-3. How the combined thermal source is built

The combined source is:

- advection contribution: `rho * thermal_ae`
- plus ohmic contribution: `thermal_src_ohmic`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:144)

This means:

- the thermal solve is not re-solving the full hydro problem
- it is correcting internal energy using an advection estimate plus thermal source physics

#### 7B-4. How cooling is integrated in the coupled path

Cooling is not applied directly to `cons(IEN)` here.

Instead, the coupled solver calls:

- `TabularCooling::IntegrateThermalODEWithSource()`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:263)

That routine:

1. reads `cons(IDN)` for density
2. reads `eint_in`
3. reads `thermal_src`
4. computes a heating rate `thermal_src / rho`
5. integrates `de/dt = cooling_table.DeDt(e, rho) + heating_rate`
6. writes:
   - `eint_out`
   - `temp_out`

Relevant code:

- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:290)

Important difference from the legacy path:

- legacy cooling directly edits `cons(IEN)`
- coupled cooling evolves a stage-local internal-energy field first

#### 7B-5. How ohmic thermal source is built

If ohmic coupling is enabled, the coupled solver uses:

- `BuildOhmicThermalSourceFromFluxDivergence()`

Relevant code:

- [src/hydro/diffusion/resistivity.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/resistivity.cpp:465)

This routine:

1. reads magnetic field data from `prim`
2. reads a specific-internal-energy field such as `eint_sdc`
3. computes resistive energy fluxes on faces
4. converts their divergence into a cell-centered thermal source field

Conceptually:

- it estimates how much ohmic diffusion would deposit into internal energy

### C. Rebuild the accepted lagged source for the next stage

After the fixed iterations:

1. rebuild final `thermal_src_ohmic` from the final `eint_sdc`
2. build pure cooling source from final `eint_sdc`
3. combine them into `thermal_src_lagged`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:269)

Why this matters:

- `thermal_src_lagged` is the accepted source estimate that will seed the next stage’s pre-flux predictor

### D. Commit the thermal correction back to the hydro state

After the stage-local thermal solve converges for its fixed number of iterations, the solver commits the correction:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:189)

The commit step does:

1. compute `delta_eint = eint_sdc - eint_adv`
2. update `cons(IEN) += rho * delta_eint`
3. update `prim(IPR) = rho * eint_sdc * (gamma - 1)`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:205)

Interpretation:

- `eint_adv` is the advection-only result
- `eint_sdc` is the corrected thermal result
- the solver only commits the difference between them

This is one of the most important lines in the coupled design.

### E. Apply magnetic-only resistive update

If `thermal_couple_ohmic` is enabled, the coupled solver also owns the resistive magnetic update:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:73)

It:

1. resets fluxes
2. computes resistive magnetic fluxes without energy flux
3. updates `cons(IB1..IB3)` using flux divergence
4. copies those magnetic values into `prim(IB1..IB3)`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:79)
- [src/hydro/diffusion/resistivity.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/resistivity.cpp:452)

Why:

- the coupled solver wants to own both ohmic heating and the corresponding magnetic diffusion update
- this prevents double-application of resistive physics

That ownership is enforced in the normal diffusion path:

- when coupled ohmic is enabled, `CalcDiffFluxes()` skips the standard resistive diffusion path

Relevant code:

- [src/hydro/diffusion/diffusion.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/diffusion.cpp:18)

### F. Problem unsplit source terms still run afterward

After the coupled thermal solve, AthenaPK still calls `ProblemSourceUnsplit` if a problem installed one:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:383)

So the coupled thermal solver does not replace other unsplit source terms. It only replaces the legacy direct thermal source integration.

## `cons` and `prim` Sync Behavior in the Coupled Path

There are multiple sync-related moments in the coupled path.

### Full sync points

Full `cons -> prim` sync still happens through `FillDerived`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:575)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:642)

The pre-flux `FillDerived` is coupled-mode specific. The post-stage `FillDerived` is common to both paths.

### Partial manual sync points

The coupled path also performs partial syncs by hand:

1. pre-flux predictor updates `prim(IPR)` when it changes `cons(IEN)`
2. commit step updates `prim(IPR)` when it changes `cons(IEN)`
3. magnetic-only resistive commit updates `prim(IB1..IB3)` after changing `cons(IB1..IB3)`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:341)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:206)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:97)

Why partial syncs exist:

- some downstream tasks need pressure or magnetic field immediately
- full `FillDerived` is more expensive and only occurs at certain structured points

## Final Stage and End-of-Timestep Flow

On the last RK/VL stage only:

1. final Strang-split sources run
2. first-order split sources run
3. boundary exchange runs
4. final `FillDerived` rebuilds `prim`

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:619)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:642)

Then the driver may:

1. run second-half RKL2 diffusion if needed
2. reset reduction variables
3. estimate the next timestep
4. advance tracers
5. tag AMR blocks

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:649)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:656)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:676)

## Timestep Constraints in the Coupled Path

`EstimateTimestep()` still computes the next timestep in the usual way:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1181)

Important coupled-path detail:

- tabular cooling can still impose a timestep limit through `EstimateTimeStep()`
- ohmic heating can still impose its own timestep limit through `EstimateOhmicHeatingTimestep()`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1187)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1202)

Even though the thermal update method changed, the code still tracks limiting physics separately for diagnostics and stability control.

## Finalization After the Simulation Ends

At the outermost level:

1. `main()` constructs `HydroDriver`
2. `driver.Execute()` runs the full simulation loop
3. Parthenon finalizes MPI and Kokkos

Relevant code:

- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp:196)

## Short Summary

When the coupled thermal solver is enabled, AthenaPK does not directly cool `cons(IEN)` in the old source-term style. Instead it:

1. saves stage-start internal energy
2. builds a lagged thermal source
3. applies a pre-flux thermal predictor
4. computes hydro fluxes
5. measures the advection-only internal-energy change
6. runs a stage-local simplified-SDC thermal solve on scratch fields
7. commits only the thermal correction back to `cons(IEN)` and `prim(IPR)`
8. if coupled ohmic is enabled, also owns the resistive magnetic update

That is the defining difference from the legacy path.

## Compact Flow Chart

```text
Initialization
  -> main() selects problem hooks
  -> Hydro::Initialize() reads flags and validates coupled configuration
  -> register cons, prim, and thermal scratch fields

Per timestep
  -> PreStepMeshUserWorkInLoop()
     -> compute shared global quantities if needed
  -> stage 1 only:
     -> optional half-step RKL2 diffusion
     -> optional initial Strang-split sources
  -> for each stage:
     -> SaveStageStartInternalEnergy()
     -> InitializeStageLaggedThermalSource()
     -> ApplyPreFluxThermalSource()
     -> boundary exchange + FillDerived()
     -> compute fluxes from prim
     -> update cons with flux divergence
     -> UpdateStageAdvectiveInternalEnergy()
     -> AddUnsplitSources()
        -> GLM source, if enabled
        -> coupled internal-energy solve
           -> initialize eint_sdc from eint_stage_start
           -> build lagged ohmic/cooling sources
           -> iterate simplified-SDC correction
           -> rebuild accepted lagged source
           -> commit thermal correction to cons(IEN), prim(IPR)
           -> commit magnetic-only resistive update if needed
        -> problem unsplit source, if installed
     -> last stage only:
        -> final Strang-split sources
        -> first-order split sources
     -> boundary exchange
     -> FillDerived()
  -> after last stage:
     -> optional second half RKL2 diffusion
     -> estimate next dt
     -> optional tracers
     -> optional AMR tagging

Finalization
  -> driver.Execute() returns
  -> main() calls ParthenonFinalize()
```
