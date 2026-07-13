# Code Flow When the Coupled Thermal Solver Is Enabled

This note describes the execution path in AthenaPK when the coupled thermal solver is enabled. In practice, that means AthenaPK uses the simplified-SDC internal-energy coupling path instead of the older direct cooling source-term path.

This is written for a reader who is comfortable with Python and object-oriented design, but less familiar with C++.

## Core Idea

AthenaPK carries two main views of the fluid state:

- `cons`: the conserved variables. This is the state that is actually advanced by the solver.
- `prim`: the primitive variables. This is the state used by reconstruction, Riemann solves, and many source terms.

With the coupled thermal solver enabled, AthenaPK also carries stage-local thermal scratch fields such as:

- `eint_stage_start`
- `eint_adv`
- `eint_sdc`
- `eint_next`
- `thermal_ae`
- `thermal_src_lagged`
- `thermal_src_pre_flux`
- `thermal_src_ohmic_pre_flux`
- `thermal_src_ohmic_post_hydro_sts`
- `thermal_src_ohmic_sts_delta`
- `thermal_src_ohmic`
- `thermal_src_total`

You can think of the coupled path as:

1. advance the main MHD state
2. estimate what advection did to internal energy
3. solve a stage-local thermal correction problem on scratch fields
4. commit only the thermal correction back into `cons(IEN)` and `prim(IPR)`

As in the legacy path, `cons` and `prim` are not automatically kept in sync after every update. Full synchronization still happens through `FillDerived`, which calls `ConservedToPrimitive`:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:215)
- [src/eos/adiabatic_hydro.cpp](/home/ianfr/athenapk_freem386/src/eos/adiabatic_hydro.cpp:33)

## How the Coupled Thermal Path Is Chosen

The main branch is in `AddUnsplitSources()`:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:358)

The coupled path is used when:

- `thermal_source_solver/enabled = true`
- and at least one `thermal_source_solver/couple_*` flag is true

This is represented in the Hydro package as:

- `thermal_source_solver_enabled`
- `thermal_couple_cooling`
- `thermal_couple_ohmic`
- `thermal_couple_conduction`
- `thermal_couple_viscous`

If the coupled path is active, `AddUnsplitSources()` calls:

- `AddCoupledInternalEnergySources()`: [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:498)

From there the live behavior splits by diffusion integrator:

1. `diffusion/integrator = unsplit`
   - stage-local simplified-SDC correction runs inside the normal unsplit-source slot
2. `diffusion/integrator = rkl2`
   - a special midpoint thermal solve runs after the pre-hydro STS half-step
   - the usual stage-local coupled-source hook later returns immediately

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:344)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:401)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:504)

## Where Input Flags Are Read and Why They Are Checked

Most configuration is read once during Hydro package initialization:

- cooling mode:
  - [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:900)
- coupled thermal solver flags:
  - [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:917)

These flags are checked during initialization for two reasons:

1. to choose the correct algorithm before the timestep task graph is built
2. to fail early if the user asked for an unsupported combination

Relevant checks:

- at least one `couple_*` flag must be true
- coupled conduction is rejected
- coupled viscous heating is rejected
- iteration count must be positive
- only `source_predictor = end_of_interface_state` is supported
- coupled cooling requires `cooling/enable_cooling = tabular`
- coupled ohmic heating requires:
  - `diffusion/resistivity = ohmic`
  - `diffusion/resistivity_coeff = spitzer`
  - `diffusion/integrator = unsplit` or `rkl2`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:943)

There is an additional runtime guard in the driver for the `rkl2` branch:

- if the thermal solver is enabled with `diffusion/integrator = rkl2`, the live code currently supports only the midpoint path with both `couple_cooling = true` and `couple_ohmic_heating = true`

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:413)

## Where Non-`cons` and Non-`prim` Quantities Are Stored

The thermal bookkeeping fields are registered as separate one-component cell fields:

- `eint_next`
- `temp_next`
- `eint_stage_start`
- `eint_adv`
- `eint_sdc`
- `thermal_ae`
- `thermal_src_lagged`
- `thermal_src_pre_flux`
- `thermal_src_ohmic_pre_flux`
- `thermal_src_ohmic_post_hydro_sts`
- `thermal_src_ohmic_sts_delta`
- `thermal_src_ohmic`
- `thermal_src_total`
- `thermal_sdc_iter_count`

They are created with `Metadata::OneCopy`:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1026)

High-level meaning:

- `eint_stage_start`: accepted specific internal energy at stage start
- `eint_adv`: specific internal energy after the hyperbolic flux update
- `thermal_ae`: stage-local advection/compression estimate
- `eint_sdc`: current thermal iterate
- `eint_next`: next iterate produced by one thermal integration pass
- `thermal_src_lagged`: accepted combined thermal source used by the pre-flux predictor
- `thermal_src_ohmic_pre_flux`: pre-hydro ohmic source snapshot
- `thermal_src_ohmic_post_hydro_sts`: post-hydro STS ohmic source snapshot in `rkl2`
- `thermal_src_ohmic_sts_delta`: difference between post- and pre-hydro STS ohmic source

## Initialization Flow

### Stage 0: Program startup

1. `main()` initializes Parthenon, MPI, and Kokkos.
2. It installs AthenaPK package callbacks:
   - `ProcessPackages`
   - `PreStepMeshUserWorkInLoop`
3. It selects problem-specific hooks based on `job/problem_id`.

Relevant code:

- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp:37)
- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp:55)

Problem hooks still matter because problem generators may install:

- `ProblemSourceUnsplit`
- `ProblemSourceStrangSplit`
- `ProblemSourceFirstOrder`
- `ProblemEstimateTimestep`

### Stage 1: Hydro package construction

1. `Hydro::Initialize()` reads hydro, EOS, cooling, diffusion, and thermal-solver settings.
2. It registers:
   - `cons` as the main independent evolved field
   - `prim` as a derived field
   - thermal scratch fields for the coupled solver
3. It stores function pointers and parameters in the Hydro package.

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:900)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1015)

### Stage 2: Initial `cons -> prim` model

`prim` is regenerated from `cons` by the EOS conversion:

- density copied from `cons`
- velocity computed from momentum / density
- pressure computed from total energy minus kinetic energy
- magnetic pressure accounted for in MHD
- floors and ceilings may change `cons`

Relevant code:

- [src/eos/adiabatic_hydro.hpp](/home/ianfr/athenapk_freem386/src/eos/adiabatic_hydro.hpp:63)

## Main Loop Flow

Below is the main timestep flow in “to-do list” form, with the `unsplit` and `rkl2` thermal branches called out where they diverge.

### Per-cycle pre-step work

Before stage tasks are built, `PreStepMeshUserWorkInLoop()` computes global quantities needed by later source terms or timestep logic.

1. if GLM-MHD cleaning or diffusion is active, compute global minimum cell size `mindx`
2. update package parameters like:
   - `mindx`
   - `dt_hyp`
   - `dt_diff`
   - `c_h`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:103)

### Stage 1 setup before the main RK/VL update

In `HydroDriver::MakeTaskCollection()` stage 1 may:

1. allocate register `u1`
2. if diffusion uses `rkl2`, apply the first half of Strang-split STS diffusion
3. if using the coupled `rkl2` midpoint path, immediately run the midpoint thermal solve
4. apply initial Strang-split problem sources

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:523)

#### What the pre-hydro STS branch does in coupled mode

`AddSTSTasks()` calls `CalcDiffFluxes(..., magnetic_only_resistive_flux = coupled_ohmic)` for STS flux work:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:216)

When coupled ohmic heating is enabled:

- STS still advances magnetic diffusion
- the legacy resistive `IEN` flux path stays disabled

Relevant code:

- [src/hydro/diffusion/diffusion.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/diffusion.cpp:49)

This means:

- `rkl2` STS owns the magnetic update
- the thermal solver remains the sole owner of the internal-energy correction

### For each integration stage

For each RK/VL stage, the driver does the following.

#### 1. Save stage-local bookkeeping

The driver calls `SaveStageStartInternalEnergy()`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:596)

This stores:

- `eint_stage_start`
- `eint_adv`
- `thermal_ae = 0`

from the current `cons` state:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:251)

#### 2. Optional pre-flux thermal predictor

The pre-flux predictor is active only when:

- `thermal_source_solver_enabled == true`
- the solver is **not** using the special `rkl2` midpoint path
- `source_predictor = end_of_interface_state`

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:579)

If active, the stage does:

1. `InitializeStageLaggedThermalSource()`
2. `ApplyPreFluxThermalSource()`
3. boundary exchange and `FillDerived` before fluxes

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:598)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:450)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:321)

This is the normal coupled-`unsplit` behavior. The coupled `rkl2` midpoint path skips this predictor branch:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:452)

#### 3. Compute fluxes from `prim`

The solver reconstructs interface states and solves Riemann problems using `prim`.

Then flux divergence updates `cons`.

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:612)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:633)

State after this step:

- `cons` is new
- `prim` still mostly reflects the pre-flux state

#### 4. Record how advection changed internal energy

The driver calls `UpdateStageAdvectiveInternalEnergy()`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:639)

This computes:

- updated `eint_adv`
- `thermal_ae = (eint_adv - eint_stage_start) / dt_stage`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:286)

#### 5. Apply unsplit source terms

The driver calls `AddUnsplitSources()` once per stage:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:647)

The order inside `AddUnsplitSources()` is:

1. GLM-MHD source term, if using `glmmhd`
2. coupled internal-energy solve, if enabled
3. otherwise legacy tabular cooling, if enabled and not coupled
4. problem-specific unsplit source term, if installed

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:358)

### Source-term details

#### A. GLM-MHD Dedner source

If `fluid == glmmhd`, the driver applies the GLM source first.

What it writes:

- always damps `cons(IPS)`
- in extended mode, also updates momentum and total energy in `cons`

Relevant code:

- [src/hydro/glmmhd/dedner_source.cpp](/home/ianfr/athenapk_freem386/src/hydro/glmmhd/dedner_source.cpp:17)

As in the legacy path, this means `cons` is newer than `prim` until the next sync point.

#### B. Coupled internal-energy branch with `diffusion/integrator = unsplit`

This is the standard simplified-SDC stage-local thermal corrector.

Dispatch:

- `AddCoupledInternalEnergySources()` calls `RunSimplifiedSDCThermalCoupling()` when diffusion is not `rkl2`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:498)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:344)

Per outer SDC iteration the live code does:

1. rebuild ohmic thermal source from `eint_sdc`, if coupled
2. combine `thermal_ae` and ohmic source into `thermal_src_total`
3. integrate the thermal ODE from `eint_stage_start`
4. copy `eint_next -> eint_sdc`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:367)

The thermal ODE integrator itself:

- reads `cons(IDN)` and the selected thermal source field
- subcycles per cell through `TabularCooling::IntegrateThermalODEWithSource()`

Relevant code:

- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:290)

After the iterations:

1. cooling source is rebuilt from final `eint_sdc`
2. the accepted combined thermal source is stored in `thermal_src_lagged`
3. diagnostics are written
4. `CommitSimplifiedSDCInternalEnergyUpdate()` updates `cons(IEN)` and `prim(IPR)`
5. because this is the `unsplit` branch, `ApplyMagneticOnlyResistiveUpdate()` then advances `B`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:381)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:393)

This is the key ownership split in the coupled `unsplit` path:

- thermal solver owns the internal-energy update
- the old resistive `IEN` flux path stays disabled
- a final magnetic-only resistive commit updates `B`

#### C. Coupled internal-energy branch with `diffusion/integrator = rkl2`

This branch is different.

The special midpoint thermal solve runs in stage 1 immediately after the pre-hydro STS half-step:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:530)

Dispatch:

- `ApplyCoupledRKL2MidpointThermalSolve()` calls `RunRKL2MidpointThermalCoupling()`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:489)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:401)

This midpoint path begins with `BuildMidpointThermalSourceBookkeeping()`:

1. zero thermal scratch fields
2. snapshot the accepted state immediately after pre-hydro STS into `eint_stage_start`
3. build cooling source from that state
4. build and store `thermal_src_ohmic_pre_flux`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:202)

Then each outer thermal iteration does:

1. rebuild cooling source from `eint_sdc`
2. combine it with the frozen `thermal_src_ohmic_pre_flux`
3. integrate the thermal ODE from `eint_stage_start`
4. copy `eint_next -> eint_sdc`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:428)

After the iterations:

1. rebuild cooling source from final `eint_sdc`
2. set accepted ohmic source to the frozen pre-hydro source
3. rebuild `thermal_src_lagged`
4. write diagnostics
5. commit the internal-energy correction back to `cons(IEN)` and `prim(IPR)`

Relevant code:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:438)

Later, when `AddUnsplitSources()` is reached during the normal stage flow, the coupled `rkl2` branch does not run a second thermal solve. `AddCoupledInternalEnergySources()` returns immediately for the supported midpoint configuration:

- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:504)

That is the defining difference from `unsplit`:

- `unsplit`: thermal corrector lives in the normal unsplit-source slot
- `rkl2`: one midpoint thermal solve runs before the hydro stage and is then skipped later

#### D. Problem unsplit source terms

After the thermal path, the driver optionally calls `ProblemSourceUnsplit`.

Relevant hook:

- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp:131)

These source terms still follow the usual AthenaPK pattern:

- often read `prim`
- usually write `cons`

### After unsplit sources, before end-of-stage sync

At this point in a stage:

- `cons` contains the newest state
- `prim` may be only partially updated
- the coupled thermal solver has at least patched `prim(IPR)` if it committed an energy correction

This is why the driver later calls `FillDerived`.

### Final stage only: split source cleanup

On the last RK/VL stage only:

1. apply final Strang-split source terms
2. apply first-order split source terms

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:653)

### Boundary exchange

After stage-local updates, the driver exchanges boundary data:

- local/nonlocal ghost communication
- prolongation/restriction support

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:668)

### Full `cons -> prim` sync

Then the driver runs `FillDerived`, the main full synchronization point:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:676)

This rebuilds a consistent primitive state from `cons` through the Hydro derived-field path and EOS conversion:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:215)
- [src/eos/adiabatic_hydro.cpp](/home/ianfr/athenapk_freem386/src/eos/adiabatic_hydro.cpp:33)

### Optional diffusion STS cleanup

If diffusion uses `rkl2`, the driver applies the second half of the Strang-split STS diffusion after the final `FillDerived`:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:685)

In coupled mode this post-hydro STS pass still advances magnetic diffusion only. The standard resistive `IEN` flux path remains disabled:

- [src/hydro/diffusion/diffusion.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/diffusion.cpp:49)

After that accepted post-hydro STS half-step, AthenaPK records the post-hydro ohmic source bookkeeping in:

- `thermal_src_ohmic_post_hydro_sts`
- `thermal_src_ohmic_sts_delta`

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:686)
- [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:516)

## Timestep Finalization Inside the Main Loop

After the last stage of a timestep:

1. reset global reduction vars used for later calculations
2. estimate the next timestep
3. optionally advance tracers
4. optionally tag cells for AMR refinement

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:689)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:709)

### How timestep constraints are checked

`EstimateTimestep()` checks active physics and takes the minimum allowed step:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1181)

For the coupled thermal path:

1. the usual cooling-time limiter still matters if tabular cooling is enabled
2. diffusion limits still matter for resistive or other diffusive pieces
3. the active branch depends on which coupled processes are enabled

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1187)
- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:789)

## Summary of the Two Coupled Thermal Branches

### `unsplit` diffusion

1. stage starts
2. optional pre-flux thermal predictor perturbs the reconstruction state
3. hydro flux update advances `cons`
4. stage-local simplified-SDC thermal corrector runs inside `AddUnsplitSources()`
5. thermal correction commits to `IEN` and `IPR`
6. magnetic-only resistive commit updates `B`
7. later `FillDerived` performs the full sync

### `rkl2` diffusion

1. pre-hydro STS half-step advances magnetic diffusion
2. midpoint thermal solve snapshots that accepted post-STS state
3. midpoint thermal correction commits to `IEN` and `IPR`
4. hydro stage runs
5. later `AddCoupledInternalEnergySources()` returns immediately instead of solving again
6. final `FillDerived` syncs the stage result
7. post-hydro STS half-step advances magnetic diffusion again
8. post-hydro STS ohmic bookkeeping is recorded for diagnostics
