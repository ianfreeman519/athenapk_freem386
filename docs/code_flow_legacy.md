# Legacy Code Flow When the Coupled Thermal Solver Is Disabled

This note describes the legacy execution path in AthenaPK when the coupled thermal solver is disabled. In practice, that means the code does **not** use the simplified-SDC internal-energy coupling path, and instead applies cooling through the older direct source-term path.

This is written for a reader who is comfortable with Python and object-oriented design, but less familiar with C++.

## Core Idea

AthenaPK carries two main views of the fluid state:

- `cons`: the conserved variables. This is the state that is actually advanced by the solver.
- `prim`: the primitive variables. This is the state used by reconstruction, Riemann solves, and many source terms.

You can think of them as two cached representations of the same physical state:

- `cons` stores things like density, momentum, and total energy.
- `prim` stores things like density, velocity, and pressure.

They are **not** automatically kept in sync after every update. The code often updates `cons` first, then later rebuilds `prim` from `cons`.

Key sync points:

- Full sync happens through `FillDerived`, which calls `ConservedToPrimitive`:
  - [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:215)
  - [src/eos/adiabatic_hydro.cpp](/home/ianfr/athenapk_freem386/src/eos/adiabatic_hydro.cpp:33)
- The EOS conversion can also modify `cons` while creating `prim`, because it enforces floors and ceilings:
  - [src/eos/adiabatic_hydro.hpp](/home/ianfr/athenapk_freem386/src/eos/adiabatic_hydro.hpp:52)

## How the Legacy Thermal Path Is Chosen

The main branch is in `AddUnsplitSources()`:

- If the coupled thermal solver is enabled and at least one thermal process is coupled, AthenaPK uses the newer internal-energy solver path.
- Otherwise, if `cooling/enable_cooling = tabular`, AthenaPK uses the old direct cooling source-term path.

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:358)

In other words, legacy behavior means:

1. Hyperbolic update happens normally.
2. Cooling is applied afterward as a normal unsplit source term.
3. The full `cons -> prim` rebuild happens later, not immediately.

## Where Input Flags Are Read and Why They Are Checked

Most configuration is read once during Hydro package initialization:

- cooling mode:
  - [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:900)
- thermal solver flags:
  - [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:917)

These flags are checked during initialization for two reasons:

1. To choose the correct algorithm before the timestep task graph is built.
2. To fail early if the user asked for an unsupported combination.

Examples:

- coupled cooling requires tabular cooling
- coupled ohmic heating requires ohmic resistivity with Spitzer coefficients
- coupled conduction and coupled viscous heating are currently rejected

Relevant checks:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:943)

Even when the coupled solver is disabled, the flags are still stored in the package so later code can branch on them cheaply.

## Where Non-`cons` and Non-`prim` Quantities Are Stored

Additional thermal bookkeeping fields are registered as separate mesh fields, not inside `cons` or `prim`:

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

They are created as one-component cell fields with `Metadata::OneCopy`:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1025)

For the legacy path, most of these are effectively scratch or dormant fields. They exist because the newer coupled solver and diagnostics use them, but legacy tabular cooling does not depend on them for its actual update.

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

Problem hooks matter because source terms may be supplied by the problem generator:

- `ProblemSourceUnsplit`
- `ProblemSourceStrangSplit`
- `ProblemSourceFirstOrder`
- `ProblemEstimateTimestep`

Defaults are `nullptr` until a problem installs them:

- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp:21)

### Stage 1: Hydro package construction

1. `Hydro::Initialize()` reads hydro, EOS, cooling, diffusion, and thermal-solver settings.
2. It registers:
   - `cons` as the main independent evolved field
   - `prim` as a derived field
3. It stores function pointers and parameters in the Hydro package.

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:406)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1014)

### Stage 2: Initial `cons -> prim` model

`prim` is regenerated from `cons` by the EOS conversion:

- density copied from `cons`
- velocity computed from momentum / density
- pressure computed from total energy minus kinetic energy
- floors and ceilings may change `cons`

Relevant code:

- [src/eos/adiabatic_hydro.hpp](/home/ianfr/athenapk_freem386/src/eos/adiabatic_hydro.hpp:63)

This is important conceptually: syncing is not a pure read-only transformation. It may repair bad states.

## Main Loop Flow

Below is the main timestep flow in “to-do list” form.

### Per-cycle pre-step work

Before stage tasks are built, `PreStepMeshUserWorkInLoop()` computes global quantities needed by later source terms or timestep logic.

1. If GLM-MHD cleaning or diffusion is active, compute global minimum cell size `mindx`.
2. Update package parameters like:
   - `mindx`
   - `dt_hyp`
   - `dt_diff`
   - `c_h`

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:103)

Why check flags here:

- These values are needed when the driver builds the stage task graph.
- This is an early “prepare shared state before timestep execution” step.

### Stage 1 setup before the main RK/VL update

In `HydroDriver::MakeTaskCollection()`:

1. If needed, allocate register `u1`.
2. If diffusion uses RKL2, apply the first half of Strang-split STS diffusion.
3. Apply initial Strang-split problem sources.

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:442)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:497)

Important detail from the driver comments:

- these initial split sources must update both `cons` and `prim`
- the next flux calculation will use `prim`
- there is no automatic sync immediately after this point

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:510)

### For each integration stage

For each RK/VL stage, the driver does the following.

#### 1. Save stage-local bookkeeping

The driver calls `SaveStageStartInternalEnergy()`.

In legacy mode:

- if thermal diagnostics are off, this usually returns immediately
- if diagnostics are on, it stores the stage-start internal energy in scratch fields

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:562)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:251)

#### 2. Skip the pre-flux thermal predictor

The pre-flux thermal predictor only runs if:

- `thermal_source_solver_enabled == true`
- predictor is `"end_of_interface_state"`

When the coupled solver is disabled, this branch is skipped.

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:546)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:243)

#### 3. Compute fluxes from `prim`

The solver reconstructs interface states and solves Riemann problems using `prim`.

Then flux divergence updates `cons`.

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:578)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:599)

State after this step:

- `cons` is new
- `prim` still mostly reflects the old stage input state

This is one of the central ideas in the code flow.

#### 4. Record how advection changed internal energy

The driver calls `UpdateStageAdvectiveInternalEnergy()`.

In legacy mode this is mostly bookkeeping:

- compute specific internal energy from updated `cons`
- compare with the stage-start value
- store the change rate in `thermal_ae`

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:605)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:286)

#### 5. Apply unsplit source terms

The driver calls `AddUnsplitSources()` once per stage:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:613)

This is the main place where the legacy thermal source is integrated.

The order inside `AddUnsplitSources()` is:

1. GLM-MHD source term, if using `glmmhd`
2. tabular cooling through the legacy path, if enabled and not coupled
3. problem-specific unsplit source term, if installed

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:358)

### Source-term details

#### A. GLM-MHD Dedner source

If `fluid == glmmhd`, the driver applies the GLM source first.

What it reads:

- mainly `prim` for magnetic fields and `psi`
- package parameters like `c_h`, `mindx`, `glmmhd_alpha`

What it writes:

- always damps `cons(IPS)`
- in extended mode, also updates momentum and total energy in `cons`

Relevant code:

- [src/hydro/glmmhd/dedner_source.cpp](/home/ianfr/athenapk_freem386/src/hydro/glmmhd/dedner_source.cpp:17)

Important consequence:

- after this runs, `cons` is newer than `prim`
- there is still no full sync yet

#### B. Legacy tabular cooling

This is the key legacy thermal path.

Dispatch:

- `TabularCooling::SrcTerm()` selects the cooling integrator:
  - RK12 subcycling
  - RK45 subcycling
  - Townsend exact-ish update

Relevant code:

- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:278)

##### What cooling reads

Legacy cooling intentionally reads from `cons`, not `prim`, because `prim` may still contain the earlier state.

For subcycling mode:

1. read `rho = cons(IDN)`
2. reconstruct internal energy from:
   - total energy
   - minus kinetic energy
   - minus magnetic energy in MHD
3. query the cooling table using `(internal_e, rho)`

Relevant code:

- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:512)
- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:518)

The comment in the code says this explicitly:

- use `cons` because `prim` may still contain the state at `t0`

##### What cooling writes

Legacy cooling modifies:

- `cons(IEN)` directly
- `prim(IPR)` directly

Relevant code:

- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:664)
- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:668)
- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:782)
- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:785)

What it does **not** modify:

- density
- momentum
- magnetic field

##### Why `prim(IPR)` is patched manually

This is a local partial sync.

The code does this because:

- pressure is no longer consistent after changing total energy
- another task might use `prim` before the next full `FillDerived`

But this is still not a full sync:

- velocities and other primitive quantities are not recomputed here
- the real canonical sync still happens later through `ConservedToPrimitive`

#### C. Problem unsplit source terms

After legacy cooling, the driver optionally calls `ProblemSourceUnsplit`.

Example: cluster problem.

Relevant hook:

- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp:131)

Example cluster unsplit source order:

1. gravity
2. AGN feedback
3. magnetic tower fixed-field source
4. SN Ia feedback

Relevant code:

- [src/pgen/cluster.cpp](/home/ianfr/athenapk_freem386/src/pgen/cluster.cpp:63)

Gravity source behavior is a good concrete example.

What it reads:

- density from `prim`
- velocity from `prim`
- cell coordinates

What it writes:

- momentum components in `cons`
- total energy in `cons`

Relevant code:

- [src/hydro/srcterms/gravitational_field.hpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/gravitational_field.hpp:26)

This again reinforces the design pattern:

- source terms often read `prim`
- but they usually write `cons`

### After unsplit sources, before end-of-stage sync

At this point in a stage, after hyperbolic update and source terms:

- `cons` contains the newest state
- `prim` may be only partially updated
- some source terms may have patched only a subset of `prim`

This is exactly why the driver later calls `FillDerived`.

### Final stage only: split source cleanup

On the last RK/VL stage only:

1. apply final Strang-split source terms
2. apply first-order split source terms

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:619)

The driver comments again emphasize that these tasks should operate correctly even when `prim` has not yet been fully rebuilt from the latest `cons`.

### Boundary exchange

After stage-local updates, the driver exchanges boundary data:

- local/nonlocal ghost communication
- prolongation/restriction support

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:634)

### Full `cons -> prim` sync

Then the driver runs `FillDerived`, which is the main full synchronization point:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:642)

That calls the Hydro package derived-field function:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:215)

And that in turn calls EOS `ConservedToPrimitive`:

- [src/eos/adiabatic_hydro.cpp](/home/ianfr/athenapk_freem386/src/eos/adiabatic_hydro.cpp:33)

This is the main place where the code says:

- “take the updated conserved state”
- “rebuild a consistent primitive state”

It is also where floors and ceilings may repair the state.

### Optional diffusion STS cleanup

If diffusion uses RKL2, the driver applies the second half of the Strang-split STS diffusion after the final `FillDerived`.

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:649)

The STS helper explicitly assumes `cons` and `prim` are in sync on entry and on exit:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:214)

## Timestep Finalization Inside the Main Loop

After the last stage of a timestep:

1. reset global reduction vars used for later calculations
2. estimate the next timestep
3. optionally advance tracers
4. optionally tag cells for AMR refinement

Relevant code:

- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:656)
- [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:676)

### How timestep constraints are checked

`EstimateTimestep()` checks active physics and takes the minimum allowed step.

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1181)

For the legacy thermal path:

1. if tabular cooling is enabled, compute a cooling-time timestep limit
2. include it in the minimum timestep
3. record which physics set the limiter

Relevant code:

- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1187)
- [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:1242)
- [src/hydro/srcterms/tabular_cooling.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/tabular_cooling.cpp:789)

This is another place where input flags matter:

- cooling is only checked if tabular cooling is enabled
- diffusion is only checked if a diffusion integrator is active
- problem-specific timestep limits only run if a problem hook was installed

## Finalization After the Simulation Ends

At the outermost level:

1. `main()` constructs the `HydroDriver`
2. `driver.Execute()` runs the simulation loop
3. when the run finishes, Parthenon finalizes MPI and Kokkos

Relevant code:

- [src/main.cpp](/home/ianfr/athenapk_freem386/src/main.cpp:196)

## Short Summary

When the coupled thermal solver is disabled, the thermal behavior is the older source-term model:

1. hyperbolic fluxes update `cons`
2. legacy cooling runs inside `AddUnsplitSources()`
3. cooling reads `cons`, not stale `prim`
4. cooling updates `cons(IEN)` and patches `prim(IPR)`
5. later, `FillDerived` fully rebuilds `prim` from `cons`

That is the most important control-flow fact to keep in mind.

## Compact Flow Chart

```text
Initialization
  -> main() selects problem hooks
  -> Hydro::Initialize() reads flags and registers fields
  -> cons is the evolved state
  -> prim is a derived state
  -> thermal scratch fields are allocated separately

Per timestep
  -> PreStepMeshUserWorkInLoop()
     -> compute global helper quantities if needed
  -> stage 1 only:
     -> optional half-step RKL2 diffusion
     -> optional initial Strang-split sources
  -> for each stage:
     -> save thermal bookkeeping (usually no-op in legacy mode)
     -> skip pre-flux thermal predictor
     -> compute fluxes from prim
     -> update cons with flux divergence
     -> record stage advective internal energy bookkeeping
     -> AddUnsplitSources()
        -> GLM source, if enabled
        -> legacy tabular cooling, if enabled
        -> problem unsplit source, if installed
     -> last stage only:
        -> final Strang-split sources
        -> first-order split sources
     -> boundary exchange
     -> FillDerived()
        -> full cons -> prim sync
  -> after last stage:
     -> optional second half RKL2 diffusion
     -> estimate next dt
     -> optional tracers
     -> optional AMR tagging

Finalization
  -> driver.Execute() returns
  -> main() calls ParthenonFinalize()
```
