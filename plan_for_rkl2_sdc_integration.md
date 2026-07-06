# Plan For First-Pass RKL2 + Simplified-SDC Thermal Coupling In AthenaPK

## Goal

Allow runs that use:

- `diffusion/integrator = rkl2`
- `thermal_source_solver/enabled = true`
- `thermal_source_solver/couple_ohmic_heating = true`
- the existing simplified-SDC thermal corrector structure

without double-counting resistive heating and without double-applying magnetic
resistive evolution.

This plan is intentionally scoped to a first-pass implementation, not an ideal fully
time-centered coupled method. The design target is:

- STS owns magnetic resistive evolution during the split half-steps
- the midpoint thermal solve consumes only the pre-hydro STS half-step ohmic source
- the legacy resistive `IEN` flux path remains disabled in the coupled mode
- the current hydro-stage simplified-SDC wrapper remains recognizable

The key design choice for this first pass is deliberate:

- pre-hydro `rkl2` half-step on `B`
- one midpoint thermal solve using the pre-hydro STS-produced ohmic source
- hydro stage as today
- post-hydro `rkl2` half-step on `B`

This preserves both live-code assumptions that currently matter most:

- the thermal solve still consumes a frozen source over its own solve
- STS still runs with no other code touching the fluid state during its subcycles

## Current Constraints In Live Code

These are the live-code constraints this plan is written against:

- coupled ohmic heating currently hard-requires `diffusion/integrator = unsplit` in
  [src/hydro/hydro.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro.cpp:964)
- when coupled ohmic heating is enabled, `CalcDiffFluxes()` skips the standard
  resistive diffusion path in
  [src/hydro/diffusion/diffusion.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/diffusion.cpp:18)
- the simplified-SDC thermal path currently commits:
  - one accepted internal-energy correction per hydro stage
  - one magnetic-only resistive update per hydro stage
  in
  [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:232)
- `rkl2` STS runs as separate split diffusion half-steps before stage 1 and after the
  final hydro stage in
  [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:500)
  and
  [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:652)
- `AddSTSTasks()` currently assumes `prim` and `cons` are in sync on entry and exit in
  [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:216)
- the current simplified-SDC implementation rebuilds `thermal_src_ohmic` from a
  stage-local state on each iteration instead of consuming an externally accumulated STS
  source in
  [src/hydro/srcterms/internal_energy_solver.cpp](/home/ianfr/athenapk_freem386/src/hydro/srcterms/internal_energy_solver.cpp:251)
- a magnetic-only resistive flux operator already exists as
  `AddMagneticOnlyResistiveFlux()` in
  [src/hydro/diffusion/resistivity.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/resistivity.cpp:452)
- the existing thermal validation is stage-centric and explicitly notes that no
  `cons_stage_start` snapshot exists yet in
  [test/coupled_thermal_validation/validate_sdc_pre_flux_source.py](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation/validate_sdc_pre_flux_source.py:100)

## Known First-Pass Limitation

This midpoint design is feasible, but it is not a fully centered coupled method.

The midpoint thermal solve can only consume heating from the first STS half-step,
because the second STS half-step has not happened yet when the thermal correction is
committed. That means:

- the same-step thermal correction is one-sided in the resistive split ordering
- final `IEN` and final `B` are not perfectly synchronized over the full timestep
- if Spitzer resistivity depends strongly on temperature, the coupling is only
  piecewise-frozen at the half-step level

This is acceptable for the first pass as long as the code and diagnostics describe the
behavior honestly and do not pretend the midpoint solve consumed a full-step STS source.

## First-Pass 10-Step Implementation Plan

### 1. Make resistive ownership explicit

Refactor the resistive coupling code so the codepaths are explicit about which product
they own:

- magnetic resistive evolution of `B`
- thermal ohmic power source for the midpoint SDC solve

Do not leave this encoded only as a side effect of skipping `CalcDiffFluxes()` and later
calling `ApplyMagneticOnlyResistiveUpdate()`.

Expected outcome:

- one clear magnetic-only resistive operator API
- one clear thermal-source construction API

### 2. Route coupled `rkl2` STS through the existing magnetic-only resistive operator

Add a path that lets `rkl2` advance resistive magnetic diffusion without ever using the
legacy resistive `IEN` flux contribution.

The live code already provides the right low-level operator:

- `AddMagneticOnlyResistiveFlux()` in
  [src/hydro/diffusion/resistivity.cpp](/home/ianfr/athenapk_freem386/src/hydro/diffusion/resistivity.cpp:452)

The smallest intended refactor is therefore not to invent a new resistive kernel, but to
let STS dispatch to a magnetic-only diffusion-flux path in the coupled mode. That could
be:

- a new `CalcDiffFluxesMagneticOnly(...)` entry point used only by STS when coupled
  ohmic heating is active
- or a narrow mode flag passed into `CalcDiffFluxes(...)`

The requirement is strict:

- STS may update `IB1..IB3`
- STS may not update `IEN` directly when coupled ohmic heating is enabled
- STS must still leave `prim` and `cons` synchronized at the end, because the current
  task graph assumes that property

### 3. Keep the legacy resistive `IEN` flux path disabled in the coupled `rkl2` mode

The current coupled unsplit path already enforces the correct ownership split:

- `CalcDiffFluxes()` skips the standard resistive branch entirely when
  `thermal_source_solver_enabled && thermal_couple_ohmic`
- the coupled thermal path owns ohmic heating separately
- the magnetic field is committed separately through a magnetic-only resistive update

Preserve that same safety property for `rkl2`.

Success criterion:

- no codepath exists where coupled `rkl2` STS advances `B` and the legacy resistive
  `cons.flux(..., IEN, ...)` contribution is also active
- the coupled path remains the sole owner of ohmic thermal source construction

### 4. Reuse existing thermal bookkeeping fields first; add new STS fields only if needed

The live code already has stage-local source bookkeeping fields:

- `thermal_src_ohmic_pre_flux`
- `thermal_src_ohmic`
- `thermal_src_pre_flux`
- `thermal_src_lagged`
- `thermal_src_total`

and stage snapshots:

- `eint_stage_start`
- `eint_adv`
- `eint_sdc`

Use those existing fields first unless they prove semantically insufficient for the STS
split bookkeeping. In particular:

- `thermal_src_ohmic_pre_flux` is the natural reusable slot for the pre-hydro STS ohmic
  source consumed by the first-pass midpoint solve
- `thermal_src_ohmic` is the natural candidate for the accepted or post-STS ohmic source
- `thermal_src_total` should remain the thermal-solver-owned combined source field, not a
  generic STS scratch field

Only add dedicated `*_sts_*` fields if the reuse path becomes too ambiguous or creates
ownership confusion in the task graph.

### 5. Accumulate or reconstruct the pre-hydro STS ohmic source with clear first-pass semantics

During the RKL2 substeps, produce an ohmic volumetric source associated with the
pre-hydro STS half-step.

The live code already has a thermal-source builder, `BuildOhmicThermalSource(...)`,
that reconstructs the ohmic thermal source from `prim` plus a supplied internal-energy
field. Reuse that construction path if possible rather than building a second
incompatible source definition.

For the first pass, the important requirement is semantic clarity, not perfect temporal
centering:

- the midpoint thermal solve must consume a source representing the pre-hydro STS
  half-step
- the code must track the later post-hydro STS source distinctly
- the implementation must not pretend the midpoint solve consumed a full-step
  STS-averaged source if it did not

Whether the first implementation stores a true time-weighted half-step average or a
single accepted half-step reconstruction is secondary to making that meaning explicit in
code and diagnostics.

### 6. Insert one midpoint thermal solve after the pre-hydro STS half-step

In the coupled `rkl2` mode, add one thermal solve immediately after the first
`AddSTSTasks(..., 0.5 * dt)` call and before the hydro stage begins.

This midpoint solve should:

- consume the pre-hydro STS-produced ohmic source field
- preserve the existing simplified-SDC thermal iteration structure as much as practical
- keep the thermal solve frozen with respect to the source it consumes
- avoid violating the current STS assumption that no other code mutates the active state
  during the STS subcycles themselves

This midpoint solve should not claim to represent the post-hydro STS half-step. That
source belongs to a later point in the split ordering and is not available when the
midpoint thermal correction is applied.

### 7. Keep the hydro stage recognizable, but branch the coupled thermal behavior in `rkl2`

Once the midpoint thermal solve exists, the current hydro-stage coupled thermal path must
stop owning coupled ohmic heating in the same way it does for `unsplit`.

Required behavior:

- `unsplit`: keep the current end-of-stage simplified-SDC coupled path, including the
  final magnetic-only resistive commit
- `rkl2`: use the new midpoint thermal solve fed by the pre-hydro STS source
- `rkl2`: do not also rebuild and consume an end-of-stage ohmic source through the old
  stage-local coupled path if that would duplicate the midpoint source semantics
- `rkl2`: keep the hydro stage otherwise recognizable, including the existing
  stage-start bookkeeping and pre-flux predictor structure where still useful

This step is where the first-pass design must be explicit about the limitation:

- the midpoint solve uses only the pre-hydro STS source
- the post-hydro STS source is tracked separately and is not same-step consumed by that
  midpoint solve

### 8. Disable the one-shot magnetic-only resistive commit in `rkl2` mode

Once STS owns magnetic resistive evolution, the current end-of-stage call to
`ApplyMagneticOnlyResistiveUpdate(md, dt)` must not run in the coupled `rkl2` mode.

Required behavior:

- `unsplit`: keep the current end-of-stage magnetic-only commit
- `rkl2`: magnetic fields are already advanced by STS, so skip the one-shot commit

This avoids double-applying the magnetic resistive operator.

### 9. Relax the runtime guard only after the new path is real and instrumented

Update the hard configuration requirement in `hydro.cpp` only after the new coupled
`rkl2` path is implemented.

Replace the current blanket rule:

- coupled ohmic heating requires `diffusion/integrator = unsplit`

with a more precise rule:

- `unsplit` is supported
- `rkl2` is supported only through the new midpoint-thermal path
- any unsupported combination still fails fast with a clear message

Do not relax this guard until diagnostics exist that can distinguish:

- pre-hydro STS source produced
- midpoint source consumed
- post-hydro STS source produced but not same-step consumed

### 10. Add diagnostics and validate in two layers

Split validation into:

- smoke/stability validation
- source-accounting validation

Smoke validation should prove:

- the coupled `rkl2` path runs
- iteration counts are sensible
- no obvious instability or negative internal energy appears

Source-accounting validation should prove:

- the pre-hydro STS-accumulated ohmic source is committed exactly once to thermal energy
- magnetic resistive evolution is applied exactly once
- the legacy resistive `IEN` path remains disabled
- the post-hydro STS source is tracked distinctly and not silently treated as already
  consumed by the midpoint thermal solve

Build this on top of the existing reduced validation scripts under
[test/simplified_sdc_reduced_validation](/home/ianfr/athenapk_freem386/test/simplified_sdc_reduced_validation)
and the more detailed checks under
[test/coupled_thermal_validation](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation).

Also add `cons_stage_start` if it is cheap enough. Without it, validation still cannot
prove the full conserved-energy accounting cleanly.

## Recommended Work Breakdown

The lowest-risk order is:

1. refactor resistive ownership APIs
2. add magnetic-only STS resistive support in coupled `rkl2`
3. add pre/post STS ohmic-source accumulation fields
4. insert the midpoint thermal solve and wire it to the pre-hydro STS source
5. skip the old one-shot magnetic commit in `rkl2`
6. add diagnostics and validation
7. only then relax the runtime guard

## Copy-Pasteable Prompts For Fresh Codex Sessions

Each prompt below is meant to be pasted into a new Codex session directly.

### Prompt 1: Code Audit And Midpoint Design Extraction

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: audit the live implementation relevant to adding `diffusion/integrator = rkl2`
support for the simplified-SDC thermal coupling path with coupled Spitzer ohmic heating,
using a first-pass single-midpoint thermal solve design.

Focus first on these files:
- `src/hydro/srcterms/internal_energy_solver.cpp`
- `src/hydro/diffusion/diffusion.cpp`
- `src/hydro/diffusion/resistivity.cpp`
- `src/hydro/hydro.cpp`
- `src/hydro/hydro_driver.cpp`

Answer from code:
- where magnetic resistive evolution is currently applied in the coupled unsplit path
- where the legacy resistive `IEN` flux path is disabled
- where the `rkl2` STS tasks are launched and what they assume about `cons` / `prim`
- whether a magnetic-only resistive operator already exists and how STS could reuse it
- what fields already exist that could be reused or extended for pre/post STS ohmic-source bookkeeping
- what configuration guard currently blocks coupled ohmic heating with `rkl2`

Deliver:
- a concise design note
- exact file/line references
- a list of the smallest refactors needed before code changes begin
- a note on the first-pass limitation that the midpoint thermal solve can only consume the pre-hydro STS source
```

### Prompt 2: Refactor Resistive Ownership APIs

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: make the resistive ownership boundaries explicit for a future coupled
`rkl2` + simplified-SDC implementation that uses a single midpoint thermal solve.

Requirements:
- inspect the live implementation first
- prefer minimal targeted changes
- do not change runtime behavior yet

Focus on:
- `src/hydro/diffusion/diffusion.cpp`
- `src/hydro/diffusion/diffusion.hpp`
- `src/hydro/diffusion/resistivity.cpp`
- `src/hydro/srcterms/internal_energy_solver.cpp`

Goal:
- separate the concepts of:
  - magnetic-only resistive evolution
  - thermal ohmic source construction
- prepare the code for an STS path that advances `B` without touching `IEN`
- prepare the code for a midpoint thermal solve that consumes a pre-hydro STS source

Deliver:
- the code changes
- a short explanation of the new API boundaries
- any behavior-preservation checks you ran
```

### Prompt 3: Add Magnetic-Only STS Resistive Support

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: implement the first mechanical step needed for coupled `rkl2` + simplified-SDC:
allow the STS path to advance magnetic resistive diffusion without using the legacy
resistive `IEN` flux contribution.

Requirements:
- inspect the live implementation first
- preserve existing behavior for non-coupled runs
- keep the legacy resistive `IEN` path disabled when coupled ohmic heating is enabled

Focus on:
- `src/hydro/hydro_driver.cpp`
- `src/hydro/diffusion/diffusion.cpp`
- `src/hydro/diffusion/diffusion.hpp`
- `src/hydro/diffusion/resistivity.cpp`

Use the explicit resistive ownership APIs that now exist in the live code:
- `AddOhmicResistiveFlux(...)`
- `AddMagneticOnlyResistiveFlux(...)`
- `BuildOhmicThermalSource(...)`

Deliver:
- code changes
- a concise explanation of how STS now gets magnetic-only resistive fluxes
- note any remaining gaps before the midpoint thermal solver can consume STS-owned ohmic heating
```

### Prompt 4: Add Pre/Post STS Ohmic-Source Accumulation

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: add first-pass STS-side ohmic-source accumulation fields and bookkeeping so the
single midpoint thermal solve can consume the pre-hydro STS half-step source, while the
post-hydro source is still tracked separately.

Requirements:
- inspect the live implementation first
- keep changes minimal and targeted
- preserve current unsplit behavior
- keep `thermal_src_*` fields volumetric and `eint_*` fields specific energies

Focus on:
- `src/hydro/hydro.cpp`
- `src/hydro/hydro_driver.cpp`
- `src/hydro/diffusion/resistivity.cpp`
- `src/hydro/diffusion/diffusion.cpp`
- `src/hydro/srcterms/internal_energy_solver.cpp`

Goals:
- reuse existing thermal bookkeeping fields first, especially `thermal_src_ohmic_pre_flux`
- assume the STS magnetic-only resistive flux path from Prompt 3 is already in place;
  this prompt is about ohmic-source bookkeeping, not reworking STS magnetic evolution
- only add new Hydro fields if the existing fields prove semantically insufficient
- produce clearly named pre-hydro and post-hydro STS ohmic-source bookkeeping
- either accumulate a time-weighted source over each STS half-step or reconstruct an accepted half-step source, but make the semantics explicit in code and diagnostics
- avoid reusing `thermal_src_total` for STS bookkeeping
- document clearly that the pre-hydro source is the one consumed by the midpoint solve in this first pass

Deliver:
- code changes
- a short explanation of the accumulation semantics
- exact note on how the new fields should be interpreted in validation
```

### Prompt 5: Insert The Midpoint Thermal Solve

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: in the coupled `rkl2` mode, insert the first-pass midpoint thermal solve after the
pre-hydro STS half-step and before the hydro stage, using the already-bookkept
pre-hydro STS ohmic source.

Requirements:
- inspect the live implementation first
- preserve the current unsplit coupled path
- do not remove existing diagnostics unless their semantics become invalid
- keep the first-pass limitation explicit: the midpoint solve does not consume the post-hydro STS source
- start from the live bookkeeping now present:
  - `thermal_src_ohmic_pre_flux` is the accepted pre-hydro STS half-step source
  - `thermal_src_ohmic_post_hydro_sts` is the accepted post-hydro STS half-step source
  - both fields are reconstructed accepted source snapshots, not time-integrated energy increments

Focus on:
- `src/hydro/hydro_driver.cpp`
- `src/hydro/srcterms/internal_energy_solver.cpp`
- `src/hydro/srcterms/internal_energy_solver.hpp`
- any supporting Hydro field declarations in `src/hydro/hydro.cpp`

Use the explicit ownership split already present in the live code:
- STS-side magnetic evolution should go through the magnetic-only resistive path
- thermal ohmic source construction should stay thermal-solver-owned via `BuildOhmicThermalSource(...)`

Goals:
- branch on diffusion integrator mode
- keep the simplified-SDC thermal corrector recognizable
- use `thermal_src_ohmic_pre_flux` as the `rkl2` midpoint-solve ohmic input
- do not reinterpret `thermal_src_ohmic_post_hydro_sts` as same-step consumed heat
- do not reintroduce implicit ownership where the hydro-stage path rebuilds and owns the same ohmic source semantics that the midpoint path already consumed
- skip any old one-shot resistive source rebuild/consume path if it would double-count the midpoint source semantics
- ensure the magnetic-only one-shot resistive commit is skipped in `rkl2`

Deliver:
- code changes
- concise explanation of the new `unsplit` vs `rkl2` behavior
- note explicitly that the midpoint solve consumes the accepted pre-hydro half-step source snapshot, while the post-hydro source remains bookkeeping only in this first pass
- note any unresolved temporal-accuracy limitations
```

### Prompt 6: Relax Runtime Guard And Add Diagnostics

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: audit the now-implemented coupled `rkl2` midpoint path, confirm that the runtime
guard has been relaxed correctly, tighten any still-unsupported combinations if needed,
and add the diagnostics needed to validate source accounting.

Requirements:
- inspect the live implementation first
- confirm the live runtime rules before changing them
- only change the runtime guard if the live code still permits an unsupported combination
- fail fast for any still-unsupported combinations
- preserve the current field semantics:
  - `thermal_src_*` are volumetric sources
  - `eint_*` are specific energies
  - `thermal_src_ohmic_pre_flux` is the accepted pre-hydro STS source consumed by the midpoint solve
  - `thermal_src_ohmic_post_hydro_sts` is the accepted post-hydro STS source produced for bookkeeping

Focus on:
- `src/hydro/hydro.cpp`
- `src/hydro/hydro_driver.cpp`
- `src/hydro/srcterms/internal_energy_solver.cpp`
- validation scripts under `test/simplified_sdc_reduced_validation/`
- validation scripts under `test/coupled_thermal_validation/`

Deliver:
- code changes
- the exact live runtime rules after your audit
- any new diagnostics fields and how to interpret them
- validation commands or scripts you updated
- a note that pre-hydro STS source consumed and post-hydro STS source produced are intentionally distinct first-pass diagnostics
- a note that the STS diagnostics are reconstructed accepted half-step sources, not substep-integrated source accumulators
```

### Prompt 7: Validation Audit For Coupled RKL2 + Midpoint SDC

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: audit and extend the reduced validation coverage for the first-pass coupled
`rkl2` + simplified-SDC implementation that uses a single midpoint thermal solve.

Requirements:
- inspect the live implementation first
- use the existing reduced validation scripts under `test/simplified_sdc_reduced_validation/`
- use the more detailed checks under `test/coupled_thermal_validation/`
- distinguish smoke-test stability from source-accounting correctness
- confirm the exact live `rkl2` thermal runtime rule before proposing validation changes:
  - supported: `diffusion/integrator = rkl2` with
    `thermal_source_solver/enabled = true`,
    `thermal_source_solver/couple_cooling = true`, and
    `thermal_source_solver/couple_ohmic_heating = true`
  - any other thermal-enabled `rkl2` combination should fail fast and be treated as
    unsupported validation scope
- explicitly validate the live `rkl2` branch behavior:
  - the midpoint thermal solve runs after the pre-hydro STS half-step and before hydro
  - the hydro-stage coupled thermal source path is skipped in `rkl2`
  - the magnetic-only one-shot resistive commit is skipped in `rkl2`

Answer from code and outputs:
- whether the accepted pre-hydro STS half-step ohmic source in `thermal_src_ohmic_pre_flux` is consumed exactly once
- whether magnetic resistive evolution is applied exactly once
- whether the legacy resistive `IEN` path remains disabled
- what the new diagnostics prove
- how the accepted post-hydro STS source is tracked in `thermal_src_ohmic_post_hydro_sts` in this first pass
- whether `thermal_src_ohmic_sts_delta` correctly reports
  `thermal_src_ohmic_post_hydro_sts - thermal_src_ohmic_pre_flux`
- whether validation is treating the STS diagnostics correctly as volumetric accepted source snapshots rather than integrated energy increments
- what remains ambiguous without `cons_stage_start`

Deliver:
- a direct audit note
- file/line references for the code claims
- any script changes needed to keep these checks reproducible
- make sure `validate_sdc_pre_flux_source.py` is used only for cases that actually
  exercise the consumed pre-flux bookkeeping identity
```

## Non-Goals For This First Pass

These are intentionally out of scope unless the implementation forces them:

- interleaving a thermal solve inside every individual RKL2 substep
- redesigning the thermal solver into a generic substep-local corrector
- implementing a two-half-step thermal solve wrapped around hydro
- implementing coupled conduction or viscous heating
- proving high-order temporal accuracy of the new `rkl2` coupling
- pretending the midpoint thermal solve consumed a full-step STS-averaged source
- removing the known ambiguity caused by the lack of `cons_stage_start`, except if that
  snapshot is easy to add during the diagnostics work

## Practical Success Criteria

The first pass is good enough if all of the following are true:

- I can run with `diffusion/integrator = rkl2` and the simplified-SDC thermal solver
  enabled
- the run does not double-apply magnetic resistive evolution
- the run does not double-count ohmic heating through the legacy resistive `IEN` path
- the midpoint thermal solver consumes a source that represents the pre-hydro STS
  half-step better than a single stage-local state-local source would
- the diagnostics can distinguish:
  - pre-hydro STS source produced
  - midpoint source consumed
  - post-hydro STS source produced
- the reduced validation scripts can distinguish stability from accounting correctness
