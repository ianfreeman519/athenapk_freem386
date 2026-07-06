# Plan For First-Pass RKL2 + Simplified-SDC Thermal Coupling In AthenaPK

## Goal

Allow runs that use:

- `diffusion/integrator = rkl2`
- `thermal_source_solver/enabled = true`
- `thermal_source_solver/couple_ohmic_heating = true`
- the existing simplified-SDC thermal corrector structure

without double-counting resistive heating and without letting the STS magnetic update
drift out of sync with the internal-energy bookkeeping.

This plan is intentionally scoped to a first-pass implementation, not an ideal fully
time-centered coupled method. The design target is:

- STS owns magnetic resistive evolution during the split half-steps
- the thermal solver consumes an STS-aggregated ohmic volumetric source
- the legacy resistive `IEN` flux path remains disabled in the coupled mode
- the current hydro-stage simplified-SDC wrapper remains recognizable

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
  [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:497)
  and
  [src/hydro/hydro_driver.cpp](/home/ianfr/athenapk_freem386/src/hydro/hydro_driver.cpp:652)

## First-Pass 10-Step Implementation Plan

### 1. Make resistive ownership explicit

Refactor the resistive coupling code so the codepaths are explicit about which product
they own:

- magnetic resistive evolution of `B`
- thermal ohmic power source for the SDC solver

Do not leave this encoded only as a side effect of skipping `CalcDiffFluxes()` and later
calling `ApplyMagneticOnlyResistiveUpdate()`.

Expected outcome:

- one clear magnetic-only resistive operator API
- one clear thermal-source construction API

### 2. Introduce an STS-compatible magnetic-only resistive flux path

Add a path that lets `rkl2` advance magnetic resistive diffusion without ever using the
legacy resistive `IEN` flux contribution.

This likely means one of:

- a new `CalcDiffFluxesMagneticOnly(...)` entry point used by STS in the coupled mode
- a flag or mode enum passed into `CalcDiffFluxes(...)`

The requirement is strict:

- STS may update `IB1..IB3`
- STS may not update `IEN` directly when coupled ohmic heating is enabled

### 3. Keep the legacy resistive `IEN` flux path disabled in the coupled `rkl2` mode

The current coupled unsplit path is correct to avoid the legacy resistive energy-flux
update when the thermal solver is owning ohmic heating.

Preserve that same safety property for `rkl2`.

Success criterion:

- no codepath exists where STS magnetic diffusion and the legacy resistive `IEN` fluxes
  are both active under `thermal_couple_ohmic = true`

### 4. Add STS ohmic-source accumulation fields

Add dedicated fields for STS-side source bookkeeping. Suggested names:

- `thermal_src_ohmic_sts_accum`
- `thermal_src_ohmic_sts_avg`
- optional split diagnostics:
  - `thermal_src_ohmic_sts_pre`
  - `thermal_src_ohmic_sts_post`

These fields should represent volumetric sources, consistent with the existing
`thermal_src_*` semantics.

Do not overload `thermal_src_total` for this. It is already semantically overloaded
inside the current thermal iteration and should not carry a second unrelated meaning.

### 5. Accumulate a time-weighted ohmic source over each STS half-step

During the RKL2 substeps, compute the instantaneous ohmic volumetric source from the
active magnetic state and accumulate a time-weighted average over the half-step.

The important detail is that the thermal solver should consume the integrated or averaged
heating over the STS half-step, not a single-point source evaluated from one state.

This is the core bookkeeping addition that makes the first-pass `rkl2` coupling
plausible.

### 6. Feed the SDC thermal solve from the STS-owned ohmic source

In the coupled `rkl2` mode, `RunSimplifiedSDCThermalCoupling()` should stop rebuilding
`thermal_src_ohmic` directly from one stage-local state on each iteration.

Instead:

- the thermal solver should consume the STS-produced averaged ohmic source field
- the hydro-stage SDC loop should still own the cooling solve and accepted
  internal-energy commit

The first-pass target is to preserve the stage-local simplified-SDC wrapper while
changing the origin of the ohmic source it consumes.

### 7. Disable the one-shot magnetic-only resistive commit in `rkl2` mode

Once STS owns magnetic resistive evolution, the current end-of-stage call to
`ApplyMagneticOnlyResistiveUpdate(md, dt)` must not run in the coupled `rkl2` mode.

Required behavior:

- `unsplit`: keep the current end-of-stage magnetic-only commit
- `rkl2`: magnetic fields are already advanced by STS, so skip the one-shot commit

This avoids double-applying the magnetic resistive operator.

### 8. Relax the runtime guard only after the new path is real

Update the hard configuration requirement in `hydro.cpp` only after the new coupled
`rkl2` path is implemented.

Replace the current blanket rule:

- coupled ohmic heating requires `diffusion/integrator = unsplit`

with a more precise rule:

- `unsplit` is supported
- `rkl2` is supported only through the new STS-coupled path
- any unsupported combination still fails fast with a clear message

### 9. Add diagnostics that can actually prove the new bookkeeping

The current diagnostics are hydro-stage-centric:

- `eint_stage_start`
- `eint_adv`
- `eint_sdc`
- `thermal_ae`

Add STS-side diagnostics so the new source accounting can be checked directly:

- accumulated pre-hydro STS ohmic source
- accumulated post-hydro STS ohmic source
- final STS-averaged ohmic source consumed by the thermal corrector

Also add `cons_stage_start`. Without it, validation still cannot prove the full
conserved-energy accounting cleanly.

### 10. Validate in two layers

Split validation into:

- smoke/stability validation
- source-accounting validation

Smoke validation should prove:

- the coupled `rkl2` path runs
- iteration counts are sensible
- no obvious instability or negative internal energy appears

Source-accounting validation should prove:

- the STS-accumulated ohmic source is committed exactly once to thermal energy
- magnetic resistive evolution is applied exactly once
- the legacy resistive `IEN` path remains disabled

Build this on top of the existing reduced validation scripts under
[test/coupled_thermal_validation](/home/ianfr/athenapk_freem386/test/coupled_thermal_validation).

## Recommended Work Breakdown

The lowest-risk order is:

1. refactor resistive ownership APIs
2. add magnetic-only STS resistive support
3. add STS ohmic-source accumulation
4. wire the thermal solver to consume the STS-owned source
5. skip the old one-shot magnetic commit in `rkl2`
6. add diagnostics and validation
7. only then relax the runtime guard

## Copy-Pasteable Prompts For Fresh Codex Sessions

Each prompt below is meant to be pasted into a new Codex session directly.

### Prompt 1: Code Audit And Design Extraction

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: audit the live implementation relevant to adding `diffusion/integrator = rkl2`
support for the simplified-SDC thermal coupling path with coupled Spitzer ohmic heating.

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
- what fields already exist that could be reused for STS-side ohmic-source accumulation
- what configuration guard currently blocks coupled ohmic heating with `rkl2`

Deliver:
- a concise design note
- exact file/line references
- a list of the smallest refactors needed before code changes begin
```

### Prompt 2: Refactor Resistive Ownership APIs

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: make the resistive ownership boundaries explicit for a future coupled
`rkl2` + simplified-SDC implementation.

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

Deliver:
- code changes
- a concise explanation of how STS now gets magnetic-only resistive fluxes
- note any remaining gaps before the thermal solver can consume STS-owned ohmic heating
```

### Prompt 4: Add STS Ohmic-Source Accumulation

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: add first-pass STS-side ohmic-source accumulation fields and bookkeeping so the
simplified-SDC thermal solver can later consume an STS-averaged ohmic volumetric source.

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

Goals:
- add new Hydro fields for STS-side ohmic-source accumulation
- accumulate a time-weighted ohmic source over each STS half-step
- avoid reusing `thermal_src_total` for this purpose

Deliver:
- code changes
- a short explanation of the accumulation semantics
- exact note on how the new fields should be interpreted in validation
```

### Prompt 5: Wire Simplified-SDC To Consume STS-Owned Ohmic Source

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: in the coupled `rkl2` mode, change the simplified-SDC thermal solver so it
consumes the STS-produced ohmic volumetric source instead of rebuilding the ohmic source
directly from one stage-local state on each iteration.

Requirements:
- inspect the live implementation first
- preserve the current unsplit coupled path
- do not remove existing diagnostics unless their semantics become invalid

Focus on:
- `src/hydro/srcterms/internal_energy_solver.cpp`
- `src/hydro/srcterms/internal_energy_solver.hpp`
- any supporting Hydro field declarations in `src/hydro/hydro.cpp`

Goals:
- branch on diffusion integrator mode
- keep the stage-local thermal corrector recognizable
- use STS-owned ohmic source input in `rkl2`
- ensure the magnetic-only one-shot resistive commit is skipped in `rkl2`

Deliver:
- code changes
- concise explanation of the new `unsplit` vs `rkl2` behavior
- note any unresolved temporal-accuracy limitations
```

### Prompt 6: Relax Runtime Guard And Add Diagnostics

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: now that the coupled `rkl2` path exists, relax the runtime guard that currently
forbids `thermal_couple_ohmic` with `diffusion/integrator = rkl2`, and add the
diagnostics needed to validate source accounting.

Requirements:
- inspect the live implementation first
- only relax the guard if the codepath is actually implemented
- fail fast for any still-unsupported combinations

Focus on:
- `src/hydro/hydro.cpp`
- `src/hydro/hydro_driver.cpp`
- `src/hydro/srcterms/internal_energy_solver.cpp`
- validation scripts under `test/coupled_thermal_validation/`

Deliver:
- code changes
- the exact new runtime rules
- any new diagnostics fields and how to interpret them
- validation commands or scripts you updated
```

### Prompt 7: Validation Audit For Coupled RKL2 + SDC

```text
You are working in my local AthenaPK repository at `/home/ianfr/athenapk_freem386`.

Task: audit and extend the reduced validation coverage for the first-pass coupled
`rkl2` + simplified-SDC implementation.

Requirements:
- inspect the live implementation first
- use the existing reduced validation scripts under `test/coupled_thermal_validation/`
- distinguish smoke-test stability from source-accounting correctness

Answer from code and outputs:
- whether the STS-accumulated ohmic source is committed exactly once
- whether magnetic resistive evolution is applied exactly once
- whether the legacy resistive `IEN` path remains disabled
- what the new diagnostics prove
- what remains ambiguous without `cons_stage_start`

Deliver:
- a direct audit note
- file/line references for the code claims
- any script changes needed to keep these checks reproducible
```

## Non-Goals For This First Pass

These are intentionally out of scope unless the implementation forces them:

- interleaving a thermal solve inside every individual RKL2 substep
- redesigning the thermal solver into a generic substep-local corrector
- implementing coupled conduction or viscous heating
- proving high-order temporal accuracy of the new `rkl2` coupling
- removing the known ambiguity caused by the lack of `cons_stage_start`, except if that
  snapshot is easy to add during the diagnostics work

## Practical Success Criteria

The first pass is good enough if all of the following are true:

- I can run with `diffusion/integrator = rkl2` and the simplified-SDC thermal solver
  enabled
- the run does not double-apply magnetic resistive evolution
- the run does not double-count ohmic heating through the legacy resistive `IEN` path
- the thermal solver consumes a source that actually represents the STS half-step
  resistive evolution better than a single state-local source would
- the reduced validation scripts can distinguish stability from accounting correctness
