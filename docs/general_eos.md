# Singularity-EOS integration

The `general-eos-support` branch starts at official AthenaPK upstream commit
`99dd6f4bd675f02412813fd487a91c9f357f259a`. The existing `hydro/eos=adiabatic`
and `hydro/gamma` inputs select Singularity `IdealGas`. This is the only runtime
material model currently enabled. There is no legacy EOS build switch.

Initialize the dependency with:

```sh
git submodule update --init external/singularity-eos
git -C external/singularity-eos submodule update --init utils/ports-of-call
```

Singularity is pinned by its gitlink, shares Parthenon's Kokkos target, and uses
header-only analytic models. Fortran, mixed-material closures, EOSPAC, and Spiner
are disabled by default. The minimum CMake version is now 3.22, matching the pinned Kokkos requirement.

Float-field builds use float EOS arithmetic as well. A generated compatibility
header fixes one mixed float/double literal in upstream `robust::make_bounded`;
the dependency checkout itself is unchanged.

## Thermodynamic contract

`src/eos/thermodynamics.hpp` provides device-callable pressure, specific energy,
temperature, heat capacity, bulk modulus, acoustic speed, and pressure derivatives.
Material construction is centralized in hydro initialization. Fluid wrappers accept
an adapter rather than constructing an ideal gas themselves. `GetOnDevice()` and
`Finalize()` provide explicit host-side lifecycle boundaries; resource-owning
models must finalize each allocation exactly once, and copies borrow resources.
A default adapter is only an initialization placeholder. Fluid EOS constructors
now accept `GasThermodynamics`; `GetGamma()` has been removed from the wrappers.
The `AdiabaticIndex` package parameter remains for analytic/reference consumers.

The adapter calls the concrete model's generic Singularity API, including its
`InternalEnergyFromDensityPressure` method. There is no variant visitation,
virtual thermodynamic dispatch, or allocation in a cell loop.

Density, pressure, and specific energy use AthenaPK code units. Temperature is
Kelvin when units and gas composition are configured. The ideal-gas Cv is derived
from the existing `mbar_over_kb` convention; otherwise Cv=1 defines a dimensionless
temperature scale. Conserved energy is an energy density, including kinetic and,
for MHD, magnetic energy. Those contributions are removed before EOS evaluation.

The fluid wrappers retain existing floors, ceilings, scalar conversion, and
conserved-energy corrections. Raw internal energy is retained in the validity check so Singularity pressure
clamping cannot hide invalid conserved states. Riemann energies and signal speeds use the adapter;
hydro HLLC and hydro/MHD HLLE use Davis bounds instead of ideal-gas shock/Roe
formulas. This numerical change can affect dissipation and shock error norms.

Cooling rates, adaptive cooling updates, conduction temperature gradients,
cluster feedback, accretion, and thermodynamic diagnostics use the adapter.
Townsend's analytic cooling transform is retained for constant Cv; a future
variable-Cv model uses adaptive RK45 instead. Conduction retains its historical
coefficient convention using a reference gas constant times EOS temperature.
Spitzer coefficients retain the established atomic-mass convention through this
normalization, while hydrodynamic temperature retains its hydrogen-mass convention.
Disabled infinite velocity ceilings skip their unused velocity-norm calculation;
finite ceilings retain their previous arithmetic.

Analytic linear-wave eigenvectors, CPAW error references, diffusion reference
states, and the cluster astrophysical entropy proxy remain explicitly ideal-gas
reference definitions. They are not replacements for general thermodynamic
closures.

## Validation and timing

The [recorded validation results](general_eos_validation.md) describe the tested
builds, measured overhead, and unresolved regression and performance gates.

A small CPU build without output dependencies can exercise the EOS tests:

```sh
cmake -S . -B build-eos-cpu -DCMAKE_BUILD_TYPE=Release \
  -DPARTHENON_DISABLE_MPI=ON -DPARTHENON_DISABLE_HDF5=ON \
  -DPARTHENON_DISABLE_OPENPMD=ON -DAthenaPK_ENABLE_TESTING=OFF \
  -DAthenaPK_ENABLE_EOS_TESTING=ON
cmake --build build-eos-cpu -j4
ctest --test-dir build-eos-cpu -R eos_device --output-on-failure
build-eos-cpu/bin/eos_benchmark
```

For full regressions, enable AthenaPK testing and configure compatible MPI/HDF5
and Python dependencies. Run `ctest --test-dir BUILD --output-on-failure`.
Performance tests have a separate `performance` label and should be measured
on controlled hardware, not mixed with correctness checks.

The device test covers ideal-gas round trips, sound speed, derivatives, a
stiffened-gas adapter fixture, density/pressure floors, velocity/internal-energy
ceilings, magnetic-energy subtraction, and scalars. The kernel benchmark compares
legacy algebra and generic calls with device fences and an observable checksum.
Use Kokkos profiling to separate conversion, flux, timestep, and source kernels.

Build unchanged upstream and the candidate with identical compilers, optimization,
precision, Kokkos/Parthenon commits, and architecture flags. Then run:

```sh
python3 scripts/compare_eos_performance.py \
  --baseline /absolute/baseline/build/bin/athenaPK \
  --candidate /absolute/candidate/build/bin/athenaPK \
  --input inputs/linear_wave3d.in --backend cpu \
  --pairs 10 --size 64 --steps 100 --exclusive \
  --output build-eos-results/cpu.json
```

Repeat with `--backend gpu` and matching GPU builds. Use `--exclusive` only when
hardware is reserved and affinity/runtime settings are controlled. Otherwise the
runner intentionally reports inconclusive results. It alternates executable
order, warms up both, excludes five initial cycles, disables output, and retains
raw logs, binary hashes, CMake settings, paired ratios, and conservative 95%
confidence intervals. Runs shorter than two seconds are inconclusive; increase
`--steps`. The acceptance threshold is an upper confidence bound below 1% slowdown
for every case on both CPU and GPU. A microbenchmark alone cannot satisfy it.

The runner currently covers 3D hydro/MHD linear waves and two block sizes. Shock,
turbulence, thermal-source, AMR, restart, and multi-rank performance measurements
must supplement this matrix before claiming the full performance acceptance gate.

## Subsequent table work

No table reader or runtime non-ideal selection is implemented here. Enabling one
requires model construction in the selected code-unit system, validated domain
and inversion policy, explicit device transfer/finalization of table storage, and
solver tests using a real non-ideal material. The adapter fixture is not proof of
an arbitrary material's stability or admissibility.

Temperature-based limits and jet injection currently retain the ideal-gas
setup-time conversion to specific-energy limits. A density-dependent table must
apply temperature limits at each cell's density and determine injected material
energy at its actual density. The table phase must also define whether cluster
entropy profiles remain conventional pressure/density proxies or use true EOS
entropy. These requirements must be resolved before exposing tabulated EOS in
production inputs.
