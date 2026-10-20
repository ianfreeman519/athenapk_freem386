# Tabulated equation of state

AthenaPK supports the version-1 three-chart HDF5 EOS format through
`hydro/eos = tabulated`. The thermodynamic primitive remains pressure, while the
inverse charts provide direct pressure-to-energy and temperature-to-energy mappings.
No iterative inversion is performed at runtime.

The required groups are:

```text
/eos_rho_e/axes/log10_density
/eos_rho_e/axes/log10_specific_internal_energy
/eos_rho_e/log10_pressure
/eos_rho_e/log10_temperature
/eos_rho_e/log10_sound_speed
/eos_rho_e/Zbar
/eos_rho_p/axes/log10_density
/eos_rho_p/axes/log10_pressure
/eos_rho_p/log10_specific_internal_energy
/eos_rho_T/axes/log10_density
/eos_rho_T/axes/log10_temperature
/eos_rho_T/log10_specific_internal_energy
```

All two-dimensional datasets use `[density_index, thermodynamic_coordinate_index]`.
Axes must be uniform in base-10 logarithmic space. Version 1 uses kg/m^3, J/kg, Pa,
m/s, and eV, with the exact units recorded in `linear_units` attributes. AthenaPK
normalizes the logarithmic arrays once during loading.

An input block must provide the table and normalization:

```text
<tabulated_eos>
filename = inputs/tables/tabulated_eos.h5
density_to_code = 1.0
specific_internal_energy_to_code = 1.0
pressure_to_code = 1.0
sound_speed_to_code = 1.0
temperature_to_kelvin = 11604.51812155008
mean_ion_mass_hydrogen = 27.0
```

The factors multiply decoded table values. `mean_ion_mass_hydrogen` supplies the
effective ion mass `A`; cell-local composition is `mu=A/(1+Zbar)` and `mu_e=A/Zbar`.

Interpolation and extrapolation are bilinear in stored logarithmic space. Excursions
within one decade warn at a host synchronization point. Larger excursions terminate
after device diagnostics have been reduced. The table fingerprint is inserted into
the parameter stream and verified when restarting.

The first supported evolution combinations are Euler-HLLE, GLM-MHD-HLLE, and
UCT-HLLE MHD. HLLC and HLLD remain ideal-gas-only. Adaptive tabular cooling and
analytic free-free cooling query the EOS at each integration stage. See
[cooling notes](cooling_notes.md#eos-aware-free-free-cooling) for the mean-charge
free-free model and its required Gaunt factor.

The sample [`tabulated_sod.in`](../inputs/tabulated_sod.in) uses SI-scaled code units
and the aluminum table in `inputs/tables/tabulated_eos.h5`.

After configuring and building AthenaPK, run the focused regression from the build
directory with:

```text
ctest -R 'regression(_mpi)?_test:tabulated_eos' --output-on-failure
```

The test validates the on-disk contract, grid-point and midpoint interpolation,
edge-pair extrapolation, eV-to-kelvin conversion, both inverse charts, derived
composition fields, and a one-step Euler-HLLE evolution. To run the fixture directly:

```text
./bin/athenaPK -i ../inputs/tabulated_sod.in \
  tabulated_eos/filename=../inputs/tables/tabulated_eos.h5
```

Adjust the relative paths if the build directory is not immediately below the source
tree. Table filenames are resolved relative to AthenaPK's working directory, not the
input file. The fatal more-than-one-decade diagnostic is exercised by setting a Sod density
or pressure beyond the declared chart margin; this is expected to terminate after the
next EOS diagnostic checkpoint.
