"""Reference mean-charge free-free rates for explicit plasma states.

This is a reference calculator, not a temperature-only cooling table generator.
Input CSV columns: density [g cm^-3], temperature [K], mean charge [dimensionless].
Output adds emissivity [erg cm^-3 s^-1] and specific energy loss [erg g^-1 s^-1].
The constant frequency-integrated Gaunt factor and mean ion mass must be supplied.
"""

import argparse

import numpy as np

# Match AthenaPK's Units atomic mass unit so code-unit comparisons are exact.
ATOMIC_MASS_UNIT_G = 1.660538921e-24


def free_free_emissivity(density_cgs, temperature_K, zbar, *, mean_ion_mass, gaunt_factor):
    """Return Q_ff in erg cm^-3 s^-1, broadcasting over explicit state arrays.

    Single-component, optically thin, single-temperature mean-charge model:
    Q = 1.426e-27 gB Zbar^2 n_e n_i sqrt(T), n_e = Zbar n_i.
    This replaces the charge-resolved second moment by Zbar^2.
    """
    density, temperature, charge = np.broadcast_arrays(
        np.asarray(density_cgs, dtype=float),
        np.asarray(temperature_K, dtype=float),
        np.asarray(zbar, dtype=float),
    )
    if not np.isfinite(mean_ion_mass) or mean_ion_mass <= 0:
        raise ValueError("mean_ion_mass must be finite and positive")
    if not np.isfinite(gaunt_factor) or gaunt_factor <= 0:
        raise ValueError("gaunt_factor must be finite and positive")
    if not (
        np.isfinite(density).all() and (density > 0).all()
        and np.isfinite(temperature).all() and (temperature > 0).all()
        and np.isfinite(charge).all() and (charge >= 0).all()
    ):
        raise ValueError("States require finite density>0, temperature>0, and zbar>=0")
    ni = density / (mean_ion_mass * ATOMIC_MASS_UNIT_G)
    ne = charge * ni
    return 1.426e-27 * gaunt_factor * charge**2 * ne * ni * np.sqrt(temperature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", help="CSV with density_cgs, temperature_K, zbar columns")
    parser.add_argument("output", help="Output CSV of states, emissivity, specific loss")
    parser.add_argument("--mean-ion-mass", type=float, required=True, help="A in units of m_u")
    parser.add_argument("--gaunt-factor", type=float, required=True)
    args = parser.parse_args()
    states = np.loadtxt(args.states, delimiter=",", ndmin=2)
    if states.shape[1] != 3:
        parser.error("States CSV must contain exactly three columns: density, T, Zbar")
    try:
        q = free_free_emissivity(
            *states.T, mean_ion_mass=args.mean_ion_mass, gaunt_factor=args.gaunt_factor
        )
    except ValueError as error:
        parser.error(str(error))
    np.savetxt(
        args.output, np.column_stack((states, q, -q / states[:, 0])), delimiter=",",
        header="rho[g/cm^3],T[K],Zbar,Q[erg/cm^3/s],deps_dt[erg/g/s]",
    )


if __name__ == "__main__":
    main()
