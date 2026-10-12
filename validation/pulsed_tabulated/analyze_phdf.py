#!/usr/bin/env python3
"""Summarize tabulated-EOS pulsed-reconnection PHDF outputs."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def finite_summary(name, values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    if finite.any():
        print(f"{name}: min={values[finite].min():.16e} max={values[finite].max():.16e} "
              f"nan={np.isnan(values).sum()} inf={np.isinf(values).sum()}")
    else:
        print(f"{name}: no finite values nan={np.isnan(values).sum()} inf={np.isinf(values).sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("outputs", nargs="+", type=Path)
    args = parser.parse_args()

    with h5py.File(args.table) as table:
        lr = table["eos_rho_e/axes/log10_density"][:]
        le = table["eos_rho_e/axes/log10_specific_internal_energy"][:]
        pressure = RegularGridInterpolator(
            (lr, le), table["eos_rho_e/log10_pressure"][:], bounds_error=False,
            fill_value=None)
        temperature = RegularGridInterpolator(
            (lr, le), table["eos_rho_e/log10_temperature"][:], bounds_error=False,
            fill_value=None)
        bounds = (lr[0], lr[-1], le[0], le[-1])

    for path in args.outputs:
        print(f"\nFILE {path}")
        with h5py.File(path) as data:
            info = data["Info"].attrs
            print(f"time={info.get('Time')} cycle={info.get('NCycle')}")
            cons = data["cons"][:]
            prim = data["prim"][:]
            rho = cons[:, 0]
            kinetic = np.sum(cons[:, 1:4] ** 2, axis=1) / (2.0 * rho)
            magnetic = 0.5 * np.sum(cons[:, 5:8] ** 2, axis=1)
            material = cons[:, 4] - kinetic - magnetic
            eps = material / rho
            points = np.column_stack((np.log10(rho.ravel() / 1.0e-3),
                                      np.log10(eps.ravel() / 1.0e-10)))
            p_ref = 10.0 ** pressure(points) * 1.0e-13
            t_ref = 10.0 ** temperature(points) * 11604.51812155008

            for name, values in (
                ("rho", rho), ("pressure", prim[:, 4]),
                ("temperature", data["eos_temperature"][:]),
                ("sound_speed", data["eos_sound_speed"][:]),
                ("zbar", data["eos_zbar"][:]), ("material_energy", material),
                ("specific_internal_energy", eps), ("total_energy", cons[:, 4]),
                ("divB", data["divB"][:])):
                finite_summary(name, values)
            print("EOS coordinates: "
                  f"log10(rho_SI)=[{points[:, 0].min():.16e},{points[:, 0].max():.16e}] "
                  f"table=[{bounds[0]:.16e},{bounds[1]:.16e}]")
            print("EOS coordinates: "
                  f"log10(eps_SI)=[{points[:, 1].min():.16e},{points[:, 1].max():.16e}] "
                  f"table=[{bounds[2]:.16e},{bounds[3]:.16e}]")
            p = prim[:, 4].ravel()
            temp = data["eos_temperature"][:].ravel()
            print(f"interpolation max_abs_log10_pressure_error="
                  f"{np.max(np.abs(np.log10(p) - np.log10(p_ref))):.16e}")
            print(f"interpolation max_abs_log10_temperature_error="
                  f"{np.max(np.abs(np.log10(temp) - np.log10(t_ref))):.16e}")


if __name__ == "__main__":
    main()
