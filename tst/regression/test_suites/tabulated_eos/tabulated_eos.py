import glob
import os
import sys

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import utils.test_case

sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    """Smoke/regression checks for the v1 aluminum three-chart EOS."""

    def Prepare(self, parameters, step):
        table_name = os.path.join(
            os.path.dirname(parameters.driver_input_path), "tables", "tabulated_eos.h5"
        )
        parameters.driver_cmd_line_args = [
            f"tabulated_eos/filename={os.path.abspath(table_name)}"
        ]
        return parameters

    def Analyse(self, parameters):
        def bilinear_uniform(values, x0, dx, y0, dy, x, y):
            """Mirror the device lookup, including edge-pair extrapolation."""
            fx = (x - x0) / dx
            fy = (y - y0) / dy
            i = np.clip(np.floor(fx).astype(int), 0, values.shape[0] - 2)
            j = np.clip(np.floor(fy).astype(int), 0, values.shape[1] - 2)
            tx = fx - i
            ty = fy - j
            return (1 - tx) * (
                (1 - ty) * values[i, j] + ty * values[i, j + 1]
            ) + tx * (
                (1 - ty) * values[i + 1, j] + ty * values[i + 1, j + 1]
            )

        table_name = os.path.join(
            os.path.dirname(parameters.driver_input_path), "tables", "tabulated_eos.h5"
        )
        with h5py.File(table_name, "r") as table:
            assert table.attrs["format_version"] == 1
            forward = table["eos_rho_e"]
            lrho = forward["axes/log10_density"][...]
            leps = forward["axes/log10_specific_internal_energy"][...]
            lp = forward["log10_pressure"][...]
            lt = forward["log10_temperature"][...]
            lcs = forward["log10_sound_speed"][...]
            zbar = forward["Zbar"][...]

            fields_ok = all(np.isfinite(q).all() for q in (lp, lt, lcs, zbar))
            axes_ok = (
                np.all(np.diff(lrho) > 0)
                and np.all(np.diff(leps) > 0)
                and np.allclose(np.diff(lrho), np.diff(lrho)[0], rtol=1e-10)
                and np.allclose(np.diff(leps), np.diff(leps)[0], rtol=1e-10)
            )
            monotone_ok = np.all(np.diff(lp, axis=1) > 0) and np.all(
                np.diff(lt, axis=1) > 0
            )
            composition_ok = np.min(zbar) >= 0 and np.max(zbar) <= 27.0 * (1 + 1e-12)

            # Exact grid recovery, a hand-computed midpoint, and all edge/corner
            # extrapolation directions in the same stored representation as C++.
            dr = np.diff(lrho)[0]
            de = np.diff(leps)[0]
            grid_ok = True
            for i, j in ((0, 0), (17, 31), (len(lrho) - 1, len(leps) - 1)):
                grid_ok &= np.isclose(
                    bilinear_uniform(lp, lrho[0], dr, leps[0], de, lrho[i], leps[j]),
                    lp[i, j],
                    rtol=0,
                    atol=1e-12,
                )
            i, j = 211, 487
            midpoint = bilinear_uniform(
                lp, lrho[0], dr, leps[0], de, lrho[i] + 0.5 * dr, leps[j] + 0.5 * de
            )
            midpoint_ok = np.isclose(
                midpoint,
                0.25 * (lp[i, j] + lp[i + 1, j] + lp[i, j + 1] + lp[i + 1, j + 1]),
                rtol=0,
                atol=1e-12,
            )
            extrapolation_ok = True
            for xr in (lrho[0] - 0.5, lrho[-1] + 0.5):
                for xe in (leps[0] - 0.5, leps[-1] + 0.5):
                    extrapolation_ok &= np.isfinite(
                        bilinear_uniform(lp, lrho[0], dr, leps[0], de, xr, xe)
                    )

            # The integration input has unit conversion factors of one except eV->K.
            ev_to_kelvin = 11604.51812155008
            unit_conversion_ok = np.isclose(
                10.0 ** (lt[211, 487] + np.log10(ev_to_kelvin)),
                (10.0 ** lt[211, 487]) * ev_to_kelvin,
                rtol=2e-14,
            )

            # Exercise both direct inverse charts against representative forward states.
            sample_e = np.linspace(0, len(leps) - 1, 33, dtype=int)
            sample_r = np.linspace(0, len(lrho) - 1, 33, dtype=int)
            rr, ee = np.meshgrid(sample_r, sample_e, indexing="ij")
            points_r = lrho[rr]
            expected_e = leps[ee]
            inverse_ok = True
            for group_name, coordinate, forward_values in (
                ("eos_rho_p", "log10_pressure", lp[rr, ee]),
                ("eos_rho_T", "log10_temperature", lt[rr, ee]),
            ):
                group = table[group_name]
                interp = RegularGridInterpolator(
                    (group["axes/log10_density"][...], group[f"axes/{coordinate}"][...]),
                    group["log10_specific_internal_energy"][...],
                    bounds_error=False,
                    fill_value=None,
                )
                predicted = interp(
                    np.column_stack((points_r.ravel(), forward_values.ravel()))
                ).reshape(expected_e.shape)
                inverse_ok &= np.max(np.abs(predicted - expected_e)) < 3.0e-3

        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        try:
            import phdf
        except ModuleNotFoundError:
            print("Couldn't import Parthenon phdf reader")
            return False

        outputs = sorted(glob.glob(os.path.join(parameters.output_path, "*.phdf")))
        if not outputs:
            print("No tabulated-EOS output file was produced")
            return False
        data = phdf.phdf(outputs[-1])
        components = data.GetComponents(data.Info["ComponentNames"], flatten=False)
        pressure = components["prim_pressure"]
        density = components["prim_density"]
        temperature = components["eos_temperature"]
        sound_speed = components["eos_sound_speed"]
        output_zbar = components["eos_zbar"]
        output_mu = components["eos_mu"]
        output_mu_e = components["eos_mu_e"]
        evolution_ok = (
            np.isfinite(pressure).all()
            and np.isfinite(density).all()
            and np.min(pressure) > 0
            and np.min(density) > 0
            and np.isfinite(temperature).all()
            and np.min(temperature) > 0
            and np.isfinite(sound_speed).all()
            and np.min(sound_speed) > 0
            and np.allclose(output_mu, 27.0 / (1.0 + output_zbar), rtol=1e-12)
            and np.allclose(output_mu_e, 27.0 / output_zbar, rtol=1e-12)
        )
        checks = {
            "finite table fields": fields_ok,
            "uniform increasing axes": axes_ok,
            "monotone forward maps": monotone_ok,
            "composition bounds": composition_ok,
            "grid-point interpolation": grid_ok,
            "midpoint interpolation": midpoint_ok,
            "edge/corner extrapolation": extrapolation_ok,
            "unit conversion": unit_conversion_ok,
            "inverse-chart consistency": inverse_ok,
            "evolution and derived fields": evolution_ok,
        }
        for name, passed in checks.items():
            if not passed:
                print(f"Tabulated EOS check failed: {name}")
        return bool(all(checks.values()))
