# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt

import utils.test_case

sys.dont_write_bytecode = True

ETA = 0.01
TLIM = 0.1
AMP = 1.0e-6
TWO_PI = 2.0 * np.pi

# Keep the expensive three-dimensional convergence test deliberately small. The one- and
# two-dimensional tests provide the tighter asymptotic convergence measurements.
CONFIGS = (
    [("uniform", 0, 16, 16, 16)]
    # UCT corner reconstruction requires a non-degenerate transverse mesh. The
    # initialized state is still independent of x2, so this remains a 1D test.
    + [("sin1d", 1, n, 4, 1) for n in (16, 32, 64)]
    # Ey and Ex are active CT edges only in a 3D mesh. The fields still vary along
    # exactly one coordinate, while four transverse cells make all edge families active.
    + [("rot_jy", 2, 64, 4, 4), ("rot_jx", 3, 4, 64, 4)]
    + [("fourier2d", 10, n, n, 1) for n in (16, 32, 64)]
    + [("abc3d", 20, n, n, n) for n in (8, 16)]
)


def output_name(config):
    name, _, nx1, nx2, nx3 = config
    return f"{name}_{nx1}_{nx2}_{nx3}"


def cell_volumes(data_file):
    return np.einsum(
        "ai,aj,ak->aijk",
        np.diff(data_file.zf),
        np.diff(data_file.yf),
        np.diff(data_file.xf),
    )


def weighted_l1(error, weights):
    return np.sum(np.abs(error) * weights) / np.sum(weights)


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        assert parameters.num_ranks <= 2, "Use <= 2 ranks for resistive CT tests."
        config = CONFIGS[step - 1]
        _, iprob, nx1, nx2, nx3 = config

        # Make at least two blocks so every configuration is valid in the two-rank test.
        # Split only one active dimension to keep the small 3D cases inexpensive.
        mb1, mb2, mb3 = nx1, nx2, nx3
        if nx1 >= 8:
            mb1 = nx1 // 2
        elif nx2 >= 8:
            mb2 = nx2 // 2
        elif nx3 >= 8:
            mb3 = nx3 // 2
        parameters.driver_cmd_line_args = [
            f"problem/resistive_diffusion/iprob={iprob}",
            f"parthenon/mesh/nx1={nx1}",
            f"parthenon/mesh/nx2={nx2}",
            f"parthenon/mesh/nx3={nx3}",
            f"parthenon/meshblock/nx1={mb1}",
            f"parthenon/meshblock/nx2={mb2}",
            f"parthenon/meshblock/nx3={mb3}",
            f"parthenon/output0/id={output_name(config)}",
        ]
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        try:
            import phdf
        except ModuleNotFoundError:
            print("Couldn't find module to read Parthenon HDF5 files.")
            return False

        passed = True
        errors = {"sin1d": [], "fourier2d": [], "abc3d": []}
        decay_1d = np.exp(-ETA * TWO_PI**2 * TLIM)
        decay_2d = np.exp(-ETA * 2.0 * TWO_PI**2 * TLIM)

        for config in CONFIGS:
            name, iprob, nx1, nx2, nx3 = config
            filename = os.path.join(
                parameters.output_path,
                f"parthenon.{output_name(config)}.final.phdf",
            )
            data = phdf.phdf(filename)
            components = data.GetComponents(
                [
                    "prim_magnetic_field_1",
                    "prim_magnetic_field_2",
                    "prim_magnetic_field_3",
                ],
                flatten=False,
            )
            bx = components["prim_magnetic_field_1"]
            by = components["prim_magnetic_field_2"]
            bz = components["prim_magnetic_field_3"]
            z, y, x = data.GetVolumeLocations(flatten=False)
            vol = cell_volumes(data)

            if iprob == 0:
                err = max(
                    np.max(np.abs(bx - 0.25)),
                    np.max(np.abs(by + 0.125)),
                    np.max(np.abs(bz - 0.0625)),
                )
                print(f"[resistive CT uniform] Linf error = {err:.8e}")
                passed &= err < 5.0e-13
                continue

            if iprob in (1, 2, 3):
                phase = np.sin(TWO_PI * (x if iprob in (1, 2) else y))
                field = by if iprob == 1 else bz
                expected = AMP * decay_1d * phase
                err = weighted_l1(field - expected, vol) / AMP
                sine_amp = np.sum(field * phase * vol) / np.sum(phase**2 * vol)
                cosine = np.cos(TWO_PI * (x if iprob in (1, 2) else y))
                cosine_amp = np.sum(field * cosine * vol) / np.sum(cosine**2 * vol)
                amp_err = abs(sine_amp / (AMP * decay_1d) - 1.0)
                phase_err = abs(cosine_amp) / AMP
                print(
                    f"[resistive CT {name} {nx1}x{nx2}x{nx3}] "
                    f"normalized L1={err:.8e}, amplitude error={amp_err:.8e}, "
                    f"quadrature amplitude={phase_err:.8e}"
                )
                if name == "sin1d":
                    errors[name].append((nx1, err))
                else:
                    passed &= amp_err < 2.0e-2 and phase_err < 2.0e-3
                continue

            if iprob == 10:
                expected_bx = AMP * TWO_PI * np.sin(TWO_PI * x) * np.cos(TWO_PI * y)
                expected_by = -AMP * TWO_PI * np.cos(TWO_PI * x) * np.sin(TWO_PI * y)
                scale = AMP * TWO_PI
                err = (
                    weighted_l1(bx - decay_2d * expected_bx, vol)
                    + weighted_l1(by - decay_2d * expected_by, vol)
                ) / (2.0 * scale)
                errors[name].append((nx1, err))
                print(f"[resistive CT Fourier2D N={nx1}] normalized L1={err:.8e}")
                continue

            # The unit-box ABC field with A=B=C=AMP obeys curl(B)=2*pi*B.
            expected_bx = AMP * (np.sin(TWO_PI * z) + np.cos(TWO_PI * y))
            expected_by = AMP * (np.sin(TWO_PI * x) + np.cos(TWO_PI * z))
            expected_bz = AMP * (np.sin(TWO_PI * y) + np.cos(TWO_PI * x))
            err = (
                weighted_l1(bx - decay_1d * expected_bx, vol)
                + weighted_l1(by - decay_1d * expected_by, vol)
                + weighted_l1(bz - decay_1d * expected_bz, vol)
            ) / (3.0 * AMP)
            errors[name].append((nx1, err))
            print(f"[resistive CT ABC3D N={nx1}] normalized L1={err:.8e}")

        for name, minimum_rate in (("sin1d", 1.8), ("fourier2d", 1.8), ("abc3d", 1.6)):
            resolutions, values = np.asarray(errors[name]).T
            if np.any(values <= 0.0) or np.any(~np.isfinite(values)):
                print(f"ERROR: invalid {name} convergence errors: {values}")
                passed = False
                continue
            rate = -np.polyfit(np.log(resolutions), np.log(values), 1)[0]
            print(f"[resistive CT {name}] measured convergence rate = {rate:.6f}")
            if rate < minimum_rate:
                print(f"ERROR: {name} convergence rate is below {minimum_rate}")
                passed = False

        fig, axis = plt.subplots()
        for name, entries in errors.items():
            resolutions, values = np.asarray(entries).T
            axis.loglog(resolutions, values, "o-", label=name)
        axis.set_xlabel("linear resolution")
        axis.set_ylabel("normalized volume-weighted L1 error")
        axis.grid(True, which="both")
        axis.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(parameters.output_path, "resistive_ct_convergence.png"))
        plt.close(fig)

        return bool(passed)
