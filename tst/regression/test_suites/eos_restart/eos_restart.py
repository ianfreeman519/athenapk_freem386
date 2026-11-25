# Copyright (c) 2026, AthenaPK Collaboration. All rights reserved.
# Licensed under the 3-Clause BSD License (the LICENSE file).

import os
import sys

import numpy as np
import utils.test_case


class TestCase(utils.test_case.TestCaseAbs):
    """Compare continuous integration with a checkpoint/restart of the EOS state."""

    def Prepare(self, parameters, step):
        name = {1: "continuous", 2: "split", 3: "resumed"}[step]
        args = []
        if step == 3:
            args += [
                "-r",
                os.path.join(parameters.output_path, "split.restart.final.rhdf"),
            ]
        args += [
            f"parthenon/job/problem_id={name}",
            f"parthenon/time/nlim={10 if step == 2 else 20}",
            "parthenon/time/tlim=1e20",
            "parthenon/time/ncycle_out=0",
            "parthenon/mesh/refinement=none",
            "parthenon/mesh/nx1=16",
            "parthenon/mesh/nx2=16",
            "parthenon/mesh/nx3=16",
            "parthenon/meshblock/nx1=8",
            "parthenon/meshblock/nx2=8",
            "parthenon/meshblock/nx3=8",
            "hydro/fluid=glmmhd",
            "hydro/riemann=hlld",
            "hydro/reconstruction=plm",
            "hydro/gamma=1.4",
            "problem/linear_wave/compute_error=false",
            "parthenon/output0/dt=-1",
            "parthenon/output1/file_type=rst",
            "parthenon/output1/id=restart",
            "parthenon/output1/dt=1e20",
        ]
        parameters.driver_cmd_line_args = args
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            0,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        import phdf

        continuous = phdf.phdf(
            os.path.join(parameters.output_path, "continuous.restart.final.rhdf")
        )
        resumed = phdf.phdf(
            os.path.join(parameters.output_path, "resumed.restart.final.rhdf")
        )
        try:
            assert continuous.NCycle == resumed.NCycle == 20
            np.testing.assert_allclose(continuous.Time, resumed.Time, rtol=1e-13)
            names = list(continuous.Info["ComponentNames"])
            assert names == list(resumed.Info["ComponentNames"])
            expected = continuous.GetComponents(names, flatten=False)
            actual = resumed.GetComponents(names, flatten=False)
            for name in names:
                assert np.isfinite(
                    actual[name]
                ).all(), f"Nonfinite {name} after restart"
                assert np.isfinite(expected[name]).all(), f"Nonfinite continuous {name}"
                np.testing.assert_allclose(
                    actual[name],
                    expected[name],
                    rtol=1e-12,
                    atol=1e-14,
                    err_msg=f"Restart changed {name}",
                )
        except (AssertionError, KeyError) as error:
            print(error)
            return False
        return True
