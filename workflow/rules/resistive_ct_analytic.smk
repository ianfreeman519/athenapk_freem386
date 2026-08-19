from pathlib import Path


RESISTIVE_CT = config["tests"]["resistive_ct_analytic"]
RESISTIVE_CT_REPO = Path(workflow.basedir).resolve().parent


def resistive_ct_path(value):
    path = Path(value)
    return str(path if path.is_absolute() else RESISTIVE_CT_REPO / path)


RESISTIVE_CT_EXECUTABLE = resistive_ct_path(RESISTIVE_CT["executable"])
RESISTIVE_CT_INPUT = resistive_ct_path(RESISTIVE_CT["input"])
RESISTIVE_CT_RESULTS = resistive_ct_path(RESISTIVE_CT["results_root"])
RESISTIVE_CT_OUT = f"{RESISTIVE_CT_RESULTS}/{RESISTIVE_CT['dirname']}"
RESISTIVE_CT_ANALYZER = str(
    RESISTIVE_CT_REPO / "workflow/diagnostics_scripts/resistive_ct_analytic.py"
)
RESISTIVE_CT_PARTHENON_TOOLS = str(
    RESISTIVE_CT_REPO
    / "external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools"
)


# Each entry is one independently schedulable AthenaPK run. The rotated modes use
# thin 3D domains because Ex and Ey are active CT edges only for a 3D mesh.
RESISTIVE_CT_CASES = {
    "uniform_16_16_16": {"iprob": 0, "mesh": (16, 16, 16)},
    # UCT edge assembly requires a genuine transverse mesh.  The state remains
    # exactly one-dimensional because the initial data are independent of x2.
    "sin1d_16_4_1": {"iprob": 1, "mesh": (16, 4, 1)},
    "sin1d_32_4_1": {"iprob": 1, "mesh": (32, 4, 1)},
    "sin1d_64_4_1": {"iprob": 1, "mesh": (64, 4, 1)},
    "rot_jy_64_4_4": {"iprob": 2, "mesh": (64, 4, 4)},
    "rot_jx_4_64_4": {"iprob": 3, "mesh": (4, 64, 4)},
    "fourier2d_16_16_1": {"iprob": 10, "mesh": (16, 16, 1)},
    "fourier2d_32_32_1": {"iprob": 10, "mesh": (32, 32, 1)},
    "fourier2d_64_64_1": {"iprob": 10, "mesh": (64, 64, 1)},
    "abc3d_8_8_8": {"iprob": 20, "mesh": (8, 8, 8)},
    "abc3d_16_16_16": {"iprob": 20, "mesh": (16, 16, 16)},
}


def resistive_ct_case(wildcards):
    return RESISTIVE_CT_CASES[wildcards.case]


def resistive_ct_meshblock(mesh):
    """Create exactly two blocks by splitting the first sufficiently large axis."""
    block = list(mesh)
    for axis, cells in enumerate(mesh):
        if cells >= 8:
            block[axis] = cells // 2
            break
    return tuple(block)


def resistive_ct_run_output(case):
    return f"{RESISTIVE_CT_OUT}/runs/{case}/parthenon.{case}.final.phdf"


RESISTIVE_CT_RUN_OUTPUTS = [
    resistive_ct_run_output(case) for case in RESISTIVE_CT_CASES
]
RESISTIVE_CT_PASS = f"{RESISTIVE_CT_OUT}/resistive_ct_analytic.passed"

resistive_ct_analytic_targets = []
if RESISTIVE_CT["enabled"]:
    resistive_ct_analytic_targets = [RESISTIVE_CT_PASS]


rule run_resistive_ct_analytic:
    input:
        executable=RESISTIVE_CT_EXECUTABLE,
        deck=RESISTIVE_CT_INPUT,
    output:
        phdf=f"{RESISTIVE_CT_OUT}/runs/{{case}}/parthenon.{{case}}.final.phdf",
    log:
        out=f"{RESISTIVE_CT_OUT}/runs/{{case}}/run.out",
        err=f"{RESISTIVE_CT_OUT}/runs/{{case}}/run.err",
    wildcard_constraints:
        case="|".join(RESISTIVE_CT_CASES),
    params:
        rundir=lambda wc: f"{RESISTIVE_CT_OUT}/runs/{wc.case}",
        iprob=lambda wc: resistive_ct_case(wc)["iprob"],
        nx1=lambda wc: resistive_ct_case(wc)["mesh"][0],
        nx2=lambda wc: resistive_ct_case(wc)["mesh"][1],
        nx3=lambda wc: resistive_ct_case(wc)["mesh"][2],
        mb1=lambda wc: resistive_ct_meshblock(resistive_ct_case(wc)["mesh"])[0],
        mb2=lambda wc: resistive_ct_meshblock(resistive_ct_case(wc)["mesh"])[1],
        mb3=lambda wc: resistive_ct_meshblock(resistive_ct_case(wc)["mesh"])[2],
        eta=RESISTIVE_CT["eta"],
        tlim=RESISTIVE_CT["tlim"],
        amp=RESISTIVE_CT["amplitude"],
    resources:
        runtime=RESISTIVE_CT.get("runtime", 30),
        mem_mb=RESISTIVE_CT.get("mem_mb", 4000),
        gpu=1,
    shell:
        r"""
        mkdir -p {params.rundir}
        cd {params.rundir}

        {input.executable} -i {input.deck} \
          problem/resistive_diffusion/iprob={params.iprob} \
          problem/resistive_diffusion/amp={params.amp} \
          diffusion/ohm_diff_coeff_code={params.eta} \
          parthenon/time/tlim={params.tlim} \
          parthenon/output0/dt={params.tlim} \
          parthenon/output0/id={wildcards.case} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.mb1} \
          parthenon/meshblock/nx2={params.mb2} \
          parthenon/meshblock/nx3={params.mb3} \
          > {log.out} 2> {log.err}
        """


rule analyze_resistive_ct_analytic:
    input:
        phdf=RESISTIVE_CT_RUN_OUTPUTS,
        analyzer=RESISTIVE_CT_ANALYZER,
    output:
        csv=f"{RESISTIVE_CT_OUT}/resistive_ct_errors.csv",
        summary=f"{RESISTIVE_CT_OUT}/resistive_ct_summary.txt",
        plot=f"{RESISTIVE_CT_OUT}/resistive_ct_convergence.png",
        passed=RESISTIVE_CT_PASS,
    log:
        f"{RESISTIVE_CT_OUT}/analysis.log",
    params:
        inputs=" ".join(
            f"--input {case}={resistive_ct_run_output(case)}"
            for case in RESISTIVE_CT_CASES
        ),
        parthenon_tools=RESISTIVE_CT_PARTHENON_TOOLS,
        eta=RESISTIVE_CT["eta"],
        tlim=RESISTIVE_CT["tlim"],
        amp=RESISTIVE_CT["amplitude"],
    resources:
        runtime=10,
        mem_mb=2000,
    shell:
        r"""
        python {input.analyzer} \
          {params.inputs} \
          --parthenon-tools {params.parthenon_tools} \
          --eta {params.eta} \
          --tlim {params.tlim} \
          --amplitude {params.amp} \
          --output-csv {output.csv} \
          --output-summary {output.summary} \
          --output-plot {output.plot} \
          --pass-marker {output.passed} \
          > {log} 2>&1
        """
