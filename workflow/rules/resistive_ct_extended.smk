from pathlib import Path


RESISTIVE_EXT = config["tests"]["resistive_ct_extended"]
RESISTIVE_EXT_REPO = Path(workflow.basedir).resolve().parent


def resistive_ext_path(value):
    path = Path(value)
    return str(path if path.is_absolute() else RESISTIVE_EXT_REPO / path)


RESISTIVE_EXT_EXE = resistive_ext_path(RESISTIVE_EXT["executable"])
RESISTIVE_EXT_DIFF_INPUT = resistive_ext_path(RESISTIVE_EXT["resistive_input"])
RESISTIVE_EXT_SHEET_INPUT = resistive_ext_path(RESISTIVE_EXT["current_sheet_input"])
RESISTIVE_EXT_OUT = (
    f"{resistive_ext_path(RESISTIVE_EXT['results_root'])}/"
    f"{RESISTIVE_EXT['dirname']}"
)
RESISTIVE_EXT_ANALYZER = str(
    RESISTIVE_EXT_REPO / "workflow/diagnostics_scripts/resistive_ct_extended.py"
)
RESISTIVE_EXT_GPU_LAUNCHER = str(RESISTIVE_EXT_REPO / "workflow/gpu_lock_launcher.py")
RESISTIVE_EXT_PYTHON = resistive_ext_path(RESISTIVE_EXT["python"])
RESISTIVE_EXT_GPU_IDS = ",".join(str(gpu) for gpu in RESISTIVE_EXT["gpu_ids"])
RESISTIVE_EXT_GPU_LOCK_DIR = f"{RESISTIVE_EXT_OUT}/.gpu-locks"
RESISTIVE_EXT_PARTHENON_TOOLS = str(
    RESISTIVE_EXT_REPO
    / "external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools"
)
RESISTIVE_EXT_FLUIDS = list(RESISTIVE_EXT["fluids"])
RESISTIVE_EXT_ETAS = {
    f"eta{eta:g}".replace(".", "p"): float(eta)
    for eta in RESISTIVE_EXT["reconnection"]["etas"]
}


def resistive_ext_run_dir(kind, fluid, eta_label=None):
    suffix = f"/{eta_label}" if eta_label else ""
    return f"{RESISTIVE_EXT_OUT}/runs/{kind}/{fluid}{suffix}"


def resistive_ext_prefix(kind, fluid, eta_label=None):
    return f"{kind}_{fluid}" + (f"_{eta_label}" if eta_label else "")


RESISTIVE_EXT_HEATING_FINALS = [
    f"{resistive_ext_run_dir('heating', fluid)}/"
    f"parthenon.{resistive_ext_prefix('heating', fluid)}.final.phdf"
    for fluid in RESISTIVE_EXT_FLUIDS
]
RESISTIVE_EXT_RECON_FINALS = [
    f"{resistive_ext_run_dir('reconnection', fluid, eta_label)}/"
    f"parthenon.{resistive_ext_prefix('reconnection', fluid, eta_label)}.final.phdf"
    for fluid in RESISTIVE_EXT_FLUIDS
    for eta_label in RESISTIVE_EXT_ETAS
]
RESISTIVE_EXT_HSTS = [
    f"{resistive_ext_run_dir('heating', fluid)}/parthenon.out1.hst"
    for fluid in RESISTIVE_EXT_FLUIDS
] + [
    f"{resistive_ext_run_dir('reconnection', fluid, eta_label)}/parthenon.out1.hst"
    for fluid in RESISTIVE_EXT_FLUIDS
    for eta_label in RESISTIVE_EXT_ETAS
]
RESISTIVE_EXT_PASS = f"{RESISTIVE_EXT_OUT}/resistive_ct_extended.passed"

resistive_ct_extended_targets = []
if RESISTIVE_EXT["enabled"]:
    resistive_ct_extended_targets = [RESISTIVE_EXT_PASS]


rule run_resistive_ct_heating:
    input:
        executable=RESISTIVE_EXT_EXE,
        deck=RESISTIVE_EXT_DIFF_INPUT,
        launcher=RESISTIVE_EXT_GPU_LAUNCHER,
    output:
        initial=f"{RESISTIVE_EXT_OUT}/runs/heating/{{fluid}}/parthenon.heating_{{fluid}}.00000.phdf",
        final=f"{RESISTIVE_EXT_OUT}/runs/heating/{{fluid}}/parthenon.heating_{{fluid}}.final.phdf",
        hst=f"{RESISTIVE_EXT_OUT}/runs/heating/{{fluid}}/parthenon.out1.hst",
    log:
        out=f"{RESISTIVE_EXT_OUT}/runs/heating/{{fluid}}/run.out",
        err=f"{RESISTIVE_EXT_OUT}/runs/heating/{{fluid}}/run.err",
    wildcard_constraints:
        fluid="|".join(RESISTIVE_EXT_FLUIDS),
    params:
        rundir=lambda wc: resistive_ext_run_dir("heating", wc.fluid),
        riemann=lambda wc: RESISTIVE_EXT["fluids"][wc.fluid],
        n=RESISTIVE_EXT["heating"]["resolution"],
        eta=RESISTIVE_EXT["heating"]["eta"],
        tlim=RESISTIVE_EXT["heating"]["tlim"],
        amp=RESISTIVE_EXT["heating"]["amplitude"],
        python=RESISTIVE_EXT_PYTHON,
        gpus=RESISTIVE_EXT_GPU_IDS,
        lock_dir=RESISTIVE_EXT_GPU_LOCK_DIR,
    resources:
        runtime=RESISTIVE_EXT.get("runtime", 30),
        mem_mb=RESISTIVE_EXT.get("mem_mb", 4000),
        gpu=1,
    shell:
        r"""
        mkdir -p {params.rundir}
        cd {params.rundir}
        {params.python} {input.launcher} --gpus {params.gpus} --lock-dir {params.lock_dir} -- \
          {input.executable} -i {input.deck} \
          hydro/fluid={wildcards.fluid} hydro/riemann={params.riemann} \
          problem/resistive_diffusion/iprob=20 \
          problem/resistive_diffusion/amp={params.amp} \
          diffusion/ohm_diff_coeff_code={params.eta} \
          parthenon/time/tlim={params.tlim} \
          parthenon/mesh/nx1={params.n} parthenon/mesh/nx2={params.n} parthenon/mesh/nx3={params.n} \
          parthenon/meshblock/nx1={params.n} parthenon/meshblock/nx2={params.n} parthenon/meshblock/nx3={params.n} \
          parthenon/job/problem_id=parthenon \
          parthenon/output0/dt={params.tlim} parthenon/output0/id=heating_{wildcards.fluid} \
          parthenon/output1/file_type=hst parthenon/output1/dt=-1 parthenon/output1/dn=1 \
          > {log.out} 2> {log.err}
        """


rule run_resistive_ct_reconnection:
    input:
        executable=RESISTIVE_EXT_EXE,
        deck=RESISTIVE_EXT_SHEET_INPUT,
        launcher=RESISTIVE_EXT_GPU_LAUNCHER,
    output:
        initial=f"{RESISTIVE_EXT_OUT}/runs/reconnection/{{fluid}}/{{eta_label}}/parthenon.reconnection_{{fluid}}_{{eta_label}}.00000.phdf",
        final=f"{RESISTIVE_EXT_OUT}/runs/reconnection/{{fluid}}/{{eta_label}}/parthenon.reconnection_{{fluid}}_{{eta_label}}.final.phdf",
        hst=f"{RESISTIVE_EXT_OUT}/runs/reconnection/{{fluid}}/{{eta_label}}/parthenon.out1.hst",
    log:
        out=f"{RESISTIVE_EXT_OUT}/runs/reconnection/{{fluid}}/{{eta_label}}/run.out",
        err=f"{RESISTIVE_EXT_OUT}/runs/reconnection/{{fluid}}/{{eta_label}}/run.err",
    wildcard_constraints:
        fluid="|".join(RESISTIVE_EXT_FLUIDS),
        eta_label="|".join(RESISTIVE_EXT_ETAS),
    params:
        rundir=lambda wc: resistive_ext_run_dir("reconnection", wc.fluid, wc.eta_label),
        riemann=lambda wc: RESISTIVE_EXT["fluids"][wc.fluid],
        eta=lambda wc: RESISTIVE_EXT_ETAS[wc.eta_label],
        nx1=RESISTIVE_EXT["reconnection"]["nx1"],
        nx2=RESISTIVE_EXT["reconnection"]["nx2"],
        tlim=RESISTIVE_EXT["reconnection"]["tlim"],
        python=RESISTIVE_EXT_PYTHON,
        gpus=RESISTIVE_EXT_GPU_IDS,
        lock_dir=RESISTIVE_EXT_GPU_LOCK_DIR,
    resources:
        runtime=RESISTIVE_EXT.get("runtime", 30),
        mem_mb=RESISTIVE_EXT.get("mem_mb", 4000),
        gpu=1,
    shell:
        r"""
        mkdir -p {params.rundir}
        cd {params.rundir}
        {params.python} {input.launcher} --gpus {params.gpus} --lock-dir {params.lock_dir} -- \
          {input.executable} -i {input.deck} \
          hydro/fluid={wildcards.fluid} hydro/riemann={params.riemann} hydro/reconstruction=plm \
          problem/current_sheet/amp=0.0 \
          diffusion/integrator=unsplit diffusion/cfl=0.3 \
          diffusion/conduction=none diffusion/viscosity=none \
          diffusion/resistivity=ohmic diffusion/resistivity_coeff=fixed \
          diffusion/ohm_diff_coeff_code={params.eta} \
          parthenon/time/tlim={params.tlim} \
          parthenon/mesh/nx1={params.nx1} parthenon/mesh/nx2={params.nx2} parthenon/mesh/nx3=1 \
          parthenon/meshblock/nx1={params.nx1} parthenon/meshblock/nx2={params.nx2} parthenon/meshblock/nx3=1 \
          parthenon/job/problem_id=parthenon \
          parthenon/output0/dt={params.tlim} parthenon/output0/id=reconnection_{wildcards.fluid}_{wildcards.eta_label} \
          parthenon/output1/file_type=hst parthenon/output1/dt=-1 parthenon/output1/dn=1 \
          > {log.out} 2> {log.err}
        """


rule analyze_resistive_ct_extended:
    input:
        heating=RESISTIVE_EXT_HEATING_FINALS,
        reconnection=RESISTIVE_EXT_RECON_FINALS,
        histories=RESISTIVE_EXT_HSTS,
        analyzer=RESISTIVE_EXT_ANALYZER,
    output:
        csv=f"{RESISTIVE_EXT_OUT}/resistive_ct_extended.csv",
        summary=f"{RESISTIVE_EXT_OUT}/resistive_ct_extended_summary.txt",
        plot=f"{RESISTIVE_EXT_OUT}/resistive_ct_extended.png",
        passed=RESISTIVE_EXT_PASS,
    log:
        f"{RESISTIVE_EXT_OUT}/analysis.log",
    params:
        heating=" ".join(
            f"--heating {fluid}={resistive_ext_run_dir('heating', fluid)}"
            for fluid in RESISTIVE_EXT_FLUIDS
        ),
        reconnection=" ".join(
            f"--reconnection {fluid},{eta}={resistive_ext_run_dir('reconnection', fluid, label)}"
            for fluid in RESISTIVE_EXT_FLUIDS
            for label, eta in RESISTIVE_EXT_ETAS.items()
        ),
        parthenon_tools=RESISTIVE_EXT_PARTHENON_TOOLS,
        heating_eta=RESISTIVE_EXT["heating"]["eta"],
        heating_tlim=RESISTIVE_EXT["heating"]["tlim"],
        reconnection_tlim=RESISTIVE_EXT["reconnection"]["tlim"],
        python=RESISTIVE_EXT_PYTHON,
    resources:
        runtime=10,
        mem_mb=2000,
    shell:
        r"""
        {params.python} {input.analyzer} {params.heating} {params.reconnection} \
          --parthenon-tools {params.parthenon_tools} \
          --heating-eta {params.heating_eta} --heating-tlim {params.heating_tlim} \
          --reconnection-tlim {params.reconnection_tlim} \
          --output-csv {output.csv} --output-summary {output.summary} \
          --output-plot {output.plot} --pass-marker {output.passed} \
          > {log} 2>&1
        """
