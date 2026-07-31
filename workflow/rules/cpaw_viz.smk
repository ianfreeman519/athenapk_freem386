CPAW_VIZ = config["tests"]["cpaw_viz"]
CPAW_VIZ_SCENARIOS = CPAW_VIZ["scenarios"]
CPAW_VIZ_SCENARIO_NAMES = list(CPAW_VIZ_SCENARIOS)
CPAW_VIZ_N = CPAW_VIZ["resolution"]


def cpaw_viz_out(scenario):
    return (
        f"{config['results_root']}/{CPAW_VIZ['dimension']}/{CPAW_VIZ['fluid']}/"
        f"{CPAW_VIZ['dirname']}/{scenario}"
    )


cpaw_viz_targets = []
if CPAW_VIZ["enabled"]:
    cpaw_viz_targets = expand(
        "{outdir}/cpaw_Bz.gif",
        outdir=[cpaw_viz_out(scenario) for scenario in CPAW_VIZ_SCENARIO_NAMES],
    )


rule run_cpaw_viz:
    output:
        done=(
            f"{config['results_root']}/{CPAW_VIZ['dimension']}/"
            f"{CPAW_VIZ['fluid']}/{CPAW_VIZ['dirname']}/"
            "{scenario}/phdf-files/run.done"
        )
    log:
        out=(
            f"{config['results_root']}/{CPAW_VIZ['dimension']}/"
            f"{CPAW_VIZ['fluid']}/{CPAW_VIZ['dirname']}/{{scenario}}/run.out"
        ),
        err=(
            f"{config['results_root']}/{CPAW_VIZ['dimension']}/"
            f"{CPAW_VIZ['fluid']}/{CPAW_VIZ['dirname']}/{{scenario}}/run.err"
        )
    params:
        outdir=lambda wc: cpaw_viz_out(wc.scenario),
        rundir=lambda wc: f"{cpaw_viz_out(wc.scenario)}/phdf-files",
        problem_id=lambda wc: f"cpaw_viz_{wc.scenario}_N{CPAW_VIZ_N}",
        nx1=2 * CPAW_VIZ_N,
        nx2=CPAW_VIZ_N,
        ang_2=CPAW_VIZ["mesh"]["ang_2"],
        v_par=lambda wc: CPAW_VIZ_SCENARIOS[wc.scenario]["v_par"],
        direction=lambda wc: CPAW_VIZ_SCENARIOS[wc.scenario]["dir"],
    resources:
        runtime=60
    wildcard_constraints:
        scenario="|".join(CPAW_VIZ_SCENARIO_NAMES)
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {params.rundir}/cpaw-errors.dat
        rm -f {output.done}
        cd {params.rundir}

        {config[athenapk]} -i {CPAW_VIZ[input]} \
          parthenon/job/problem_id={params.problem_id} \
          problem/cpaw/compute_error=false \
          problem/cpaw/ang_2={params.ang_2} \
          problem/cpaw/v_par={params.v_par} \
          problem/cpaw/dir={params.direction} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3=1 \
          parthenon/meshblock/nx1={params.nx1} \
          parthenon/meshblock/nx2={params.nx2} \
          parthenon/meshblock/nx3=1 \
          parthenon/time/tlim=1.0 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={CPAW_VIZ[fluid]} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/gamma=1.666666666666667 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt={CPAW_VIZ[output_dt]} \
          parthenon/output0/variables=prim \
          > {log.out} 2> {log.err}

        touch {output.done}
        """


rule make_cpaw_viz:
    input:
        done=(
            f"{config['results_root']}/{CPAW_VIZ['dimension']}/"
            f"{CPAW_VIZ['fluid']}/{CPAW_VIZ['dirname']}/"
            "{scenario}/phdf-files/run.done"
        )
    output:
        gif=report(
            f"{config['results_root']}/{CPAW_VIZ['dimension']}/"
            f"{CPAW_VIZ['fluid']}/{CPAW_VIZ['dirname']}/"
            "{scenario}/cpaw_Bz.gif",
            caption="../report/cpaw_Bz.rst",
            category="2D Tests",
            subcategory="ctmhd / Circularly Polarized Alfven Wave / {scenario}",
            labels={"fluid": CPAW_VIZ["fluid"], "scenario": "{scenario}", "quantity": "Bz"},
        )
    params:
        outdir=lambda wc: cpaw_viz_out(wc.scenario),
        phdf=lambda wc: f"{cpaw_viz_out(wc.scenario)}/phdf-files",
    resources:
        runtime=30
    wildcard_constraints:
        scenario="|".join(CPAW_VIZ_SCENARIO_NAMES)
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {CPAW_VIZ[plotting_script]} \
          {params.phdf} -o cpaw_Bz.gif --bmax 0.1 --label {wildcards.scenario}
        """
