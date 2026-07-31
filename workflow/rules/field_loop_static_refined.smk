FIELD_LOOP_STATIC_REFINED = config["tests"]["field_loop_static_refined"]
FIELD_LOOP_STATIC_REFINED_MESH = FIELD_LOOP_STATIC_REFINED["mesh"]
FIELD_LOOP_STATIC_REFINED_REGIONS = FIELD_LOOP_STATIC_REFINED["refinement_regions"]


if FIELD_LOOP_STATIC_REFINED["enabled"] and config["dimension"] != "3D":
    raise ValueError("field_loop_static_refined is a 3D-only test")


def field_loop_static_refined_out(fluid):
    return (
        f"{config['results_root']}/{config['dimension']}/{fluid}/"
        f"{FIELD_LOOP_STATIC_REFINED['dirname']}"
    )


def field_loop_static_refinement_args():
    args = ["parthenon/mesh/refinement=static"]
    for index, region in enumerate(FIELD_LOOP_STATIC_REFINED_REGIONS, start=1):
        block = f"parthenon/static_refinement{index}"
        for parameter in (
            "x1min",
            "x1max",
            "x2min",
            "x2max",
            "x3min",
            "x3max",
            "level",
        ):
            args.append(f"{block}/{parameter}={region[parameter]}")
    return " ".join(args)


FIELD_LOOP_STATIC_REFINEMENT_ARGS = field_loop_static_refinement_args()


field_loop_static_refined_targets = []
if FIELD_LOOP_STATIC_REFINED["enabled"]:
    field_loop_static_refined_targets = expand(
        "{outdir}/field_loop_Bmag_{plane}.mp4",
        outdir=[
            field_loop_static_refined_out(fluid) for fluid in config["fluids"]
        ],
        plane=["z", "y", "x"],
    )


rule run_field_loop_static_refined:
    input:
        executable=config["athenapk"],
        deck=FIELD_LOOP_STATIC_REFINED["input"]
    output:
        done=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{FIELD_LOOP_STATIC_REFINED['dirname']}/phdf-files/run.done"
        )
    log:
        out=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{FIELD_LOOP_STATIC_REFINED['dirname']}/run.out"
        ),
        err=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{FIELD_LOOP_STATIC_REFINED['dirname']}/run.err"
        )
    params:
        outdir=lambda wc: field_loop_static_refined_out(wc.fluid),
        rundir=lambda wc: f"{field_loop_static_refined_out(wc.fluid)}/phdf-files",
        problem_id=FIELD_LOOP_STATIC_REFINED["problem_id"],
        nx1=FIELD_LOOP_STATIC_REFINED_MESH["nx1"],
        nx2=FIELD_LOOP_STATIC_REFINED_MESH["nx2"],
        nx3=FIELD_LOOP_STATIC_REFINED_MESH["nx3"],
        mb_nx1=FIELD_LOOP_STATIC_REFINED_MESH["mb_nx1"],
        mb_nx2=FIELD_LOOP_STATIC_REFINED_MESH["mb_nx2"],
        mb_nx3=FIELD_LOOP_STATIC_REFINED_MESH["mb_nx3"],
        iprob=FIELD_LOOP_STATIC_REFINED_MESH["iprob"],
        refinement_args=FIELD_LOOP_STATIC_REFINEMENT_ARGS
    resources:
        runtime=1440,
        mem_mb=30000
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {input.executable} -i {input.deck} \
          problem/field_loop/iprob={params.iprob} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.mb_nx1} \
          parthenon/meshblock/nx2={params.mb_nx2} \
          parthenon/meshblock/nx3={params.mb_nx3} \
          {params.refinement_args} \
          parthenon/time/tlim=2.0 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/gamma=1.666666666666667 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt=0.02 \
          parthenon/output0/variables=prim \
          parthenon/output1/file_type=hst \
          parthenon/output1/dt=0.005 \
          > {log.out} 2> {log.err}

        touch {output.done}
        """


rule make_field_loop_static_refined_videos:
    input:
        done=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{FIELD_LOOP_STATIC_REFINED['dirname']}/phdf-files/run.done"
        )
    output:
        xy=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{FIELD_LOOP_STATIC_REFINED['dirname']}/field_loop_Bmag_z.mp4",
            caption="../report/field_loop_Bmag.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Field Loop",
            labels={
                "fluid": "{fluid}",
                "mesh": "static",
                "quantity": "magnetic field magnitude",
                "diagnostic": "static grid overlay",
                "slice": "xy",
            },
        ),
        xz=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{FIELD_LOOP_STATIC_REFINED['dirname']}/field_loop_Bmag_y.mp4",
            caption="../report/field_loop_Bmag.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Field Loop",
            labels={
                "fluid": "{fluid}",
                "mesh": "static",
                "quantity": "magnetic field magnitude",
                "diagnostic": "static grid overlay",
                "slice": "xz",
            },
        ),
        yz=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{FIELD_LOOP_STATIC_REFINED['dirname']}/field_loop_Bmag_x.mp4",
            caption="../report/field_loop_Bmag.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Field Loop",
            labels={
                "fluid": "{fluid}",
                "mesh": "static",
                "quantity": "magnetic field magnitude",
                "diagnostic": "static grid overlay",
                "slice": "yz",
            },
        )
    params:
        outdir=lambda wc: field_loop_static_refined_out(wc.fluid),
        phdf=lambda wc: f"{field_loop_static_refined_out(wc.fluid)}/phdf-files"
    resources:
        runtime=180,
        mem_mb=10000
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {FIELD_LOOP_STATIC_REFINED[plotting_script]} \
          {params.phdf} -o field_loop
        """
