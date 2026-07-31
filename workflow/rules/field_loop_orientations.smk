FIELD_LOOP_ORIENTATIONS = config["tests"]["field_loop_orientations"]
FIELD_LOOP_ORIENTATIONS_MESH = FIELD_LOOP_ORIENTATIONS["mesh"]["2D"]
FIELD_LOOP_ORIENTATION_VECTORS = FIELD_LOOP_ORIENTATIONS["orientations"]

if FIELD_LOOP_ORIENTATIONS["enabled"] and config["dimension"] != "2D":
    raise ValueError("field_loop_orientations is currently a 2D-only test")


def field_loop_orientation_out(fluid, orientation):
    return (
        f"{config['results_root']}/2D/{fluid}/"
        f"{FIELD_LOOP_ORIENTATIONS['dirname']}/{orientation}"
    )


field_loop_orientation_targets = []
if FIELD_LOOP_ORIENTATIONS["enabled"]:
    field_loop_orientation_targets = expand(
        (
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{FIELD_LOOP_ORIENTATIONS['dirname']}/{{orientation}}/"
            "field_loop_Bmag_xy.gif"
        ),
        fluid=config["fluids"],
        orientation=FIELD_LOOP_ORIENTATION_VECTORS.keys(),
    )


rule run_field_loop_orientation:
    input:
        executable=config["athenapk"],
        deck=FIELD_LOOP_ORIENTATIONS["input"]
    output:
        done=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{FIELD_LOOP_ORIENTATIONS['dirname']}/{{orientation}}/"
            "phdf-files/run.done"
        )
    log:
        out=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{FIELD_LOOP_ORIENTATIONS['dirname']}/{{orientation}}/run.out"
        ),
        err=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{FIELD_LOOP_ORIENTATIONS['dirname']}/{{orientation}}/run.err"
        )
    params:
        outdir=lambda wc: field_loop_orientation_out(wc.fluid, wc.orientation),
        rundir=lambda wc: (
            f"{field_loop_orientation_out(wc.fluid, wc.orientation)}/phdf-files"
        ),
        problem_id=lambda wc: (
            f"field_loop_{wc.orientation.replace('-', '_')}_{wc.fluid}_"
            f"Nx{FIELD_LOOP_ORIENTATIONS_MESH['nx1']}x"
            f"{FIELD_LOOP_ORIENTATIONS_MESH['nx2']}x"
            f"{FIELD_LOOP_ORIENTATIONS_MESH['nx3']}"
        ),
        vflow1=lambda wc: FIELD_LOOP_ORIENTATION_VECTORS[wc.orientation]["vflow1"],
        vflow2=lambda wc: FIELD_LOOP_ORIENTATION_VECTORS[wc.orientation]["vflow2"],
        nx1=FIELD_LOOP_ORIENTATIONS_MESH["nx1"],
        nx2=FIELD_LOOP_ORIENTATIONS_MESH["nx2"],
        nx3=FIELD_LOOP_ORIENTATIONS_MESH["nx3"],
        mb_nx1=FIELD_LOOP_ORIENTATIONS_MESH["mb_nx1"],
        mb_nx2=FIELD_LOOP_ORIENTATIONS_MESH["mb_nx2"],
        mb_nx3=FIELD_LOOP_ORIENTATIONS_MESH["mb_nx3"],
        iprob=FIELD_LOOP_ORIENTATIONS_MESH["iprob"]
    resources:
        runtime=120
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
          problem/field_loop/vflow=0.0 \
          problem/field_loop/vflow1={params.vflow1} \
          problem/field_loop/vflow2={params.vflow2} \
          problem/field_loop/vflow3=0.0 \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.mb_nx1} \
          parthenon/meshblock/nx2={params.mb_nx2} \
          parthenon/meshblock/nx3={params.mb_nx3} \
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
          > {log.out} 2> {log.err}

        touch {output.done}
        """


rule make_field_loop_orientation_video:
    input:
        done=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{FIELD_LOOP_ORIENTATIONS['dirname']}/{{orientation}}/"
            "phdf-files/run.done"
        )
    output:
        xy=report(
            (
                f"{config['results_root']}/2D/{{fluid}}/"
                f"{FIELD_LOOP_ORIENTATIONS['dirname']}/{{orientation}}/"
                "field_loop_Bmag_xy.gif"
            ),
            caption="../report/field_loop_orientations.rst",
            category="2D Tests",
            subcategory="{fluid} / Field Loop / Direction Sweep",
            labels={
                "fluid": "{fluid}",
                "mesh": "uniform",
                "direction": "{orientation}",
                "quantity": "magnetic field magnitude",
                "slice": "xy",
            },
        )
    params:
        outdir=lambda wc: field_loop_orientation_out(wc.fluid, wc.orientation),
        phdf=lambda wc: (
            f"{field_loop_orientation_out(wc.fluid, wc.orientation)}/phdf-files"
        )
    resources:
        runtime=60
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {FIELD_LOOP_ORIENTATIONS[plotting_script]} \
          {params.phdf} -o field_loop
        """
