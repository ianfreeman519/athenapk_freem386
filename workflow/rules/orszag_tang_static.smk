ORSZAG_TANG_STATIC = config["tests"]["orszag_tang_static"]
ORSZAG_TANG_STATIC_RESOLUTION = ORSZAG_TANG_STATIC["resolution"]
ORSZAG_TANG_STATIC_MESHBLOCK = ORSZAG_TANG_STATIC["meshblock"]
ORSZAG_TANG_STATIC_FLUIDS = ORSZAG_TANG_STATIC.get("fluids", config["fluids"])

if ORSZAG_TANG_STATIC["enabled"] and config["dimension"] != "2D":
    raise ValueError("orszag_tang_static is a two-dimensional test")


def orszag_tang_static_out(fluid):
    return (
        f"{config['results_root']}/{config['dimension']}/{fluid}/"
        f"{ORSZAG_TANG_STATIC['dirname']}"
    )


def orszag_tang_static_refinement_args():
    args = ["parthenon/mesh/refinement=static"]
    for index, region in enumerate(ORSZAG_TANG_STATIC["refinement_regions"], start=1):
        prefix = f"parthenon/static_refinement{index}"
        for parameter in ("x1min", "x1max", "x2min", "x2max", "level"):
            args.append(f"{prefix}/{parameter}={region[parameter]}")
    return " ".join(args)


orszag_tang_static_targets = []
if ORSZAG_TANG_STATIC["enabled"]:
    orszag_tang_static_targets = expand(
        f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
        f"{ORSZAG_TANG_STATIC['dirname']}/orszag_tang_static.mp4",
        fluid=ORSZAG_TANG_STATIC_FLUIDS,
    )


rule run_orszag_tang_static:
    input:
        executable=config["athenapk"],
        deck=ORSZAG_TANG_STATIC["input"]
    output:
        done=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{ORSZAG_TANG_STATIC['dirname']}/phdf-files/run.done"
        )
    log:
        out=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{ORSZAG_TANG_STATIC['dirname']}/run.out"
        ),
        err=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{ORSZAG_TANG_STATIC['dirname']}/run.err"
        )
    params:
        outdir=lambda wc: orszag_tang_static_out(wc.fluid),
        rundir=lambda wc: f"{orszag_tang_static_out(wc.fluid)}/phdf-files",
        problem_id=lambda wc: (
            f"orszag_tang_static_{wc.fluid}_"
            f"Nx{ORSZAG_TANG_STATIC_RESOLUTION[0]}x{ORSZAG_TANG_STATIC_RESOLUTION[1]}"
        ),
        nx1=ORSZAG_TANG_STATIC_RESOLUTION[0],
        nx2=ORSZAG_TANG_STATIC_RESOLUTION[1],
        nx1_mb=ORSZAG_TANG_STATIC_MESHBLOCK[0],
        nx2_mb=ORSZAG_TANG_STATIC_MESHBLOCK[1],
        refinement_args=orszag_tang_static_refinement_args(),
        output_dt=ORSZAG_TANG_STATIC["output_dt"],
        history_dt=ORSZAG_TANG_STATIC["history_dt"]
    resources:
        runtime=180,
        mem_mb=10000
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {input.executable} -i {input.deck} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3=1 \
          parthenon/meshblock/nx1={params.nx1_mb} \
          parthenon/meshblock/nx2={params.nx2_mb} \
          parthenon/meshblock/nx3=1 \
          {params.refinement_args} \
          parthenon/time/tlim=1.0 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/gamma=1.666666666666667 \
          hydro/pfloor=1e-15 \
          hydro/scratch_level=1 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt={params.output_dt} \
          parthenon/output0/variables=prim \
          parthenon/output1/file_type=hst \
          parthenon/output1/dt={params.history_dt} \
          > {log.out} 2> {log.err}

        touch {output.done}
        """


rule make_orszag_tang_static_movie:
    input:
        done=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{ORSZAG_TANG_STATIC['dirname']}/phdf-files/run.done"
        )
    output:
        movie=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{ORSZAG_TANG_STATIC['dirname']}/orszag_tang_static.mp4",
            caption="../report/orszag_tang_static_pressure.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Orszag-Tang / Static Mesh Refinement",
            labels={
                "fluid": "{fluid}",
                "quantity": "pressure",
                "diagnostic": "static grid overlay",
            },
        )
    params:
        phdf_dir=lambda wc: f"{orszag_tang_static_out(wc.fluid)}/phdf-files",
        plotting_bin=config["plotting_python"].rsplit("/", 1)[0]
    resources:
        runtime=120,
        mem_mb=10000
    shell:
        """
        PATH={params.plotting_bin}:$PATH \
          {config[plotting_python]} {ORSZAG_TANG_STATIC[movie_script]} \
          {params.phdf_dir} \
          --output {output.movie}
        """
