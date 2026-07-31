LINEAR_WAVE_STATIC = config["tests"]["linear_wave_static"]
LINEAR_WAVE_STATIC_RESOLUTIONS = LINEAR_WAVE_STATIC["resolutions"]
LINEAR_WAVE_STATIC_WAVES = LINEAR_WAVE_STATIC["waves"]
LINEAR_WAVE_STATIC_REGIONS = LINEAR_WAVE_STATIC["refinement_regions"]


if LINEAR_WAVE_STATIC["enabled"] and config["dimension"] != "3D":
    raise ValueError("linear_wave_static is a 3D-only test")


def linear_wave_static_out(fluid):
    return (
        f"{config['results_root']}/{config['dimension']}/{fluid}/"
        f"{LINEAR_WAVE_STATIC['dirname']}"
    )


def linear_wave_static_mesh_params(N):
    N = int(N)
    return {
        "nx1": 2 * N,
        "nx2": N,
        "nx3": N,
        "nx1_mb": 2 * (N // 4),
        "nx2_mb": N // 2,
        "nx3_mb": N // 2,
    }


def linear_wave_static_refinement_args():
    args = ["parthenon/mesh/refinement=static"]
    for index, region in enumerate(LINEAR_WAVE_STATIC_REGIONS, start=1):
        block = f"parthenon/static_refinement{index}"
        for parameter in ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max", "level"):
            args.append(f"{block}/{parameter}={region[parameter]}")
    return " ".join(args)


LINEAR_WAVE_STATIC_REFINEMENT_ARGS = linear_wave_static_refinement_args()


linear_wave_static_targets = []
if LINEAR_WAVE_STATIC["enabled"]:
    linear_wave_static_targets = expand(
        "{outdir}/wave_{wave}/linear_wave_static_orders-{wave}.txt",
        outdir=[linear_wave_static_out(fluid) for fluid in config["fluids"]],
        wave=LINEAR_WAVE_STATIC_WAVES,
    )
    linear_wave_static_targets.append(
        f"{config['results_root']}/{config['dimension']}/"
        f"{LINEAR_WAVE_STATIC['dirname']}/mesh/mesh_structure.png"
    )


rule run_linear_wave_static:
    input:
        executable=config["athenapk_mpi"],
        deck=LINEAR_WAVE_STATIC["input"]
    output:
        dat=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/wave_{{wave}}/runs/N{{N}}/"
            "linearwave-errors-{wave}.dat"
        )
    log:
        out=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/wave_{{wave}}/runs/N{{N}}/run.out"
        ),
        err=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/wave_{{wave}}/runs/N{{N}}/run.err"
        )
    params:
        rundir=lambda wc: (
            f"{linear_wave_static_out(wc.fluid)}/wave_{wc.wave}/runs/N{wc.N}"
        ),
        problem_id=lambda wc: f"linear_wave_static_{wc.fluid}_w{wc.wave}_N{wc.N}",
        nx1=lambda wc: linear_wave_static_mesh_params(wc.N)["nx1"],
        nx2=lambda wc: linear_wave_static_mesh_params(wc.N)["nx2"],
        nx3=lambda wc: linear_wave_static_mesh_params(wc.N)["nx3"],
        nx1_mb=lambda wc: linear_wave_static_mesh_params(wc.N)["nx1_mb"],
        nx2_mb=lambda wc: linear_wave_static_mesh_params(wc.N)["nx2_mb"],
        nx3_mb=lambda wc: linear_wave_static_mesh_params(wc.N)["nx3_mb"],
        refinement_args=LINEAR_WAVE_STATIC_REFINEMENT_ARGS
    resources:
        runtime=360,
        nodes=1,
        tasks=40,
        mpi="srun",
        mem_mb_per_cpu=4000
    shell:
        """
        mkdir -p {params.rundir}
        cd {params.rundir}
        rm -f {output.dat}

        {resources.mpi} -n {resources.tasks} {input.executable} -i {input.deck} \
          parthenon/job/problem_id={params.problem_id} \
          problem/linear_wave_mhd/wave_flag={wildcards.wave} \
          problem/linear_wave_mhd/compute_error=true \
          problem/linear_wave_mhd/ang_2=-999.9 \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.nx1_mb} \
          parthenon/meshblock/nx2={params.nx2_mb} \
          parthenon/meshblock/nx3={params.nx3_mb} \
          {params.refinement_args} \
          parthenon/time/tlim=1.0 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/gamma=1.666666666666667 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt=-0.01 \
          parthenon/output0/variables=prim \
          > {log.out} 2> {log.err}
        """


rule combine_linear_wave_static_errors:
    input:
        lambda wc: expand(
            f"{linear_wave_static_out(wc.fluid)}/wave_{wc.wave}/runs/N{{N}}/"
            f"linearwave-errors-{wc.wave}.dat",
            N=LINEAR_WAVE_STATIC_RESOLUTIONS,
        )
    output:
        dat=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/wave_{{wave}}/"
            "linearwave-errors-{wave}.dat"
        )
    resources:
        runtime=10
    shell:
        "cat {input} > {output.dat}"


rule compute_linear_wave_static_orders:
    input:
        dat=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/wave_{{wave}}/"
            "linearwave-errors-{wave}.dat"
        ),
        script=LINEAR_WAVE_STATIC["order_script"]
    output:
        txt=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/wave_{{wave}}/"
            "linear_wave_static_orders-{wave}.txt",
            caption="../report/linear_wave_static_orders.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Linear MHD Wave",
            labels={
                "problem": "linear MHD wave",
                "mesh": "static",
                "fluid": "{fluid}",
                "wave": "{wave}",
                "diagnostic": "convergence orders",
            },
        )
    params:
        outdir=lambda wc: f"{linear_wave_static_out(wc.fluid)}/wave_{wc.wave}"
    resources:
        runtime=10
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {input.script} {input.dat} > {output.txt}
        """


rule make_linear_wave_static_mesh_plot:
    input:
        executable=config["athenapk"],
        deck=LINEAR_WAVE_STATIC["input"],
        gnuplot=config["gnuplot"]
    output:
        dat=(
            f"{config['results_root']}/{config['dimension']}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/mesh/mesh_structure.dat"
        ),
        png=report(
            f"{config['results_root']}/{config['dimension']}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/mesh/mesh_structure.png",
            caption="../report/linear_wave_static_mesh.rst",
            category=f"{config['dimension']} Tests",
            subcategory="Linear MHD Wave / Mesh Structure",
            labels={
                "problem": "linear MHD wave",
                "mesh": "static",
                "diagnostic": "mesh structure",
            },
        )
    log:
        out=(
            f"{config['results_root']}/{config['dimension']}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/mesh/mesh.out"
        ),
        err=(
            f"{config['results_root']}/{config['dimension']}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/mesh/mesh.err"
        )
    params:
        rundir=(
            f"{config['results_root']}/{config['dimension']}/"
            f"{LINEAR_WAVE_STATIC['dirname']}/mesh"
        ),
        fluid=config["fluids"][0],
        nx1=linear_wave_static_mesh_params(LINEAR_WAVE_STATIC["mesh_plot_resolution"])["nx1"],
        nx2=linear_wave_static_mesh_params(LINEAR_WAVE_STATIC["mesh_plot_resolution"])["nx2"],
        nx3=linear_wave_static_mesh_params(LINEAR_WAVE_STATIC["mesh_plot_resolution"])["nx3"],
        nx1_mb=linear_wave_static_mesh_params(LINEAR_WAVE_STATIC["mesh_plot_resolution"])["nx1_mb"],
        nx2_mb=linear_wave_static_mesh_params(LINEAR_WAVE_STATIC["mesh_plot_resolution"])["nx2_mb"],
        nx3_mb=linear_wave_static_mesh_params(LINEAR_WAVE_STATIC["mesh_plot_resolution"])["nx3_mb"],
        refinement_args=LINEAR_WAVE_STATIC_REFINEMENT_ARGS
    resources:
        runtime=10
    shell:
        """
        mkdir -p {params.rundir}
        cd {params.rundir}
        rm -f {output.dat} {output.png}

        {input.executable} -i {input.deck} -m 1 \
          problem/linear_wave_mhd/ang_2=-999.9 \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.nx1_mb} \
          parthenon/meshblock/nx2={params.nx2_mb} \
          parthenon/meshblock/nx3={params.nx3_mb} \
          {params.refinement_args} \
          hydro/fluid={params.fluid} \
          > {log.out} 2> {log.err}

        {input.gnuplot} -e 'set terminal pngcairo size 1400,1000; set output "{output.png}"; set view equal xyz; set xlabel "x1"; set ylabel "x2"; set zlabel "x3"; set title "AthenaPK 3-D static mesh structure"; splot "{output.dat}" using 1:2:3 with lines notitle'
        """
