CPAW_STATIC = config["tests"]["cpaw_static"]
CPAW_STATIC_RESOLUTIONS = CPAW_STATIC["resolutions"]
CPAW_STATIC_REGIONS = CPAW_STATIC["refinement_regions"]


if CPAW_STATIC["enabled"] and config["dimension"] != "3D":
    raise ValueError("cpaw_static is a 3D-only test")


def cpaw_static_out(fluid):
    return (
        f"{config['results_root']}/{config['dimension']}/{fluid}/"
        f"{CPAW_STATIC['dirname']}"
    )


def cpaw_static_mesh_params(N):
    N = int(N)
    return {
        "nx1": 2 * N,
        "nx2": N,
        "nx3": N,
        "nx1_mb": 2 * (N // 4),
        "nx2_mb": N // 2,
        "nx3_mb": N // 2,
    }


def cpaw_static_refinement_args():
    args = ["parthenon/mesh/refinement=static"]
    for index, region in enumerate(CPAW_STATIC_REGIONS, start=1):
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


CPAW_STATIC_REFINEMENT_ARGS = cpaw_static_refinement_args()


cpaw_static_targets = []
if CPAW_STATIC["enabled"]:
    cpaw_static_targets = expand(
        "{outdir}/cpaw_static_orders.txt",
        outdir=[cpaw_static_out(fluid) for fluid in config["fluids"]],
    )
    cpaw_static_targets.append(
        f"{config['results_root']}/{config['dimension']}/"
        f"{CPAW_STATIC['dirname']}/mesh/mesh_structure.png"
    )


rule run_cpaw_static:
    input:
        executable=config["athenapk_mpi"],
        deck=CPAW_STATIC["input"]
    output:
        dat=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{CPAW_STATIC['dirname']}/runs/N{{N}}/cpaw-errors.dat"
        )
    log:
        out=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{CPAW_STATIC['dirname']}/runs/N{{N}}/run.out"
        ),
        err=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{CPAW_STATIC['dirname']}/runs/N{{N}}/run.err"
        )
    params:
        rundir=lambda wc: f"{cpaw_static_out(wc.fluid)}/runs/N{wc.N}",
        problem_id=lambda wc: f"cpaw_static_{wc.fluid}_N{wc.N}",
        nx1=lambda wc: cpaw_static_mesh_params(wc.N)["nx1"],
        nx2=lambda wc: cpaw_static_mesh_params(wc.N)["nx2"],
        nx3=lambda wc: cpaw_static_mesh_params(wc.N)["nx3"],
        nx1_mb=lambda wc: cpaw_static_mesh_params(wc.N)["nx1_mb"],
        nx2_mb=lambda wc: cpaw_static_mesh_params(wc.N)["nx2_mb"],
        nx3_mb=lambda wc: cpaw_static_mesh_params(wc.N)["nx3_mb"],
        refinement_args=CPAW_STATIC_REFINEMENT_ARGS
    resources:
        runtime=360,
        nodes=1,
        tasks=40,
        mpi="srun",
        mem_mb_per_cpu=2000
    shell:
        """
        mkdir -p {params.rundir}
        cd {params.rundir}
        rm -f {output.dat}

        {resources.mpi} -n {resources.tasks} {input.executable} -i {input.deck} \
          parthenon/job/problem_id={params.problem_id} \
          problem/cpaw/compute_error=true \
          problem/cpaw/ang_2=-999.9 \
          problem/cpaw/v_par=1.0 \
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


rule combine_cpaw_static_errors:
    input:
        lambda wc: expand(
            f"{cpaw_static_out(wc.fluid)}/runs/N{{N}}/cpaw-errors.dat",
            N=CPAW_STATIC_RESOLUTIONS,
        )
    output:
        dat=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{CPAW_STATIC['dirname']}/cpaw-static-errors.dat"
        )
    resources:
        runtime=10
    shell:
        "cat {input} > {output.dat}"


rule compute_cpaw_static_orders:
    input:
        dat=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{CPAW_STATIC['dirname']}/cpaw-static-errors.dat"
        ),
        script=CPAW_STATIC["order_script"]
    output:
        txt=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{CPAW_STATIC['dirname']}/cpaw_static_orders.txt",
            caption="../report/cpaw_static_orders.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Circularly Polarized Traveling Alfven Wave",
            labels={
                "fluid": "{fluid}",
                "mesh": "static",
                "diagnostic": "convergence orders",
            },
        )
    params:
        outdir=lambda wc: cpaw_static_out(wc.fluid)
    resources:
        runtime=10
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {input.script} {input.dat} > {output.txt}
        """


rule make_cpaw_static_mesh_plot:
    input:
        executable=config["athenapk"],
        deck=CPAW_STATIC["input"],
        gnuplot=config["gnuplot"]
    output:
        dat=(
            f"{config['results_root']}/{config['dimension']}/"
            f"{CPAW_STATIC['dirname']}/mesh/mesh_structure.dat"
        ),
        png=report(
            f"{config['results_root']}/{config['dimension']}/"
            f"{CPAW_STATIC['dirname']}/mesh/mesh_structure.png",
            caption="../report/cpaw_static_mesh.rst",
            category=f"{config['dimension']} Tests",
            subcategory="CPAW / Mesh Structure",
            labels={
                "problem": "CPAW",
                "mesh": "static",
                "diagnostic": "mesh structure",
            },
        )
    log:
        out=(
            f"{config['results_root']}/{config['dimension']}/"
            f"{CPAW_STATIC['dirname']}/mesh/mesh.out"
        ),
        err=(
            f"{config['results_root']}/{config['dimension']}/"
            f"{CPAW_STATIC['dirname']}/mesh/mesh.err"
        )
    params:
        rundir=(
            f"{config['results_root']}/{config['dimension']}/"
            f"{CPAW_STATIC['dirname']}/mesh"
        ),
        fluid=config["fluids"][0],
        nx1=cpaw_static_mesh_params(CPAW_STATIC["mesh_plot_resolution"])["nx1"],
        nx2=cpaw_static_mesh_params(CPAW_STATIC["mesh_plot_resolution"])["nx2"],
        nx3=cpaw_static_mesh_params(CPAW_STATIC["mesh_plot_resolution"])["nx3"],
        nx1_mb=cpaw_static_mesh_params(CPAW_STATIC["mesh_plot_resolution"])["nx1_mb"],
        nx2_mb=cpaw_static_mesh_params(CPAW_STATIC["mesh_plot_resolution"])["nx2_mb"],
        nx3_mb=cpaw_static_mesh_params(CPAW_STATIC["mesh_plot_resolution"])["nx3_mb"],
        refinement_args=CPAW_STATIC_REFINEMENT_ARGS
    resources:
        runtime=10
    shell:
        """
        mkdir -p {params.rundir}
        cd {params.rundir}
        rm -f {output.dat} {output.png}

        {input.executable} -i {input.deck} -m 1 \
          problem/cpaw/ang_2=-999.9 \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.nx1_mb} \
          parthenon/meshblock/nx2={params.nx2_mb} \
          parthenon/meshblock/nx3={params.nx3_mb} \
          {params.refinement_args} \
          hydro/fluid={params.fluid} \
          parthenon/time/integrator={config[integrator]} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          > {log.out} 2> {log.err}

        {input.gnuplot} -e 'set terminal pngcairo size 1400,1000; set output "{output.png}"; set view equal xyz; set xlabel "x1"; set ylabel "x2"; set zlabel "x3"; set title "AthenaPK 3-D CPAW static mesh structure"; splot "{output.dat}" using 1:2:3 with lines notitle'
        """
