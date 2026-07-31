ORSZAG_TANG_REFINED = config["tests"]["orszag_tang_refined"]
ORSZAG_TANG_REFINED_RESOLUTION = ORSZAG_TANG_REFINED["resolution"]
ORSZAG_TANG_REFINED_FLUIDS = ORSZAG_TANG_REFINED.get("fluids", config["fluids"])

def orszag_tang_refined_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{ORSZAG_TANG_REFINED['dirname']}"

orszag_tang_refined_targets = []
if ORSZAG_TANG_REFINED["enabled"]:
    orszag_tang_refined_targets = expand(
        f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG_REFINED['dirname']}/orszag_tang_refined.mp4",
        fluid=ORSZAG_TANG_REFINED_FLUIDS,
    )

rule run_orszag_tang_refined:
    output:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG_REFINED['dirname']}/phdf-files/run.done"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG_REFINED['dirname']}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG_REFINED['dirname']}/run.err"
    params:
        outdir=lambda wc: orszag_tang_refined_out(wc.fluid),
        rundir=lambda wc: f"{orszag_tang_refined_out(wc.fluid)}/phdf-files",
        problem_id=lambda wc: f"orszag_tang_refined_{wc.fluid}_Nx{ORSZAG_TANG_REFINED_RESOLUTION[0]}x{ORSZAG_TANG_REFINED_RESOLUTION[1]}",
        nx1=ORSZAG_TANG_REFINED_RESOLUTION[0],
        nx2=ORSZAG_TANG_REFINED_RESOLUTION[1],
        nx1_mb=(ORSZAG_TANG_REFINED_RESOLUTION[0] // 4),
        nx2_mb=(ORSZAG_TANG_REFINED_RESOLUTION[1] // 4)
    resources:
        runtime=360,
        mem_mb=20000
        #mem_mb=(default for now) mb means megabyte
        #slurm_partition=(default for now)
        # see https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
        # for more options (number of tasks per job, cpu per task, mem per cpu, etc)
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {config[athenapk]} -i {ORSZAG_TANG_REFINED[input]} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3=1 \
          parthenon/mesh/refinement=adaptive \
          parthenon/mesh/numlevel=3 \
          parthenon/meshblock/nx1={params.nx1_mb} \
          parthenon/meshblock/nx2={params.nx2_mb} \
          parthenon/meshblock/nx3=1 \
          refinement/type=pressure_gradient \
          refinement/threshold_pressure_gradient=0.2 \
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
          parthenon/output0/dt=0.01 \
          parthenon/output0/variables=prim \
          parthenon/output1/file_type=hst \
          parthenon/output1/dt=0.005 \
          > {log.out} 2> {log.err}

        touch {output.done}
        """


rule make_orszag_tang_refined_movie:
    input:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG_REFINED['dirname']}/phdf-files/run.done"
    output:
        movie=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG_REFINED['dirname']}/orszag_tang_refined.mp4",
            caption="../report/orszag_tang_refined_pressure.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Orszag-Tang / Adaptive Mesh Refinement",
            labels={
                "fluid": "{fluid}",
                "quantity": "pressure",
                "diagnostic": "AMR grid overlay",
            },
        )
    params:
        phdf_dir=lambda wc: f"{orszag_tang_refined_out(wc.fluid)}/phdf-files",
        plotting_bin=config["plotting_python"].rsplit("/", 1)[0]
    resources:
        runtime=120,
        mem_mb=10000
    shell:
        """
        PATH={params.plotting_bin}:$PATH \
          {config[plotting_python]} {ORSZAG_TANG_REFINED[movie_script]} \
          {params.phdf_dir} \
          --output {output.movie}
        """
