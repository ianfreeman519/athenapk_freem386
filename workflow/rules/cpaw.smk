CPAW = config["tests"]["cpaw"]
CPAW_RESOLUTIONS = CPAW["resolutions"]

def cpaw_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{CPAW['dirname']}"

cpaw_targets = []
if CPAW["enabled"]:
    cpaw_targets = expand(
        "{outdir}/cpaw_orders.txt",
        outdir=[cpaw_out(fluid) for fluid in config["fluids"]],
    )


rule run_cpaw:
    output:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/runs/N{{N}}/cpaw-errors.dat"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/runs/N{{N}}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/runs/N{{N}}/run.err"
    params:
        outdir=lambda wildcards: cpaw_out(wildcards.fluid),
        rundir=lambda wildcards: f"{cpaw_out(wildcards.fluid)}/runs/N{wildcards.N}",
        problem_id=lambda wildcards: f"cpaw_{wildcards.fluid}_N{wildcards.N}",
        nx1=lambda wildcards: 2 * int(wildcards.N),
        nx2=lambda wildcards: int(wildcards.N),
        nx3=lambda wc: 1 if config["dimension"] == "2D" else int(wc.N),
        ang_2=lambda wc: CPAW["mesh"]["2D"]["ang_2"] if config["dimension"] == "2D" else -999.9,
        nx1_mb=lambda wildcards: 2 * (int(wildcards.N) // 4),
        nx2_mb=lambda wildcards: (int(wildcards.N) // 2),
        nx3_mb=lambda wc: 1 if config["dimension"] == "2D" else (int(wc.N) // 2)
    resources:
        runtime=120,
        nodes=1,
        tasks=16,
        mpi="srun",
        mem_mb_per_cpu=4000
        #slurm_partition=(default for now)
        # see https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
        # for more options (number of tasks per job, cpu per task, mem per cpu, etc)
        #hydro/pfloor=1e-15 \
    shell:
        """
        mkdir -p {params.rundir}
        cd {params.rundir}
        rm -f {output.dat}

        {resources.mpi} -n {resources.tasks} {config[athenapk_mpi]} -i {CPAW[input]} \
          parthenon/job/problem_id={params.problem_id} \
          problem/cpaw/compute_error=true \
          problem/cpaw/ang_2={params.ang_2} \
          problem/cpaw/v_par=1.0 \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.nx1_mb} \
          parthenon/meshblock/nx2={params.nx2_mb} \
          parthenon/meshblock/nx3={params.nx3_mb} \
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

rule combine_cpaw_errors:
    input:
        lambda wildcards: expand(
            f"{cpaw_out(wildcards.fluid)}/runs/N{{N}}/cpaw-errors.dat",
            N=CPAW_RESOLUTIONS,
        )
    output:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/cpaw-errors.dat"
    resources:
        runtime=10
    shell:
        """
        cat {input} > {output.dat}
        """

rule compute_cpaw_orders:
    input:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/cpaw-errors.dat"
    output:
        txt=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/cpaw_orders.txt",
            caption="../report/cpaw_orders.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Circularly Polarized Traveling Alfven Wave",
            labels={"fluid": "{fluid}", "mesh": "uniform", "diagnostic": "convergence orders"}
        )
    params:
        outdir=lambda wildcards: cpaw_out(wildcards.fluid),
    resources:
        runtime=10
    shell:
        """
        cd {params.outdir}
        python {CPAW[order_script]} {input.dat} > {output.txt}
        """
