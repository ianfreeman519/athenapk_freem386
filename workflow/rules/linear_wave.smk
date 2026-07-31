LIN_WAVE = config["tests"]["linear_wave"]
LIN_WAVE_RESOLUTIONS = LIN_WAVE["resolutions"]
LIN_WAVE_WAVES = LIN_WAVE["waves"]

def linear_wave_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{LIN_WAVE['dirname']}"

linear_wave_targets = []
if LIN_WAVE["enabled"]:
    linear_wave_targets = expand(
        "{outdir}/wave_{wave}/linear_wave_orders-{wave}.txt",
        outdir=[linear_wave_out(fluid) for fluid in config["fluids"]],
        wave=LIN_WAVE_WAVES,
    )

rule run_linear_wave:
    output:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{LIN_WAVE['dirname']}/wave_{{wave}}/runs/N{{N}}/linearwave-errors-{{wave}}.dat"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{LIN_WAVE['dirname']}/wave_{{wave}}/runs/N{{N}}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{LIN_WAVE['dirname']}/wave_{{wave}}/runs/N{{N}}/run.err"
    params:
        outdir=lambda wildcards: linear_wave_out(wildcards.fluid),
        rundir=lambda wildcards: f"{linear_wave_out(wildcards.fluid)}/wave_{wildcards.wave}/runs/N{wildcards.N}",
        problem_id=lambda wildcards: f"linear_wave_{wildcards.fluid}_w{wildcards.wave}_N{wildcards.N}",
        nx1=lambda wildcards: 2 * int(wildcards.N),
        nx2=lambda wildcards: int(wildcards.N),
        nx3=lambda wc: 1 if config["dimension"] == "2D" else int(wc.N),
        nx1_mb=lambda wildcards: 2 * (int(wildcards.N) // 4),
        nx2_mb=lambda wildcards: (int(wildcards.N) // 2),
        nx3_mb=lambda wc: 1 if config["dimension"] == "2D" else (int(wc.N) // 2),
        ang_2=lambda wc: LIN_WAVE["mesh"]["2D"]["ang_2"] if config["dimension"] == "2D" else -999.9
    resources:
        runtime=120,
        nodes=1,
        tasks=16,
        mpi="srun",
        mem_mb_per_cpu=2000
        #slurm_partition=(default for now)
        # see https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
        # for more options (number of tasks per job, cpu per task, mem per cpu, etc)
        #hydro/pfloor=1e-15 \
    shell:
        """
        mkdir -p {params.rundir}
        cd {params.rundir}
        rm -f {output.dat}

        {resources.mpi} -n {resources.tasks} {config[athenapk_mpi]} -i {LIN_WAVE[input]} \
          parthenon/job/problem_id={params.problem_id} \
          problem/linear_wave_mhd/wave_flag={wildcards.wave} \
          problem/linear_wave_mhd/compute_error=true \
          problem/linear_wave_mhd/ang_2={params.ang_2} \
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

rule combine_linear_wave_errors:
    input:
        lambda wildcards: expand(
            f"{linear_wave_out(wildcards.fluid)}/wave_{wildcards.wave}/runs/N{{N}}/linearwave-errors-{wildcards.wave}.dat",
            N=LIN_WAVE_RESOLUTIONS,
        )
    output:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{LIN_WAVE['dirname']}/wave_{{wave}}/linearwave-errors-{{wave}}.dat"
    resources:
        runtime=10
    shell:
        """
        cat {input} > {output.dat}
        """

rule compute_linear_wave_orders:
    input:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{LIN_WAVE['dirname']}/wave_{{wave}}/linearwave-errors-{{wave}}.dat"
    output:
        txt=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{LIN_WAVE['dirname']}/wave_{{wave}}/linear_wave_orders-{{wave}}.txt",
            caption="../report/linear_wave_orders.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Linear MHD Wave",
            labels={"fluid": "{fluid}", "mesh": "uniform", "wave": "{wave}", "diagnostic": "convergence orders"}
        )
    params:
        outdir=lambda wildcards: linear_wave_out(wildcards.fluid),
    resources:
        runtime=10
    shell:
        """
        cd {params.outdir}/wave_{wildcards.wave}
        python {LIN_WAVE[order_script]} {input.dat} > {output.txt}
        """
