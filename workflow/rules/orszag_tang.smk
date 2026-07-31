ORSZAG_TANG = config["tests"]["orszag_tang"]
ORSZAG_TANG_RESOLUTION = ORSZAG_TANG["resolution"]

def orszag_tang_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{ORSZAG_TANG['dirname']}"

orszag_tang_targets = []
if ORSZAG_TANG["enabled"]:
    orszag_tang_targets = expand(
        "{outdir}/{name}",
        outdir=[orszag_tang_out(fluid) for fluid in config["fluids"]],
        name=[
            "orszag_tang_pressure.gif",
            "orszag_tang_Bmag.gif",
            "orszag_tang_pressure_rot180_abs.gif",
        ],
    )

rule run_orszag_tang:
    output:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/phdf-files/run.done"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/run.err"
    params:
        outdir=lambda wc: orszag_tang_out(wc.fluid),
        rundir=lambda wc: f"{orszag_tang_out(wc.fluid)}/phdf-files",
        problem_id=lambda wc: f"orszag_tang_{wc.fluid}_Nx{ORSZAG_TANG_RESOLUTION[0]}x{ORSZAG_TANG_RESOLUTION[1]}",
        nx1=ORSZAG_TANG_RESOLUTION[0],
        nx2=ORSZAG_TANG_RESOLUTION[1],
    resources:
        runtime=30
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

        {config[athenapk]} -i {ORSZAG_TANG[input]} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3=1 \
          parthenon/meshblock/nx1={params.nx1} \
          parthenon/meshblock/nx2={params.nx2} \
          parthenon/meshblock/nx3=1 \
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
          > {log.out} 2> {log.err}
    
        touch {output.done}
        """

rule make_orszag_tang_videos:
    input:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/phdf-files/run.done"
    output:
        pressure=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/orszag_tang_pressure.gif",
            caption="../report/orszag_tang_pressure.rst",
            category="2D Tests",
            subcategory="{fluid} / Orszag-Tang",
            labels={"fluid": "{fluid}", "quantity": "pressure"}
        ),
        bmag=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/orszag_tang_Bmag.gif",
            caption="../report/orszag_tang_Bmag.rst",
            category="2D Tests",
            subcategory="{fluid} / Orszag-Tang",
            labels={"fluid": "{fluid}", "quantity": "magnetic field magnitude"}
        ),
        pressure_rot180_abs=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/orszag_tang_pressure_rot180_abs.gif",
            caption="../report/orszag_tang_pressure_rot180_abs.rst",
            category="2D Tests",
            subcategory="{fluid} / Orszag-Tang",
            labels={"fluid": "{fluid}", "quantity": "180 degree pressure symmetry error"}
        )
    params:
        outdir=lambda wc: orszag_tang_out(wc.fluid),
        phdf=lambda wc: f"{orszag_tang_out(wc.fluid)}/phdf-files",
    resources:
        runtime=30
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {ORSZAG_TANG[plotting_script]} {params.phdf} -o orszag_tang
        """
