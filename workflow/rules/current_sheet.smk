CURRENT_SHEET = config["tests"]["current_sheet"]
CURRENT_SHEET_RESOLUTION = CURRENT_SHEET["resolution"]
CURRENT_SHEET_FLUIDS = CURRENT_SHEET.get("fluids", config["fluids"])

def current_sheet_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{CURRENT_SHEET['dirname']}"

current_sheet_targets = []
if CURRENT_SHEET["enabled"]:
    current_sheet_targets = expand(
        "{outdir}/{name}",
        outdir=[current_sheet_out(fluid) for fluid in CURRENT_SHEET_FLUIDS],
        name=[
            "current_sheet_By.gif",
            "current_sheet_field_lines_initial.png",
            "current_sheet_field_lines_middle.png",
            "current_sheet_field_lines_final.png",
        ],
    )

rule run_current_sheet:
    output:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/phdf-files/run.done"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/run.err"
    params:
        outdir=lambda wc: current_sheet_out(wc.fluid),
        rundir=lambda wc: f"{current_sheet_out(wc.fluid)}/phdf-files",
        problem_id=lambda wc: f"current_sheet_{wc.fluid}_Nx{CURRENT_SHEET_RESOLUTION[0]}x{CURRENT_SHEET_RESOLUTION[1]}",
        nx1=CURRENT_SHEET_RESOLUTION[0],
        nx2=CURRENT_SHEET_RESOLUTION[1],
    resources:
        runtime=30
        #mem_mb=(default for now) mb means megabyte
        #slurm_partition=(default for now)
        # see https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
        # for more options (number of tasks per job, cpu per task, mem per cpu, etc)
        #hydro/pfloor=1e-15 \
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {config[athenapk]} -i {CURRENT_SHEET[input]} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3=1 \
          parthenon/meshblock/nx1={params.nx1} \
          parthenon/meshblock/nx2={params.nx2} \
          parthenon/meshblock/nx3=1 \
          parthenon/time/tlim=10.0 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/gamma=1.666666666666667 \
          hydro/ct_energy_correction=true \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt=0.05 \
          parthenon/output0/variables=prim \
          > {log.out} 2> {log.err}
    
        touch {output.done}
        """

rule make_current_sheet_videos:
    input:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/phdf-files/run.done"
    output:
        gif=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/current_sheet_By.gif",
            caption="../report/current_sheet_By.rst",
            category="2D Tests",
            subcategory="{fluid} / Current Sheet",
            labels={"fluid": "{fluid}", "quantity": "magnetic field By"}
        )
    params:
        outdir=lambda wc: current_sheet_out(wc.fluid),
        phdf=lambda wc: f"{current_sheet_out(wc.fluid)}/phdf-files",
    resources:
        runtime=30
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {CURRENT_SHEET[plotting_script]} {params.phdf} -o current_sheet
        """

rule make_current_sheet_field_lines:
    input:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/phdf-files/run.done"
    output:
        initial=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/current_sheet_field_lines_initial.png",
            caption="../report/current_sheet_field_lines.rst",
            category="2D Tests",
            subcategory="{fluid} / Current Sheet",
            labels={"fluid": "{fluid}", "snapshot": "initial", "diagnostic": "magnetic field lines"}
        ),
        middle=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/current_sheet_field_lines_middle.png",
            caption="../report/current_sheet_field_lines.rst",
            category="2D Tests",
            subcategory="{fluid} / Current Sheet",
            labels={"fluid": "{fluid}", "snapshot": "middle", "diagnostic": "magnetic field lines"}
        ),
        final=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/current_sheet_field_lines_final.png",
            caption="../report/current_sheet_field_lines.rst",
            category="2D Tests",
            subcategory="{fluid} / Current Sheet",
            labels={"fluid": "{fluid}", "snapshot": "final", "diagnostic": "magnetic field lines"}
        )
    params:
        outdir=lambda wc: current_sheet_out(wc.fluid),
        phdf=lambda wc: f"{current_sheet_out(wc.fluid)}/phdf-files",
    resources:
        runtime=30
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {CURRENT_SHEET[field_line_script]} {params.phdf} -o current_sheet
        """
