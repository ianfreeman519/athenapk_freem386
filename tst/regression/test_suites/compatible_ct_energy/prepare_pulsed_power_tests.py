#!/usr/bin/env python3
"""Prepare the small pulsed-power matrix and launch scripts. Never runs a driver."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from prepare_pilots import parse, write


def launcher(cases, ranks):
    # One process per GPU in the screen; two processes per confirmation case.
    names = ' '.join(cases)
    return f'''#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then echo "Usage: $0 /absolute/path/to/rebuilt/athenaPK" >&2; exit 2; fi
driver=$(realpath -- "$1")
[[ -x "$driver" ]] || {{ echo "Executable not found: $driver" >&2; exit 2; }}
root=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
export OMP_NUM_THREADS=1
IFS=, read -ra devices <<< "${{CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
[[ ${{#devices[@]}} -ge 4 ]] || {{ echo "Four visible GPUs required" >&2; exit 2; }}
names=({names})
for name in "${{names[@]}}"; do
  [[ ! -e "$root/$name/run.log" ]] || {{ echo "Run already exists: $name" >&2; exit 2; }}
done
pids=()
for index in "${{!names[@]}}"; do
  name=${{names[index]}}
  first=$((index * {ranks}))
  gpu_list=${{devices[first]}}
  if [[ {ranks} -eq 2 ]]; then gpu_list+=",${{devices[first+1]}}"; fi
  (
    cd -- "$root/$name"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu_list" timeout --signal=TERM --kill-after=30s \
      "${{CASE_TIMEOUT:-{'45m' if ranks == 1 else '90m'}}}" \
      mpirun -n {ranks} --bind-to none "$driver" -i pulsed_reconnection.in \
        --kokkos-map-device-id-by=mpi_rank > run.log 2>&1
    code=$?
    printf '%s\\n' "$code" > exit_status.txt
    exit "$code"
  ) &
  pids+=("$!")
  echo "Started $name on GPU(s) $gpu_list"
done
failed=0
for index in "${{!pids[@]}}"; do
  if wait "${{pids[index]}}"; then
    echo "Finished ${{names[index]}}; check final physical time before accepting"
  else
    echo "Stopped/failed ${{names[index]}}; inspect run.log and exit_status.txt" >&2
    failed=1
  fi
done
exit "$failed"
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--eos-table', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    source = parse(args.input.read_text())
    units = source['units']
    if tuple(float(units[k]) for k in ('code_length_cgs', 'code_mass_cgs', 'code_time_cgs')) != (1., 1e-10, 1e-7):
        ap.error('Required normalization: L=1 cm, M=1e-10 g, time=1e-7 s')
    table = args.eos_table.resolve()
    if not table.is_file():
        ap.error(f'Missing EOS table: {table}')
    base = deepcopy(source)
    base['problem/pulsed_reconnection'].update(
        reservoir_mask_enable='false', inner_reservoir_support='none',
        rho_background='1e-6', T_background='1e3', rho_inner_reservoir='1e-1',
        density_perturb_amplitude='0', temperature_perturb_amplitude='0', N='0',
        drive_enable='true', B_peak_gauss='0', drive_B_peak_gauss='2e6',
        drive_t_peak_ns='300', v0='0', drive_velocity_mode='none',
        drive_rho_profile='none', drive_T_profile='none',
        initial_force_balance='false', drive_force_balance='false')
    # Pin the geometry, profiles and boundaries to the investigated case.
    base['problem/pulsed_reconnection'].update(
        array_separation='6', inner_reservoir_radius='1.5',
        inner_reservoir_rho_transition_width='.45',
        inner_reservoir_T_transition_width='.45', T_inner_reservoir='1e3',
        initial_rho_profile='annulus', rho_wire='3.4e-3',
        initial_rho_inner_radius='1.95', initial_rho_outer_radius='2.05',
        initial_rho_transition_width='.05',
        initial_T_profile='annulus', T_wire='1.16e3',
        initial_T_inner_radius='1.95', initial_T_outer_radius='2.05',
        initial_T_transition_width='.05', drive_magnetic_profile='confined_current',
        drive_magnetic_conductor_radius='1', drive_magnetic_conductor_transition_width='.5',
        drive_magnetic_wire_inner_radius='1.9', drive_magnetic_wire_outer_radius='2.05',
        drive_magnetic_wire_transition_width='.099999999999')
    base['hydro'].update(fluid='ucthllemhd', eos='tabulated', riemann='hlle',
                         reconstruction='plm', dfloor='1e4', Tfloor='1e3')
    base['tabulated_eos'].update(filename=str(table), density_to_code='1e7',
        specific_internal_energy_to_code='1e-10', pressure_to_code='1e-3',
        sound_speed_to_code='1e-5', temperature_to_kelvin='11604.51812155008',
        mean_ion_mass_hydrogen='27')
    base['tabulated_eos'].pop('fingerprint', None)
    base['diffusion'].update(integrator='unsplit', cfl='.15', conduction='none',
        viscosity='none', resistivity='ohmic', resistivity_coeff='spitzer',
        spitzer_log_lambda='10', spitzer_eta_min='1e3', spitzer_eta_max='1e7')
    base['cooling']['enable_cooling'] = 'none'
    base['parthenon/mesh'].update(refinement='none', numlevel='1', nghost='2', nx3='1',
        x1min='-6', x1max='6', x2min='-6', x2max='6', x3min='-1', x3max='1',
        ix1_bc='outflow', ox1_bc='outflow', ix2_bc='outflow', ox2_bc='outflow',
        ix3_bc='periodic', ox3_bc='periodic')
    for key in list(base):
        if key == 'refinement' or key.startswith(('parthenon/refinement', 'parthenon/static_refinement', 'parthenon/output')):
            del base[key]
    # Start with a fresh time block so no copied forced/floored timestep survives.
    base['parthenon/time'] = dict(integrator='rk2', cfl='.4', tlim='.8',
                                  ncycle_out='100', dt_min='1e-8', dt_min_cycle_limit='100')
    matrix = [
        ('screen_legacy_off', 512, 'legacy', False, True),
        ('screen_legacy_on', 512, 'legacy', True, True),
        ('screen_compatible', 512, 'compatible', False, True),
        ('screen_no_drive', 512, 'compatible', False, False),
        ('confirm_legacy_off', 1024, 'legacy', False, True),
        ('confirm_compatible', 1024, 'compatible', False, True),
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    fields = ['prim', 'cons', 'Bface', 'divB', 'curlBz', 'eta', 'beta',
              'eos_temperature', 'eos_zbar']
    for name, n, mode, old_correction, drive in matrix:
        cfg = deepcopy(base)
        cfg['hydro'].update(ct_energy_correction_mode=mode,
            ct_energy_correction=str(old_correction).lower(),
            ct_energy_diagnostics=str(mode == 'compatible').lower())
        cfg['problem/pulsed_reconnection']['drive_enable'] = str(drive).lower()
        cfg['parthenon/mesh'].update(nx1=str(n), nx2=str(n))
        cfg['parthenon/meshblock'].update(nx1='128', nx2='128', nx3='1')
        cfg['parthenon/time']['nlim'] = '100000' if n == 512 else '250000'
        variables = fields + (['ct_energy_exchange', 'ct_energy_diagnostics'] if mode == 'compatible' else [])
        cfg['parthenon/output0'] = dict(file_type='hdf5', dt='.02', id='out0', variables=', '.join(variables))
        cfg['parthenon/output1'] = dict(file_type='hst', dn='100')
        cfg['parthenon/output2'] = dict(file_type='rst', dt='.2')
        directory = args.output/name
        directory.mkdir()
        write(directory/'pulsed_reconnection.in', cfg)
        manifest.append(dict(name=name, nx=n, mode=mode, drive=drive,
                             expected_end_ns=80, dx_cm=12/n,
                             transition_cells=.05/(12/n)))
    (args.output/'matrix.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'run_screen.sh').write_text(launcher([m[0] for m in matrix[:4]], 1))
    (args.output/'run_confirmation.sh').write_text(launcher([m[0] for m in matrix[4:]], 2))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
