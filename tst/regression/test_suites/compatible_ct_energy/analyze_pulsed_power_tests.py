#!/usr/bin/env python3
"""Summarize prepared pulsed-power runs; never launches a simulation.

Regions and integrated quantities are in the input's code units. Temperatures
are kelvin. The z extent is included in integrals, as in the history output.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import h5py
import numpy as np


def snapshot(path):
    with h5py.File(path) as f:
        nx, ny, nz = map(int, f['Info'].attrs['MeshBlockSize'])
        nb = int(f['Info'].attrs['NumMeshBlocks'])
        assert nz == 1, 'These measurements are defined for the 2D matrix'
        x, y = f['VolumeLocations/x'][:], f['VolumeLocations/y'][:]
        dx = np.diff(f['Locations/x'][:], axis=1)
        dy = np.diff(f['Locations/y'][:], axis=1)
        dz = np.diff(f['Locations/z'][:], axis=1)[:, 0]
        volume = (dx[:, None, :]*dy[:, :, None]*dz[:, None, None]).reshape(nb, -1)
        X = np.broadcast_to(x[:, None, :], (nb, ny, nx)).reshape(nb, -1)
        Y = np.broadcast_to(y[:, :, None], (nb, ny, nx)).reshape(nb, -1)
        radius = np.hypot(X, abs(Y)-3.)
        u = f['cons'][:].reshape(nb, 8, -1)
        kinetic = .5*np.sum(u[:, 1:4]**2, axis=1)/u[:, 0]
        magnetic = .5*np.sum(u[:, 5:8]**2, axis=1)
        thermal = u[:, 4]-kinetic-magnetic
        T = f['eos_temperature'][:].reshape(nb, -1)
        eta = f['eta'][:].reshape(nb, -1)
        q = f['ct_energy_exchange'][:].reshape(nb, 2, -1) if 'ct_energy_exchange' in f else None
        row = dict(file=path.name, time_ns=float(f['Info'].attrs['Time'])*100,
                   dt_ns=float(f['Info'].attrs['dt'])*100,
                   cycle=int(f['Info'].attrs['NCycle']), blocks=nb,
                   mass=float(np.sum(u[:, 0]*volume)),
                   total_energy=float(np.sum(u[:, 4]*volume)),
                   kinetic_energy=float(np.sum(kinetic*volume)),
                   magnetic_energy=float(np.sum(magnetic*volume)),
                   thermal_energy=float(np.sum(thermal*volume)),
                   minimum_density_cgs=float(np.min(u[:, 0])*1e-10),
                   maximum_temperature_K=float(np.max(T)),
                   nonpositive_thermal_cells=int(np.count_nonzero(thermal <= 0)))
        b = f['Bface'][:]
        div = ((b[:, 0, 0, :ny, 1:nx+1]-b[:, 0, 0, :ny, :nx])/dx[:, None, :]
               +(b[:, 1, 0, 1:ny+1, :nx]-b[:, 1, 0, :ny, :nx])/dy[:, :, None])
        row['maximum_divB'] = float(np.max(abs(div)))
        scale = float(np.max(np.sqrt(2*magnetic)))
        row['normalized_divB'] = row['maximum_divB']*float(min(dx.min(), dy.min()))/scale if scale > 0 else 0.
        if q is not None:
            row['ideal_exchange'] = float(np.sum(q[:, 0]*volume))
            row['eos_exchange'] = float(np.sum(q[:, 1]*volume))
        for region, lo, hi in [('core', 0, 1.05), ('gap', 1.5, 1.85), ('return', 1.85, 2.1)]:
            mask = (radius > lo) & (radius < hi)
            energy = float(np.sum(thermal[mask]*volume[mask]))
            row[region+'_thermal_energy'] = energy
            for percentile in [1, 50, 99, 100]:
                row[f'{region}_T_p{percentile}_K'] = float(np.percentile(T[mask], percentile))
            row[region+'_density_min_cgs'] = float(np.min(u[:, 0][mask])*1e-10)
            row[region+'_eta_p99_cgs'] = float(np.percentile(eta[mask], 99))
            row[region+'_at_density_floor_fraction'] = float(np.mean(u[:, 0][mask] <= 1e4*(1+1e-8)))
            if q is not None:
                supplied = float(np.sum(q[:, 1][mask]*volume[mask]))
                row[region+'_eos_exchange'] = supplied
                row[region+'_eos_fraction_of_thermal'] = supplied/energy if energy > 0 else None
        return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    root = args.directory.resolve()
    matrix = json.loads((root/'matrix.json').read_text())
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    reports, all_rows = [], {}
    for case in matrix:
        directory = root/case['name']
        rows = sorted((snapshot(p) for p in directory.glob('*.phdf')), key=lambda r: r['time_ns'])
        all_rows[case['name']] = rows
        report = dict(name=case['name'], nx=case['nx'], mode=case['mode'])
        log = (directory/'run.log').read_text(errors='replace') if (directory/'run.log').exists() else ''
        status_file = directory/'exit_status.txt'
        status = int(status_file.read_text()) if status_file.exists() else None
        warnings = [line for line in log.splitlines() if 'CTMHD_EOS_LOW' in line and 'ghost=0' in line]
        matches = list(re.finditer(r'cycle=(\d+) time=([0-9.eE+-]+) dt=([0-9.eE+-]+)', log))
        cycles = int(matches[-1][1]) if matches else 0
        report.update(exit_status=status, interior_raw_energy_warnings=len(warnings),
                      warnings_per_1000_cycles=1000*len(warnings)/cycles if cycles else None,
                      fatal_eos_messages=sum(log.count(v) for v in
                          ['TABULATED_EOS_FAR', 'TABULATED_EOS_INVALID', 'CT_ENERGY_NONPOSITIVE_PRESSURE']),
                      last_logged_time_ns=float(matches[-1][2])*100 if matches else None)
        if rows:
            keys = sorted(set().union(*(row.keys() for row in rows)))
            with (output/(case['name']+'.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            report['last_snapshot'] = rows[-1]
            reached = rows[-1]['time_ns'] >= case['expected_end_ns']-1e-6
            report['status'] = 'completed' if reached and status == 0 else 'incomplete_or_censored'
            if not log:
                report['status'] = 'not_launched_with_saved_log'
        else:
            report['status'] = 'not_started' if not log else 'failed_or_no_output'
        reports.append(report)
        print(case['name'], report['status'], report.get('last_logged_time_ns'),
              'ns; raw-energy warnings:', len(warnings))
    comparisons = []
    for coarse_name, fine_name in [('screen_compatible', 'confirm_compatible'),
                                    ('screen_legacy_off', 'confirm_legacy_off')]:
        for time in [40., 60., 80.]:
            coarse, fine = all_rows[coarse_name], all_rows[fine_name]
            item = dict(coarse=coarse_name, fine=fine_name, target_ns=time)
            if not coarse or not fine:
                item['status'] = 'missing_run'
            else:
                a = min(coarse, key=lambda r: abs(r['time_ns']-time))
                b = min(fine, key=lambda r: abs(r['time_ns']-time))
                if max(abs(a['time_ns']-time), abs(b['time_ns']-time)) > 1.:
                    item['status'] = 'missing_common_time'
                else:
                    item.update(status='sampled', coarse_time_ns=a['time_ns'], fine_time_ns=b['time_ns'])
                    for key in ['gap_thermal_energy', 'gap_T_p99_K', 'return_T_p99_K']:
                        item[key+'_relative_change'] = abs(a[key]-b[key])/max(abs(b[key]), 1e-300)
            comparisons.append(item)
    (output/'summary.json').write_text(json.dumps(dict(runs=reports, resolution_comparisons=comparisons), indent=2, allow_nan=False)+'\n')
    print('Measurements saved in', output)


if __name__ == '__main__':
    main()
