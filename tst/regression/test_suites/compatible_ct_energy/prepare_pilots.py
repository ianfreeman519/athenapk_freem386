#!/usr/bin/env python3
"""Prepare three isolated 80 ns pilots; never builds, submits, or runs a driver."""
import argparse
from copy import deepcopy
from pathlib import Path


def parse(text):
    result, block = {}, None
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<') and line.endswith('>'):
            block = line[1:-1]
            result.setdefault(block, {})
        elif '=' in line and block:
            key, value = line.split('=', 1)
            result[block][key.strip()] = value.strip()
    return result


def write(path, values):
    path.write_text('\n'.join(f'<{block}>\n'+'\n'.join(
        f'{key} = {value}' for key, value in items.items())+'\n'
        for block, items in values.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--eos-table', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = parse(args.input.read_text())
    table = args.eos_table.resolve()
    if not table.is_file():
        parser.error(f'EOS table not found: {table}')
    if (float(source['units']['code_mass_cgs']) != 1e-10 or
            float(source['units']['code_length_cgs']) != 1 or
            float(source['units']['code_time_cgs']) != 1e-7):
        parser.error('Pilot normalization must be L=1 cm, M=1e-10 g, t=1e-7 s')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('legacy_off', 'legacy_on', 'compatible'):
        cfg = deepcopy(source)
        cfg['problem/pulsed_reconnection'].update(
            rho_background='1e-6', density_perturb_amplitude='0',
            reservoir_mask_enable='false')
        cfg['hydro'].update(dfloor='1e4',
            ct_energy_correction_mode='compatible' if name == 'compatible' else 'legacy',
            ct_energy_correction='true' if name == 'legacy_on' else 'false',
            ct_energy_diagnostics='true' if name == 'compatible' else 'false')
        cfg['tabulated_eos']['filename'] = str(table)
        cfg['tabulated_eos'].pop('fingerprint', None)
        cfg['parthenon/time']['tlim'] = '0.8'
        cfg['parthenon/time']['nlim'] = '-1'
        # Preserve the source resolution, integrator, transport and AMR settings
        # identically. Do not silently fix the separate adaptive-tagging bug.
        for block, values in cfg.items():
            if block.startswith('parthenon/output') and 'variables' in values:
                fields = [v.strip() for v in values['variables'].split(',')]
                fields = [v for v in fields if not v.startswith(('reservoir_mask_', 'ct_energy_'))]
                if name == 'compatible':
                    fields += ['ct_energy_exchange', 'ct_energy_diagnostics']
                values['variables'] = ', '.join(fields)
        directory = args.output/name
        directory.mkdir()
        write(directory/'pulsed_reconnection.in', cfg)
    (args.output/'README.txt').write_text(
        'Prepared only. Run each input fresh, from its own directory, with the same rebuilt driver.\n'
        'Validate analytic tests first. All source mesh/AMR settings were preserved.\n'
        'The density-AMR dispatch bug is intentionally not fixed by this change.\n'
        'Compare temperature/current morphology, dt, raw-energy warnings per cycle, and source budgets.\n'
        'Restart only a compatible-mode checkpoint into the same mode.\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
