#!/usr/bin/env python3
"""Black-box tests for a prebuilt driver. Run on a compute allocation; never builds.

Use --launcher 'mpiexec -n 2' for MPI. Supply --eos-table for tabulated/Spitzer
coverage and --baseline for bitwise comparison against an unmodified binary.
All simulations have isolated temporary output directories.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shlex
import subprocess
import tempfile

import h5py
import numpy as np


def fixture(n=32):
    return {
        'job': {'problem_id': 'resistive_diffusion'},
        'problem/resistive_diffusion': {
            'iprob': 1, 'rho': 1., 'pressure': 1., 'amp': .02,
            'Bx': .1, 'By': -.1, 'Bz': .03,
        },
        'parthenon/mesh': {
            'refinement': 'none', 'nghost': 2,
            'nx1': n, 'x1min': 0., 'x1max': 1.,
            'nx2': n, 'x2min': 0., 'x2max': 1.,
            'nx3': 1, 'x3min': 0., 'x3max': 1.,
            **{f'{side}x{d}_bc': 'periodic'
               for side in ('i', 'o') for d in (1, 2, 3)},
        },
        'parthenon/meshblock': {'nx1': n//2, 'nx2': n//2, 'nx3': 1},
        'parthenon/time': {'integrator': 'rk2', 'cfl': .3, 'tlim': .02,
                           'nlim': -1, 'ncycle_out': 100},
        'hydro': {'fluid': 'ucthllemhd', 'eos': 'adiabatic', 'riemann': 'hlle',
                  'reconstruction': 'plm', 'gamma': 5/3, 'scratch_level': 1,
                  'ct_energy_correction_mode': 'compatible',
                  'ct_energy_correction': 'false', 'ct_energy_diagnostics': 'true'},
        'diffusion': {'integrator': 'unsplit', 'cfl': .15, 'conduction': 'none',
                      'viscosity': 'none', 'resistivity': 'ohmic',
                      'resistivity_coeff': 'fixed', 'ohm_diff_coeff_code': .05},
        'parthenon/output0': {
            'file_type': 'hdf5', 'dt': .01, 'id': 'out0',
            'variables': 'cons, prim, Bface, ct_energy_exchange, ct_energy_diagnostics'},
        'parthenon/output1': {'file_type': 'hst', 'dn': 1},
    }


def legacy(config, enabled=False, explicit=True):
    c = deepcopy(config)
    c['hydro']['ct_energy_diagnostics'] = 'false'
    c['hydro']['ct_energy_correction'] = str(enabled).lower()
    if explicit:
        c['hydro']['ct_energy_correction_mode'] = 'legacy'
    else:
        c['hydro'].pop('ct_energy_correction_mode', None)
        c['hydro'].pop('ct_energy_diagnostics', None)
    c['parthenon/output0']['variables'] = 'cons, prim, Bface'
    return c


def write_input(path, config):
    path.write_text('\n'.join(f'<{block}>\n' + '\n'.join(
        f'{key} = {value}' for key, value in values.items()) + '\n'
        for block, values in config.items()))


def read(path):
    with h5py.File(path) as f:
        u = f['cons'][:, :, 0]
        nb, _, ny, nx = u.shape
        x, y = f['VolumeLocations/x'][:], f['VolumeLocations/y'][:]
        # All fixtures use one z cell, with unit physical extent.
        dx, dy = np.diff(x, axis=1)[:, 0], np.diff(y, axis=1)[:, 0]
        volume = np.broadcast_to((dx*dy)[:, None, None], (nb, ny, nx))
        ke = .5*np.sum(u[:, 1:4]**2, axis=1)/u[:, 0]
        me = .5*np.sum(u[:, 5:8]**2, axis=1)
        eth = u[:, 4]-ke-me
        q = f['ct_energy_exchange'][:, :, 0] if 'ct_energy_exchange' in f else None
        b = f['Bface'][:]
        div = ((b[:, 0, 0, :ny, 1:nx+1]-b[:, 0, 0, :ny, :nx])/dx[:, None, None]
               +(b[:, 1, 0, 1:ny+1, :nx]-b[:, 1, 0, :ny, :nx])/dy[:, None, None])
        return dict(u=u, p=f['prim'][:, :, 0], x=x, y=y, v=volume, eth=eth,
                    E=np.sum(u[:, 4]*volume), M=np.sum(me*volume),
                    K=np.sum(ke*volume), H=np.sum(eth*volume),
                    q=np.sum(q*volume[:, None], axis=(0, 2, 3)) if q is not None else np.zeros(2),
                    time=float(f['Info'].attrs['Time']), div=float(np.max(abs(div))),
                    blocks=nb, levels=f['Levels'][:],
                    exchange=q)


def snapshots(directory):
    files = list(directory.glob('*.phdf'))
    assert files, f'No PHDF outputs in {directory}'
    return sorted(((read(p), p) for p in files), key=lambda item: item[0]['time'])


def compare(a, b, *, exact=False, tolerance=1e-11):
    assert abs(a['time']-b['time']) < 1e-12
    assert a['u'].shape == b['u'].shape
    for key in ('u', 'p', 'exchange'):
        if a[key] is None or b[key] is None:
            continue
        if exact:
            np.testing.assert_array_equal(a[key], b[key])
        else:
            np.testing.assert_allclose(a[key], b[key], rtol=tolerance, atol=tolerance)


def energy_check(a, b):
    assert a['div'] < 1e-10, (
        f'Initial state has divB={a["div"]}; check initialization before judging CT evolution')
    supplied = np.sum(b['q']-a['q'])
    residual = b['E']-a['E']-supplied
    assert abs(residual) < 2e-10*max(abs(a['E']), 1.), residual
    assert b['div'] < 1e-10, b['div']
    assert np.all(np.isfinite(b['u'])) and np.min(b['eth']) > 0
    # With no floors expected, source accounting should expose any unexpected
    # floor/ceiling interventions rather than letting them hide a thermal failure.
    assert abs(b['q'][1]-a['q'][1]) < 1e-9*max(abs(a['E']), 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', required=True, type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--launcher', default='')
    parser.add_argument('--driver-arg', action='append', default=[],
                        help='Extra argument for every driver invocation; repeatable. '
                             'Use --driver-arg=--kokkos-map-device-id-by=mpi_rank for GPUs.')
    parser.add_argument('--eos-table', type=Path)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--output-dir', type=Path,
                        help='Keep all case outputs here; must not exist')
    args = parser.parse_args()
    driver = args.driver.resolve()
    assert driver.is_file(), driver
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        root = args.output_dir.resolve()
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(prefix='compatible_ct_energy_')
        root = Path(temporary.name)

    def run(name, config, *, binary=driver, failure=None, restart=None):
        directory = root/name
        directory.mkdir()
        inp = directory/'input.in'
        write_input(inp, config)
        cmd = shlex.split(args.launcher)+[str(binary)]
        cmd += (['-r', str(restart), f'parthenon/time/tlim={config["parthenon/time"]["tlim"]}']
                if restart else ['-i', str(inp)])
        cmd += args.driver_arg
        result = subprocess.run(cmd, cwd=directory, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=args.timeout)
        (directory/'run.log').write_text(result.stdout)
        if failure:
            assert result.returncode != 0, f'{name}: unsupported setup accepted'
            assert failure in result.stdout, result.stdout[-4000:]
            print(f'PASS rejection: {name}')
            return None
        assert result.returncode == 0, f'{name}: {result.stdout[-6000:]}'
        if config['hydro']['eos'] == 'tabulated':
            for warning in ('TABULATED_EOS_FAR', 'TABULATED_EOS_INVALID',
                            'Tabulated EOS extrapolated'):
                assert warning not in result.stdout, f'{name}: {warning}'
        states = snapshots(directory)
        assert abs(states[-1][0]['time']-float(config['parthenon/time']['tlim'])) < 1e-10
        print(f'completed {name}', flush=True)
        return states[0][0], states[-1][0], directory

    config = fixture()
    a = run('legacy_default', legacy(config, explicit=False))
    b = run('legacy_explicit', legacy(config))
    compare(a[1], b[1], exact=True)
    if args.baseline:
        baseline = run('legacy_baseline', legacy(config, explicit=False),
                       binary=args.baseline.resolve())
        compare(a[1], baseline[1], exact=True)

    for rk in ('rk1', 'rk2', 'rk3'):
        ideal = fixture()
        ideal['parthenon/time']['integrator'] = rk
        ideal['diffusion']['resistivity'] = 'none'
        ideal['diffusion']['integrator'] = 'none'
        old = run(f'ideal_legacy_{rk}', legacy(ideal, enabled=True))
        new = run(f'ideal_compatible_{rk}', ideal)
        # Without Ohmic induction the physical and ideal-only CT fields must
        # coincide. Check this before comparing evolved states, so a reference
        # curl/sign error is reported directly rather than as downstream drift.
        with h5py.File(snapshots(new[2])[-1][1]) as f:
            residual = np.max(abs(f['ct_energy_diagnostics'][:, 1]))
            assert residual < 1e-12, (
                f'{rk}: nonzero resistive magnetic change with resistivity=none: '
                f'{residual}')
        compare(old[1], new[1], tolerance=3e-12)
        energy_check(new[0], new[1])
        for iprob in (1, 2, 10):
            resistive = fixture()
            resistive['parthenon/time']['integrator'] = rk
            resistive['problem/resistive_diffusion']['iprob'] = iprob
            case = run(f'resistive_{rk}_{iprob}', resistive)
            energy_check(case[0], case[1])
            loss = case[0]['M']-case[1]['M']
            assert loss > 0
            gain = case[1]['H']-case[0]['H']+case[1]['K']-case[0]['K']
            np.testing.assert_allclose(gain, loss+np.sum(case[1]['q']-case[0]['q']),
                                       rtol=1e-6, atol=1e-10)
            assert gain > .5*loss

    # The recorded EOS adjustment must include floor interventions at startup.
    cfg = fixture()
    cfg['problem/resistive_diffusion'].update(iprob=0, pressure=.1)
    cfg['hydro']['pfloor'] = .2
    case = run('floor_accounting', cfg)
    np.testing.assert_allclose(case[0]['q'][1], .1/(5/3-1), rtol=1e-12, atol=1e-12)

    # Fully specified ideal-gas Spitzer path, clamped to a known coefficient.
    cfg = fixture()
    cfg['units'] = dict(code_length_cgs=1., code_mass_cgs=1e-10, code_time_cgs=1e-7)
    cfg['hydro'].update(mean_molecular_weight=6.75, mean_ionization_state=3.)
    cfg['diffusion'].update(resistivity_coeff='spitzer', spitzer_log_lambda=10.,
                            spitzer_eta_min=1e5, spitzer_eta_max=1e5)
    case = run('ideal_gas_spitzer', cfg)
    energy_check(case[0], case[1])

    errors = []
    for n in (16, 32, 64):
        cfg = fixture(n)
        case = run(f'fourier_convergence_{n}', cfg)
        final = case[1]
        x = np.broadcast_to(final['x'][:, None, :], final['eth'].shape)
        expected = .02*np.exp(-.05*(2*np.pi)**2*.02)*np.sin(2*np.pi*x)
        errors.append(np.sum(abs(final['u'][:, 6]-expected)*final['v'])/.02)
        energy_check(case[0], final)
    assert errors[-1] < .05 and errors[-1] < errors[0]/2, errors

    errors = []
    for n in (32, 64, 128):
        cfg = fixture(n)
        cfg['problem/resistive_diffusion'].update(
            iprob=30, pressure=1e-8, amp=.01, circular_force_balance='true')
        cfg['diffusion'].update(resistivity='none', integrator='none')
        cfg['parthenon/time']['tlim'] = .002
        cfg['parthenon/output0']['dt'] = .002
        case = run(f'circular_low_beta_{n}', cfg)
        final = case[1]
        r = np.hypot(final['x'][:, None, :]-.5, final['y'][:, :, None]-.5)
        band = (r > .14) & (r < .28)
        relative = (final['eth']-1e-8/(5/3-1))/(1e-8/(5/3-1))
        errors.append(np.sqrt(np.sum(relative[band]**2*final['v'][band])/
                              np.sum(final['v'][band])))
        energy_check(case[0], final)
    assert errors[-1] < errors[0]/2, errors

    # Force a coarse/fine interface without depending on the deferred AMR tagging fix.
    cfg = fixture(32)
    # Use face-integrated fields from a discrete curl of A_z. A face-center
    # sine sample (iprob=1) is solenoidal on a uniform grid, but its coarse
    # value differs from the average of fine samples. Restricting that normal
    # field at a coarse/fine interface injects divergence at initialization.
    cfg['problem/resistive_diffusion']['iprob'] = 10
    cfg['parthenon/mesh'].update(refinement='static', numlevel=2)
    cfg['parthenon/meshblock'].update(nx1=8, nx2=8)
    cfg['parthenon/static_refinement0'] = dict(x1min=.25, x1max=.75,
                                              x2min=.25, x2max=.75, level=1)
    case = run('static_refinement', cfg)
    assert len(np.unique(case[1]['levels'])) == 2
    energy_check(case[0], case[1])

    cfg = fixture()
    cfg['hydro'].update(fluid='ucthlldmhd', riemann='hlld')
    case = run('uct_hlld', cfg)
    energy_check(case[0], case[1])

    cfg = fixture()
    # Binary-exact checkpoint/end times and an identical safe timestep sequence
    # ensure this compares restart handling rather than different final-step cuts.
    cfg['parthenon/time'].update(tlim=.03125, dt_force=1/4096)
    full = run('restart_reference', cfg)
    half = deepcopy(cfg)
    half['parthenon/time']['tlim'] = .015625
    half['parthenon/output2'] = dict(file_type='rst', dt=.015625)
    first = run('restart_first_half', half)
    checkpoints = sorted(first[2].glob('*.rhdf'))
    checkpoint = next(p for p in checkpoints if abs(read_restart_time(p)-.015625) < 1e-10)
    resumed = run('restart_continuation', cfg, restart=checkpoint)
    compare(full[1], resumed[1], tolerance=2e-10)
    np.testing.assert_allclose(full[1]['q'], resumed[1]['q'], rtol=1e-9, atol=1e-11)

    bad = fixture()
    bad['hydro']['ct_energy_correction'] = 'true'
    run('reject_boolean', bad, failure='requires ct_energy_correction=false')
    bad = fixture()
    bad['hydro']['ct_energy_correction_mode'] = 'unknown'
    run('reject_mode', bad, failure='must be legacy or compatible')
    bad = fixture()
    bad['diffusion']['integrator'] = 'rkl2'
    run('reject_rkl2', bad, failure='compatible CT energy supports')
    for operator in ('conduction', 'viscosity'):
        bad = fixture()
        bad['diffusion'][operator] = 'isotropic'
        bad['diffusion'][f'{operator}_coeff'] = 'fixed'
        bad['diffusion']['thermal_diff_coeff_code' if operator == 'conduction'
                         else 'mom_diff_coeff_code'] = .01
        run(f'reject_{operator}', bad, failure='compatible CT energy supports')
    bad = fixture()
    bad['parthenon/mesh']['nx3'] = 4
    bad['parthenon/meshblock']['nx3'] = 4
    run('reject_3d', bad, failure='requires 2D UCT')

    if args.eos_table:
        table = args.eos_table.resolve()
        assert table.is_file(), table
        for spitzer in (False, True):
            cfg = fixture()
            cfg['hydro'].update(eos='tabulated', Tfloor=1000, dfloor=1e4)
            cfg['units'] = dict(code_length_cgs=1., code_mass_cgs=1e-10, code_time_cgs=1e-7)
            cfg['tabulated_eos'] = dict(filename=table, density_to_code=1e7,
                specific_internal_energy_to_code=1e-10, pressure_to_code=1e-3,
                sound_speed_to_code=1e-5, temperature_to_kelvin=11604.51812155008,
                mean_ion_mass_hydrogen=27.)
            cfg['problem/resistive_diffusion'].update(rho=1e4, pressure=3., amp=.01)
            if spitzer:
                cfg['diffusion'].update(resistivity_coeff='spitzer', spitzer_log_lambda=10.,
                                        spitzer_eta_min=1e3, spitzer_eta_max=1e7)
            cfg['parthenon/output0']['variables'] += ', eos_temperature'
            case = run(f'tabulated_{"spitzer" if spitzer else "fixed"}', cfg)
            energy_check(case[0], case[1])
            with h5py.File(snapshots(case[2])[-1][1]) as f:
                assert np.all(np.isfinite(f['eos_temperature'][:]))
                assert np.min(f['eos_temperature'][:]) >= 999.99
    else:
        print('SKIP tabulated EOS cases: supply --eos-table for full coverage')
    # Use an existing valid masked input, only changing its mode. This checks
    # the actual problem-installed mask, not an unused input key.
    source = Path(__file__).resolve().parents[4]/'inputs/ct_glm_compatible/pulsed_reconnection_reservoir_mask.in'
    text = source.read_text()
    import re
    text = re.sub(r'(?m)^ct_energy_correction\s*=.*$', 'ct_energy_correction = false', text)
    # The fixture uses an ideal EOS and needs no external table.
    text = text.replace('<hydro>', '<hydro>\nct_energy_correction_mode = compatible')
    directory = root/'reject_mask'
    directory.mkdir()
    (directory/'input.in').write_text(text)
    result = subprocess.run(shlex.split(args.launcher)+[str(driver), '-i', 'input.in']
                            + args.driver_arg,
                            cwd=directory, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=args.timeout)
    (directory/'run.log').write_text(result.stdout)
    assert result.returncode != 0 and 'does not yet support the reservoir mask' in result.stdout
    print('PASS compatible CT energy tests; outputs:', root)
    if temporary:
        temporary.cleanup()


def read_restart_time(path):
    with h5py.File(path) as f:
        return float(f['Info'].attrs['Time'])


if __name__ == '__main__':
    main()
