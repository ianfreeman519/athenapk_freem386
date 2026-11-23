"""Exact force-free diffusion references and independent heating verification."""
import json
from copy import deepcopy

import numpy as np

ETA = .05
GAMMA = 5/3
P0 = .1


def reference(state, oblique, eta=ETA):
    kx, ky = 2*np.pi, 4*np.pi if oblique else 0.
    wave = np.hypot(kx, ky)
    x = state['x'][:, None, :]
    y = state['y'][:, :, None]
    dx = np.diff(state['x'], axis=1)[:, :1, None]
    dy = np.diff(state['y'], axis=1)[:, :1, None]
    average = np.sinc(kx*dx/(2*np.pi))*np.sinc(ky*dy/(2*np.pi))
    phase = kx*x+ky*y
    decay = np.exp(-eta*wave**2*state['time'])
    field = np.stack(np.broadcast_arrays(-ky/wave*average*np.cos(phase),
                                         kx/wave*average*np.cos(phase),
                                         average*np.sin(phase)), axis=1)*decay
    heat = .5*(-np.expm1(-2*eta*wave**2*state['time']))
    return field, heat


def metrics(state, oblique, eta=ETA):
    expected, heat = reference(state, oblique, eta)
    volume = state['v']
    mean = lambda values: float(np.sum(values*volume)/np.sum(volume))
    return dict(time=state['time'],
                magnetic_error=mean(np.mean(abs(state['u'][:, 5:8]-expected), axis=1)),
                heating_error=mean(abs(state['eth']-P0/(GAMMA-1)-heat))/max(heat, 1e-14),
                normalized_divergence=float(state.get('div', 0.)*min(np.min(np.diff(state['x'])), np.min(np.diff(state['y'])))),
                density_error=mean(abs(state['u'][:, 0]-1)),
                velocity_rms=np.sqrt(mean(np.sum((state['u'][:, 1:4]/state['u'][:, :1])**2, axis=1))),
                ideal_exchange=float(state['q'][0]), eos_exchange=float(state['q'][1]))


def accept(row):
    assert row['normalized_divergence'] < 1e-10, row
    assert row['magnetic_error'] < .02, row
    assert row['heating_error'] < .02, row
    assert row['density_error'] < .01, row
    assert row['velocity_rms'] < .01, row


def run_suite(run, fixture, legacy, snapshots, energy_check, root, compare_mpi=False):
    rows = []

    def execute(name, cfg, oblique, eta=ETA, diagnostic=False):
        try:
            initial, final, directory = run(name, cfg)
        except AssertionError as error:
            if not diagnostic:
                raise
            rows.append(dict(case=name, diagnostic_failure=str(error)))
            (root/'analytic_results.json').write_text(json.dumps(rows, indent=2))
            print(f'DIAGNOSTIC FAILURE {name}; see run.log', flush=True)
            return None, None
        if not diagnostic:
            energy_check(initial, final)
        states = [state for state, _ in snapshots(directory) if state['time'] > 1e-12]
        result = []
        for state in states:
            assert np.all(np.isfinite(state['u'])) and np.min(state['eth']) > 0
            if not diagnostic:
                energy_check(initial, state)
            row = dict(case=name, **metrics(state, oblique, eta))
            rows.append(row)
            (root/'analytic_results.json').write_text(json.dumps(rows, indent=2))
            result.append(row)
        return final, result[-1]

    for oblique in (False, True):
        label = 'oblique' if oblique else 'axis'
        sequence = []
        for n in (32, 64, 128):
            cfg = fixture(n)
            cfg['problem/resistive_diffusion'].update(iprob=41 if oblique else 40,
                amp=1., rho=1., pressure=P0, mode_x=1, mode_y=2)
            if not oblique:
                cfg['parthenon/mesh']['nx2'] = 4
                cfg['parthenon/meshblock']['nx2'] = 4
            end = .2/(ETA*(2*np.pi)**2*(5 if oblique else 1))
            cfg['parthenon/time']['tlim'] = end
            cfg['parthenon/output0']['dt'] = end/2
            final, row = execute(f'{label}_{n}', cfg, oblique)
            sequence.append(row)
        if compare_mpi:
            _, serial, _ = run(f'{label}_serial_reference', cfg, launcher='')
            def ordered(state):
                x, y = np.broadcast_arrays(state['x'][:, None, :], state['y'][:, :, None])
                order = np.lexsort((x.ravel(), y.ravel()))
                return np.moveaxis(state['u'], 1, -1).reshape(-1, 8)[order]
            np.testing.assert_allclose(ordered(final), ordered(serial), rtol=1e-10, atol=1e-11)
        for row in rows:
            if row['case'] == f'{label}_128':
                accept(row)
        accept(sequence[-1])
        for key in ('magnetic_error', 'heating_error'):
            rate = np.log2(sequence[-2][key]/sequence[-1][key])
            assert np.isfinite(rate) and rate >= 1.7, (label, key, rate)
        half = deepcopy(cfg)
        half['diffusion']['cfl'] *= .5
        _, row = execute(f'{label}_half_cfl', half, oblique)
        accept(row)
        for rk in ('rk1', 'rk3'):
            other = deepcopy(cfg)
            other['parthenon/time']['integrator'] = rk
            _, row = execute(f'{label}_{rk}', other, oblique)
            accept(row)
        for enabled in (False, True):
            execute(f'{label}_legacy_{enabled}', legacy(cfg, enabled), oblique,
                    diagnostic=True)
        ideal = deepcopy(cfg)
        ideal['diffusion'].update(resistivity='none', integrator='none')
        final, _ = execute(f'{label}_ideal', ideal, oblique, eta=0.)
        assert np.max(abs(final['eth']-P0/(GAMMA-1))) < .01

    uniform = fixture()
    uniform['problem/resistive_diffusion'].update(iprob=0, pressure=P0)
    initial, final, _ = run('analytic_uniform', uniform)
    energy_check(initial, final)
    np.testing.assert_allclose(final['u'], initial['u'], atol=1e-12, rtol=1e-12)
    (root/'analytic_results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label in ('axis', 'oblique'):
        selected = [next(r for r in reversed(rows) if r['case'] == f'{label}_{n}')
                    for n in (32, 64, 128)]
        for ax, key in zip(axes, ('magnetic_error', 'heating_error')):
            ax.loglog((32, 64, 128), [r[key] for r in selected], 'o-', label=label)
            ax.set(xlabel='N', ylabel=key)
            ax.legend()
            ax.grid(True)
    fig.tight_layout()
    fig.savefig(root/'analytic_convergence.png')
    plt.close(fig)


def self_test():
    # Magnetic decay alone must never let missing thermal heating pass.
    n = 16
    centers = (np.arange(n)+.5)/n
    state = dict(x=centers[None], y=centers[None], v=np.ones((1,n,n))/n**2,
                 time=.1, u=np.zeros((1,8,n,n)), eth=np.full((1,n,n), P0/(GAMMA-1)),
                 q=np.zeros(2))
    state['u'][:,0] = 1
    state['u'][:,5:8], heat = reference(state, True)
    bad = metrics(state, True)
    assert bad['magnetic_error'] == 0 and abs(bad['heating_error']-1) < 1e-12
    try:
        accept(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing heating was accepted')
    state['eth'] += heat
    accept(metrics(state, True))
    print('PASS analytic reference and missing-heating rejection')


if __name__ == '__main__':
    self_test()
