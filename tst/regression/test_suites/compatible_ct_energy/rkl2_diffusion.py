"""RKL2 compatible-energy validation against exact and converged MHD references.

This module never builds a driver. Its self-test uses no driver or GPU.
"""
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np

from analytic_diffusion import reference as fixed_reference

GAMMA = 5/3
P0 = .1
ETA = .05
# Same documented physical constants as Units, evaluated independently in Python.
MH = 1.007947*1.660538921e-24
KB = 1.3806488e-16
TEMPERATURE_FACTOR = MH/KB*1.e14  # mu=1, length=1 cm, time=1e-7 s


def average(values, volume):
    return float(np.sum(values*volume)/np.sum(volume))


def vector_metrics(actual, expected, volume, initial_rms):
    """Signed errors everywhere; angles exclude only weak *reference* fields."""
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    norm = max(initial_rms, 1.e-30)
    error = actual-expected
    a, b = np.moveaxis(actual, 1, -1), np.moveaxis(expected, 1, -1)
    good = np.linalg.norm(b, axis=-1) >= 1.e-6*norm
    assert np.any(good), 'No reference field directions above cutoff'
    angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b), axis=-1),
                                 np.sum(a*b, axis=-1)))
    component = [average(abs(error[:,i]),volume)/norm for i in range(3)]
    return dict(magnetic_error=max(component), component_l1=component,
                direction_rms_deg=np.sqrt(average(angle[good]**2,volume[good])),
                direction_max_deg=float(np.max(angle[good])),
                excluded_volume=average(~good,volume)), angle


def compare_reference(state, reference, initial):
    assert state['u'].shape == reference['u'].shape
    assert abs(state['time']-reference['time']) < 1.e-10
    np.testing.assert_allclose(state['x'],reference['x'],rtol=0,atol=1.e-13)
    np.testing.assert_allclose(state['y'],reference['y'],rtol=0,atol=1.e-13)
    initial_rms = np.sqrt(average(np.sum(initial['u'][:,5:8]**2,axis=1),initial['v']))
    row, angle = vector_metrics(state['u'][:,5:8],reference['u'][:,5:8],
                                state['v'],initial_rms)
    increment = average(abs(reference['eth']-initial['eth']),state['v'])
    assert increment > 1.e-12*max(average(abs(initial['eth']),initial['v']),1.e-30), \
        'Reference has no resolved thermal increment'
    row['heating_error'] = average(abs(state['eth']-reference['eth']),state['v'])/increment
    for name in reference.get('eos',{}):
        assert name in state['eos'], f'Missing {name}'
        row[name+'_relative_error'] = average(abs(state['eos'][name]-reference['eos'][name]),
                                             state['v'])/max(average(abs(reference['eos'][name]),state['v']),1.e-30)
    return row, angle


def order_state(state):
    """Canonical cell order for comparisons across partition/rank layouts."""
    x,y = np.broadcast_arrays(state['x'][:,None,:],state['y'][:,:,None])
    order = np.lexsort((x.ravel(),y.ravel()))
    u = np.moveaxis(state['u'],1,-1).reshape(-1,8)[order]
    return u


def measured_timestep(directory):
    files=list(directory.glob('*.hst'))
    assert len(files)==1, f'Expected one history file in {directory}'
    history=np.loadtxt(files[0],comments='#',ndmin=2)
    assert history.shape[0]>1 and history.shape[1]>=3
    dt=history[:-1,1]
    assert np.all(np.isfinite(dt)) and np.all(dt>0)
    return float(np.max(dt))


def require_accuracy(row):
    assert row['magnetic_error'] < .02, row
    assert row['heating_error'] < .02, row
    assert row['direction_rms_deg'] < .5, row
    assert row['direction_max_deg'] < 2., row
    for key in ('eos_temperature_relative_error','eos_zbar_relative_error'):
        if key in row:
            assert row[key] < .02, row


def spitzer_eta(pressure, rho=1.):
    temperature = TEMPERATURE_FACTOR*pressure/rho
    return 1.02688e12*3.*10.*1.e-7/temperature**1.5


def nonlinear_reference(t, oblique):
    from scipy.integrate import solve_ivp
    k2 = (2*np.pi)**2*(5 if oblique else 1)
    def rhs(_, amplitude):
        pressure = (GAMMA-1)*(P0/(GAMMA-1)+.5*(1-amplitude[0]**2))
        return [-k2*spitzer_eta(pressure)*amplitude[0]]
    solution = solve_ivp(rhs,(0.,t),[1.],rtol=1.e-11,atol=1.e-13)
    assert solution.success
    amplitude = float(solution.y[0,-1])
    heat = .5*(1-amplitude**2)
    return amplitude,heat,spitzer_eta(P0+(GAMMA-1)*heat)


def exact_state(state, oblique, amplitude=1., pressure=P0, nonlinear=False):
    expected, unit_heat = fixed_reference(state,oblique,ETA)
    eta_final = ETA
    if nonlinear:
        a,heat,eta_final = nonlinear_reference(state['time'],oblique)
        expected, _ = fixed_reference(dict(state,time=0.),oblique,ETA)
        expected *= a
    else:
        expected *= amplitude
        heat = amplitude**2*unit_heat
    reference = dict(state,u=state['u'].copy(),eth=np.full_like(state['eth'],pressure/(GAMMA-1)+heat),eos={})
    reference['u'][:,5:8] = expected
    return reference,eta_final


def fourier_metrics(state, reference, oblique, scale):
    x,y = np.broadcast_arrays(state['x'][:,None,:],state['y'][:,:,None])
    phase = 2*np.pi*(x+(2*y if oblique else 0.))
    weights=state['v']
    numerical=[]
    expected=[]
    for component in range(3):
        for basis in (np.cos(phase),np.sin(phase)):
            denominator=average(basis*basis,weights)
            numerical.append(average(state['u'][:,5+component]*basis,weights)/denominator)
            expected.append(average(reference['u'][:,5+component]*basis,weights)/denominator)
    return dict(fourier_signed=numerical,fourier_reference=expected,
                fourier_error=float(np.max(abs(np.array(numerical)-expected))/scale))


def config(fixture,n=64,iprob=41):
    cfg=fixture(n)
    cfg['problem/resistive_diffusion'].update(iprob=iprob,mode_x=1,mode_y=2,
                                             amp=1.,pressure=P0,rho=1.)
    cfg['diffusion'].update(integrator='rkl2',rkl2_max_dt_ratio=100)
    end=.2/(ETA*(2*np.pi)**2*(1 if iprob==40 else 5))
    cfg['parthenon/time']['tlim']=end
    cfg['parthenon/output0']['dt']=end/2
    if iprob==40:
        cfg['parthenon/mesh']['nx2']=4
        cfg['parthenon/meshblock']['nx2']=4
    return cfg


def make_spitzer(cfg,clamped=False):
    cfg=deepcopy(cfg)
    cfg['units']=dict(code_length_cgs=1.,code_mass_cgs=1.e-10,code_time_cgs=1.e-7)
    cfg['hydro'].update(mean_molecular_weight=1.,mean_ionization_state=3.)
    cfg['diffusion'].update(resistivity_coeff='spitzer',spitzer_log_lambda=10.)
    if clamped:
        cfg['diffusion'].update(spitzer_eta_min=ETA/1.e-7,spitzer_eta_max=ETA/1.e-7)
    else:
        end=.2/(spitzer_eta(P0)*(2*np.pi)**2*(1 if cfg['problem/resistive_diffusion']['iprob']==40 else 5))
        cfg['parthenon/time']['tlim']=end
        cfg['parthenon/output0']['dt']=end/2
    return cfg


def make_tabulated(cfg,table):
    cfg=make_spitzer(cfg)
    cfg['hydro'].update(eos='tabulated',Tfloor=1000,dfloor=1.e4)
    cfg['problem/resistive_diffusion'].update(rho=1.e4,pressure=3.,amp=1.)
    cfg['tabulated_eos']=dict(filename=str(table.resolve()),density_to_code=1.e7,
        specific_internal_energy_to_code=1.e-10,pressure_to_code=1.e-3,
        sound_speed_to_code=1.e-5,temperature_to_kelvin=11604.51812155008,
        mean_ion_mass_hydrogen=27.)
    cfg['diffusion'].update(spitzer_eta_min=1.e3,spitzer_eta_max=1.e7)
    cfg['parthenon/time']['tlim']=.01
    cfg['parthenon/output0']['dt']=.005
    cfg['parthenon/output0']['variables']+=', eos_temperature, eos_zbar'
    return cfg


def plot_case(root,name,state,reference,angle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(9,9))
    # Physical x ordering makes the phase and component signs easy to inspect.
    u=order_state(state)
    ref=order_state(reference)
    take=slice(None,None,max(1,len(u)//2000))
    for i,label in enumerate(('Bx','By','Bz')):
        axes[0].plot(u[take,5+i],label=label)
        axes[0].plot(ref[take,5+i],'--',alpha=.7,label=label+' reference')
    axes[0].set_ylabel('signed magnetic field')
    axes[0].legend(ncol=3)
    x,y=np.broadcast_arrays(state['x'][:,None,:],state['y'][:,:,None])
    order=np.lexsort((x.ravel(),y.ravel()))
    axes[1].plot(angle.ravel()[order][take])
    axes[1].set_ylabel('direction error (degrees)')
    axes[2].plot(state['eth'].ravel()[order][take],label='numerical')
    axes[2].plot(reference['eth'].ravel()[order][take],'--',label='reference')
    axes[2].set(ylabel='thermal energy density',xlabel='sampled cell')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(root/(name+'.png'))
    plt.close(fig)


def run_suite(run,fixture,snapshots,energy_check,root,*,section='all',eos_table=None,compare_mpi=False):
    rows=[]
    failures=[]
    (root/('rkl2_'+section+'_PASS')).unlink(missing_ok=True)
    def save():
        (root/('rkl2_'+section+'_results.json')).write_text(json.dumps(rows,indent=2,allow_nan=False))
    def execute(name,cfg,**kwargs):
        initial,final,directory=run(name,cfg,**kwargs)
        energy_check(initial,final)
        assert abs(final['q'][1]-initial['q'][1]) < 1.e-12*max(abs(initial['E']),1.), \
            f'{name}: EOS intervention in smooth test'
        rows.append(dict(case=name+'_budget',
            budget_residual=float(final['E']-initial['E']-np.sum(final['q']-initial['q'])),
            ideal_exchange=float(final['q'][0]-initial['q'][0]),
            eos_exchange=float(final['q'][1]-initial['q'][1]),divergence=final['div']))
        save()
        return initial,final,directory
    def check_exact(name,cfg,oblique,nonlinear=False):
        initial,final,directory=execute(name,cfg)
        last=None
        for state,_ in snapshots(directory):
            if state['time']<=0: continue
            energy_check(initial,state)
            reference,eta_final=exact_state(state,oblique,
                cfg['problem/resistive_diffusion']['amp'],
                cfg['problem/resistive_diffusion']['pressure'],nonlinear)
            row,angle=compare_reference(state,reference,initial)
            row.update(case=name,time=state['time'],eta_final=eta_final,
                       **fourier_metrics(state,reference,oblique,cfg['problem/resistive_diffusion']['amp']))
            rows.append(row)
            save()
            if cfg['parthenon/mesh']['nx1']==128:
                require_accuracy(row)
            last=row
        plot_case(root,name,final,reference,angle)
        if compare_mpi and cfg['parthenon/mesh']['nx1']==128:
            _,serial,_=execute(name+'_serial',cfg,launcher='')
            np.testing.assert_allclose(order_state(final),order_state(serial),rtol=1.e-10,atol=1.e-11)
        return last
    def reference_run(name,cfg):
        # Full MHD reference at the identical spatial mesh; only time is refined.
        ref=deepcopy(cfg)
        ref['diffusion']['integrator']='unsplit'
        dt=min(float(cfg['parthenon/time']['tlim'])/(256 if name=='temporal' else 16),
               .15*.25/(cfg['parthenon/mesh']['nx1']**2*ETA))
        previous=None
        previous_dt=None
        for level in range(9):
            ref['hydro']['max_dt']=dt
            initial,current,directory=execute(f'{name}_reference_{level}',ref)
            actual_dt=measured_timestep(directory)
            if previous is not None:
                row,_=compare_reference(current,previous,initial)
                if (actual_dt <= .6*previous_dt and
                    row['magnetic_error']<.001 and row['heating_error']<.001 and
                    row['direction_rms_deg']<.05 and
                    all(value<.001 for key,value in row.items() if key.endswith('_relative_error'))):
                    rows.append(dict(case=name+'_reference_convergence',level=level,
                                     measured_dt=actual_dt,previous_dt=previous_dt,**row))
                    save()
                    return current
            previous=current
            previous_dt=actual_dt
            dt=min(dt/2,actual_dt/2)
        raise AssertionError(f'{name}: unsplit reference failed timestep convergence')
    def full_mhd(name,cfg):
        reference=reference_run(name,cfg)
        initial,final,_=execute(name,cfg)
        row,angle=compare_reference(final,reference,initial)
        rows.append(dict(case=name,**row));save()
        require_accuracy(row)
        plot_case(root,name,final,reference,angle)
        # Confirm this fixture really changes direction, rather than only amplitude.
        if cfg['problem/resistive_diffusion']['iprob']==42:
            change,_=vector_metrics(reference['u'][:,5:8],initial['u'][:,5:8],initial['v'],
                                   np.sqrt(average(np.sum(initial['u'][:,5:8]**2,axis=1),initial['v'])))
            assert change['direction_rms_deg']>.05, 'Mixed fixture did not resolve direction evolution'
        if compare_mpi:
            _,serial,_=execute(name+'_serial',cfg,launcher='')
            np.testing.assert_allclose(order_state(final),order_state(serial),rtol=1.e-10,atol=1.e-11)
    if section in ('all','fixed'):
        for oblique in (False,True):
            label='oblique' if oblique else 'axis'
            results=[]
            for n in (32,64,128):
                row=check_exact(f'rkl2_{label}_{n}',config(fixture,n,41 if oblique else 40),oblique)
                results.append(row)
            require_accuracy(results[-1])
            assert results[-1]['fourier_error']<.02
            for key in ('magnetic_error','heating_error'):
                rate=float(np.log2(results[-2][key]/results[-1][key]))
                rows.append(dict(case=label+'_convergence',metric=key,order=rate));save()
                assert rate>=1.7,(label,key,rate)
        full_mhd('rkl2_mixed',config(fixture,64,42))
        # Actual accepted superstep ratios, not just configured caps.
        for target in (2,10,50):
            cfg=config(fixture,128,41)
            cfg['problem/resistive_diffusion'].update(amp=.25,rho=100.)
            explicit=.15*.25/(128**2*ETA)
            cfg['hydro']['max_dt']=2*target*explicit
            cfg['diffusion']['rkl2_max_dt_ratio']=2*target
            row=check_exact(f'rkl2_ratio_{target}',cfg,True)
            require_accuracy(row)
            log=(root/f'rkl2_ratio_{target}'/'run.log').read_text()
            ratios=[float(value) for value in re.findall(r'superstep_ratio=([\d.eE+-]+)',log)]
            stages=[int(value) for value in re.findall(r'Taking (\d+) steps',log)]
            assert ratios and stages and max(stages)>1, 'No RKL2 stage evidence'
            assert .8*target<=max(ratios)<=1.2*target,(target,ratios)
            rows.append(dict(case=f'ratio_{target}',measured_ratios=ratios,stage_counts=stages));save()
    if section in ('all','spitzer'):
        for oblique in (False,True):
            label='oblique' if oblique else 'axis'
            results=[]
            for n in (32,64,128):
                cfg=make_spitzer(config(fixture,n,41 if oblique else 40))
                row=check_exact(f'spitzer_{label}_{n}',cfg,oblique,True)
                results.append(row)
            require_accuracy(row)
            assert abs(row['eta_final']/spitzer_eta(P0)-1)>=.2
            for key in ('magnetic_error','heating_error'):
                rate=float(np.log2(results[-2][key]/results[-1][key]))
                rows.append(dict(case='spitzer_'+label+'_convergence',metric=key,order=rate));save()
                if not rate>=1.7:
                    failures.append((label,key,rate))
                    print('FAIL Spitzer convergence:',failures[-1],flush=True)
            cfg=make_spitzer(config(fixture,128,41 if oblique else 40),True)
            row=check_exact('clamped_'+label,cfg,oblique)
            require_accuracy(row)
            fixed=deepcopy(cfg)
            fixed['diffusion']['resistivity_coeff']='fixed'
            fixed['diffusion']['ohm_diff_coeff_code']=ETA
            _,fixed_state,_=execute('clamped_'+label+'_fixed_reference',fixed)
            clamped_state=snapshots(root/('clamped_'+label))[-1][0]
            np.testing.assert_allclose(order_state(clamped_state),order_state(fixed_state),
                                       rtol=1.e-10,atol=1.e-11)
    if section in ('all','tabulated'):
        table=Path(eos_table) if eos_table else Path(__file__).resolve().parents[4]/'aluminum_saha_eos_three_chart.h5'
        assert table.is_file(), 'Supply --eos-table for required tabulated Spitzer coverage'
        for iprob in (40,41,42):
            full_mhd(f'tabulated_spitzer_{iprob}',make_tabulated(config(fixture,64,iprob),table))
    if section in ('all','temporal'):
        cfg=config(fixture,64,42)
        reference=reference_run('temporal',cfg)
        errors=[]
        timesteps=[]
        # Powers of two divide the chosen end time to avoid final-cut artifacts.
        for count in (16,32,64):
            test=deepcopy(cfg)
            test['hydro']['max_dt']=cfg['parthenon/time']['tlim']/count
            initial,final,directory=execute(f'temporal_{count}',test)
            timesteps.append(measured_timestep(directory))
            row,_=compare_reference(final,reference,initial)
            errors.append(row)
            rows.append(dict(case=f'temporal_{count}',measured_dt=timesteps[-1],**row));save()
        ratio=timesteps[-2]/timesteps[-1]
        assert 1.8<ratio<2.2, ('Temporal timesteps did not halve',timesteps)
        for key in ('magnetic_error','heating_error'):
            rate=float(np.log(errors[-2][key]/errors[-1][key])/np.log(ratio))
            assert rate>=1.7,(key,rate)
    if section in ('all','controls'):
        cfg=config(fixture,32,0)
        initial,final,_=execute('rkl2_uniform',cfg)
        np.testing.assert_allclose(order_state(initial),order_state(final),rtol=1.e-12,atol=1.e-12)
        cfg=config(fixture,32,41)
        cfg['diffusion']['resistivity']='none'
        initial,final,_=execute('rkl2_zero_eta',cfg)
        other=deepcopy(cfg);other['diffusion']['integrator']='none'
        _,ideal,_=execute('rkl2_zero_eta_ideal_reference',other)
        np.testing.assert_allclose(order_state(final),order_state(ideal),rtol=1.e-12,atol=1.e-12)
        cfg=config(fixture,32,41)
        cfg['parthenon/mesh'].update(refinement='static',numlevel=2)
        cfg['parthenon/meshblock'].update(nx1=8,nx2=8)
        cfg['parthenon/static_refinement0']=dict(x1min=.25,x1max=.75,x2min=.25,x2max=.75,level=1)
        _,final,_=execute('rkl2_static_refinement',cfg)
        assert len(np.unique(final['levels']))==2
        cfg=config(fixture,32,41)
        cfg['parthenon/time'].update(tlim=.03125,dt_force=1/4096)
        cfg['parthenon/output0']['dt']=.015625
        _,complete,_=execute('rkl2_restart_full',cfg)
        half=deepcopy(cfg);half['parthenon/time']['tlim']=.015625
        half['parthenon/output2']=dict(file_type='rst',dt=.015625)
        _,_,directory=execute('rkl2_restart_first',half)
        from check_compatible_energy import read_restart_time
        checkpoint=next(path for path in directory.glob('*.rhdf') if abs(read_restart_time(path)-.015625)<1.e-12)
        _,resumed,_=execute('rkl2_restart_resumed',cfg,restart=checkpoint)
        np.testing.assert_allclose(order_state(complete),order_state(resumed),rtol=2.e-10,atol=2.e-10)
        np.testing.assert_allclose(complete['q'],resumed['q'],rtol=1.e-9,atol=1.e-11)
    save()
    plot_convergence(root,rows,section)
    assert not failures, ('Spitzer convergence failures',failures)
    (root/('rkl2_'+section+'_PASS')).write_text('All requested acceptance checks passed.\n')


def plot_convergence(root,rows,section):
    if section not in ('all','fixed','spitzer','temporal'):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for label in ('axis','oblique','spitzer_axis','spitzer_oblique','temporal'):
        sizes=(16,32,64) if label=='temporal' else (32,64,128)
        names=[f'temporal_{n}' if label=='temporal' else
               f'{label}_{n}' if label.startswith('spitzer_') else
               f'rkl2_{label}_{n}' for n in sizes]
        if not all(any(r['case']==name for r in rows) for name in names):
            continue
        selected=[next(r for r in reversed(rows) if r['case']==name) for name in names]
        for ax,key in zip(axes,('magnetic_error','heating_error')):
            ax.loglog(sizes,[r[key] for r in selected],'o-',label=label)
            ax.set(xlabel='N or inverse timestep',ylabel=key)
            ax.grid(True);ax.legend()
    fig.tight_layout()
    fig.savefig(root/('rkl2_'+section+'_convergence.png'))
    plt.close(fig)


def self_test():
    from analytic_diffusion import self_test as heating_self_test
    heating_self_test()
    shape=(1,16,16)
    x=(np.arange(16)+.5)/16
    phase=2*np.pi*x[None,None,:]+np.zeros(shape)
    field=np.stack((np.zeros(shape),np.cos(phase),np.sin(phase)),axis=1)
    volume=np.ones(shape)/256
    correct,_=vector_metrics(field,field,volume,1.)
    assert correct['direction_max_deg']==0.
    weak=field.copy();weak[:,:,0,0]=0.
    weak_result,_=vector_metrics(field,weak,volume,1.)
    assert weak_result['excluded_volume']==1/256
    assert weak_result['magnetic_error']>0
    for wrong in (-field,field*np.array([1,-1,1])[None,:,None,None],
                  np.stack((np.zeros(shape),np.cos(phase+.2),np.sin(phase+.2)),axis=1)):
        row,_=vector_metrics(wrong,field,volume,1.)
        row['heating_error']=0.
        try: require_accuracy(row)
        except AssertionError: pass
        else: raise AssertionError('Incorrect field direction accepted')
    reversed_row,_=vector_metrics(-field,field,volume,1.)
    np.testing.assert_allclose(reversed_row['direction_rms_deg'],180.)
    # Independent ODE integrates amplitude and thermal gain consistently.
    t=.2/(spitzer_eta(P0)*(2*np.pi)**2)
    a,heat,eta=nonlinear_reference(t,False)
    assert 0<a<1 and heat>0 and eta/spitzer_eta(P0)<.8
    np.testing.assert_allclose(.5*a*a+P0/(GAMMA-1)+heat,.65)
    print('PASS RKL2 signed-vector, reversal, phase and nonlinear Spitzer references')


if __name__=='__main__':
    self_test()
