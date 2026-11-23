"""Small coarse-to-fine advection test, using the already compiled field loop."""
import json
import re

import numpy as np

AMP=.1
H=1/64  # finest dx=dy; domain [-1,1] x [-.5,.5]


def configuration(fixture,mesh,integrator,coefficient):
    cfg=fixture(64)
    cfg['job']['problem_id']='field_loop'
    cfg.pop('problem/resistive_diffusion')
    # The generator multiplies vflow1 by domain length: .5 * 2 = vx=1.
    cfg['problem/field_loop']=dict(iprob=1,rad=.15,amp=AMP,vflow=0.,
                                  vflow1=.5,vflow2=0.,vflow3=0.,drat=2.)
    cfg['parthenon/mesh'].update(nx1=64,nx2=32,x1min=-1.,x1max=1.,
                                x2min=-.5,x2max=.5,nx3=1,x3min=-.5,x3max=.5,)
    cfg['parthenon/meshblock'].update(nx1=8,nx2=8,nx3=1)
    if mesh=='smr':
        cfg['parthenon/mesh'].update(refinement='static',numlevel=2)
        cfg['parthenon/static_refinement0']=dict(x1min=.25,x1max=.75,
                                                x2min=-.5,x2max=.5,level=1)
    else:
        cfg['parthenon/mesh'].update(nx1=128,nx2=64,refinement='none')
        cfg['parthenon/meshblock'].update(nx1=32,nx2=16)
    cfg['parthenon/time'].update(tlim=.55,ncycle_out=100)
    cfg['parthenon/output0']['dt']=.05
    cfg['diffusion'].update(integrator=integrator,ohm_diff_coeff_code=.02,
                            rkl2_max_dt_ratio=50)
    if coefficient=='ideal':
        cfg['diffusion'].update(resistivity='none',integrator='none')
    elif coefficient=='spitzer':
        cfg['units']=dict(code_length_cgs=1.,code_mass_cgs=1.e-10,code_time_cgs=1.5e-7)
        cfg['hydro'].update(mean_molecular_weight=1.,mean_ionization_state=3.)
        cfg['diffusion'].update(resistivity_coeff='spitzer',spitzer_log_lambda=10.)
    return cfg


def scalar_geometry(state):
    x,y=np.broadcast_arrays(state['x'][:,None,:],state['y'][:,:,None])
    return x,y


def project(reference,values,target):
    """Volume-average a uniform finest-grid field onto the actual target leaves.

    Evaluate thermal energy on the fine grid BEFORE averaging: subtracting the
    magnetic energy of an averaged B would create artificial coarse-cell heat.
    """
    if values.ndim==3: values=values[:,None]
    count=values.shape[1]
    grid=np.full((count,64,128),np.nan)
    coverage=np.zeros((64,128),dtype=int)
    for b in range(len(reference['x'])):
        dx=np.diff(reference['x'][b])[0];dy=np.diff(reference['y'][b])[0]
        assert abs(dx-H)<1.e-12 and abs(dy-H)<1.e-12
        ny,nx=values.shape[-2:]
        i=int(round((reference['x'][b,0]-.5*dx+1)/H))
        j=int(round((reference['y'][b,0]-.5*dy+.5)/H))
        grid[:,j:j+ny,i:i+nx]=values[b]
        coverage[j:j+ny,i:i+nx]+=1
    assert np.all(coverage==1) and np.all(np.isfinite(grid))
    result=[]
    for b in range(len(target['x'])):
        dx=np.diff(target['x'][b])[0];dy=np.diff(target['y'][b])[0]
        rx=int(round(dx/H));ry=int(round(dy/H))
        assert rx in (1,2) and ry in (1,2)
        nx=len(target['x'][b]);ny=len(target['y'][b])
        i=int(round((target['x'][b,0]-.5*dx+1)/H))
        j=int(round((target['y'][b,0]-.5*dy+.5)/H))
        patch=grid[:,j:j+ny*ry,i:i+nx*rx]
        result.append(patch.reshape(count,ny,ry,nx,rx).mean(axis=(2,4)))
    return np.asarray(result)


def compare(state,expected_b,expected_h,initial):
    volume=state['v'];m0=initial['M']
    bscale=np.sqrt(2*m0/np.sum(initial['v']))
    x,_=scalar_geometry(state)
    band=abs(x-.25)<2/32  # two coarse cells around the entry interface
    thermal=state['eth']-expected_h
    actual=np.moveaxis(state['u'][:,5:8],1,-1)
    expected=np.moveaxis(expected_b,1,-1)
    strong=np.linalg.norm(expected,axis=-1)>.01*AMP
    angles=np.degrees(np.arctan2(np.linalg.norm(np.cross(actual,expected),axis=-1),
                                 np.sum(actual*expected,axis=-1)))
    return dict(magnetic_l1=float(np.sum(np.mean(abs(state['u'][:,5:8]-expected_b),axis=1)*volume)/np.sum(volume)/bscale),
        thermal_l1_over_initial_magnetic_energy=float(np.sum(abs(thermal)*volume)/m0),
        thermal_l1_over_reference_thermal_energy=float(np.sum(abs(thermal)*volume)/np.sum(abs(expected_h)*volume)),
        signed_thermal_difference_over_initial_magnetic_energy=float(np.sum(thermal*volume)/m0),
        entry_band_thermal_linf_over_magnetic_density=float(np.max(abs(thermal[band]))/(.5*AMP**2)),
        direction_rms_deg=float(np.sqrt(np.sum(angles[strong]**2*volume[strong])/np.sum(volume[strong]))) if np.any(strong) else None,
        excluded_direction_volume=float(np.sum(volume[~strong])/np.sum(volume)))


def fine_marker_fraction(state):
    excess=np.maximum(state['u'][:,0]-1.,0.)*state['v']
    levels=np.asarray(state['levels']).reshape(-1)
    fine=np.broadcast_to((levels==levels.max())[:,None,None],state['eth'].shape)
    assert len(np.unique(levels))==2
    assert np.sum(excess)>1.e-10, 'No density marker: check fixture center and drat'
    return float(np.sum(excess[fine])/np.sum(excess))


def level_summary(state):
    levels=np.asarray(state['levels']).reshape(-1)
    result={}
    magnetic=.5*np.sum(state['u'][:,5:8]**2,axis=1)
    for level in np.unique(levels):
        mask=np.broadcast_to((levels==level)[:,None,None],state['eth'].shape)
        volume=state['v'][mask]
        result[str(int(level))]=dict(
            mass=float(np.sum(state['u'][:,0][mask]*volume)),
            total_energy=float(np.sum(state['u'][:,4][mask]*volume)),
            thermal_energy=float(np.sum(state['eth'][mask]*volume)),
            magnetic_energy=float(np.sum(magnetic[mask]*volume)),
            ideal_exchange=float(np.sum(state['exchange'][:,0][mask]*volume)),
            eos_exchange=float(np.sum(state['exchange'][:,1][mask]*volume)))
    return result


def run_suite(run,fixture,snapshots,energy_check,root):
    rows=[];failures=[];states={}
    (root/'refinement_flow_PASS').unlink(missing_ok=True)
    # Complete simulations before plotting/reanalyzing cached histories. Otherwise
    # repeated analysis can consume a short batch before a remaining run finishes.
    executions={}
    execution_failures={}
    for coefficient in ('ideal','fixed','spitzer'):
        for mesh in ('fine','smr'):
            for integrator in (('unsplit',) if coefficient=='ideal' else ('unsplit','rkl2')):
                name=f'flow_{coefficient}_{mesh}_{integrator}'
                try:
                    executions[name]=run(name,configuration(fixture,mesh,integrator,coefficient))
                except AssertionError:
                    stamp=root/name/'execution.json'
                    if not stamp.exists() or json.loads(stamp.read_text())['returncode']==0:
                        raise
                    execution_failures[name]=json.loads(stamp.read_text())['returncode']
                    failures.append(f'{name}: driver failed; see {name}/run.log')
    def save():
        (root/'refinement_flow_results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    def check(condition,message):
        if not condition: failures.append(message)
    for coefficient in ('ideal','fixed','spitzer'):
        integrations=('unsplit',) if coefficient=='ideal' else ('unsplit','rkl2')
        for mesh in ('fine','smr'):
            for integrator in integrations:
                name=f'flow_{coefficient}_{mesh}_{integrator}'
                if name in execution_failures:
                    rows.append(dict(case=name,returncode=execution_failures[name],
                                     failure='Driver runtime failure; see run.log'))
                    save()
                    continue
                initial,final,directory=executions[name]
                states[(coefficient,mesh,integrator)]=(initial,final)
                series=[]
                for state,_ in snapshots(directory):
                    energy_check(initial,state)
                    assert abs(state['q'][1]-initial['q'][1])<1.e-11
                    mass=float(np.sum(state['u'][:,0]*state['v']))
                    mass0=float(np.sum(initial['u'][:,0]*initial['v']))
                    assert abs(mass-mass0)<1.e-10*mass0
                    x,_=scalar_geometry(state)
                    band=abs(x-.25)<2/32
                    series.append(dict(time=state['time'],
                        relative_mass_change=float((mass-mass0)/mass0),
                        eos_exchange=float(state['q'][1]-initial['q'][1]),
                        budget_residual=float(state['E']-initial['E']-np.sum(state['q']-initial['q'])),
                        ideal_exchange_over_M0=float((state['q'][0]-initial['q'][0])/initial['M']),
                        absolute_ideal_exchange_over_M0=float(np.sum(abs(state['exchange'][:,0]-initial['exchange'][:,0])*state['v'])/initial['M']),
                        entry_pressure_min=float(np.min(state['p'][:,4][band])),
                        entry_pressure_max=float(np.max(state['p'][:,4][band])),
                        entry_thermal_min=float(np.min(state['eth'][band])),
                        entry_thermal_max=float(np.max(state['eth'][band])),
                        divB=state['div'],
                        entry_ideal_exchange_over_M0=float(np.sum((state['exchange'][:,0]-initial['exchange'][:,0])[band]*state['v'][band])/initial['M']),
                        levels=level_summary(state)))
                record=dict(case=name,history=series,initial_magnetic_energy=initial['M'])
                if integrator=='rkl2':
                    log=(directory/'run.log').read_text()
                    ratios=[float(value) for value in re.findall(r'superstep_ratio=([\d.eE+-]+)',log)]
                    stages=[int(value) for value in re.findall(r'Taking (\d+) steps',log)]
                    record['superstep_ratio_range']=[min(ratios),max(ratios)] if ratios else []
                    record['stage_counts']=sorted(set(stages))
                    check(bool(ratios) and max(ratios)>1 and bool(stages) and max(stages)>1,
                          f'{name}: no measured super-time-stepping coverage')
                if mesh=='smr':
                    before=fine_marker_fraction(initial);after=fine_marker_fraction(final)
                    record.update(initial_marker_in_fine=before,final_marker_in_fine=after)
                    check(before<.05 and after>.65,f'{name}: insufficient coarse-to-fine material transport')
                    fine=states[(coefficient,'fine',integrator)][1]
                    b=project(fine,fine['u'][:,5:8],final)
                    eth=project(fine,fine['eth'],final)[:,0]
                    errors=compare(final,b,eth,initial)
                    record['vs_uniform_fine']=errors
                    check(errors['magnetic_l1']<.15,f'{name}: SMR magnetic discrepancy >15%')
                    check(errors['thermal_l1_over_initial_magnetic_energy']<.3,
                          f'{name}: SMR thermal discrepancy >30% initial magnetic energy')
                    check(errors['entry_band_thermal_linf_over_magnetic_density']<.5,
                          f'{name}: entry-band thermal spike >50% initial magnetic energy density')
                rows.append(record);save()
                plot_history(root,name,series)
        if coefficient!='ideal':
            for mesh in ('fine','smr'):
                if (coefficient,mesh,'rkl2') not in states:
                    continue
                initial,reference=states[(coefficient,mesh,'unsplit')]
                _,actual=states[(coefficient,mesh,'rkl2')]
                errors=compare(actual,reference['u'][:,5:8],reference['eth'],initial)
                rows.append(dict(case=f'{coefficient}_{mesh}_rkl2_vs_unsplit',**errors));save()
                check(errors['magnetic_l1']<.05,f'{coefficient}/{mesh}: integrator B discrepancy >5%')
                check(errors['thermal_l1_over_initial_magnetic_energy']<.1,
                      f'{coefficient}/{mesh}: integrator thermal discrepancy >10% initial magnetic energy')
                check(errors['direction_rms_deg'] is not None and errors['direction_rms_deg']<2.,
                      f'{coefficient}/{mesh}: integrator direction RMS >2 degrees')
    plot_comparison(root,states)
    save()
    (root/'refinement_flow_failures.json').write_text(json.dumps(failures,indent=2))
    assert not failures,failures
    (root/'refinement_flow_PASS').write_text('Coarse-to-fine smoke checks passed.\n')


def plot_history(root,name,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(7,6))
    time=[r['time'] for r in history]
    axes[0].plot(time,[r['entry_pressure_min'] for r in history],label='entry band minimum')
    axes[0].plot(time,[r['entry_pressure_max'] for r in history],label='entry band maximum')
    axes[0].set_ylabel('pressure');axes[0].legend()
    axes[1].plot(time,[r['ideal_exchange_over_M0'] for r in history],label='net correction/M0')
    axes[1].plot(time,[r['absolute_ideal_exchange_over_M0'] for r in history],label='absolute local correction/M0')
    axes[1].plot(time,[r['entry_ideal_exchange_over_M0'] for r in history],label='entry-band net correction/M0')
    axes[1].set(xlabel='time',ylabel='cumulative ideal energy correction');axes[1].legend()
    for ax in axes:
        ax.axvspan(.1,.4,color='grey',alpha=.15)
    fig.tight_layout();fig.savefig(root/(name+'_history.png'));plt.close(fig)


def plot_comparison(root,states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for row,coefficient in enumerate(('fixed','spitzer')):
        for mesh in ('smr','fine'):
            for integrator in ('unsplit','rkl2'):
                if (coefficient,mesh,integrator) not in states:
                    continue
                _,state=states[(coefficient,mesh,integrator)]
                x,y=scalar_geometry(state)
                dy=np.diff(state['y'],axis=1)[:,0,None,None]
                take=abs(y)<.6*dy
                order=np.argsort(x[take])
                style='.-' if mesh=='smr' else '--'
                label=integrator+' '+mesh
                axes[row,0].plot(x[take][order],state['p'][:,4][take][order],style,label=label)
                axes[row,1].plot(x[take][order],state['u'][:,0][take][order],style,label=label)
        for ax,label in zip(axes[row],('pressure','density')):
            ax.axvline(.25,color='k',ls='--');ax.axvline(.75,color='k',ls='--')
            ax.set(xlabel='x',ylabel=label,title=coefficient);ax.legend()
    fig.tight_layout();fig.savefig(root/'refinement_flow_profiles.png');plt.close(fig)


def self_test():
    # Validate conservative restriction, not interpolation of B^2 into heat.
    x=-1+(np.arange(128)+.5)*H;y=-.5+(np.arange(64)+.5)*H
    fine=dict(x=x[None],y=y[None])
    coarse=dict(x=(-1+(np.arange(64)+.5)*2*H)[None],
                y=(-.5+(np.arange(32)+.5)*2*H)[None])
    field=np.broadcast_to(x[None,None,:],(1,64,128))
    restricted=project(fine,field,coarse)[:,0]
    np.testing.assert_allclose(restricted,np.broadcast_to(coarse['x'][:,None,:],restricted.shape))
    thermal=np.full_like(field,1.5)
    np.testing.assert_allclose(project(fine,thermal,coarse),1.5)
    print('PASS conservative fine-to-leaf projection and thermal reference')




def run_guard_suite(run,fixture,snapshots,energy_check,root):
    """Original failing Spitzer/SMR run and same-mesh converged unsplit reference.

    Keep the original physical parameters, final time, and integrator comparison
    thresholds. Reference convergence is tested separately from RKL agreement.
    """
    marker=root/'rkl2_guard_PASS'
    marker.unlink(missing_ok=True)
    records=[]
    def execute(name,integrator,cap=None):
        cfg=configuration(fixture,'smr',integrator,'spitzer')
        if cap is not None: cfg['hydro']['max_dt']=cap
        cfg['parthenon/output2']=dict(file_type='rst',dt=.01,id='checkpoint')
        initial,final,directory=run(name,cfg)
        for state,_ in snapshots(directory):
            energy_check(initial,state)
            assert abs(state['q'][1]-initial['q'][1])<1.e-11
        return initial,final,directory
    initial,actual,directory=execute('flow_spitzer_smr_rkl2','rkl2')
    log=(directory/'run.log').read_text()
    ratios=[float(v) for v in re.findall(r'superstep_ratio=([\d.eE+-]+)',log)]
    assert ratios and max(ratios)>1., 'No measured super-time-stepping coverage'
    previous=None
    converged=False
    for level in range(5):
        cap=.00025/2**level
        _,reference,_=execute(f'flow_spitzer_smr_unsplit_ref{level}','unsplit',cap)
        if previous is not None:
            delta=compare(reference,previous['u'][:,5:8],previous['eth'],initial)
            heat_scale=np.sum(abs(reference['eth']-initial['eth'])*reference['v'])
            thermal_change=float(np.sum(abs(reference['eth']-previous['eth'])*reference['v'])/
                                 max(heat_scale,np.finfo(float).tiny))
            record=dict(reference_dt_cap=cap,convergence=delta,
                        thermal_change_over_reference_increment=thermal_change)
            records.append(record)
            (root/'rkl2_guard_results.json').write_text(json.dumps(records,indent=2))
            if delta['magnetic_l1']<.001 and thermal_change<.001 and delta['direction_rms_deg']<.05:
                converged=True
                break
        previous=reference
    assert converged, 'Unsplit reference did not converge within five timestep caps'
    errors=compare(actual,reference['u'][:,5:8],reference['eth'],initial)
    records.append(dict(rkl2_vs_unsplit=errors,superstep_ratio_range=[min(ratios),max(ratios)],
                        stage_counts=sorted(set(map(int,re.findall(r'Taking (\d+) steps',log))))))
    (root/'rkl2_guard_results.json').write_text(json.dumps(records,indent=2))
    assert errors['magnetic_l1']<.05, errors
    assert errors['thermal_l1_over_initial_magnetic_energy']<.1, errors
    assert errors['direction_rms_deg']<2., errors
    marker.write_text('Original Spitzer SMR run and converged unsplit comparison passed.\n')


def guard_self_test():
    import tempfile
    from pathlib import Path
    from check_compatible_energy import fixture
    shape=(1,8,8)
    initial=dict(x=np.linspace(-.8,.95,8)[None],y=np.linspace(-.4,.4,8)[None],
                 u=np.zeros((1,8,8,8)),eth=np.full(shape,1.5),v=np.full(shape,2./64),
                 q=np.zeros(2),M=.01)
    initial['u'][:,5]=.1
    final=dict(initial,eth=initial['eth']+.001)
    def exercise(reverse):
        states={};caps=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def run(name,cfg):
                directory=root/name;directory.mkdir()
                (directory/'run.log').write_text('superstep_ratio=3 Taking 3 steps')
                assert cfg['parthenon/time']['tlim']==.55
                assert cfg['diffusion']['resistivity_coeff']=='spitzer'
                if 'max_dt' in cfg['hydro']: caps.append(cfg['hydro']['max_dt'])
                out=final
                if reverse and cfg['diffusion']['integrator']=='rkl2':
                    out=dict(final,u=final['u'].copy());out['u'][:,5:8]*=-1
                states[directory]=[(initial,None),(out,None)]
                return initial,out,directory
            run_guard_suite(run,fixture,lambda d: states[d],lambda a,b: None,root)
            assert caps==[.00025,.000125]
            assert (root/'rkl2_guard_PASS').is_file()
    exercise(False)
    try:
        exercise(True)
    except AssertionError:
        pass
    else:
        raise AssertionError('Guard analyzer accepted a reversed RKL magnetic field')
    print('PASS Spitzer guard reference selection, timestep halving and reversal rejection')


if __name__=='__main__':
    self_test()
    guard_self_test()
