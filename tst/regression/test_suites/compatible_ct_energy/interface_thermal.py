"""Smooth-helix localization of coarse/fine thermal errors; never compiles."""
from copy import deepcopy
import json
import hashlib
import math
import re
from pathlib import Path

import h5py
import numpy as np

A=.1
ETA=.02
END=.1
LABELS=('current_before_sync','current_after_sync','eta_j2','ohmic_ez_before_sync',
        'ohmic_ez_after_sync','update_total','update_magnetic','update_kinetic',
        'update_thermal','boundary_thermal','boundary_face_magnetic','center_thermal',
        'pressure_work_rate','resistive_transport_rate','baseline_total',
        'baseline_magnetic','baseline_kinetic','baseline_face_magnetic',
        'rkl_thermal_increment','rkl_balance_residual','probe_active',
        'rkl_transport_increment','rkl_previous_transport','rkl_initial_transport_rate',
        'current_before_boundary','current_after_boundary')


def timestep(n):
    explicit=.25*.1/(2*n)**2/ETA
    return END/math.ceil(END/(.8*explicit))


def configuration(fixture,n,mesh,vx,case,dt=None,end=END):
    cfg=fixture(n)
    cfg['problem/resistive_diffusion'].update(iprob=0 if case=='uniform' else 40,
        rho=1.,pressure=1.,amp=A,Bx=.03,By=.08,Bz=.06,vx=vx,mode_x=1)
    cfg['parthenon/meshblock'].update(nx1=n//4,nx2=n//4)
    if mesh=='smr':
        cfg['parthenon/mesh'].update(refinement='static',numlevel=2)
        cfg['parthenon/static_refinement0']=dict(x1min=.25,x1max=.75,
                                                x2min=0.,x2max=1.,level=1)
    else:
        cfg['parthenon/mesh'].update(nx1=2*n,nx2=2*n)
        cfg['parthenon/meshblock'].update(nx1=n,nx2=n)
    cfg['parthenon/time'].update(tlim=end,dt_force=dt or timestep(n),ncycle_out=100)
    cfg['hydro']['ct_interface_diagnostics']='true'
    cfg['diffusion'].update(integrator='rkl2' if case=='rkl2' else 'unsplit',
        cfl=.1,ohm_diff_coeff_code=ETA,rkl2_max_dt_ratio=100)
    if case=='ideal':
        cfg['diffusion'].update(resistivity='none',integrator='none')
    cfg['parthenon/output0'].update(dt=end/4,
        variables=cfg['parthenon/output0']['variables']+',ct_interface_diagnostics')
    if n>=64 or (dt is not None and dt<timestep(n)):
        # Expensive diagnostic cases may span several <60 s process batches.
        # Checkpoints do not cut timesteps or change the requested integration.
        cfg['parthenon/output2']=dict(file_type='rst',dt=end/8)
    return cfg


def geometry(state,n):
    x,y=np.broadcast_arrays(state['x'][:,None,:],state['y'][:,:,None])
    band=(abs(x-.25)<2/n)|(abs(x-.75)<2/n)
    return x,y,band


def exact(state,case,vx):
    x=state['x'][:,None,:]
    dx=np.diff(state['x'],axis=1)[:,0,None,None]
    amp=A*np.exp(-ETA*(2*np.pi)**2*state['time']) if case not in ('ideal','uniform') else A
    if case=='uniform':
        b=np.broadcast_to(np.array([.03,.08,.06])[None,:,None,None],state['u'][:,5:8].shape)
        heat=0.
    else:
        phase=2*np.pi*(x-vx*state['time'])
        b=np.zeros_like(state['u'][:,5:8])
        b[:,1]=amp*np.sinc(dx)*np.cos(phase)
        b[:,2]=amp*np.sinc(dx)*np.sin(phase)
        heat=.5*(A*A-amp*amp)
    return b,1.5+heat,heat


def norms(values,volume,mask):
    assert np.any(mask)
    return dict(mean=float(np.sum(values[mask]*volume[mask])/np.sum(volume[mask])),
                l1=float(np.sum(abs(values[mask])*volume[mask])/np.sum(volume[mask])),
                minimum=float(np.min(values[mask])),maximum=float(np.max(values[mask])),
                linf=float(np.max(abs(values[mask]))))


def metrics(state,case,vx,n):
    _,_,band=geometry(state,n)
    b,h,heat=exact(state,case,vx)
    error=state['eth']-h
    v=state['v']
    return dict(time=state['time'],reference_heat_gain=heat,
        magnetic_l1=float(np.sum(np.mean(abs(state['u'][:,5:8]-b),axis=1)*v)/np.sum(v)/A),
        thermal={label:norms(error,v,mask) for label,mask in
                 (('interface',band),('bulk',~band),('all',np.ones_like(band)))},
        thermal_l1_over_exact_heat_gain=float(np.sum(abs(error)*v)/np.sum(v)/heat) if heat>0 else None)


def read_probe(path):
    with h5py.File(path) as f:
        assert 'ct_interface_diagnostics' in f, 'Recompile with interface diagnostics before running'
        d=f['ct_interface_diagnostics'][:,:,0]
        assert d.shape[1]==len(LABELS), d.shape
        assert np.all(np.isfinite(d)), 'Nonfinite interface diagnostics'
        return d


def difference(a,b,n):
    # Temporal references have identical leaf layouts, including block order.
    np.testing.assert_array_equal(a['x'],b['x'])
    np.testing.assert_array_equal(a['y'],b['y'])
    np.testing.assert_array_equal(a['levels'],b['levels'])
    _,_,band=geometry(a,n)
    gain=.5*A*A*(1-np.exp(-2*ETA*(2*np.pi)**2*a['time']))
    v=a['v']
    return dict(magnetic=float(np.sum(np.mean(abs(a['u'][:,5:8]-b['u'][:,5:8]),axis=1)*v)/np.sum(v)/A),
        thermal=float(np.sum(abs(a['eth']-b['eth'])*v)/np.sum(v)/gain),
        interface_thermal=float(np.sum(abs(a['eth'][band]-b['eth'][band])*v[band])/np.sum(v[band])/gain))


def plot(root,name,states,case,vx,n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(9,12))
    for state,path in states:
        x,_,_=geometry(state,n)
        b,h,_=exact(state,case,vx)
        xx=np.unique(x)
        profile=lambda field: [float(np.mean(field[x==q])) for q in xx]
        axes[0].plot(xx,profile(state['eth']-h),label=f't={state["time"]:.5g}')
    state,path=states[-1]
    d=read_probe(path);x,_,_=geometry(state,n);xx=np.unique(x)
    for component in (0,1,24,25):
        axes[1].plot(xx,profile(d[:,component]),label=LABELS[component])
    for component in (5,6,7,8,9,10,11):
        axes[2].plot(xx,profile(d[:,component]),label=LABELS[component])
    with h5py.File(path) as f:
        stage=f['ct_energy_diagnostics'][:,:,0]
    axes[2].plot(xx,profile(stage[:,0]),label='ideal_correction')
    for component in (2,12,13):
        axes[3].plot(xx,profile(d[:,component]),label=LABELS[component])
    for ax in axes:
        for interface in (.25,.75):
            ax.axvspan(interface-2/n,interface+2/n,color='grey',alpha=.12)
        ax.set_xlabel('x');ax.legend(fontsize=7)
    axes[0].set_ylabel('signed thermal error');axes[1].set_ylabel('Jz');axes[2].set_ylabel('latest-stage energy changes')
    axes[3].set_ylabel('latest-stage rates (not integrated)')
    fig.tight_layout();fig.savefig(root/(name+'.png'));plt.close(fig)


def run_suite(run,fixture,snapshots,energy_check,root,section='all'):
    rows=[];failures=[];finals={};evolution={}
    marker=root/f'interface_thermal_{section}_PASS'
    marker.unlink(missing_ok=True)
    results=root/'interface_thermal_results.json'
    cached={r['case']:r for r in json.loads(results.read_text())} if results.exists() else {}
    analyzer=Path(__file__).read_bytes()
    def save():
        (root/'interface_thermal_results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        (root/'interface_thermal_failures.json').write_text(json.dumps(failures,indent=2))
    def check(ok,message):
        if not ok: failures.append(message)
    def execute(name,cfg,case,vx,n,phase='evolution'):
        try:
            initial,final,directory=run(name,cfg)
        except AssertionError:
            stamp=root/name/'execution.json'
            if not stamp.exists() or json.loads(stamp.read_text())['returncode']==0: raise
            failures.append(name+': solver runtime failure; see run.log');save();return None
        signature=hashlib.sha256(analyzer+(directory/'input.in').read_bytes()+
                                 (directory/'execution.json').read_bytes()).hexdigest()
        if name in cached and cached[name].get('analysis_sha256')==signature:
            row=cached[name];rows.append(row);save()
            failures.extend(row['failures'])
            return initial,final,row
        start_failures=len(failures)
        history=[]
        for state,path in snapshots(directory):
            energy_check(initial,state)
            check(abs(state['q'][1]-initial['q'][1])<1.e-11,name+': EOS intervention')
            mass0=np.sum(initial['u'][:,0]*initial['v'])
            check(abs(np.sum(state['u'][:,0]*state['v'])/mass0-1)<1.e-10,name+': mass conservation')
            record=metrics(state,case,vx,n)
            d=read_probe(path)
            _,_,band=geometry(state,n)
            record['diagnostics']={label:{region:norms(d[:,component],state['v'],mask)
                for region,mask in (('interface',band),('bulk',~band))}
                for component,label in enumerate(LABELS) if component not in (14,15,16,17,20,22,23)}
            with h5py.File(path) as f:
                stage=f['ct_energy_diagnostics'][:,:,0]
            for label,field in (('stage_ideal_correction',stage[:,0]),
                                ('stage_resistive_magnetic_change',stage[:,1]),
                                ('stage_resistive_transport',stage[:,2]),
                                ('cumulative_ideal_exchange',state['exchange'][:,0]),
                                ('cumulative_eos_exchange',state['exchange'][:,1])):
                record['diagnostics'][label]={region:norms(field,state['v'],mask)
                    for region,mask in (('interface',band),('bulk',~band))}
            record.update(ideal_exchange=float(state['q'][0]-initial['q'][0]),
                eos_exchange=float(state['q'][1]-initial['q'][1]),
                budget_residual=float(state['E']-initial['E']-np.sum(state['q']-initial['q'])),
                magnetic_energy=float(state['M']),kinetic_energy=float(state['K']),thermal_energy=float(state['H']))
            history.append(record)
            if case=='uniform':
                check(np.max(abs(state['u']-initial['u']))<1.e-11,name+': uniform state changed')
                check(np.max(abs(d[:,0:5]))<1.e-11,name+': uniform state has current or EMF')
            if case=='rkl2' and state['time']>0:
                check(np.max(abs(d[:,19]))<1.e-10,name+': RKL recurrence thermal balance failed')
        log=(directory/'run.log').read_text()
        ratios=[float(v) for v in re.findall(r'superstep_ratio=([\d.eE+-]+)',log)]
        actual_dt=measured_dt(directory)
        check(abs(actual_dt/float(cfg['parthenon/time']['dt_force'])-1)<1.e-8,
              name+': measured macrostep differs from forced timestep')
        row=dict(case=name,phase=phase,n=n,mesh='smr' if len(np.unique(final['levels']))>1 else 'fine',
                 vx=vx,history=history,dt=float(cfg['parthenon/time']['dt_force']),
                 measured_timestep=actual_dt,
                 analysis_sha256=signature,failures=failures[start_failures:],
                 superstep_ratio_range=[min(ratios),max(ratios)] if ratios else [],
                 stage_counts=sorted(set(map(int,re.findall(r'Taking (\d+) steps',log)))))
        rows.append(row);save();plot(root,name,snapshots(directory),case,vx,n)
        return initial,final,row
    def paired(n,mesh,vx,case):
        name=f'interface_{n}_{mesh}_v{vx}_{case}'
        cfg=configuration(fixture,n,mesh,vx,case)
        # A separate cheap one-step run gives the first update without hundreds of dumps.
        first=deepcopy(cfg);first['parthenon/time']['tlim']=timestep(n)
        first['parthenon/output0']['dt']=timestep(n)
        execute(name+'_first',first,case,vx,n,'first_step')
        result=execute(name,cfg,case,vx,n)
        if result:
            finals[n,mesh,vx,case]=result[1]
            evolution[n,mesh,vx,case]=result[2]
    if section in ('all','base'):
        for vx in (0,1):
            for case in ('uniform','ideal','unsplit','rkl2'):
                for mesh in ('fine','smr'): paired(32,mesh,vx,case)
        for vx in (0,1):
            for mesh in ('fine','smr'):
                keys=[(32,mesh,vx,c) for c in ('unsplit','rkl2')]
                if all(k in finals for k in keys):
                    rows.append(dict(case=f'integrators_{mesh}_v{vx}',**difference(*(finals[k] for k in keys),32)));save()
    if section in ('all','resolution'):
        for n in (32,64):
            for case in ('ideal','unsplit','rkl2'):
                for mesh in ('fine','smr'):
                    if (n,mesh,1,case) not in finals: paired(n,mesh,1,case)
        for case in ('ideal','unsplit','rkl2'):
            for mesh in ('fine','smr'):
                keys=[(n,mesh,1,case) for n in (32,64)]
                if all(k in evolution for k in keys):
                    errors=[evolution[k]['history'][-1]['thermal']['interface']['l1'] for k in keys]
                    check(errors[1]<errors[0] or max(errors)<1.e-12,
                          f'{mesh}/{case}: interface thermal error did not decrease with refinement')
                    rows.append(dict(case=f'resolution_{mesh}_{case}',interface_errors=errors,
                                     observed_order=float(np.log2(errors[0]/errors[1])) if min(errors)>0 else None));save()
    if section in ('all','temporal'):
        n=32;mesh='smr';vx=1;previous=None;reference=None
        for divisor in (2,4,8,16,32,64):
            cfg=configuration(fixture,n,mesh,vx,'unsplit',dt=timestep(n)/divisor)
            result=execute(f'interface_reference_dt{divisor}',cfg,'unsplit',vx,n,'reference')
            if result is None: break
            if previous is not None:
                changes=difference(result[1],previous,n)
                rows.append(dict(case=f'reference_change_{divisor}',**changes));save()
                if max(changes.values())<1.e-3:
                    reference=result[1];break
            previous=result[1]
        check(reference is not None,'Unsplit temporal reference did not converge below 0.1%')
        if reference is not None:
            errors=[]
            for divisor in (1,2):
                cfg=configuration(fixture,n,mesh,vx,'rkl2',dt=timestep(n)/divisor)
                result=execute(f'interface_rkl_dt{divisor}',cfg,'rkl2',vx,n,'temporal')
                if result:
                    errors.append(difference(result[1],reference,n))
                    rows.append(dict(case=f'temporal_error_{divisor}',**errors[-1]));save()
            if len(errors)==2:
                for key in errors[0]:
                    check(errors[1][key]<errors[0][key] or max(e[key] for e in errors)<1.e-12,
                          f'RKL timestep halving did not reduce {key} error')
            explicit=.25*.1/(2*n)**2/ETA
            dt=END/math.ceil(END/(6*explicit))
            cfg=configuration(fixture,n,mesh,vx,'rkl2',dt=dt)
            result=execute('interface_rkl_accelerated',cfg,'rkl2',vx,n,'accelerated')
            if result:
                ratios=result[2]['superstep_ratio_range']
                check(bool(ratios) and 2.4<=ratios[-1]<=3.6,'Accelerated run did not measure ratio near 3')
                check(max(result[2]['stage_counts'],default=0)>1,'No multistage RKL evidence')
                rows.append(dict(case='accelerated_vs_reference',**difference(result[1],reference,n)));save()
    save()
    assert not failures,failures
    marker.write_text('Requested interface diagnostic checks passed.\n')


def measured_dt(directory):
    # History output rounds to six significant digits. PHDF stores the actual
    # macrostep in full precision, including the initial forced timestep.
    files=list(directory.glob('*.phdf'))
    assert files
    values=[]
    for path in files:
        with h5py.File(path) as f:
            values.append(float(f['Info'].attrs['dt']))
    return max(values)


def self_test():
    from check_compatible_energy import fixture
    source=Path(__file__).resolve().parents[4]/'src/hydro/ctmhd/compatible_energy.cpp'
    declaration=source.read_text().split('pkg->AddField("ct_interface_diagnostics",',1)[1]
    declaration=declaration.split('std::vector<std::string>{',1)[1].split('}',1)[0]
    assert tuple(re.findall(r'"([a-z0-9_]+)"',declaration))==LABELS, 'C++/analyzer component mismatch'
    x=(np.arange(16)+.5)/16
    shape=(1,16,16)
    state=dict(x=x[None],y=x[None],u=np.zeros((1,8,16,16)),v=np.ones(shape)/256,
               eth=np.full(shape,1.5),time=0.)
    b,h,gain=exact(state,'unsplit',1)
    np.testing.assert_allclose(np.sum(b*b,axis=1),A*A*np.sinc(1/16)**2)
    state['time']=END;b,h,gain=exact(state,'unsplit',1)
    state['u'][:,5:8]=b;state['eth'][:]=h
    assert metrics(state,'unsplit',1,32)['thermal']['interface']['l1']==0
    _,_,band=geometry(state,32)
    for sign in (-1,1):
        state['eth'][:]=h;state['eth'][band]+=sign*1.e-3
        m=metrics(state,'unsplit',1,32)
        assert m['thermal']['interface']['mean']*sign>0
        assert m['thermal']['bulk']['linf']==0
        assert m['thermal_l1_over_exact_heat_gain']>0
    state['eth'][:]=1.5
    np.testing.assert_allclose(metrics(state,'unsplit',1,32)['thermal_l1_over_exact_heat_gain'],1.)
    # Equal hot/cold interface errors must survive a zero global signed mean.
    xx,_,band=geometry(state,32)
    state['eth'][:]=h
    state['eth'][band]+=np.where(xx[band]<.5,1.e-3,-1.e-3)
    m=metrics(state,'unsplit',1,32)
    assert abs(m['thermal']['all']['mean'])<1.e-14
    assert m['thermal']['interface']['l1']>9.e-4
    state['u'][:,5:8]=-b
    assert metrics(state,'unsplit',1,32)['magnetic_l1']>.5
    state['u'][:,5:8]=b
    c=configuration(fixture,32,'smr',1,'rkl2')
    assert c['problem/resistive_diffusion']['vx']==1
    assert c['parthenon/time']['dt_force']<.25*.1/(64**2)/ETA
    assert c['hydro']['ct_interface_diagnostics']=='true'
    # Exercise the PHDF diagnostic reader and plot path without a solver build.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'synthetic.phdf'
        with h5py.File(path,'w') as f:
            f['ct_interface_diagnostics']=np.zeros((1,len(LABELS),1,16,16))
            f['ct_energy_diagnostics']=np.zeros((1,5,1,16,16))
            f.create_group('Info').attrs['dt']=1/4100
        assert read_probe(path).shape==(1,len(LABELS),16,16)
        (Path(tmp)/'rounded.hst').write_text('0 2.43902e-4\n0.1 2.43902e-4\n')
        assert measured_dt(Path(tmp))==1/4100
        plot(Path(tmp),'synthetic',[(state,path)],'unsplit',1,32)
        assert (Path(tmp)/'synthetic.png').stat().st_size>0
    print('PASS exact advected helix, signed/cancelling heating and cooling, missing heat, field reversal, configurations, diagnostic I/O and plots')


if __name__=='__main__':
    self_test()
