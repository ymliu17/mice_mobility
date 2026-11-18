"""Gate-to-first-dig AST analysis with animal-level summaries and QC sensitivity.

Uses raw custom DLC poses, cached camera timestamps and the reviewed plotting
event table. Does not change Stage 6 outputs. Target zones remain provisional.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from behavior_methodlab.plot_ast_timecourses import smooth_segments, COLORS
except ModuleNotFoundError:
    from plot_ast_timecourses import smooth_segments, COLORS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/06_fiber_joint/ast_timecourses'
OUT = SOURCE / 'approach_trials'
GROUPS = ['Control', 'Social isolation']
STAGES = ['IDS', 'EDS']
SEED = 20261108
METRICS = {
    'approach_latency_s': 'First approach latency (s)',
    'dig_latency_s': 'First dig latency (s)',
    'path_bl': 'Observed path length (BL)',
    'mean_speed_bl_s': 'Mean speed (BL/s)',
    'total_turn_deg': 'Observed total turning (deg)',
    'mean_turn_deg_s': 'Mean turning speed (deg/s)',
    'mean_target_bl': 'Mean distance to nearest zone (BL)*',
    'first5_speed_bl_s': 'Speed in first 5 s (BL/s)',
}


def build_trials(events, video_start, video_end):
    """Keep all gates; first dig defines choice, with explicit censoring flags."""
    events = events.drop_duplicates(['time', 'event', 'stage']).sort_values('time')
    gates = events[events.event == 'GateOpen'].reset_index(drop=True)
    result = []
    for i, gate in gates.iterrows():
        start = float(gate.time)
        next_gate = float(gates.iloc[i+1].time) if i+1 < len(gates) else np.inf
        end = min(start+180, next_gate, video_end)
        stage_events = events[(events.time >= start) & (events.time < end) & (events.stage == gate.stage)]
        digs = stage_events[stage_events.event.isin(['CorrectDigStart', 'IncorrectDigStart'])]
        apps = stage_events[stage_events.event.isin(['ApproachCorrect', 'ApproachIncorrect'])]
        miss = stage_events[stage_events.event == 'Miss']
        outcome = 'no_dig_unresolved'
        dig_t = np.nan
        tied = False
        if len(digs):
            dig_t = float(digs.iloc[0].time)
            tied = digs[digs.time == dig_t].event.nunique() > 1
            outcome = 'ambiguous' if tied else ('correct' if digs.iloc[0].event == 'CorrectDigStart' else 'incorrect')
        elif len(miss):
            outcome = 'annotated_miss'
        stop = dig_t if np.isfinite(dig_t) else end
        apps = apps[apps.time <= stop]
        conflict = bool(len(miss) and np.isfinite(dig_t) and miss.time.min() <= dig_t)
        # Include a stimulus that starts before the gate and is still active.
        pinch_events = events[(events.time <= stop) & events.event.isin(['TailPinchStart', 'TailPinchStop'])]
        before = pinch_events[pinch_events.time < start]
        active = len(before) > 0 and before.iloc[-1].event == 'TailPinchStart'
        stimulated = bool(active or ((pinch_events.time >= start) & (pinch_events.event == 'TailPinchStart')).any())
        truncated = start < video_start or (not np.isfinite(dig_t) and end < start+180)
        result.append(dict(trial_number=i+1, stage=gate.stage, gate_time_s=start,
                           interval_end_s=end, window_end_s=stop, outcome=outcome,
                           dig_latency_s=dig_t-start if np.isfinite(dig_t) else np.nan,
                           approach_latency_s=float(apps.iloc[0].time)-start if len(apps) else np.nan,
                           first_approach=apps.iloc[0].event if len(apps) else '',
                           missing_approach=len(apps)==0, n_dig_events=len(digs),
                           both_dig_types=digs.event.nunique()>1, miss_before_first_dig=conflict,
                           simultaneous_conflicting_digs=tied, tail_pinch=stimulated,
                           censored_or_outside_video=truncated, gate_in_video=video_start<=start<video_end,
                           event_eligible=outcome in ['correct','incorrect'] and not conflict and not stimulated
                           and not tied and video_start<=start<video_end and stop>start))
    return pd.DataFrame(result)


def load_motion(path, times, fps, zones):
    raw = pd.read_hdf(path).droplevel(0, axis=1)
    n = min(len(raw),len(times))
    raw = raw.iloc[:n]; times = times[:n]
    assert np.all(np.diff(times)>0), 'Camera timestamps must increase'
    def point(name):
        xy=raw[name][['x','y']].to_numpy(float)
        good=(raw[name]['likelihood'].to_numpy()>=.6)&np.isfinite(xy).all(1)
        return smooth_segments(xy,good)
    center,neck,tail = point('body_center'),point('neck'),point('tail_base')
    bl=float(np.nanmedian(np.linalg.norm(neck-tail,axis=1)))
    assert np.isfinite(bl) and bl>0
    dt=np.r_[np.nan,np.diff(times)]
    valid_dt=(dt>0)&(dt<1.5/fps)
    step=np.r_[np.nan,np.linalg.norm(np.diff(center,axis=0),axis=1)]/bl
    step[~valid_dt]=np.nan
    direction=np.arctan2((neck-tail)[:,1],(neck-tail)[:,0])
    turn=np.abs(np.angle(np.exp(1j*np.r_[np.nan,np.diff(direction)])))*180/np.pi
    turn[~valid_dt]=np.nan
    target=np.min(np.linalg.norm(center[:,None,:]-zones[None,:,:],axis=2),axis=1)/bl
    return dict(t=times,dt=dt,step=step,turn=turn,target=target,bl=bl)


def window_metrics(m,start,end):
    duration=end-start
    # Steps belong only if both endpoints are inside this trial window.
    valid=(m['t']>=start)&(m['t']<end)&(m['t']-m['dt']>=start)
    motion=valid & np.isfinite(m['step'])
    turning=valid & np.isfinite(m['turn'])
    target=valid & np.isfinite(m['target']) & np.isfinite(m['dt'])
    sm=float(m['dt'][motion].sum());st=float(m['dt'][turning].sum());sz=float(m['dt'][target].sum())
    path=float(m['step'][motion].sum()) if sm>0 else np.nan
    total_turn=float(m['turn'][turning].sum()) if st>0 else np.nan
    return dict(duration_s=duration, coverage_speed=min(1,sm/duration) if duration>0 else 0,
                coverage_turn=min(1,st/duration) if duration>0 else 0,
                coverage_target=min(1,sz/duration) if duration>0 else 0,
                path_bl=path,mean_speed_bl_s=path/sm if sm>0 else np.nan,
                total_turn_deg=total_turn,mean_turn_deg_s=total_turn/st if st>0 else np.nan,
                mean_target_bl=float(np.average(m['target'][target],weights=m['dt'][target])) if sz>0 else np.nan)


def animal_summary(trials,threshold):
    keep=trials.event_eligible & (trials.duration_s>=.5)
    if threshold>0:
        keep &= trials[['coverage_speed','coverage_turn','coverage_target']].min(axis=1)>=threshold
    d=trials[keep].copy()
    # A missing first-5-second value stays missing: never use post-choice motion.
    d.loc[d.first5_coverage<threshold,'first5_speed_bl_s']=np.nan
    per_session=d.groupby(['group','mouse','session','stage','outcome'])[list(METRICS)].median()
    per_mouse=per_session.groupby(['group','mouse','stage','outcome']).mean().reset_index()
    counts=d.groupby(['group','mouse','stage','outcome']).size().rename('n_trials').reset_index()
    out=per_mouse.merge(counts)
    out['coverage_threshold']=threshold
    return out


def effect_table(animals):
    rng=np.random.default_rng(SEED);rows=[]
    for (threshold,stage,outcome),d in animals.groupby(['coverage_threshold','stage','outcome']):
        for metric in METRICS:
            a=d.loc[d.group==GROUPS[0],metric].dropna().to_numpy()
            b=d.loc[d.group==GROUPS[1],metric].dropna().to_numpy()
            delta=b.mean()-a.mean() if len(a) and len(b) else np.nan
            lo=hi=np.nan
            if min(len(a),len(b))>=3:
                differences=np.mean(rng.choice(b,(2000,len(b))),axis=1)-np.mean(rng.choice(a,(2000,len(a))),axis=1)
                lo,hi=np.percentile(differences,[2.5,97.5])
            rows.append(dict(coverage_threshold=threshold,stage=stage,outcome=outcome,metric=metric,
                             n_control=len(a),n_isolation=len(b),mean_control=a.mean() if len(a) else np.nan,
                             mean_isolation=b.mean() if len(b) else np.nan,
                             isolation_minus_control=delta,ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows)


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def figures(trials,animals,effects):
    rng=np.random.default_rng(SEED)
    for stage in STAGES:
        fig,axes=plt.subplots(2,4,figsize=(15,8))
        d=animals[(animals.coverage_threshold==.8)&(animals.stage==stage)]
        for ax,(metric,label) in zip(axes.flat,METRICS.items()):
            for i,outcome in enumerate(['correct','incorrect']):
                for j,group in enumerate(GROUPS):
                    values=d.loc[(d.outcome==outcome)&(d.group==group),metric].dropna().to_numpy()
                    x=i+(j-.5)*.3
                    ax.scatter(x+rng.uniform(-.045,.045,len(values)),values,color=COLORS[group],s=24,alpha=.8)
                    if len(values):ax.plot([x-.09,x+.09],[values.mean()]*2,color=COLORS[group],lw=2.5)
                    ax.text(x,.98,f'n={len(values)}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8,color=COLORS[group])
            ax.set_xticks([0,1],['Correct first dig','Incorrect first dig'],fontsize=8)
            ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.2);ax.margins(y=.2)
        fig.suptitle(f'{stage}: GateOpen to first dig | tracking coverage >=80%',fontsize=14)
        fig.text(.1,.025,'Blue: Control; orange: Social isolation. Points: animal summaries; bars: group means. *Fixed target zones are provisional. First-5-s panel requires no dig before 5 s.',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.94));save(fig,f'{stage}_approach_metrics')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for r,stage in enumerate(STAGES):
        d=trials[(trials.stage==stage)&trials.gate_in_video]
        for j,group in enumerate(GROUPS):
            g=d[d.group==group]
            for c,(field,label) in enumerate([('coverage_speed','Speed coverage'),('event_eligible','Eligible first-dig fraction'),('no_dig','No annotated dig fraction')]):
                h=g.assign(no_dig=g.outcome.isin(['annotated_miss','no_dig_unresolved']))
                vals=h.groupby('mouse')[field].mean().dropna()
                axes[r,c].scatter(j+rng.uniform(-.08,.08,len(vals)),vals,color=COLORS[group],s=25)
                if len(vals):axes[r,c].plot([j-.15,j+.15],[vals.mean()]*2,color=COLORS[group],lw=3)
                axes[r,c].set_title(f'{stage}: {label}');axes[r,c].set_xticks([0,1],['Control','Isolation']);axes[r,c].set_ylim(0,1.05)
    fig.suptitle('Trial QC and annotation completeness | all gates retained')
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'trial_QC')
    fig,axes=plt.subplots(2,4,figsize=(15,8))
    for ax,(metric,label) in zip(axes.flat,METRICS.items()):
        for j,(stage,outcome) in enumerate([(s,o) for s in STAGES for o in ['correct','incorrect']]):
            d=effects[(effects.metric==metric)&(effects.stage==stage)&(effects.outcome==outcome)].sort_values('coverage_threshold')
            ax.plot(d.coverage_threshold,d.isolation_minus_control,'o-',ms=3,label=f'{stage} {outcome}')
        ax.axhline(0,color='gray',lw=.7);ax.set_title(label,fontsize=9);ax.set_xlabel('Minimum tracking coverage');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('Sensitivity: isolation minus control | animal mean differences')
    fig.text(.1,.01,'Coverage thresholds select different animals/trials; this is not exact quality matching. Bootstrap intervals and counts are in group_effects.csv.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));save(fig,'coverage_sensitivity')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    events=pd.read_csv(SOURCE/'parsed_events.csv')
    qc=pd.read_csv(SOURCE/'session_QC.csv').set_index('session')
    zones=pd.read_csv(SOURCE/'target_zones.csv').set_index('session')
    index=json.loads((ROOT/'outputs/06_fiber_joint/index.json').read_text())
    all_trials=[];audit=[]
    for session,v in sorted(index.items()):
        if v['test']!='AST':continue
        ev=events[events.session==session]
        if not (ev.event=='GateOpen').any():
            audit.append(dict(session=session,status='no_gate_annotations',n_gates=0));continue
        t=np.load(ROOT/'outputs/06_fiber_joint/sessions'/f'{session}.npz')['cam_t']
        z=zones.loc[session,['left_x','left_y','right_x','right_y']].to_numpy(float).reshape(2,2)
        m=load_motion(v['h5'],t,float(qc.loc[session,'fps']),z)
        trials=build_trials(ev,float(m['t'][0]),float(m['t'][-1]))
        metrics=[]
        for trial in trials.itertuples():
            row=window_metrics(m,trial.gate_time_s,trial.window_end_s)
            first5=window_metrics(m,trial.gate_time_s,trial.gate_time_s+5)
            full5=trial.window_end_s>=trial.gate_time_s+5
            row['first5_coverage']=first5['coverage_speed'] if full5 else 0
            row['first5_speed_bl_s']=first5['mean_speed_bl_s'] if full5 else np.nan
            metrics.append(row)
        trials=pd.concat([trials,pd.DataFrame(metrics)],axis=1)
        trials['session']=session;trials['mouse']=qc.loc[session,'mouse'];trials['group']=v['group'];trials['camera']=v['cam']
        trials['body_length_px']=m['bl'];all_trials.append(trials)
        audit.append(dict(session=session,status='processed',n_gates=len(trials)))
        print(session,len(trials),flush=True)
    trials=pd.concat(all_trials,ignore_index=True)
    trials.to_csv(OUT/'trial_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'session_audit.csv',index=False)
    animals=pd.concat([animal_summary(trials,x) for x in [0,.5,.8,.9]],ignore_index=True)
    animals.to_csv(OUT/'animal_stage_outcome_metrics.csv',index=False)
    effects=effect_table(animals);effects.to_csv(OUT/'group_effects.csv',index=False)
    cam1_animals=pd.concat([animal_summary(trials[trials.camera=='Cam1'],x) for x in [0,.5,.8,.9]],ignore_index=True)
    effect_table(cam1_animals).to_csv(OUT/'cam1_only_group_effects.csv',index=False)
    trials.groupby(['group','stage','outcome']).size().rename('n_trials').to_csv(OUT/'outcome_counts.csv')
    figures(trials,animals,effects)
    print(trials.groupby(['group','stage','outcome']).size().to_string())
    print('Eligible:',trials.event_eligible.sum(),'of',len(trials))


if __name__=='__main__':
    run()
