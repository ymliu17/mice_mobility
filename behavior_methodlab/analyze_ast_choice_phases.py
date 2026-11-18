"""Split choice behavior at the first annotated approach; retain boundary audits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from behavior_methodlab.analyze_ast_approach_trials import (
        ROOT, SOURCE, COLORS, GROUPS, STAGES, SEED, load_motion, window_metrics)
except ModuleNotFoundError:
    from analyze_ast_approach_trials import (
        ROOT, SOURCE, COLORS, GROUPS, STAGES, SEED, load_motion, window_metrics)

OUT=SOURCE/'choice_phases'
PHASES={'approach':'Gate to first approach',
        'post_approach':'First approach to first dig',
        'choice':'Gate to first dig (whole choice)'}
METRICS={'duration_s':'Duration (s)','mean_speed_bl_s':'Mean speed (BL/s)',
         'path_bl':'Observed path (BL)','mean_turn_deg_s':'Turning speed (deg/s)',
         'mean_target_bl':'Nearest fixed-zone distance (BL)*'}


def boundaries(trial,events):
    """Do not interpret a Miss as a terminal event without a confirmed definition."""
    d=events[(events.time>=trial.gate_time_s)&(events.time<trial.interval_end_s)&(events.stage==trial.stage)].sort_values('time')
    digs=d[d.event.isin(['CorrectDigStart','IncorrectDigStart'])]
    dig=float(digs.iloc[0].time) if len(digs) else np.nan
    apps=d[d.event.isin(['ApproachCorrect','ApproachIncorrect'])]
    before=apps[apps.time<=dig] if np.isfinite(dig) else apps
    approach=float(before.iloc[0].time) if len(before) else np.nan
    tied=bool(len(before) and before[before.time==approach].event.nunique()>1)
    miss=d[d.event=='Miss']
    miss_t=float(miss.time.min()) if len(miss) else np.nan
    return dict(first_approach_time_s=approach,first_dig_time_s=dig,
                first_miss_time_s=miss_t,ambiguous_approach=tied,
                approach_only_after_dig=bool(len(apps) and np.isfinite(dig) and apps.time.min()>dig),
                strict_miss_terminal_end_s=min(trial.interval_end_s,miss_t) if np.isfinite(miss_t) else trial.interval_end_s,
                first_dig_survives_miss_terminal=bool(np.isfinite(dig) and (not np.isfinite(miss_t) or dig<miss_t)),
                n_approaches_before_dig=len(before))


def summarize(data,threshold,mode):
    base=data.phase_eligible & (data.duration_s>=.5)
    passing=base & (data[['coverage_speed','coverage_turn','coverage_target']].min(axis=1)>=threshold)
    if mode in ['same_trials','event_only']:
        # Same trials for all three phases: paired durations decompose exactly.
        counts=data[passing].groupby(['session','trial_number']).phase.nunique()
        keys=set(counts[counts==3].index)
        passing &= pd.Series([(s,n) in keys for s,n in zip(data.session,data.trial_number)],index=data.index)
    d=data[passing]
    keys=['group','mouse','session','stage','outcome','phase']
    session=d.groupby(keys)[list(METRICS)].mean()
    mouse=session.groupby(['group','mouse','stage','outcome','phase']).mean().reset_index()
    counts=d.groupby(['group','mouse','stage','outcome','phase']).size().rename('n_trials').reset_index()
    result=mouse.merge(counts);result['coverage_threshold']=threshold;result['selection']=mode
    if mode=='event_only':
        result.loc[:,[k for k in METRICS if k!='duration_s']]=np.nan
    return result


def effects(animals):
    rng=np.random.default_rng(SEED);rows=[]
    for key,d in animals.groupby(['selection','coverage_threshold','stage','outcome','phase']):
        for metric in METRICS:
            a=d.loc[d.group==GROUPS[0],metric].dropna().to_numpy()
            b=d.loc[d.group==GROUPS[1],metric].dropna().to_numpy()
            lo=hi=np.nan
            if min(len(a),len(b))>=3:
                diff=rng.choice(b,(2000,len(b))).mean(1)-rng.choice(a,(2000,len(a))).mean(1)
                lo,hi=np.percentile(diff,[2.5,97.5])
            rows.append(dict(zip(['selection','coverage_threshold','stage','outcome','phase'],key),
                             metric=metric,n_control=len(a),n_isolation=len(b),
                             mean_control=a.mean() if len(a) else np.nan,mean_isolation=b.mean() if len(b) else np.nan,
                             isolation_minus_control=b.mean()-a.mean() if len(a) and len(b) else np.nan,
                             ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows)


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)


def plot(animals,phase_table):
    rng=np.random.default_rng(SEED)
    d=animals[(animals.coverage_threshold==.8)&(animals.selection=='same_trials')]
    for stage in STAGES:
        for outcome in ['correct','incorrect']:
            fig,axes=plt.subplots(3,5,figsize=(17,10))
            for row,(phase,title) in enumerate(PHASES.items()):
                sub=d[(d.stage==stage)&(d.outcome==outcome)&(d.phase==phase)]
                for col,(metric,label) in enumerate(METRICS.items()):
                    ax=axes[row,col]
                    for j,group in enumerate(GROUPS):
                        vals=sub.loc[sub.group==group,metric].dropna().to_numpy()
                        ax.scatter(j+rng.uniform(-.09,.09,len(vals)),vals,s=23,color=COLORS[group],alpha=.8)
                        if len(vals):ax.plot([j-.18,j+.18],[vals.mean()]*2,lw=2.5,color=COLORS[group])
                        ax.text(j,.98,f'n={len(vals)}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
                    ax.set_xticks([0,1],['Control','Isolation'],fontsize=8);ax.set_xlim(-.5,1.5)
                    ax.set_title(label,fontsize=9);ax.grid(axis='y',alpha=.2);ax.margins(y=.25)
                    if col==0:ax.set_ylabel(title,fontsize=10)
            fig.suptitle(f'{stage} | {outcome} first dig | same trials across all phases',fontsize=14)
            fig.text(.1,.015,'Points = animal means; lines = group means. Each phase >=0.5 s and tracking coverage >=80%. *Provisional zones. Approach is annotation-defined.',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.95));save(fig,f'{stage}_{outcome}_phase_comparison')
    d=animals[animals.selection=='event_only']
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=False)
    for ax,(stage,outcome) in zip(axes.flat,[(s,o) for s in STAGES for o in ['correct','incorrect']]):
        sub=d[(d.stage==stage)&(d.outcome==outcome)]
        for i,group in enumerate(GROUPS):
            a=sub[(sub.group==group)&(sub.phase=='approach')].set_index('mouse').duration_s
            b=sub[(sub.group==group)&(sub.phase=='post_approach')].set_index('mouse').duration_s
            shared=a.index.intersection(b.index)
            if len(shared):
                av,bv=a.loc[shared].mean(),b.loc[shared].mean()
                ax.bar(i,av,color=COLORS[group],alpha=.4)
                ax.bar(i,bv,bottom=av,color=COLORS[group])
                ax.text(i,av+bv,f'n={len(shared)}',ha='center',va='bottom',fontsize=9)
            else:
                ax.text(i,.05,'No eligible animals',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.set_xticks([0,1],GROUPS,fontsize=9);ax.set_ylabel('Mean choice duration (s)');ax.set_title(f'{stage} | {outcome}');ax.margins(y=.2)
    fig.suptitle('Choice duration: first approach (light) + post-approach interval (dark)')
    fig.text(.1,.01,'Event-only timing; no tracking filter. Same ordered trials in both intervals; ambiguous trials excluded.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));save(fig,'choice_duration_decomposition')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(SOURCE/'approach_trials/trial_metrics.csv')
    events=pd.read_csv(SOURCE/'parsed_events.csv')
    qc=pd.read_csv(SOURCE/'session_QC.csv').set_index('session')
    zones=pd.read_csv(SOURCE/'target_zones.csv').set_index('session')
    index=json.loads((ROOT/'outputs/06_fiber_joint/index.json').read_text())
    rows=[];audit=[]
    for session,trials in old.groupby('session'):
        t=np.load(ROOT/'outputs/06_fiber_joint/sessions'/f'{session}.npz')['cam_t']
        z=zones.loc[session,['left_x','left_y','right_x','right_y']].to_numpy(float).reshape(2,2)
        m=load_motion(index[session]['h5'],t,float(qc.loc[session,'fps']),z)
        ev=events[events.session==session]
        for trial in trials.itertuples():
            bound=boundaries(trial,ev)
            audit.append(dict(session=session,trial_number=trial.trial_number,stage=trial.stage,
                              outcome=trial.outcome,miss_before_first_dig=trial.miss_before_first_dig,**bound))
            approach,dig=bound['first_approach_time_s'],bound['first_dig_time_s']
            ordered=np.isfinite(approach) and np.isfinite(dig) and trial.gate_time_s<approach<dig
            for phase,(start,end) in zip(PHASES,[(trial.gate_time_s,approach),(approach,dig),(trial.gate_time_s,dig)]):
                valid=np.isfinite(start) and np.isfinite(end) and start<end
                metrics=window_metrics(m,start,end) if valid else {k:np.nan for k in [*METRICS,'coverage_speed','coverage_turn','coverage_target']}
                rows.append(dict(session=session,trial_number=trial.trial_number,mouse=trial.mouse,
                                 group=trial.group,camera=trial.camera,stage=trial.stage,outcome=trial.outcome,
                                 phase=phase,start_s=start,end_s=end,ordered_boundaries=ordered,
                                 phase_eligible=bool(trial.event_eligible and valid and not bound['ambiguous_approach'] and (ordered or phase=='choice')),
                                 **metrics))
        print(session,flush=True)
    data=pd.DataFrame(rows);data.to_csv(OUT/'trial_phase_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'trial_boundary_audit.csv',index=False)
    animals=pd.concat([summarize(data,t,mode) for mode in ['same_trials','phase_specific'] for t in [0,.5,.8,.9]]+[summarize(data,0,'event_only')],ignore_index=True)
    animals.to_csv(OUT/'animal_phase_metrics.csv',index=False)
    effects(animals).to_csv(OUT/'phase_group_effects.csv',index=False)
    cam1=pd.concat([summarize(data[data.camera=='Cam1'],t,'same_trials') for t in [0,.5,.8,.9]],ignore_index=True)
    effects(cam1).to_csv(OUT/'cam1_only_phase_effects.csv',index=False)
    plot(animals,data)
    assert len(data)==3*len(old)
    paired=animals[(animals.selection=='same_trials')].pivot(index=['coverage_threshold','group','mouse','stage','outcome'],columns='phase',values='duration_s')
    assert np.allclose(paired.approach+paired.post_approach,paired.choice)
    print('PASS: all gates retained in three phase rows; paired mean durations sum to whole choice.')


if __name__=='__main__':
    run()
