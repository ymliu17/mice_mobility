"""Three slide-sized discussion figures drawn only from cached analysis tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/06_fiber_joint/ast_timecourses'
OUT=BASE/'slide_discussion'
GROUPS=['Control','Social isolation']
COLORS=['#26799c','#d8663a']
INK='#25343d'


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#9ea9ae')
    ax.tick_params(labelsize=11,colors=INK)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)


def canvas(title,subtitle):
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.07,.93,title,fontsize=25,weight='bold',color=INK)
    fig.text(.07,.885,subtitle,fontsize=13,color='#576970')
    return fig


def legend(fig):
    fig.legend(handles=[Line2D([0],[0],color=c,marker='o',lw=2,label=g) for g,c in zip(GROUPS,COLORS)],
               loc='upper right',bbox_to_anchor=(.94,.855),frameon=False,ncol=2,fontsize=12)


def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=200,facecolor='white')
    plt.close(fig)


def activity():
    d=pd.read_csv(BASE/'group_timecourses.csv')
    d=d[(d.metric=='speed')&(d.time_minutes<=60)]
    fig=canvas('Overall locomotor speed: substantial overlap',
               'AST | First 60 minutes from the first GateOpen | 5-second bins')
    ax=fig.add_axes([.09,.38,.84,.40]);nax=fig.add_axes([.09,.19,.84,.13],sharex=ax)
    for group,color in zip(GROUPS,COLORS):
        x=d[d.group==group].sort_values('time_minutes')
        ax.plot(x.time_minutes,x['mean'],color=color,lw=1.1)
        ax.fill_between(x.time_minutes,x.ci_low,x.ci_high,color=color,alpha=.16)
        nax.plot(x.time_minutes,x.n_mice,color=color,lw=1.1)
    style(ax);style(nax);ax.set_xlim(0,60);ax.set_ylim(bottom=0)
    ax.set_ylabel('Speed (body lengths/s)',fontsize=13)
    ax.tick_params(labelbottom=False);nax.set_ylabel('Mice with data',fontsize=12)
    nax.set_xlabel('Minutes from first GateOpen',fontsize=13);nax.set_yticks([0,5,10,15]);nax.set_ylim(0,16)
    legend(fig)
    fig.text(.09,.09,'Shading: pointwise 95% animal-bootstrap CI. Available animals change over time.',fontsize=11,color=INK)
    fig.text(.09,.055,'Descriptive overlap is not an equivalence test; task stages are not aligned across animals.',fontsize=11,color='#576970')
    save(fig,'01_overall_activity')


def phases():
    a=pd.read_csv(BASE/'choice_phases/animal_phase_metrics.csv')
    a=a[(a.selection=='event_only')&(a.outcome=='correct')]
    e=pd.read_csv(BASE/'choice_phases/phase_group_effects.csv')
    e=e[(e.selection=='event_only')&(e.outcome=='correct')&(e.metric=='duration_s')]
    fig=canvas('Where does choice time differ?',
               'Correct first-dig trials | GateOpen -> first approach -> first dig | Event-only timing')
    rng=np.random.default_rng(34)
    for col,stage in enumerate(['IDS','EDS']):
        left=.09+col*.46
        ax=fig.add_axes([left,.42,.37,.35]);delta=fig.add_axes([left,.19,.37,.14])
        for i,phase in enumerate(['approach','post_approach']):
            for j,(group,color) in enumerate(zip(GROUPS,COLORS)):
                v=a[(a.stage==stage)&(a.phase==phase)&(a.group==group)].duration_s.to_numpy()
                x=i+(j-.5)*.27
                ax.scatter(x+rng.uniform(-.04,.04,len(v)),v,c=color,s=35,alpha=.8,edgecolors='white',linewidths=.4)
                ax.plot([x-.09,x+.09],[v.mean()]*2,color=color,lw=3)
            row=e[(e.stage==stage)&(e.phase==phase)].iloc[0]
            y=1-i
            delta.plot([row.ci_low,row.ci_high],[y,y],color=INK,lw=2)
            delta.scatter(row.isolation_minus_control,y,color=INK,s=35)
        style(ax);style(delta);ax.set_title(stage,loc='left',fontsize=18,weight='bold',pad=10)
        ax.set_xticks([0,1],['Gate -> approach','Approach -> dig'],fontsize=12)
        ax.set_xlim(-.45,1.45);ax.set_ylim(0,48);ax.set_ylabel('Duration (s)',fontsize=12)
        ax.text(.98,.98,'15 Control / 8 isolation',transform=ax.transAxes,ha='right',va='top',fontsize=10,color='#576970')
        delta.axvline(0,color='#a1a9ae',ls='--',lw=1)
        delta.set_yticks([1,0],['Approach','Post-approach'],fontsize=10)
        delta.set_ylim(-.6,1.6);delta.set_xlim(-10,5);delta.set_xlabel('Isolation - control (s), 95% CI',fontsize=11)
    legend(fig)
    fig.text(.09,.09,'Dots: animal means; bars: group means. Same eligible trials across phases; no tracking filter for timing.',fontsize=11,color=INK)
    fig.text(.09,.055,'Both phase-duration difference intervals include zero. Exploratory, unadjusted CIs; approach definition needs validation.',fontsize=11,color='#576970')
    save(fig,'02_choice_timing')


def accuracy():
    """First-dig accuracy, without pose coverage or approach-duration selection."""
    t=pd.read_csv(BASE/'approach_trials/trial_metrics.csv')
    assert not t.duplicated(['session','trial_number']).any()
    t=t[t.stage.isin(['IDS','EDS'])].copy()
    t['included']=t.event_eligible & t.outcome.isin(['correct','incorrect'])
    t['correct']=t.included & t.outcome.eq('correct')
    t['no_dig']=t.outcome.isin(['annotated_miss','no_dig_unresolved'])
    t['excluded_other']=~t.included & ~t.no_dig
    keys=['group','mouse','session','stage']
    sessions=t.groupby(keys).agg(n_gates=('included','size'),n_eligible=('included','sum'),
                                n_correct=('correct','sum'),n_no_dig=('no_dig','sum'),
                                n_excluded_other=('excluded_other','sum')).reset_index()
    sessions['accuracy_pct']=100*sessions.n_correct/sessions.n_eligible.replace(0,np.nan)
    assert (sessions.n_eligible+sessions.n_no_dig+sessions.n_excluded_other==sessions.n_gates).all()
    animals=sessions.groupby(['group','mouse','stage']).agg(
        accuracy_pct=('accuracy_pct','mean'),n_sessions=('session','size'),
        n_eligible=('n_eligible','sum'),n_correct=('n_correct','sum')).reset_index()
    sessions.to_csv(OUT/'choice_accuracy_session_audit.csv',index=False)
    animals.to_csv(OUT/'choice_accuracy_animals.csv',index=False)
    audit=sessions.groupby(['group','stage'])[['n_gates','n_eligible','n_no_dig','n_excluded_other']].sum()
    audit['no_dig_pct']=100*audit.n_no_dig/audit.n_gates
    audit['excluded_other_pct']=100*audit.n_excluded_other/audit.n_gates
    audit.to_csv(OUT/'choice_accuracy_exclusions.csv')
    fig=canvas('Does faster choice come with a change in accuracy?',
               'First-dig choice accuracy | Event-defined trials | No tracking-quality filter')
    rng=np.random.default_rng(36);effects=[]
    for col,stage in enumerate(['IDS','EDS']):
        left=.09+col*.46
        ax=fig.add_axes([left,.39,.37,.38]);delta=fig.add_axes([left,.21,.37,.075])
        values=[]
        for j,(group,color) in enumerate(zip(GROUPS,COLORS)):
            v=animals.loc[(animals.stage==stage)&(animals.group==group),'accuracy_pct'].dropna().to_numpy()
            values.append(v)
            ax.scatter(j+rng.uniform(-.08,.08,len(v)),v,s=48,color=color,alpha=.85,edgecolors='white',linewidths=.5)
            ax.plot([j-.15,j+.15],[v.mean()]*2,color=color,lw=3)
            n=int(audit.loc[(group,stage),'n_eligible'])
            ax.text(j,103,f'{len(v)} mice / {n} trials',ha='center',fontsize=11,color=color)
        a,b=values
        diff=b.mean()-a.mean()
        boot=rng.choice(b,(5000,len(b))).mean(axis=1)-rng.choice(a,(5000,len(a))).mean(axis=1)
        lo,hi=np.percentile(boot,[2.5,97.5])
        effects.append(dict(stage=stage,n_control=len(a),n_isolation=len(b),
                            control_accuracy_pct=a.mean(),isolation_accuracy_pct=b.mean(),
                            difference_pp=diff,ci_low_pp=lo,ci_high_pp=hi))
        style(ax);ax.set_title(stage,loc='left',fontsize=18,weight='bold',pad=12)
        ax.set_xlim(-.5,1.5);ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100])
        ax.set_xticks([0,1],['Control','Social isolation'],fontsize=13)
        ax.set_ylabel('First-dig accuracy (%)',fontsize=13)
        style(delta);delta.axvline(0,color='#a1a9ae',ls='--',lw=1)
        delta.plot([lo,hi],[0,0],color=INK,lw=3);delta.scatter(diff,0,color=INK,s=45)
        delta.set_xlim(-40,40);delta.set_ylim(-1,1);delta.set_yticks([])
        delta.set_xlabel('Isolation - control (percentage points), 95% CI',fontsize=11)
    legend(fig)
    fig.text(.09,.105,'Dots: animal accuracies (session rates averaged per animal); bars: equal-weight animal means.',fontsize=11,color=INK)
    fig.text(.09,.065,'No-dig, ambiguous/Miss-conflict and other ineligible trials excluded, not counted as errors. See exclusion audit.',fontsize=11,color='#576970')
    fig.text(.09,.035,'Exploratory animal-bootstrap CIs, unadjusted. Accuracy alone does not establish a speed-accuracy trade-off.',fontsize=11,color='#576970')
    pd.DataFrame(effects).to_csv(OUT/'choice_accuracy_group_effects.csv',index=False)
    save(fig,'03_choice_accuracy')


def quality():
    t=pd.read_csv(BASE/'approach_trials/trial_metrics.csv')
    t=t[t.gate_in_video]
    fig=canvas('What needs validation before interpretation?',
               'Tracking coverage, event definitions and unequal recording conditions')
    rng=np.random.default_rng(35)
    for i,stage in enumerate(['IDS','EDS']):
        ax=fig.add_axes([.09+i*.245,.26,.19,.49])
        for j,(group,color) in enumerate(zip(GROUPS,COLORS)):
            vals=t[(t.stage==stage)&(t.group==group)].groupby('mouse').coverage_speed.mean()*100
            ax.scatter(j+rng.uniform(-.08,.08,len(vals)),vals,s=42,color=color,alpha=.85)
            ax.plot([j-.13,j+.13],[vals.mean()]*2,color=color,lw=3)
            ax.text(j,102,f'n={len(vals)}',ha='center',fontsize=10,color=color)
        style(ax);ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100])
        ax.set_xticks([0,1],['Control','Isolation'],fontsize=11)
        ax.set_title(stage,fontsize=18,loc='left',weight='bold')
        if i==0:ax.set_ylabel('Mean trial speed coverage (%)',fontsize=12)
    notes=[('01  Validate event definitions',
            '63 trials contain Miss before the first dig.\nConfirm Miss/CR meaning and the first-approach criterion.'),
           ('02  Check selection and recording effects',
            'Short approach windows lose many trials under strict QC.\nCam2 occurs only in Control; date/batch effects remain.'),
           ('03  Validate the behavioral interpretation',
            'Review pauses, revisits and first-choice outcomes on video.\nTarget zones are provisional; shorter latency is not better learning.')]
    for y,(heading,body) in zip([.72,.53,.34],notes):
        fig.text(.59,y,heading,fontsize=15,weight='bold',color=INK)
        fig.text(.59,y-.045,body,fontsize=11.5,color='#576970',va='top',linespacing=1.6)
    fig.text(.09,.12,'Each dot is one animal; coverage is averaged across all retained GateOpen trials within a stage.',fontsize=11,color=INK)
    fig.text(.09,.075,'QC plots describe the data, not a biological group effect. One Control recording lacks GateOpen annotations.',fontsize=11,color='#576970')
    save(fig,'03_validation_and_QC')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    activity();phases();accuracy();quality()
    print(f'Saved 3 slide figures plus QC backup (PNG, PDF, SVG) to {OUT}')
