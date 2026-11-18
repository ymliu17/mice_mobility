"""AST time courses; fixed target-zone estimates are explicitly provisional."""
from pathlib import Path
import json
import re
import warnings
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/06_fiber_joint/ast_timecourses'
# Approximate fixed task-zone centers in native 640x480 images, visually read
# from 120-second reference frames. Not trial-specific pot tracking.
ZONES = [(255,285,402,285),(265,295,415,295),(240,300,385,300),
         (250,285,400,285),(225,280,370,280),(225,280,370,280),
         (245,280,390,280),(255,280,405,280),(215,275,365,275),
         (230,280,380,280),(225,265,375,265),(220,280,375,280),
         (290,275,420,275),(240,280,385,280),(240,280,385,280),
         (245,270,390,270),(240,265,390,265),(235,280,385,280),
         (240,265,385,265),(240,260,390,260),(235,350,385,350),
         (235,350,385,350),(235,350,385,350),(235,350,385,350)]
METRICS = ['speed','turn','distance','target']
LABELS = ['Speed (BL/s)','Turning speed (deg/s)',
          'Observed cumulative distance (BL)','Nearest target-zone distance (BL)\n[provisional fixed zones]']
COLORS = {'Control':'#26799c','Social isolation':'#d8663a'}
EVENT_STYLE = {
    'GateOpen': ('Gate open', '#555555'),
    'ApproachCorrect': ('Approach correct', '#268f84'),
    'ApproachIncorrect': ('Approach incorrect', '#d58431'),
    'CorrectDigStart': ('Dig correct', '#3164b0'),
    'IncorrectDigStart': ('Dig incorrect', '#c74d54'),
    'EatingStart': ('Eating start', '#8064a2'),
    'Miss': ('Miss', '#222222'),
}


def redraw_events():
    """Redraw from cached tables, preserving the original time alignment."""
    from matplotlib.lines import Line2D
    data=pd.read_csv(OUT/'session_timecourses_5s.csv')
    ev=pd.read_csv(OUT/'parsed_events.csv')
    qc=pd.read_csv(OUT/'session_QC.csv')
    with PdfPages(OUT/'all_AST_timecourses.pdf') as pdf:
        for s,b in data.groupby('session',sort=True):
            b=b.sort_values('time_seconds');q=qc.set_index('session').loc[s]
            stop=b.time_seconds.max()/60
            e=ev[(ev.session==s)&(ev.relative_seconds>=0)&(ev.relative_seconds<=stop*60)]
            fig,aa=plt.subplots(5,1,figsize=(13,11),sharex=True,
                               gridspec_kw={'height_ratios':[1,1,1,1,1.1]})
            for a,metric,label in zip(aa[:4],METRICS,LABELS):
                a.plot(b.time_seconds/60,b[metric],color=COLORS[q['group']],lw=.7)
                a.set_ylabel(label,fontsize=9);a.grid(alpha=.2)
            for stage,g in e[e.event=='GateOpen'].groupby('stage',sort=False):
                x=g.relative_seconds.min()/60
                for a in aa:a.axvline(x,color='gray',ls=':',lw=.7)
                aa[0].text(x,.98,stage,transform=aa[0].get_xaxis_transform(),fontsize=8,va='top')
            for k,(event,(label,color)) in enumerate(EVENT_STYLE.items()):
                x=e.loc[e.event==event,'relative_seconds']/60
                aa[4].scatter(x,np.full(len(x),k),marker='|',s=65,color=color,linewidths=.9)
            aa[4].set_yticks(range(len(EVENT_STYLE)),[x[0] for x in EVENT_STYLE.values()],fontsize=8)
            aa[4].set_ylim(len(EVENT_STYLE)-.5,-.5);aa[4].grid(axis='x',alpha=.2)
            if e.empty:aa[4].text(.5,.5,'Event annotations unavailable',transform=aa[4].transAxes,ha='center')
            aa[4].set_xlabel('Minutes from first GateOpen' if q.gate_aligned else 'Minutes from first video timestamp (GateOpen unavailable)')
            aa[4].set_xlim(0,stop)
            fig.suptitle(f'{q.mouse} | {q["group"]} | {s.split("_")[0]}',fontsize=11)
            fig.text(.12,.01,'Ticks = annotated event onsets; dotted lines = task-stage starts. Target zones remain provisional.',fontsize=9)
            fig.tight_layout(rect=(0,.03,1,.96));save(fig,'per_video/'+s);pdf.savefig(fig);plt.close(fig)
    summary=pd.read_csv(OUT/'group_timecourses.csv')
    fig,axes=plt.subplots(7,1,figsize=(14,15),sharex=True,
                          gridspec_kw={'height_ratios':[1,1,1,1,.5,1.4,1]})
    for group,g in summary.groupby('group'):
        for i,metric in enumerate(METRICS):
            d=g[g.metric==metric].sort_values('time_minutes')
            axes[i].plot(d.time_minutes,d['mean'],color=COLORS[group],lw=.8,label=group)
            axes[i].fill_between(d.time_minutes,d.ci_low,d.ci_high,color=COLORS[group],alpha=.18)
            axes[i].set_ylabel(LABELS[i],fontsize=9)
            if i==0:axes[4].plot(d.time_minutes,d.n_mice,color=COLORS[group])
    axes[0].legend();axes[4].set_ylabel('N mice\n(speed)',fontsize=9)
    for ax,group in zip(axes[5:],['Control','Social isolation']):
        eligible=qc[(qc['group']==group)&qc.gate_aligned].sort_values('mouse')
        mice=eligible.mouse.unique().tolist()
        for k,mouse in enumerate(mice):
            sessions=eligible[eligible.mouse==mouse].session
            e=ev[ev.session.isin(sessions)&(ev.relative_seconds>=0)]
            for event,(label,color) in EVENT_STYLE.items():
                x=e.loc[e.event==event,'relative_seconds']/60
                ax.scatter(x,np.full(len(x),k),marker='|',s=26,color=color,linewidths=.7,alpha=.7)
        ax.set_yticks(range(len(mice)),mice,fontsize=6)
        ax.set_ylim(len(mice)-.5,-.5);ax.set_ylabel(group+'\nevents by mouse',fontsize=9)
    handles=[Line2D([0],[0],color=color,marker='|',ls='',label=label) for label,color in EVENT_STYLE.values()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.97),ncol=4,fontsize=8,frameon=False)
    for a in axes:a.grid(axis='x',alpha=.18)
    axes[-1].set_xlabel('Minutes from first GateOpen')
    fig.suptitle('AST group time courses and individual event timing',fontsize=14)
    fig.text(.12,.01,'Event ticks are individual annotations, not group-average events. CI: animal bootstrap. Fixed target zones are provisional.',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.93));save(fig,'group_mean_timecourses');plt.close(fig)
    print(f'Redrawn {data.session.nunique()} individual figures and group figure with event markers.')


def events(folder):
    if not list(folder.glob('*.tnt')) or not list(folder.glob('*SavedNotes.nxt')):
        return pd.DataFrame(columns=['time','event','stage','camera'])
    names = next(folder.glob('*.tnt')).read_text(errors='replace').splitlines()[1:]
    rows=[]
    for line in next(folder.glob('*SavedNotes.nxt')).read_text(errors='replace').splitlines():
        m=re.match(r'(\S+)\s+@\s+([\d.]+)\s+->\s+(\d+)',line)
        if not m or not 0<int(m[3])<=len(names): continue
        name=names[int(m[3])-1].strip()
        stage=re.search(r'-(SD|CD|RL|IDS|EDS)$',name)
        rows.append((float(m[2]),re.sub(r'-(SD|CD|RL|IDS|EDS)$','',name),stage[1] if stage else 'EDS',m[1]))
    return pd.DataFrame(rows,columns=['time','event','stage','camera']).sort_values('time')


def smooth_segments(x, valid):
    out=np.full_like(x,np.nan)
    edges=np.diff(np.r_[False,valid,False].astype(int))
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        out[a:b]=savgol_filter(x[a:b],9,2,axis=0) if b-a>=9 else x[a:b]
    return out


def save(fig,name):
    for ext in ['png','pdf']: fig.savefig(OUT/f'{name}.{ext}',dpi=160,bbox_inches='tight')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'per_video').mkdir(exist_ok=True)
    index=json.loads((ROOT/'outputs/06_fiber_joint/index.json').read_text())
    sessions=sorted((s,v) for s,v in index.items() if v['test']=='AST')
    assert len(sessions)==len(ZONES)
    tables=[]; qc=[]; evs=[]; roi=[]
    contact,axs=plt.subplots(6,4,figsize=(16,18))
    with PdfPages(OUT/'all_AST_timecourses.pdf') as pdf:
        for j,(s,v) in enumerate(sessions):
            folder=Path(v['tdt']); video=next(folder.glob('*.avi'))
            cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS)
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.set(cv2.CAP_PROP_POS_MSEC,120000)
            ok,im=cap.read();cap.release()
            assert ok and im.shape[:2]==(480,640)
            zones=np.array(ZONES[j]).reshape(2,2)
            ax=axs.flat[j];ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            ax.scatter(zones[:,0],zones[:,1],c='#ff3155',marker='+',s=100)
            ax.set_title(s.split('_')[1][:9],fontsize=9);ax.axis('off')
            roi.append(dict(session=s,left_x=zones[0,0],left_y=zones[0,1],right_x=zones[1,0],right_y=zones[1,1],status='provisional fixed task zones; not moving pot centers'))
            raw=pd.read_hdf(v['h5']).droplevel(0,axis=1)
            cache=np.load(ROOT/'outputs/06_fiber_joint/sessions'/f'{s}.npz')
            times=cache['cam_t'];n=min(len(times),len(raw));times=times[:n];raw=raw.iloc[:n]
            assert np.all(np.diff(times)>0) and abs(frames-len(raw))<10
            ev=events(folder);ev=ev[ev.camera==v['cam']].copy()
            gates=ev[ev.event=='GateOpen']
            aligned=len(gates)>0
            t0=gates.time.min() if aligned else times[0];t=times-t0
            ev['session']=s;ev['relative_seconds']=ev.time-t0;evs.append(ev)
            def point(bp):
                xy=raw[bp][['x','y']].to_numpy();good=(raw[bp]['likelihood'].to_numpy()>=.6)&np.isfinite(xy).all(1)
                return smooth_segments(xy,good)
            ctr=point('body_center');neck=point('neck');tail=point('tail_base')
            bl=float(np.nanmedian(np.linalg.norm(neck-tail,axis=1)))
            dt=np.r_[np.nan,np.diff(times)];timely=(dt>0)&(dt<1.5/fps)
            step=np.r_[np.nan,np.linalg.norm(np.diff(ctr,axis=0),axis=1)]/bl
            step[~timely]=np.nan;speed=step/dt
            theta=np.arctan2((neck-tail)[:,1],(neck-tail)[:,0])
            turn=np.abs(np.angle(np.exp(1j*np.r_[np.nan,np.diff(theta)])))*180/np.pi/dt
            turn[~timely]=np.nan
            target=np.min(np.linalg.norm(ctr[:,None,:]-zones[None,:,:],axis=2),axis=1)/bl
            keep=t>=0
            frame=pd.DataFrame(dict(t=t[keep],speed=speed[keep],turn=turn[keep],target=target[keep],step=step[keep]))
            frame['bin']=(frame.t//5).astype(int)
            b=frame.groupby('bin')[['speed','turn','target']].mean()
            count=frame.groupby('bin')[['speed','turn','target']].count()
            b=b.where(count>=.5*5*fps)
            obs=frame.groupby('bin').step.sum(min_count=1)
            b['distance']=obs.fillna(0).cumsum().where(count.speed>=.5*5*fps)
            b['time_seconds']=b.index*5+2.5
            mouse='-'.join(s.split('_')[1].split('-')[:2])
            b['session']=s;b['mouse']=mouse;b['group']=v['group'];b['gate_aligned']=aligned;tables.append(b.reset_index())
            qc.append(dict(session=s,mouse=mouse,group=v['group'],video_frames=frames,prediction_frames=len(raw),timestamp_frames=len(times),fps=fps,body_length_px=bl,first_gate_seconds=t0 if aligned else np.nan,gate_aligned=aligned,valid_speed_fraction=np.isfinite(speed[keep]).mean()))
            fig,aa=plt.subplots(4,1,figsize=(12,9),sharex=True)
            for a,metric,label in zip(aa,METRICS,LABELS):
                a.plot(b.time_seconds/60,b[metric],color=COLORS[v['group']],lw=.7);a.set_ylabel(label,fontsize=9);a.grid(alpha=.2)
                for stage,d in gates.groupby('stage',sort=False):
                    x=(d.time.min()-t0)/60;a.axvline(x,color='gray',ls=':',lw=.7)
                    if a is aa[0]:a.text(x,.98,stage,transform=a.get_xaxis_transform(),fontsize=8,va='top')
            aa[-1].set_xlabel('Minutes from first GateOpen' if aligned else 'Minutes from first video timestamp (GateOpen unavailable)')
            fig.suptitle(f'{mouse} | {v["group"]} | {s.split("_")[0]}',fontsize=11)
            fig.text(.12,.01,'5-s bins; gaps retained. Distance sums observed movement only. Target zones are provisional.',fontsize=9)
            fig.tight_layout(rect=(0,.03,1,.96));save(fig,'per_video/'+s);pdf.savefig(fig);plt.close(fig)
            print(j+1,s,flush=True)
    contact.suptitle('Provisional fixed target zones (+); 120-s reference frames',fontsize=16)
    contact.tight_layout(rect=(0,0,1,.97));save(contact,'target_zone_QC');plt.close(contact)
    data=pd.concat(tables,ignore_index=True);data.to_csv(OUT/'session_timecourses_5s.csv',index=False)
    pd.concat(evs).to_csv(OUT/'parsed_events.csv',index=False)
    pd.DataFrame(roi).to_csv(OUT/'target_zones.csv',index=False)
    pd.DataFrame(qc).to_csv(OUT/'session_QC.csv',index=False)
    # Average sessions within animals first; resample animals, not frames.
    animals=data[data.gate_aligned].groupby(['group','mouse','bin'])[METRICS].mean().reset_index()
    rng=np.random.default_rng(20261108);summaries=[]
    fig,axes=plt.subplots(5,1,figsize=(12,11),sharex=True)
    for group,g in animals.groupby('group'):
        for i,metric in enumerate(METRICS):
            p=g.pivot(index='mouse',columns='bin',values=metric).reindex(columns=range(int(g.bin.max())+1))
            a=p.to_numpy();num=np.isfinite(a).sum(0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                mean=np.nanmean(a,axis=0)
                boots=np.array([np.nanmean(a[rng.integers(len(a),size=len(a))],axis=0) for _ in range(500)])
                lo,hi=np.nanpercentile(boots,[2.5,97.5],axis=0)
            visible=num>=3;mean[~visible]=np.nan;lo[~visible]=np.nan;hi[~visible]=np.nan
            x=(p.columns.to_numpy()*5+2.5)/60
            axes[i].plot(x,mean,color=COLORS[group],lw=.9,label=f'{group} (total n={len(a)})')
            axes[i].fill_between(x,lo,hi,color=COLORS[group],alpha=.18)
            axes[i].set_ylabel(LABELS[i],fontsize=9)
            if i==0:axes[4].plot(x,num,color=COLORS[group],label=group)
            summaries.append(pd.DataFrame(dict(group=group,metric=metric,time_minutes=x,mean=mean,ci_low=lo,ci_high=hi,n_mice=num)))
    axes[0].legend();axes[4].set_ylabel('Mice with speed data');axes[4].set_xlabel('Minutes from first GateOpen')
    for a in axes:a.grid(alpha=.2)
    fig.suptitle('AST group time courses | animal mean and pointwise 95% bootstrap CI')
    fig.text(.1,.01,'First-gate alignment mixes task stages. Curves hidden at n<3. Fixed target zones provisional; no group significance test.',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.97));save(fig,'group_mean_timecourses');plt.close(fig)
    pd.concat(summaries).to_csv(OUT/'group_timecourses.csv',index=False)


if __name__=='__main__':
    import sys
    if '--events-only' in sys.argv:
        redraw_events()
    else:
        run()
        redraw_events()
