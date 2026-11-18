"""Evaluate the 50 annotated AST frames without changing behavioral outputs."""
from pathlib import Path
import json
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/methodlab/ast_tracking_qc_20250621'


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = next((ROOT / 'Data/Fiber photometry/labeled-data').glob('*250621*/CollectedData*.csv'))
    video = next((ROOT / 'Data/Fiber photometry/AST').rglob('*1864968-1-250621-152216_Cam1.avi'))
    new_path = next(video.parent.glob('*snapshot_best-190.csv'))
    old_path = next((ROOT / 'outputs/01_dlc_raw').rglob('*1864968-1-250621-152216*.h5'))
    gt = pd.read_csv(labels, header=[0, 1, 2], index_col=[0, 1, 2]).droplevel(0, axis=1)
    new = pd.read_csv(new_path, header=[0, 1, 2], index_col=0).droplevel(0, axis=1)
    old = pd.read_hdf(old_path).droplevel(0, axis=1).xs('animal0', level=0, axis=1)
    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(new)
    assert len(old) == (len(new) + 1) // 2
    points = gt.columns.get_level_values(0).unique().tolist()
    mapping = {'body_center': 'mouse_center', 'tail_tip': 'tail_end'}
    rows, alignment = [], []
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for j, (idx, truth) in enumerate(gt.iterrows()):
        frame = int(re.search(r'img(\d+)', idx[-1]).group(1))
        im = cv2.imread(str(labels.parent / idx[-1]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, native = cap.read()
        assert ok and native.shape == im.shape
        mae = float(np.abs(im.astype(float) - native.astype(float)).mean())
        alignment.append({'frame': frame, 'image_mae_0_255': mae})
        for bp in points:
            xy = truth[bp][['x', 'y']].to_numpy(float)
            if not np.isfinite(xy).all():
                continue
            # The old run was resampled with ffmpeg fps=10. Keep the timing
            # uncertainty explicit rather than interpolating pose coordinates.
            for model, data, f, part in [('new', new, frame, bp), ('old', old, frame // 2, mapping.get(bp, bp))]:
                pred = data.iloc[f][part]
                px = pred[['x', 'y']].to_numpy(float)
                likelihood = float(pred['likelihood'])
                valid = np.isfinite(px).all() and np.isfinite(likelihood) and likelihood >= 0
                rows.append(dict(frame=frame, keypoint=bp, model=model, prediction_row=f,
                                 definition='proxy' if bp in mapping else 'same_name',
                                 gt_x=xy[0], gt_y=xy[1], pred_x=px[0], pred_y=px[1],
                                 likelihood=likelihood, valid_prediction=valid,
                                 error_px=float(np.linalg.norm(px-xy)) if valid else np.nan))
        if j < 6:
            ax = axes.flat[j]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            for model, color, marker in [('gt', '#28a745', 'o'), ('new', '#e87924', 'x'), ('old', '#287ac0', '+')]:
                xy = np.array([truth[b][['x','y']].to_numpy(float) if model == 'gt' else
                               (new.iloc[frame][b] if model == 'new' else old.iloc[frame//2][mapping.get(b,b)])[['x','y']].to_numpy(float) for b in points])
                ax.scatter(xy[:,0], xy[:,1], s=20, c=color, marker=marker, label=model)
            ax.set(xlim=(0,640), ylim=(480,0), title=f'Frame {frame}')
            ax.axis('off')
    cap.release()
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'overlay_examples.png', dpi=180); plt.close(fig)
    errors = pd.DataFrame(rows)
    errors.to_csv(OUT/'per_frame_keypoint_errors.csv', index=False)
    pd.DataFrame(alignment).to_csv(OUT/'frame_alignment.csv', index=False)
    summary = []
    for (model,bp), d in errors.groupby(['model','keypoint'], sort=False):
        e = d.error_px.dropna()
        summary.append(dict(model=model,keypoint=bp,definition=d.definition.iloc[0],n_labeled=len(d),
                            n_valid=len(e),mean_error_px=e.mean(),median_error_px=e.median(),
                            rmse_px=np.sqrt((e**2).mean()),p90_error_px=e.quantile(.9),
                            coverage_likelihood_ge06=(d.valid_prediction & (d.likelihood>=.6)).mean(),
                            mean_error_confident_px=d.loc[d.likelihood>=.6,'error_px'].mean()))
    s=pd.DataFrame(summary);s.to_csv(OUT/'keypoint_error_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4))
    for k,model in enumerate(['old','new']):
        vals=s[s.model==model].set_index('keypoint').reindex(points)
        ax.bar(np.arange(len(points))+(k-.5)*.36,vals.mean_error_px,.36,label=model)
    ax.set_xticks(np.arange(len(points)),[p+('*' if p in mapping else '') for p in points],rotation=20)
    ax.set_ylabel('Mean Euclidean error (pixels)');ax.legend()
    fig.text(.12,.01,'* Different landmark definitions: exploratory comparison. Missing predictions excluded; see coverage table.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ['png','pdf']:fig.savefig(OUT/f'keypoint_error_comparison.{ext}',dpi=180)
    plt.close(fig)
    metadata=dict(labels=str(labels),new_predictions=str(new_path),old_predictions=str(old_path),
                  n_annotated_frames=len(gt),image_size=[640,480],old_individual='animal0',
                  old_frame_mapping='floor(native_frame/2); ffmpeg sampling phase unverified, allow ~50 ms',
                  training_membership='unknown; these are annotation QC errors, not held-out generalization estimates',
                  landmark_proxies=mapping,max_image_mae=max(x['image_mae_0_255'] for x in alignment))
    (OUT/'evaluation_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(s.to_string(index=False));print('Alignment:',metadata['max_image_mae'])


if __name__ == '__main__':
    run()
