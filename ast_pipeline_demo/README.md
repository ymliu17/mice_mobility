# AST Pipeline Explorer (offline demo)

An interactive, researcher-oriented walkthrough of the mice mobility pipeline, using one
AST recording (mouse 1864968-1, Social isolation, 2025-06-21, Cam1):

| Page | Stage | What you can explore |
|---|---|---|
| Overview | – | Pipeline map (behavior lane S1→S2→S3, photometry lane S1b→S6), key numbers |
| S1 | Pose extraction | Real video clip with SuperAnimal (27 kp) vs custom AST (7 kp) overlays, per-keypoint yield, confidence over the session, error vs manual labels |
| S2 | Preprocessing | Step through mask → interpolate → smooth → normalize on the video and on keypoint traces; body length; arena occupancy |
| S3 | keypoint-MoSeq | Syllables on the video, full-session ethogram, usage, PCA, transitions, bout durations, AST cohort usage, key-syllable GIFs with F / S / X features |
| S6 | Photometry × kinematics | Bleaching-robust dF/F steps from the raw TDT block, linked session timeline (dF/F, speed, syllables, events), peri-event dopamine, trial table, cohort context, per-video time-course figure |
| Provenance | – | Source file for every panel, build checks, rebuild command |

## Run it

Open `index.html` in any modern browser (Chrome, Edge, Firefox, Safari).
No internet, server, or install needed; everything is in this folder (~11 MB).

Controls: drag across any time chart to zoom, double-click to reset, hover for values.
In the video player: <kbd>Space</kbd> play/pause, <kbd>←</kbd>/<kbd>→</kbd> step one frame.
The moon button toggles light/dark.

## Folder

```
index.html              entry point
assets/                 style.css, charts.js, player.js, app.js, GIFs, figure
data/demo_data.js       all numbers/arrays (generated)
frames/                 380 video frames (10 fps, 38 s clip) + arena.jpg (generated)
build/build_demo_data.py  regenerates data/ and frames/ from project outputs
```

## Rebuild (only inside the project)

```bash
cd /home/yangliu/mice_mobility
/home/yangliu/miniconda3/envs/mice/bin/python ast_pipeline_demo/build/build_demo_data.py
```

Edit `BLOCK` / `CLIP_T` at the top of the build script to feature another session or trial.
The build only reads project outputs; nothing in `outputs/` is modified.
