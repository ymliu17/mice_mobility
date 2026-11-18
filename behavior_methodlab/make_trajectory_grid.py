#!/usr/bin/env python3
"""
Composite the 12 freq50 syllable trajectory PNGs into a 2×6 grid.
Reads pre-existing individual PNGs — no recomputation needed.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TRAJ_DIR = Path("outputs/05_stage5/syllable_atlas_presence_gt10/kpms_style_trajectories")
FREQ50   = [0, 1, 4, 7, 11, 24, 26, 32, 37, 43, 45, 46]
NCOLS, NROWS = 6, 2

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(3.5 * NCOLS, 4.2 * NROWS),
                         gridspec_kw={"hspace": 0.05, "wspace": 0.02})

for ax, s in zip(axes.ravel(), FREQ50):
    png = TRAJ_DIR / f"syllable_{s:02d}_trajectory.png"
    img = mpimg.imread(png)
    ax.imshow(img)
    ax.axis("off")

fig.suptitle("freq50 syllable trajectories", fontsize=14, y=1.01)
plt.tight_layout()
out = TRAJ_DIR / "all12_trajectory_grid.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
