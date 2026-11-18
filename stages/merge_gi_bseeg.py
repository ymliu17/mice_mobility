"""Combine BSEEG composite scores with GI cohort feature data.

Reads day/night BSEEG composite scores from the Gi POD BSEEG workbook
(two sheets: Gi-A covers Gi-01..Gi-06, Gi-B covers Gi-07..Gi-12) and
merges them with GI-cohort rows from the zscored feature master.

Output: outputs/05_stage5/gi_bseeg_feature_master.csv
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

EXCEL_PATH = Path("Data/Gi mice OFT/Gi POD BSEEG (031126Wed).xlsx")
CSV_PATH = Path("outputs/05_stage5/study_group_feature_master_presence_gt10_full10meta_zscored.csv")
OUT_PATH = Path("outputs/05_stage5/gi_bseeg_feature_master.csv")


def parse_bseeg_sheet(xl: pd.ExcelFile, sheet: str, mice: list[str]) -> pd.DataFrame:
    """Extract day/night BSEEG composite scores for the given mice.

    The composite block is columns 18 (Day label), 19 (time: day/night),
    20..25 (six mouse columns). Rows begin at index 3.
    """
    raw = pd.read_excel(xl, sheet_name=sheet, header=None)
    block = raw.iloc[3:, [18, 19] + list(range(20, 20 + len(mice)))].copy()
    block.columns = ["day", "time_of_day"] + mice
    block["day"] = block["day"].ffill()
    block = block[block["time_of_day"].isin(["day", "night"])]
    block = block[block["day"].astype(str).str.match(r"^Day\d+$", na=False)]
    long = block.melt(
        id_vars=["day", "time_of_day"],
        var_name="mouse_key",
        value_name="bseeg",
    )
    long["bseeg"] = pd.to_numeric(long["bseeg"], errors="coerce")
    return long


def build_bseeg_long() -> pd.DataFrame:
    """Long table: mouse_key × day × time_of_day → bseeg."""
    xl = pd.ExcelFile(EXCEL_PATH)
    parts = [
        parse_bseeg_sheet(xl, "BSEEG all (Gi-A)", [f"Gi-0{i}" for i in range(1, 7)]),
        parse_bseeg_sheet(xl, "BSEEG all (Gi-B)",
                          [f"Gi-{i:02d}" for i in range(7, 13)]),
    ]
    return pd.concat(parts, ignore_index=True)


# time_points in the feature CSV → matching Day in the BSEEG workbook.
# "pre" is baseline (pre-surgery), so no EEG is available.
TIMEPOINT_TO_DAY = {"pre": None, "post1": "Day1", "post3": "Day3"}


def main() -> None:
    long = build_bseeg_long()
    feats = pd.read_csv(CSV_PATH)
    gi = feats[feats["study_cohort"] == "gi"].copy()

    # OFT_GI and OFT_single carry identical features per (mouse, timepoint);
    # keep only OFT_GI so each (mouse, timepoint) appears once.
    gi = gi[gi["study_context"] == "OFT_GI"].copy()
    # After this filter every mouse_id is already in "Gi-XX" form.

    # Wide BSEEG: one column per Day × time_of_day, named bseeg_DayN_{day,night}.
    long["col"] = "bseeg_" + long["day"] + "_" + long["time_of_day"]
    wide = (long.pivot_table(index="mouse_key", columns="col",
                             values="bseeg", aggfunc="first")
                 .reset_index()
                 .rename(columns={"mouse_key": "mouse_id"}))
    wide.columns.name = None

    # Keep only Day1/Day3 columns — those are the timepoints matched in this CSV.
    bseeg_keep = ["bseeg_Day1_night", "bseeg_Day3_day", "bseeg_Day3_night"]
    bseeg_keep = [c for c in bseeg_keep if c in wide.columns]
    wide = wide[["mouse_id"] + bseeg_keep]

    merged = gi.merge(wide, on="mouse_id", how="left")

    # Zero out BSEEG cells that don't correspond to the row's time_points
    # (e.g. pre rows get nothing; post1 rows keep only Day1; post3 keeps Day3).
    tp_day = merged["time_points"].map(TIMEPOINT_TO_DAY)
    for col in bseeg_keep:
        col_day = re.match(r"bseeg_(Day\d+)_", col).group(1)
        merged.loc[tp_day != col_day, col] = np.nan

    # Row-wise mean across whatever BSEEG values remain for that row.
    merged["bseeg_mean"] = merged[bseeg_keep].mean(axis=1, skipna=True)

    # Put BSEEG columns right after the feature-master metadata
    meta_cols = ["sample_id", "session_usage", "n_sessions", "mouse_id",
                 "study_cohort", "study_context", "abnormal_condition", "time_points",
                 "env_context", "repr_version", "cv_type", "label_binary"]
    bseeg_cols = bseeg_keep + ["bseeg_mean"]
    other_cols = [c for c in merged.columns if c not in meta_cols + bseeg_cols]
    merged = merged[meta_cols + bseeg_cols + other_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f"wrote {OUT_PATH}  rows={len(merged)}  cols={len(merged.columns)}")
    print("coverage by time_points (non-null bseeg_mean):")
    print(merged.groupby("time_points")["bseeg_mean"].apply(
        lambda s: f"{s.notna().sum()}/{len(s)}"))


if __name__ == "__main__":
    main()
