"""Write a raw Gi BSEEG table: Gi-01..Gi-12 × Day1..Day8 × day/night."""
from pathlib import Path
import re
import pandas as pd

EXCEL_PATH = Path("Data/Gi mice OFT/Gi POD BSEEG (031126Wed).xlsx")
OUT_PATH = Path("outputs/05_stage5/gi_bseeg_day1_to_day8.csv")


def parse_sheet(xl: pd.ExcelFile, sheet: str, mice: list[str]) -> pd.DataFrame:
    raw = pd.read_excel(xl, sheet_name=sheet, header=None)
    block = raw.iloc[3:, [18, 19] + list(range(20, 20 + len(mice)))].copy()
    block.columns = ["day", "time_of_day"] + mice
    block["day"] = block["day"].ffill()
    block = block[block["time_of_day"].isin(["day", "night"])]
    block = block[block["day"].astype(str).str.match(r"^Day\d+$", na=False)]
    long = block.melt(id_vars=["day", "time_of_day"],
                      var_name="mouse", value_name="bseeg")
    long["bseeg"] = pd.to_numeric(long["bseeg"], errors="coerce")
    return long


def main() -> None:
    xl = pd.ExcelFile(EXCEL_PATH)
    long = pd.concat([
        parse_sheet(xl, "BSEEG all (Gi-A)", [f"Gi-0{i}" for i in range(1, 7)]),
        parse_sheet(xl, "BSEEG all (Gi-B)", [f"Gi-{i:02d}" for i in range(7, 13)]),
    ], ignore_index=True)

    # Keep only Day1..Day8.
    long["day_num"] = long["day"].str.extract(r"Day(\d+)").astype(int)
    long = long[long["day_num"].between(1, 8)]

    # Drop Gi-04 (died; excluded from the feature dataset).
    long = long[long["mouse"] != "Gi-04"]

    long["col"] = "bseeg_" + long["day"] + "_" + long["time_of_day"]
    wide = (long.pivot_table(index="mouse", columns="col",
                             values="bseeg", aggfunc="first")
                .reset_index()
                .rename(columns={"mouse": "mouse_id"}))
    wide.columns.name = None

    def _key(c: str) -> tuple[int, int]:
        m = re.match(r"bseeg_Day(\d+)_(day|night)", c)
        return (int(m.group(1)), 0 if m.group(2) == "day" else 1)

    ordered = ["mouse_id"] + sorted(
        [c for c in wide.columns if c != "mouse_id"], key=_key)
    wide = wide[ordered].sort_values("mouse_id").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}  shape={wide.shape}")
    print(wide.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
