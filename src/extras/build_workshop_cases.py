"""
Build single-building DECARB example cases (workshop set) from an existing study.

From one building's inputs in a study made by the pipeline, writes one folder per
case, each with in/ and a source.txt describing where it came from and what was
changed, plus a bdg_path.txt listing every case for DECARB's run_decarb.jl:

  base_‹tariff›                       ‹tariff›_heat_0_solar_0, unchanged (existing equipment)
  heat_pump_‹tariff›                  ‹tariff›_heat_100_solar_0, unchanged (Mitsubishi ASHP)
  hp_backup_‹tariff›                  ‹tariff›_heat_100_solar_0 + the building's own furnace kept
                                      as backup at BACKUP_FRACTION of the ASHP heating capacity,
                                      fuel prices (pGcost, pLcost) = FUEL_PRICE
  hp_solar_bess_‹tariff›_hourly       ‹tariff›_heat_100_solar_100 (pipeline PV sizing) + one
                                      TeslaPowerwall_Cash installed (pBbess = 1)
  hp_solar_bess_‹tariff›_15min_summer same, with tm.csv / cfg.csv regenerated at 15-min for
                                      SUMMER (using the study's own decarb.yaml)

Before writing anything it checks that regenerating the hourly full-year tm.csv
with the current code reproduces the study's file, so the 15-min cases match.

Usage:
    python src/extras/build_workshop_cases.py --study-dir Z:/building_grid_integration_v2/out/studies/study_2 \
        --input-dir Z:/building_grid_integration_v2/in --out C:/.../decarb/examples/single_bldg_updates
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
import create_decarb_inputs as c  # noqa: E402
from config.loader import load_config, load_decarb_config  # noqa: E402

TARIFFS = ["flat", "flat_cap", "tou", "tou_cap"]
BACKUP_FRACTION = 0.2
FUEL_PRICE = 0.23                         # $/kWh, backup cases
BATTERY = "TeslaPowerwall_Cash"
SUMMER = {"start": "Jun 1", "end": "Aug 31"}
TOU_SELL_PRICE = 0.15                     # $/kWh, below the TOU off-peak price (0.16)


def read_str(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_kv(path):
    return pd.read_csv(path, header=None, index_col=0, dtype=str, keep_default_na=False)


def set_sell_price(tm_path):
    """TOU cases: sell price below the off-peak buy price (study_2 used 0.18)."""
    tm = pd.read_csv(tm_path)
    tm["pQcostSell"] = TOU_SELL_PRICE
    tm.to_csv(tm_path, index=False)


class Builder:
    def __init__(self, study_dir, input_dir, out_dir, bldg_id):
        self.study_dir, self.out_dir, self.bldg = study_dir, out_dir, bldg_id
        cfg = load_config(study_dir / "config.yaml")
        self.year, self.state, self.tz = cfg["year"], cfg["state"], cfg["timezone"]
        self.dcfg = load_decarb_config(study_dir / "decarb.yaml")
        self.tariffs = {t["name"]: t for t in self.dcfg["tariffs"]}
        meta = pd.read_parquet(input_dir / f"{self.state}_upgrade0.parquet")
        meta = meta.set_index("bldg_id" if "bldg_id" in meta.columns else "building_id")
        self.row = c.get_building_row(bldg_id, meta)
        self.cons = c.load_consumption(bldg_id, input_dir, self.year, self.state)
        self.setpoints = c.load_setpoints(bldg_id, input_dir, self.year, self.state)
        self.schedule = c.load_schedule(bldg_id, input_dir, self.state)
        self.cases = []

    def src(self, case):
        return self.study_dir / "decarb" / case / str(self.bldg) / "in"

    def copy(self, case, name):
        d = self.out_dir / name / "in"
        if d.parent.exists():
            shutil.rmtree(d.parent)
        shutil.copytree(self.src(case), d)
        self.cases.append(name)
        if case.startswith("tou"):
            set_sell_price(d / "tm.csv")
        return d

    def note(self, d, text):
        (d.parent / "source.txt").write_text(text.strip() + "\n")

    def make_tm(self, tariff, timestep, date_range):
        c._worker_state.update(year=self.year, cfg={"year": self.year, "date_range": date_range,
                                                    "timezone": self.tz})
        return c.generate_tm_data(self.cons, self.setpoints, self.row, self.schedule,
                                  self.tariffs[tariff], dict(self.dcfg, timestep=timestep))

    def check_reproduces_study(self):
        tmp = Path(tempfile.mkdtemp())
        c.create_tm_file(self.make_tm("flat", "hourly", None), tmp)
        a = pd.read_csv(tmp / "tm.csv")
        b = pd.read_csv(self.src("flat_heat_100_solar_0") / "tm.csv")
        shutil.rmtree(tmp)
        ok = (a.columns.equals(b.columns) and len(a) == len(b) and (a.pP == b.pP).all()
              and np.allclose(a.drop(columns="pP"), b.drop(columns="pP")))
        if not ok:
            raise SystemExit("Regenerated tm.csv differs from the study's: the study was made with "
                             "other code or settings; 15-min cases would not match the hourly ones.")

    def add_backup(self, d):
        hvac = read_str(d / "hvac.csv")
        ashp = hvac[hvac.ty.str.startswith("Mitsubishi ASHP")]
        if len(ashp) != 1:
            raise ValueError(f"{d}: expected one Mitsubishi ASHP row, found {len(ashp)}")
        backup_kw = BACKUP_FRACTION * float(ashp.pHVmx.iloc[0])
        base_sp = read_str(self.src("flat_heat_0_solar_0") / "sp.csv")
        heater = base_sp.loc[base_sp.pCHPz0 == "1", "pCHP0"]
        if len(heater) != 1:
            raise ValueError(f"base case has {len(heater)} installed fuel heaters; expected one")
        heater = heater.iloc[0]
        sp = read_str(d / "sp.csv")
        sp.loc[sp.pCHP0 == heater, ["pCHPz0", "pCHPyn"]] = ["1", "NO"]
        sp.to_csv(d / "sp.csv", index=False)
        chp = read_str(d / "chp.csv")
        chp.loc[chp.ty == heater, "mx"] = repr(backup_kw)
        chp.to_csv(d / "chp.csv", index=False)
        if not (read_str(d / "topo.csv").up == heater).any():
            raise ValueError(f"{d}: {heater} is not linked to the building in topo.csv")
        tm = pd.read_csv(d / "tm.csv")
        tm["pGcost"] = FUEL_PRICE
        tm["pLcost"] = FUEL_PRICE
        tm.to_csv(d / "tm.csv", index=False)
        return ashp.ty.iloc[0], heater, backup_kw

    def add_battery(self, d):
        sp = read_str(d / "sp.csv")
        if (sp.pBESS0 == BATTERY).sum() != 1:
            raise ValueError(f"{d}: {BATTERY} not in sp.csv")
        sp.loc[sp.pBESS0 == BATTERY, ["pBESSz0", "pBESSyn"]] = ["1", "NO"]
        sp.to_csv(d / "sp.csv", index=False)
        bdg = read_kv(d / "bdg_i.csv")
        bdg.loc["pBbess", 1] = "1"
        bdg.to_csv(d / "bdg_i.csv", header=False)
        return sp[sp.pPVz0 != "0"][["pPV0", "pPVz0"]].iloc[0]

    def build(self):
        b = self.bldg
        self.check_reproduces_study()

        for t in TARIFFS:
            d = self.copy(f"{t}_heat_0_solar_0", f"base_{t}")
            self.note(d, f"Building {b}: existing equipment. Tariff: {t}. Hourly, full year {self.year}.\n"
                         f"From {self.study_dir.name}/decarb/{t}_heat_0_solar_0/{b}/in, unchanged.")

        for t in TARIFFS:
            d = self.copy(f"{t}_heat_100_solar_0", f"heat_pump_{t}")
            self.note(d, f"Building {b}: Mitsubishi ASHP replaces the existing heating and cooling. "
                         f"Tariff: {t}. Hourly, full year {self.year}.\n"
                         f"From {self.study_dir.name}/decarb/{t}_heat_100_solar_0/{b}/in, unchanged.")

        for t in TARIFFS:
            d = self.copy(f"{t}_heat_100_solar_0", f"hp_backup_{t}")
            ashp, heater, kw = self.add_backup(d)
            self.note(d, f"Building {b}: {ashp} plus the existing {heater} kept as backup at "
                         f"{BACKUP_FRACTION:.0%} of the heat pump heating capacity ({kw:.2f} kW).\n"
                         f"Fuel prices (pGcost, pLcost) set to {FUEL_PRICE} $/kWh. Tariff: {t}. "
                         f"Hourly, full year {self.year}.\n"
                         f"From {self.study_dir.name}/decarb/{t}_heat_100_solar_0/{b}/in; changed "
                         f"sp.csv (heater installed), chp.csv (heater mx), tm.csv (fuel prices).")

        for t in TARIFFS:
            src = f"{t}_heat_100_solar_100"
            d = self.copy(src, f"hp_solar_bess_{t}_hourly")
            pv = self.add_battery(d)
            desc = (f"Building {b}: Mitsubishi ASHP, rooftop PV ({pv.pPVz0} x {pv.pPV0} panels, "
                    f"pipeline sizing), one {BATTERY} installed. Tariff: {t}.")
            changed = (f"From {self.study_dir.name}/decarb/{src}/{b}/in; changed sp.csv "
                       f"(battery installed), bdg_i.csv (pBbess 1)")
            self.note(d, f"{desc} Hourly, full year {self.year}.\n{changed}.")

            d = self.copy(src, f"hp_solar_bess_{t}_15min_summer")
            self.add_battery(d)
            tm = self.make_tm(t, "15min", SUMMER)
            c.create_tm_file(tm, d)
            if t.startswith("tou"):
                set_sell_price(d / "tm.csv")
            c.create_cfg_file(d, pd.read_csv(d / "cfg.csv", header=None, index_col=0),
                              self.cons, tm, "15min")
            self.note(d, f"{desc} 15-min, {SUMMER['start']} - {SUMMER['end']} {self.year} "
                         f"({len(tm)} periods).\n{changed}; tm.csv and cfg.csv regenerated with "
                         f"the pipeline (timestep 15min, date_range {SUMMER['start']} - {SUMMER['end']}).")

        rel = Path("examples") / self.out_dir.name
        order = sorted(self.cases, key=lambda n: n.endswith("_15min_summer"))   # 15-min runs last
        (self.out_dir / "bdg_path.txt").write_text(
            "".join(f"{(rel / n).as_posix()}\n" for n in order))
        return self.cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-dir", type=Path, required=True,
                    help="study folder made by decarb-inputs (with config.yaml, decarb.yaml, decarb/)")
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="pipeline in/ folder with the ResStock metadata, profiles, setpoints, schedules")
    ap.add_argument("--out", type=Path, required=True,
                    help="output folder, e.g. <DECARB repo>/examples/single_bldg_updates")
    ap.add_argument("--bldg-id", type=int, default=4836)
    args = ap.parse_args()
    cases = Builder(args.study_dir, args.input_dir, args.out, args.bldg_id).build()
    print(f"{len(cases)} cases written to {args.out}:")
    for n in cases:
        print(f"  {n}")


if __name__ == "__main__":
    main()
