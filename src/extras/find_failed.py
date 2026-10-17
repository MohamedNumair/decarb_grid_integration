import csv
from pathlib import Path

STUDY_DECARB = Path(r"D:\shared\building_grid_integration\out\studies\study_1\decarb")

for case_dir in sorted(STUDY_DECARB.iterdir()):
    if not case_dir.is_dir():
        continue

    bldgs = sorted(
        d for d in case_dir.iterdir()
        if d.is_dir() and (d / "in").is_dir()
    )

    completed = [d for d in bldgs if (d / "out" / "ts.csv").exists()]
    attempted = [d for d in bldgs if (d / "out").is_dir()]
    failed = [d for d in attempted if not (d / "out" / "ts.csv").exists()]

    print(f"{case_dir.name}: {len(completed)} ok, {len(failed)} failed, "
          f"{len(bldgs) - len(attempted)} not attempted")

    if failed:
        csv_path = case_dir / "failed_buildings.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bldg_id", "status", "error"])
            for d in failed:
                w.writerow([d.name, "error", "no ts.csv"])
        print(f"  Wrote {csv_path}")
    elif len(attempted) > 0 and len(failed) == 0:
        print(f"  All attempted buildings succeeded")
