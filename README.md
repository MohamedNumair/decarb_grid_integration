# Building-Grid Integration Pipeline

Models residential electrification and rooftop solar on a distribution network. ResStock buildings are fitted with a thermal model, matched to the residential loads of a SMART-DS feeder zone, optimized with DECARB, and fed back into OpenDSS as hourly or 15-minute load profiles.

```
preprocess  ->  decarb-inputs  ->  DECARB (Julia)  ->  network-inputs  ->  opendss  ->  results
```

The zone workflow (`scope: "zone"`, OpenDSS) is the maintained path. The statewide GenX path (`scope: "statewide"`) is not up to date.

---

## 1. Setup

**Python 3.10+**

```bash
pip install -r requirements.txt
```

**Julia + DECARB** (only for the DECARB optimizer step): DECARB itself, Gurobi, and the Julia packages `CSV`, `DataFrames`, `Dates`, `Distributed`.

## 2. Input data

Downloaded automatically by the data acquisition stage. Everything is public, and no account or API key is needed:

| Data | Source | Saved to |
|---|---|---|
| ResStock metadata (`{STATE}_upgrade0.parquet`, plus a `.csv.gz` copy for manual inspection) | OEDI S3 `oedi-data-lake`, `resstock` settings in `config.yaml` | `in/` |
| ResStock 15-min building profiles | same | `in/residential_profiles/{STATE}/` |
| SMART-DS feeder network (`.dss` files for the zone) | OEDI S3, `smartds` settings in `config.yaml` | `paths.network_dir` |

Provided by you:

| File | What it is | Where it goes |
|---|---|---|
| NEEP heat pump database | NEEP cold-climate ASHP product list export (CSV or Excel). Must include the `Brand Name`, `Status`, `HSPF (Region IV)` and the rated/min/max capacity and COP columns at 5/17/47/82/95 °F | `in/neep_database.csv`, or set `heatpump.neep_database` |
| County shapefile | County polygons for the network's state, with fields `name`, `statefp`, `countyfp` (e.g. the OpenDataSoft "US County Boundaries" export) | `paths.county_shapefile` |
| DECARB input templates | `bdg_i.csv`, `in.csv`, `abs.csv`, `pv.csv`, `bess.csv`, `wind.csv`, `ev.csv`, `chp.csv`, `topo.csv`, `hvac.csv`, `wh.csv`, `sp.csv` | `template_dir` in `decarb.yaml` (default `in/decarb_input_templates/`) |

The building state and the network do not have to match. For example, NY or MA buildings on Texas feeders is a normal setup.

## 3. Configuration

| File | Controls |
|---|---|
| `config/config.yaml` | state/year/time zone, zone, paths, test mode, which preprocessing stages run, matching, heat pump sizing, date range |
| `config/decarb.yaml` | DECARB timestep, tariffs, electrification levels, rooftop solar, which DECARB input files to write |
| `config/network.yaml` | GenX settings (statewide only), interpolation, workers |

Settings a new user usually changes:

```yaml
state: "NY"                   # ResStock state of the buildings
timezone: "America/New_York"  # time zone of the buildings (PV orientation)
zone: {name: "P101U"}         # SMART-DS zone (feeder network) to use
paths:
  network_dir: "D:/smartds"   # keep this path short on Windows (260-character path limit)
test_mode: true               # 600 buildings and one zone, for a quick end-to-end check
stages: {...: true}           # turn on the preprocessing stages you need
```

Settings that change results (heat pump sizing, matching limits, time zone, timestep, data locations) have no built-in defaults. If one is missing or invalid, the run stops with a message naming the key.

## 4. Running the pipeline

All commands are run from anywhere with the project's `src/run_pipeline.py`.

### 4.1 Preprocess

```bash
python src/run_pipeline.py preprocess
```

Runs the stages switched on under `stages:`, in this order:

| Stage | Reads | Writes |
|---|---|---|
| `data_acquisition` | S3 | metadata, building ID list (`in/resstock_bldg_ids_{state}_{year}.csv`, always the full list), building profiles, SMART-DS network (`data_acquisition.download_network`) |
| `thermal_model` | profiles, metadata | `out/thermal_model_outputs/{STATE}/baseline_internal_gains/regression_coeff_*.csv`, `skipped_buildings.csv` |
| `feeder_processing` | `network_dir/{zone}` `Loads.dss`/`Buscoords.dss`, county shapefile | `in/feeder_summaries.parquet` (residential loads with county and climate zone) |
| `building_matching` | feeder summaries, thermal results, metadata | `in/feeder_summaries_matched.parquet` (ResStock building for every load) |
| `schedules` | profiles, metadata | `in/schedules/{STATE}/`, `in/setpoint_timeseries/{STATE}/` |
| `heatpump_selection` | profiles, metadata, NEEP database | `out/heatpump_files/{STATE}/building_hp_summary.csv` |

Every stage stops with a clear error if its inputs are missing, instead of producing partial results. With `test_mode: true`, every stage uses the same `test_limit` lowest building IDs and only `zone.name`, and logs `TEST MODE: ...`.

`in/feeder_summaries*.parquet` hold one zone and state at a time. Switching `zone.name` or `state` means re-running feeder processing and matching. `decarb-inputs` and `network-inputs` stop if the matched file is for a different zone.

### 4.2 DECARB inputs

```bash
python src/run_pipeline.py decarb-inputs
```

- Checks every required file first, and lists any building with missing inputs in `out/decarb_inputs_missing_files.csv`. Then it asks for a short study description and creates `out/studies/study_N/`.
- Writes `decarb/{tariff}_heat_{0|100}_solar_{0|100}/{bldg_id}/update_0/in/` for every matched building, plus copies of the three config files and the matching file.
- The thermal coefficients in `bdg_ii.csv` are converted from the 15-min fit to the DECARB `timestep`.
- `date_range` in `config.yaml` limits `tm.csv` to that period. Preprocessing always uses the full year.

### 4.3 Run DECARB

Outside this Python pipeline: edit the settings block at the top of `src/run_decarb_study.jl` (`STUDY_DIR`, `DECARB_SRC`, `N_WORKERS`) and run it with Julia. DECARB writes `update_0/out/ts.csv` for each building.

### 4.4 Network inputs

```bash
python src/run_pipeline.py network-inputs --study N
```

Builds `out/studies/study_N/opendss/{case}/`. It copies the zone's feeders from `network_dir`, and replaces each matched residential load with that building's DECARB profile (kW and reconstructed kvar). The time axis follows the study's own `decarb.yaml` timestep and `date_range`. With `interpolation.enabled: true` in `network.yaml`, it also writes intermediate electrification/solar penetration levels (see below).

### 4.5 OpenDSS and results

```bash
python src/run_pipeline.py opendss --study N
python src/run_pipeline.py results --study N
```

The OpenDSS runner reads each case's step size and length from its `Master.dss`, so hourly and 15-min cases run the same way. Results go to `out/studies/study_N/opendss_results/{case}/zone_yearly.csv`.

## 5. Time resolution

| Part | Resolution |
|---|---|
| ResStock data, thermal fit, schedules, heat pump sizing | always 15-min |
| DECARB inputs, OpenDSS cases, interpolation | `decarb.yaml` → `timestep` (`hourly` or `15min`) |
| GenX demand | hourly only |

## 6. Interpolation

DECARB runs only the 0% and 100% bookends of each dimension (electrification, solar). For an intermediate level, a seeded random set of buildings, weighted by how many loads each represents, takes its 100% profile and the rest keep their 0% profile. The sum is the zone demand at that level. Output: `out/studies/study_N/interpolation_summary.csv`.

## 7. Folder layout

```
config/                     config.yaml, decarb.yaml, network.yaml
src/                        pipeline stages; src/extras/ holds one-off analysis scripts
in/                         downloaded and preprocessed inputs (not in git)
out/
  thermal_model_outputs/    thermal fits per state
  heatpump_files/           heat pump selection per state
  studies/
    study_log.csv           description of every study
    study_N/                one DECARB study: config snapshots, decarb/, opendss/, opendss_results/
```
