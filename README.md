# Building-Grid Integration Pipeline

Modular pipeline for modeling residential electrification and rooftop solar adoption impacts on the ERCOT grid. Takes ResStock building data through DECARB optimization and produces GenX capacity expansion or OpenDSS distribution analysis inputs.

## Setup

Requires Python 3.9+ with numpy, pandas, pyyaml, requests, and fastparquet (or pyarrow).

```bash
pip install numpy pandas pyyaml requests fastparquet
```

## Configuration

All config files live in `config/`:

| File | Purpose |
|------|---------|
| `config.yaml` | Main pipeline settings: scope (statewide/zone), paths, stage toggles, test mode |
| `decarb.yaml` | DECARB scenario parameters: tariffs, fuel costs, solar toggle, output file selection |
| `network.yaml` | Network analysis settings: GenX base case path, demand aggregation mode, OpenDSS resolution |

Set `scope: "statewide"` for GenX capacity expansion (all of Texas) or `scope: "zone"` for OpenDSS distribution analysis (single substation/feeder).

## Running the Pipeline

### Step 1: Preprocessing

Run once per dataset to prepare all building-level inputs.

```bash
python src/run_pipeline.py preprocess
```

This runs (in order):
1. Data acquisition (ResStock downloads from OEDI)
2. Thermal model fitting
3. Feeder/network processing
4. PV detection
5. Building-to-load matching
6. Schedule/setpoint generation
7. Heat pump selection (NEEP database matching)

Toggle individual stages on/off in `config.yaml` under `stages:`. Use `test_mode: true` to limit to a small subset of buildings during development.

### Step 2: DECARB Input Generation

Generate optimizer input files for the bookend cases (0% and 100% electrification, with and without solar).

```bash
python src/run_pipeline.py decarb-inputs
```

Creates a new study directory (`out/studies/study_N/`) with DECARB input files for every building at each bookend combination. Every building gets solar parameters calculated; the `mx` (max capacity) is only nonzero in the solar cases.

### Step 3: Run DECARB Optimizer

Run the DECARB optimizer externally on the generated input files. This step is outside the pipeline -- DECARB reads from each building's `update_0/in/` directory and writes results to `update_0/out/ts.csv`.

### Step 4: Network Input Generation

After DECARB has run, aggregate the per-building results into GenX demand profiles or OpenDSS loadshapes.

```bash
python src/run_pipeline.py network-inputs --study 3 --electrification 0,50,100 --solar 0,30
```

Options:
- `--study N` — which study to use (default: latest)
- `--electrification 0,25,50,75,100` — electrification penetration levels
- `--solar 0,30,60,100` — solar penetration levels

For bookend levels (0% and 100%), profiles are aggregated directly from DECARB outputs. For intermediate levels, buildings are randomly assigned to either their 0% or 100% profile and then aggregated.

### Step 5: Network Analysis

Run GenX or OpenDSS on the generated network input files. This step is outside the pipeline.

### GenX Base Case

The GenX supply side (generators, storage, fuels, variability, network topology) is built once and reused across studies. Point to it in `network.yaml`:

```yaml
genx:
  base_case: "../in/genx_base"
```

Per study, the base case directory is copied and only `Demand_data.csv` is overwritten with the scenario-specific demand profile.

## Output Structure

```
out/studies/
  study_0/
    config.yaml          # snapshot of config at generation time
    decarb.yaml          # snapshot of decarb config
    network.yaml         # snapshot of network config
    decarb/
      flat_heat_0/             # baseline (no electrification, no solar)
        {bldg_id}/update_0/
          in/                  # DECARB input files
          out/                 # DECARB output files (after optimizer runs)
      flat_heat_100/           # full electrification
      flat_heat_0_solar_100/   # baseline + all solar
      flat_heat_100_solar_100/ # full electrification + all solar
    genx/
      flat_heat_0/             # bookend GenX case
        resources/             # Thermal.csv, Vre.csv, Storage.csv (from base case)
        system/                # Demand_data.csv, Fuels_data.csv, etc.
      flat_heat_50/            # interpolated at 50% electrification
      flat_heat_50_solar_30/   # interpolated at 50% elec + 30% solar
      ...
```

## How Interpolation Works

Instead of running DECARB at every penetration level:

1. DECARB runs at 0% and 100% for each dimension (electrification, solar)
2. For an intermediate level like 50% electrification, 50% of buildings are randomly selected (seeded for reproducibility)
3. Selected buildings use their 100%-electrified demand profile; others use their 0% profile
4. Profiles are aggregated (scaled by building count from the load matching step) into system-level MW demand
5. The same logic applies to solar penetration levels independently

This means a full sweep of `--electrification 0,10,20,...,100 --solar 0,10,20,...,100` produces 121 GenX demand cases from just 4 DECARB runs.
