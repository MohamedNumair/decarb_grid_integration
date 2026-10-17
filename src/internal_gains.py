"""Internal heat gains as DECARB computes them, from ResStock 15-min data.

DECARB's internal gains are pBihgp * occupants + pBihgl * pQlight + pBihge * pQequip
(load_tm.jl, calculate_internal_heat_gain), and it bills pQlight and pQequip as
electricity. create_decarb_inputs writes pQlight / pQequip / pBppl from these
functions, and the thermal model is fitted on the same gains, so the fitted
coefficients apply to the inputs DECARB actually gets.
"""

import pandas as pd

ELEC = "out.electricity.{}.energy_consumption..kwh"

TOTAL_COL = ELEC.format("total")
LIGHTING_COLS = [ELEC.format(e) for e in ("lighting_exterior", "lighting_garage", "lighting_interior")]
HEATING_COLS = [ELEC.format(e) for e in ("heating", "heating_fans_pumps",
                                         "heating_hp_bkup", "heating_hp_bkup_fa")]
COOLING_COLS = [ELEC.format(e) for e in ("cooling", "cooling_fans_pumps")]
EV_COL = ELEC.format("ev_charging")
HOT_WATER_COLS = [ELEC.format(e) for e in ("hot_water", "hot_water_solar_th")]
OCCUPANT_SCHEDULE_COL = "out.schedules.occupants"
OCCUPANTS_META_COL = "in.occupants"


def _sum(df, cols):
    avail = [c for c in cols if c in df.columns]
    return df[avail].fillna(0).sum(axis=1) if avail else pd.Series(0.0, index=df.index)


def lighting_kwh(df):
    """Electric lighting energy per row (kWh)."""
    if not any(c in df.columns for c in LIGHTING_COLS):
        raise ValueError("No lighting consumption columns found")
    return _sum(df, LIGHTING_COLS)


def equipment_kwh(df):
    """Electric energy per row not modeled separately by DECARB (kWh): electricity
    total minus electric heating, cooling, EV, hot water and lighting. Gas, propane
    and oil end uses other than heating and hot water are not modeled."""
    if TOTAL_COL not in df.columns:
        raise ValueError(f"Electricity total column not found: {TOTAL_COL}")
    return (df[TOTAL_COL].fillna(0) - _sum(df, HEATING_COLS) - _sum(df, COOLING_COLS)
            - _sum(df, [EV_COL]) - _sum(df, HOT_WATER_COLS) - lighting_kwh(df))


def occupant_count(value):
    """ResStock in.occupants ("10+" counts as 15)."""
    raw = str(value).strip()
    return 15 if raw == "10+" else int(raw)
