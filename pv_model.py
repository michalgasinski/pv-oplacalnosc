"""Profitability model for a PV installation with an optional battery.

Single-family house in Poland, heated with a heat pump, settled under net-billing.
This module contains only calculations (no UI), so it can be tested on its own.

Method in short:
1. Monthly PV production and monthly consumption are split into a typical day per month
   (24 hourly values). Each month has two day types, sunny and cloudy, so the battery
   is not credited with a perfectly even surplus every day.
2. For every hour: PV covers the load first, the surplus charges the battery, the rest is
   exported. A deficit is covered by the battery first, the rest is imported.
3. Exported energy goes to the prosumer deposit at RCEm * 1.23. The deposit pays only the
   energy part of the bill; distribution is always paid for every imported kWh.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

MONTH_NAMES = [
    "Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec",
    "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień",
]
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Official monthly RCEm prices for 2025 [PLN/kWh net], index 0 = January.
RCEM_2025 = np.array([
    0.48001, 0.44202, 0.17884, 0.16319, 0.21697, 0.13630,
    0.28483, 0.21468, 0.27971, 0.33027, 0.38288, 0.46608,
])
# Statutory correction coefficient applied to the value of exported energy.
NET_BILLING_COEFFICIENT = 1.23
# Deposit left unused after 12 months is refunded only up to 20% of the exported value.
REFUND_CAP_SHARE = 0.20

# Share of annual PV yield per month for a south-facing system in Poland (sums to 1).
_PV_MONTHLY_RAW = np.array([25, 40, 75, 115, 135, 140, 140, 125, 85, 60, 30, 20], dtype=float)
PV_MONTHLY_SHARE = _PV_MONTHLY_RAW / _PV_MONTHLY_RAW.sum()

# Approximate day length [h] around 52°N and solar noon in local clock time
# (CET in Nov-Mar, CEST in Apr-Oct shifts noon by one hour).
DAYLIGHT_HOURS = np.array([8.2, 9.9, 11.9, 13.9, 15.6, 16.6, 16.2, 14.6, 12.6, 10.6, 8.7, 7.8])
SOLAR_NOON = np.array([11.75] * 3 + [12.75] * 7 + [11.75] * 2)

# Two day types per month, equally frequent: a sunny day gives 1.5x the average
# production and a cloudy day 0.5x. Their mean equals the monthly average.
DAY_TYPE_FACTORS = np.array([1.5, 0.5])

# Household (non-heating) use: slightly higher in winter (lighting, more time indoors).
HOUSEHOLD_SEASONAL_WEIGHTS = np.array(
    [1.15, 1.10, 1.00, 0.95, 0.90, 0.85, 0.85, 0.85, 0.90, 1.00, 1.10, 1.15]
)
# Heat pump use for space heating, roughly following heating degree days (zero in summer).
HEAT_PUMP_MONTHLY_WEIGHTS = np.array([22, 19, 14, 6, 1.5, 0, 0, 0, 1.5, 7, 13, 20], dtype=float)

# Hourly shape of household use: low at night, morning peak, high in the evening.
HOUSEHOLD_HOURLY_WEIGHTS = np.array([
    0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.9, 1.3, 1.0, 0.7, 0.7, 0.7,
    0.7, 0.7, 0.7, 0.7, 0.9, 1.3, 1.6, 1.7, 1.6, 1.3, 0.9, 0.6,
])
# Hourly shape of heat pump use: works all day, more at night and in the cold morning.
HEAT_PUMP_HOURLY_WEIGHTS = np.array([
    1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.3, 1.3, 1.3, 0.8, 0.8, 0.8,
    0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1,
])

# Number of identical days simulated in a row; only the last one is recorded, so the
# battery starts it with the charge it would really carry over from the previous evening.
_WARMUP_DAYS = 3


@dataclass(frozen=True)
class Inputs:
    pv_kwp: float = 8.0
    battery_kwh: float = 10.0
    annual_consumption_kwh: float = 7000.0
    household_kwh: float = 2800.0  # part of annual consumption that is not heating
    pv_yield_kwh_per_kwp: float = 1000.0
    grid_price: float = 1.15  # PLN/kWh gross, energy + distribution
    energy_price_part: float = 0.62  # PLN/kWh gross, the part the deposit can pay for
    pv_cost_per_kwp: float = 3500.0
    battery_cost_per_kwh: float = 2000.0
    battery_round_trip_efficiency: float = 0.90
    battery_depth_of_discharge: float = 0.90
    battery_c_rate: float = 0.5  # max charge/discharge power as a fraction of capacity per hour

    @property
    def distribution_price_part(self) -> float:
        return self.grid_price - self.energy_price_part

    @property
    def investment(self) -> float:
        return self.pv_kwp * self.pv_cost_per_kwp + self.battery_kwh * self.battery_cost_per_kwh

    @property
    def battery_investment(self) -> float:
        return self.battery_kwh * self.battery_cost_per_kwh


def monthly_consumption(inp: Inputs) -> np.ndarray:
    """Monthly consumption [kWh]: household part spread evenly-ish, heat pump part by heating need."""
    household = inp.household_kwh * HOUSEHOLD_SEASONAL_WEIGHTS / HOUSEHOLD_SEASONAL_WEIGHTS.sum()
    heat_pump_kwh = max(inp.annual_consumption_kwh - inp.household_kwh, 0.0)
    heat_pump = heat_pump_kwh * HEAT_PUMP_MONTHLY_WEIGHTS / HEAT_PUMP_MONTHLY_WEIGHTS.sum()
    return household + heat_pump


def monthly_production(inp: Inputs) -> np.ndarray:
    """Monthly PV production [kWh]."""
    return inp.pv_kwp * inp.pv_yield_kwh_per_kwp * PV_MONTHLY_SHARE


def pv_daily_shape() -> np.ndarray:
    """(12, 24) array: share of a day's PV production in each hour (rows sum to 1).

    Production follows a sine arc between sunrise and sunset, evaluated at the middle of each hour.
    """
    hour_mid = np.arange(24) + 0.5
    sunrise = SOLAR_NOON - DAYLIGHT_HOURS / 2
    phase = (hour_mid[None, :] - sunrise[:, None]) / DAYLIGHT_HOURS[:, None]
    shape = np.where((phase > 0) & (phase < 1), np.sin(np.pi * np.clip(phase, 0, 1)), 0.0)
    return shape / shape.sum(axis=1, keepdims=True)


def load_daily_profile(inp: Inputs) -> np.ndarray:
    """(12, 24) array: average hourly consumption [kWh] on a typical day of each month."""
    household = inp.household_kwh * HOUSEHOLD_SEASONAL_WEIGHTS / HOUSEHOLD_SEASONAL_WEIGHTS.sum()
    heat_pump = monthly_consumption(inp) - household
    household_shape = HOUSEHOLD_HOURLY_WEIGHTS / HOUSEHOLD_HOURLY_WEIGHTS.sum()
    heat_pump_shape = HEAT_PUMP_HOURLY_WEIGHTS / HEAT_PUMP_HOURLY_WEIGHTS.sum()
    per_day_household = (household / DAYS_IN_MONTH)[:, None] * household_shape[None, :]
    per_day_heat_pump = (heat_pump / DAYS_IN_MONTH)[:, None] * heat_pump_shape[None, :]
    return per_day_household + per_day_heat_pump


def simulate_energy(inp: Inputs) -> pd.DataFrame:
    """Hour-by-hour energy balance on typical days, summed up to months.

    Returns a DataFrame with one row per month and columns in kWh.
    """
    # Shapes below: (month, day type, hour).
    pv_per_day = monthly_production(inp) / DAYS_IN_MONTH
    pv = pv_per_day[:, None, None] * DAY_TYPE_FACTORS[None, :, None] * pv_daily_shape()[:, None, :]
    load = np.broadcast_to(load_daily_profile(inp)[:, None, :], pv.shape)

    usable_capacity = inp.battery_kwh * inp.battery_depth_of_discharge
    max_power = inp.battery_kwh * inp.battery_c_rate
    # Round-trip losses split equally between charging and discharging.
    one_way_efficiency = np.sqrt(inp.battery_round_trip_efficiency)

    soc = np.zeros(pv.shape[:2])  # energy stored in the battery [kWh]
    flows = {name: np.zeros(pv.shape) for name in ("direct", "charge", "discharge", "export", "import")}

    for day in range(_WARMUP_DAYS):
        for hour in range(24):
            p, l = pv[:, :, hour], load[:, :, hour]
            direct = np.minimum(p, l)
            surplus = p - direct
            deficit = l - direct

            # Surplus charges the battery (limited by free space and power), the rest is exported.
            charge = np.minimum.reduce([surplus, (usable_capacity - soc) / one_way_efficiency,
                                        np.full_like(soc, max_power)])
            soc = soc + charge * one_way_efficiency
            export = surplus - charge

            # Deficit is covered from the battery first, the rest is imported from the grid.
            discharge = np.minimum.reduce([deficit, soc * one_way_efficiency, np.full_like(soc, max_power)])
            soc = soc - discharge / one_way_efficiency
            grid_import = deficit - discharge

            if day == _WARMUP_DAYS - 1:
                for name, value in (("direct", direct), ("charge", charge), ("discharge", discharge),
                                    ("export", export), ("import", grid_import)):
                    flows[name][:, :, hour] = value

    def monthly(name: str) -> np.ndarray:
        # Sum the day, average over day types (equally frequent), scale to the month.
        return flows[name].sum(axis=2).mean(axis=1) * DAYS_IN_MONTH

    direct, discharge = monthly("direct"), monthly("discharge")
    charge = monthly("charge")
    return pd.DataFrame({
        "month": MONTH_NAMES,
        "consumption": monthly_consumption(inp),
        "production": monthly_production(inp),
        "direct_use": direct,
        "from_battery": discharge,
        "self_consumption": direct + discharge,
        "battery_charge": charge,
        "battery_losses": charge - discharge,
        "export": monthly("export"),
        "import": monthly("import"),
    })


@dataclass(frozen=True)
class Result:
    inputs: Inputs
    monthly: pd.DataFrame  # energy [kWh] and money [PLN] per month
    bill_before: float
    bill_after: float
    deposit_total: float  # value of exported energy credited to the deposit
    deposit_used: float  # part of the deposit that paid the energy part of bills
    deposit_refund: float  # part of the unused deposit paid back (max 20%)
    deposit_forfeited: float  # unused deposit that is lost
    savings_from_self_consumption: float
    annual_savings: float

    @property
    def simple_payback_years(self) -> float | None:
        if self.annual_savings <= 0:
            return None
        return self.inputs.investment / self.annual_savings


def analyze(inp: Inputs) -> Result:
    """Energy balance plus the annual bill under net-billing.

    The year is treated as a steady state of a running installation: deposit from any month can
    pay bills of the following 12 months, so over a whole year the deposit can cover at most the
    whole annual energy part of the bill. Whatever is left is refunded up to 20% of its value.
    """
    df = simulate_energy(inp)

    # Value of exported energy credited to the deposit: kWh * RCEm * 1.23.
    df["rcem"] = RCEM_2025
    df["export_value_per_kwh"] = RCEM_2025 * NET_BILLING_COEFFICIENT
    df["deposit_in"] = df["export"] * df["export_value_per_kwh"]
    # Imported energy split into the part the deposit can pay (energy) and the part it cannot.
    df["energy_cost"] = df["import"] * inp.energy_price_part
    df["distribution_cost"] = df["import"] * inp.distribution_price_part
    df["bill_before"] = df["consumption"] * inp.grid_price
    # Opportunity loss: an exported kWh is worth RCEm * 1.23, a self-consumed one saves the full price.
    df["export_lost_value"] = df["export"] * (inp.grid_price - df["export_value_per_kwh"])

    deposit_total = df["deposit_in"].sum()
    energy_cost_total = df["energy_cost"].sum()
    deposit_used = min(deposit_total, energy_cost_total)
    unused = deposit_total - deposit_used
    refund = min(unused, REFUND_CAP_SHARE * deposit_total)

    bill_before = df["bill_before"].sum()
    bill_after = df["distribution_cost"].sum() + energy_cost_total - deposit_used - refund
    savings_self = df["self_consumption"].sum() * inp.grid_price

    return Result(
        inputs=inp,
        monthly=df,
        bill_before=bill_before,
        bill_after=bill_after,
        deposit_total=deposit_total,
        deposit_used=deposit_used,
        deposit_refund=refund,
        deposit_forfeited=unused - refund,
        savings_from_self_consumption=savings_self,
        annual_savings=bill_before - bill_after,
    )


@dataclass(frozen=True)
class PaybackOptions:
    price_growth: float = 0.0  # yearly growth of electricity prices (also RCEm), e.g. 0.03
    degradation: float = 0.0  # yearly loss of PV output, e.g. 0.005
    battery_life_years: int | None = None  # None = no battery replacement
    horizon_years: int = 25


def cashflow(result: Result, options: PaybackOptions) -> pd.DataFrame:
    """Cumulative cash flow over the horizon.

    Year t savings = first year savings * (1 + price growth)^(t-1) * (1 - degradation)^(t-1).
    Scaling savings by degradation is an approximation: it assumes savings fall in proportion
    to production. A battery replacement costs the same as the original battery.
    """
    years = np.arange(1, options.horizon_years + 1)
    factor = ((1 + options.price_growth) * (1 - options.degradation)) ** (years - 1)
    savings = result.annual_savings * factor
    replacement = np.zeros_like(savings)
    if options.battery_life_years and result.inputs.battery_kwh > 0:
        due = (years % options.battery_life_years == 0) & (years < options.horizon_years)
        replacement[due] = result.inputs.battery_investment
    net = savings - replacement
    return pd.DataFrame({
        "year": years,
        "savings": savings,
        "battery_replacement": replacement,
        "cumulative": -result.inputs.investment + np.cumsum(net),
    })


def payback_years(result: Result, options: PaybackOptions) -> float | None:
    """Year in which cumulative cash flow becomes non-negative for good (interpolated).

    None when the investment does not pay back within the horizon.
    """
    flow = cashflow(result, options)
    cumulative = flow["cumulative"].to_numpy()
    negative = np.nonzero(cumulative < 0)[0]
    if len(negative) == 0:
        # Paid back within the first year.
        first_net = flow["savings"].iloc[0] - flow["battery_replacement"].iloc[0]
        return result.inputs.investment / first_net if first_net > 0 else None
    last_negative = negative[-1]
    if last_negative == len(cumulative) - 1:
        return None
    before = cumulative[last_negative]
    step = cumulative[last_negative + 1] - before
    return float(last_negative + 1 + (-before / step))


def sweep(base: Inputs, pv_values, battery_values, options: PaybackOptions) -> pd.DataFrame:
    """Run the full analysis for every PV size x battery size combination."""
    rows = []
    for battery in battery_values:
        for kwp in pv_values:
            inp = replace(base, pv_kwp=float(kwp), battery_kwh=float(battery))
            res = analyze(inp)
            m = res.monthly
            summer = m.index.isin(range(3, 9))  # April - September
            rows.append({
                "pv_kwp": kwp,
                "battery_kwh": battery,
                "investment": inp.investment,
                "annual_savings": res.annual_savings,
                "payback": payback_years(res, options),
                "production": m["production"].sum(),
                "self_consumption": m["self_consumption"].sum(),
                "export": m["export"].sum(),
                "export_share": m["export"].sum() / m["production"].sum(),
                "summer_export_lost_value": m.loc[summer, "export_lost_value"].sum(),
                "deposit_forfeited": res.deposit_forfeited,
            })
    return pd.DataFrame(rows)
