"""Profitability model for a PV installation with an optional battery.

Single-family house near Bydgoszcz, heated with a heat pump, new net-billing prosumer, Enea G11.
This module contains only calculations (no UI), so it can be tested on its own.

Method in short:
1. Real hourly weather for 2019-2023 (PVGIS): PV output for both roof slopes and air temperature.
2. Consumption per hour = household use (Enea standard G11 profile) + heat pump space heating
   (driven by temperature) + hot water.
3. For every hour: PV covers the load first, the surplus charges the battery, the rest is
   exported. A deficit is covered by the battery first, the rest is imported.
4. Exported energy goes to the prosumer deposit at RCE * 1.23, where RCE is the 2025 average
   hourly price for that month and hour (or the monthly RCEm as an alternative). The deposit
   pays only the energy part of the bill; distribution is always paid for every imported kWh.
5. Optionally the battery sells what the house will not need overnight in the evening peak.
6. Results are averaged over the five weather years.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

MONTH_NAMES = [
    "Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec",
    "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień",
]

# Official monthly RCEm prices for 2025 [PLN/kWh net], index 0 = January. RCEm is the average
# market price weighted by national PV output, so it already reflects when PV produces.
RCEM_2025 = np.array([
    0.48001, 0.44202, 0.17884, 0.16319, 0.21697, 0.13630,
    0.28483, 0.21468, 0.27971, 0.33027, 0.38288, 0.46608,
])
# Statutory correction coefficient applied to the deposit of prosumers settled on RCE.
NET_BILLING_COEFFICIENT = 1.23
# Deposit left unused after 12 months is refunded up to 30% of the month's value (RCE settlement).
REFUND_CAP_SHARE = 0.30

# Enea G11 tariffs for 2026 approved by the URE President on 17.12.2025, gross [PLN/kWh].
ENEA_ENERGY_PRICE_2026 = 0.6187  # Enea S.A., 0.4980 net + excise + VAT
# Enea Operator variable distribution: network 0.2456 + quality 0.0331 + RES 0.0073 + CHP 0.0030 net.
ENEA_DISTRIBUTION_NET_2026 = 0.2456 + 0.0331 + 0.0073 + 0.0030
ENEA_DISTRIBUTION_PRICE_2026 = round(ENEA_DISTRIBUTION_NET_2026 * 1.23, 4)

# Heat pump model (air-to-water, underfloor heating ~35°C):
HEATING_BASE_TEMP_C = 15.0  # no space heating when the daily mean is above this
COP_AT_0C, COP_SLOPE, COP_MIN, COP_MAX = 3.6, 0.12, 1.5, 5.0
# Hot water: tank reheated mainly in the morning and evening.
HOT_WATER_HOURLY_WEIGHTS = np.array([
    0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 2.0, 1.5, 0.8, 0.6, 0.6,
    0.8, 0.6, 0.6, 0.6, 0.8, 1.2, 1.8, 2.0, 2.0, 1.5, 0.8, 0.5,
])

# PME 2 grant (NFOŚiGW, call from 20.10.2026) for net-billing prosumers.
PME2_SHARE, PME2_PER_KWH, PME2_CAP, PME2_MIN_KWH, PME2_PV_RATIO = 0.30, 800.0, 16000.0, 10.0, 2.0
# Thermal modernisation tax relief: deduction limit per taxpayer.
TAX_RELIEF_LIMIT = 53000.0

# Evening export from the battery: allowed hours (local time) and when the night reserve ends.
EVENING_EXPORT_HOURS = range(17, 23)
MIDDAY_HOURS = slice(10, 15)
MORNING_HOUR = 7
# With evening export the battery charges only in cheap hours (price <= 1.3 x midday average)
# or in the afternoon, so it stores the cheapest midday surplus instead of the dearer morning one.
CHEAP_CHARGE_FACTOR = 1.3
AFTERNOON_CHARGE_HOUR = 14


@dataclass(frozen=True)
class Inputs:
    pv_kwp: float = 8.0
    nw_share: float = 0.0  # part of PV power on the north-west slope (rest on south-east)
    battery_kwh: float = 10.0
    annual_consumption_kwh: float = 7000.0
    household_kwh: float = 2600.0
    hot_water_kwh: float = 500.0  # 0 when the heat pump does not heat water
    energy_price: float = ENEA_ENERGY_PRICE_2026  # gross, the part the deposit can pay for
    distribution_price: float = ENEA_DISTRIBUTION_PRICE_2026  # gross, always paid per imported kWh
    pv_cost_per_kwp: float = 3500.0
    battery_cost_per_kwh: float = 2000.0
    battery_round_trip_efficiency: float = 0.90
    battery_depth_of_discharge: float = 0.90
    battery_c_rate: float = 0.5  # max charge/discharge power as a fraction of capacity per hour
    export_pricing: str = "hourly"  # "hourly" = RCE 2025 per month and hour, "rcem" = monthly RCEm
    evening_export: bool = True  # battery sells what the house will not need overnight (hourly only)
    use_pme2_grant: bool = False
    use_tax_relief: bool = True
    tax_rates: tuple[float, float] = (0.32, 0.12)  # two co-owners
    relief_limits: tuple[float, float] = (TAX_RELIEF_LIMIT, TAX_RELIEF_LIMIT)  # limit still unused

    @property
    def grid_price(self) -> float:
        return self.energy_price + self.distribution_price

    @property
    def heating_kwh(self) -> float:
        return max(self.annual_consumption_kwh - self.household_kwh - self.hot_water_kwh, 0.0)

    @property
    def pv_investment(self) -> float:
        return self.pv_kwp * self.pv_cost_per_kwp

    @property
    def battery_investment(self) -> float:
        return self.battery_kwh * self.battery_cost_per_kwh

    @property
    def investment(self) -> float:
        return self.pv_investment + self.battery_investment


# ---------------------------------------------------------------- source data

@lru_cache(maxsize=1)
def weather() -> pd.DataFrame:
    """Hourly PV output per 1 kWp for both slopes and air temperature (PVGIS, local time)."""
    return pd.read_csv(DATA / "pvgis_bydgoszcz.csv")


@lru_cache(maxsize=1)
def household_shape() -> np.ndarray:
    """Enea G11 standard profile mapped onto the weather hours, normalised to 1 per year."""
    w = weather()
    profile = pd.read_csv(DATA / "enea_g11_profile_2026.csv").set_index(["day_of_year", "hour"])["share"]
    keys = pd.MultiIndex.from_arrays([w["day_of_year"].clip(upper=365), w["hour"]])
    shape = profile.reindex(keys).to_numpy()
    return shape / pd.Series(shape).groupby(w["year"].to_numpy()).transform("sum").to_numpy()


@lru_cache(maxsize=1)
def rce_2025() -> np.ndarray:
    """(12, 24) average hourly RCE in 2025 [PLN/kWh], negative prices counted as zero."""
    rce = pd.read_csv(DATA / "rce_2025_month_hour.csv")
    return rce.pivot(index="month", columns="hour", values="price").to_numpy() / 1000


def export_price(pricing: str) -> np.ndarray:
    """Market price [PLN/kWh, before the 1.23 coefficient] for every weather hour."""
    w = weather()
    month = w["month"].to_numpy() - 1
    if pricing == "rcem":
        return RCEM_2025[month]
    return rce_2025()[month, w["hour"].to_numpy()]


def evening_export_hours(inp: Inputs) -> np.ndarray:
    """Weather hours in which the battery may sell to the grid.

    Only in the evening, only with hourly pricing, and only when the evening price after battery
    losses beats the midday price at which the same energy would otherwise have been exported.
    """
    w = weather()
    if not inp.evening_export or inp.export_pricing != "hourly":
        return np.zeros(len(w), dtype=bool)
    rce = rce_2025()
    midday = rce[:, MIDDAY_HOURS].mean(axis=1, keepdims=True)
    profitable = rce * inp.battery_round_trip_efficiency > midday
    evening = np.isin(np.arange(24), list(EVENING_EXPORT_HOURS))[None, :]
    return (profitable & evening)[w["month"].to_numpy() - 1, w["hour"].to_numpy()]


def charge_hours(inp: Inputs) -> np.ndarray:
    """Weather hours in which the battery may charge from PV surplus.

    Without evening export: always. With it: only when the typical price for that month and hour
    is low (the surplus would be exported cheaply anyway) or from 14:00 on.
    """
    w = weather()
    if not evening_export_hours(inp).any():
        return np.ones(len(w), dtype=bool)
    rce = rce_2025()
    midday = rce[:, MIDDAY_HOURS].mean(axis=1, keepdims=True)
    allowed = (rce <= midday * CHEAP_CHARGE_FACTOR) | (np.arange(24)[None, :] >= AFTERNOON_CHARGE_HOUR)
    return allowed[w["month"].to_numpy() - 1, w["hour"].to_numpy()]


def night_reserve(load: np.ndarray) -> np.ndarray:
    """For every weather hour: average house use from the next hour until 7:00 [kWh].

    The battery keeps this much for the night and sells only the rest in the evening.
    Uses the month's average hourly load, so the strategy does not know the future.
    """
    w = weather()
    month, hour = w["month"].to_numpy() - 1, w["hour"].to_numpy()
    avg = pd.DataFrame({"m": month, "h": hour, "load": load}).groupby(["m", "h"])["load"].mean()
    avg = avg.unstack().reindex(index=range(12), columns=range(24)).fillna(0).to_numpy()
    need = np.zeros((12, 24))
    for h in range(24):
        k, hours_left = (h + 1) % 24, []
        while k != MORNING_HOUR:
            hours_left.append(k)
            k = (k + 1) % 24
        need[:, h] = avg[:, hours_left].sum(axis=1) if hours_left else 0.0
    return need[month, hour]


def heat_pump_cop(temp_c: np.ndarray) -> np.ndarray:
    """Approximate COP of an air-to-water heat pump: worse in frost, better in mild weather."""
    return np.clip(COP_AT_0C + COP_SLOPE * temp_c, COP_MIN, COP_MAX)


def heating_shape() -> np.ndarray:
    """Share of yearly heating electricity in each hour, averaged so a mean year sums to 1.

    Heat need follows how far the daily mean temperature is below 15°C (heating degree days,
    the building's thermal mass smooths hourly swings); electricity = heat / COP at the hour's
    temperature. Colder years therefore use more than the average, milder years less.
    """
    w = weather()
    daily_mean = w.groupby(["year", "day_of_year"])["temp_c"].transform("mean").to_numpy()
    heat = np.maximum(HEATING_BASE_TEMP_C - daily_mean, 0.0)
    electricity = heat / heat_pump_cop(w["temp_c"].to_numpy())
    n_years = w["year"].nunique()
    return electricity / electricity.sum() * n_years


def hot_water_shape() -> np.ndarray:
    w = weather()
    weights = HOT_WATER_HOURLY_WEIGHTS[w["hour"].to_numpy()]
    return weights / pd.Series(weights).groupby(w["year"].to_numpy()).transform("sum").to_numpy()


def load_components(inp: Inputs) -> dict[str, np.ndarray]:
    """Hourly consumption [kWh] split into household, space heating and hot water."""
    return {
        "household": inp.household_kwh * household_shape(),
        "heating": inp.heating_kwh * heating_shape(),
        "hot_water": inp.hot_water_kwh * hot_water_shape(),
    }


def pv_per_kwp(nw_share: float) -> np.ndarray:
    w = weather()
    return (1 - nw_share) * w["pv_se"].to_numpy() + nw_share * w["pv_nw"].to_numpy()


# ---------------------------------------------------------------- energy simulation

def simulate_hourly(pv_unit: np.ndarray, load: np.ndarray, pv_kwp: np.ndarray, battery_kwh: np.ndarray,
                    inp: Inputs) -> dict[str, np.ndarray]:
    """Hour-by-hour energy flows for many PV/battery configurations at once.

    pv_unit and load have one value per hour; pv_kwp and battery_kwh one value per configuration.
    Returns arrays of shape (hours, configurations) in kWh. "export" includes battery_to_grid.
    """
    pv = pv_unit[:, None] * pv_kwp[None, :]
    load2d = np.broadcast_to(load[:, None], pv.shape)
    direct = np.minimum(pv, load2d)
    surplus = pv - direct
    deficit = load2d - direct

    usable = battery_kwh * inp.battery_depth_of_discharge
    max_power = battery_kwh * inp.battery_c_rate
    # Round-trip losses split equally between charging and discharging.
    eff = np.sqrt(inp.battery_round_trip_efficiency)
    charge = np.zeros_like(pv)
    discharge = np.zeros_like(pv)
    to_grid = np.zeros_like(pv)
    if np.any(battery_kwh > 0):
        sell = evening_export_hours(inp)
        can_charge = charge_hours(inp)
        reserve = night_reserve(load)
        soc = np.zeros_like(battery_kwh, dtype=float)
        for h in range(len(pv_unit)):
            if can_charge[h]:
                # Surplus charges the battery (limited by free space and power).
                c = np.minimum(np.minimum(surplus[h], (usable - soc) / eff), max_power)
                soc += c * eff
                charge[h] = c
            # Deficit is covered from the battery (limited by stored energy and power).
            d = np.minimum(np.minimum(deficit[h], soc * eff), max_power)
            soc -= d / eff
            discharge[h] = d
            if sell[h]:
                # Evening peak: sell what is stored above the night reserve (within remaining power).
                spare = np.maximum(soc - reserve[h] / eff, 0.0)
                g = np.minimum(spare * eff, max_power - d)
                soc -= g / eff
                to_grid[h] = g
    return {
        "production": pv,
        "consumption": load2d,
        "direct_use": direct,
        "battery_charge": charge,
        "from_battery": discharge,
        "battery_to_grid": to_grid,
        "export": surplus - charge + to_grid,
        "import": deficit - discharge,
    }


@lru_cache(maxsize=1)
def _month_blocks() -> tuple[np.ndarray, np.ndarray, int]:
    """Start index of every (year, month) block of consecutive hours, its month and number of years."""
    w = weather()
    key = (w["year"] * 12 + w["month"] - 1).to_numpy()
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    return starts, w["month"].to_numpy()[starts] - 1, w["year"].nunique()


def monthly_average(hourly: np.ndarray) -> np.ndarray:
    """Sum hours into (year, month), then average over weather years -> shape (12, configurations)."""
    starts, months, n_years = _month_blocks()
    sums = np.add.reduceat(hourly, starts, axis=0)
    out = np.zeros((12,) + hourly.shape[1:])
    np.add.at(out, months, sums)
    return out / n_years


# ---------------------------------------------------------------- money

@dataclass(frozen=True)
class Support:
    grant: float
    grant_note: str | None  # why the grant is not available, if it is not
    tax_relief: float


def pme2_grant(inp: Inputs) -> tuple[float, str | None]:
    """PME 2: 30% of battery costs, max 800 PLN per kWh and 16 000 PLN for net-billing.

    Conditions: battery at least 10 kWh and at least twice the PV peak power.
    """
    if not inp.use_pme2_grant or inp.battery_kwh == 0:
        return 0.0, None
    if inp.battery_kwh < PME2_MIN_KWH:
        return 0.0, f"magazyn musi mieć co najmniej {PME2_MIN_KWH:g} kWh"
    if inp.battery_kwh < PME2_PV_RATIO * inp.pv_kwp:
        return 0.0, f"magazyn musi mieć co najmniej 2 × moc PV = {PME2_PV_RATIO * inp.pv_kwp:g} kWh"
    amount = min(PME2_SHARE * inp.battery_investment, PME2_PER_KWH * inp.battery_kwh, PME2_CAP)
    return amount, None


def tax_relief(inp: Inputs, grant: float) -> float:
    """Thermal modernisation relief: costs not covered by grants are deducted from taxable income.

    Each co-owner deducts up to their remaining limit; the cost goes first to the person with the
    higher tax rate. Benefit = deducted amount * that person's tax rate.
    """
    if not inp.use_tax_relief:
        return 0.0
    remaining = inp.investment - grant
    benefit = 0.0
    for rate, limit in sorted(zip(inp.tax_rates, inp.relief_limits), reverse=True):
        deducted = min(remaining, limit)
        benefit += deducted * rate
        remaining -= deducted
    return benefit


def support(inp: Inputs) -> Support:
    grant, note = pme2_grant(inp)
    return Support(grant=grant, grant_note=note, tax_relief=tax_relief(inp, grant))


@dataclass(frozen=True)
class Result:
    inputs: Inputs
    monthly: pd.DataFrame  # energy [kWh] and money [PLN] per month, average weather year
    bill_before: float
    bill_after: float
    deposit_total: float
    deposit_used: float
    deposit_refund: float
    deposit_forfeited: float
    savings_from_self_consumption: float
    annual_savings: float
    support: Support
    deposit_at_rcem: float | None = None  # check: the same exports valued at monthly RCEm
    deposit_at_hourly_rce: float | None = None  # check: the same exports valued at hourly RCE 2025

    @property
    def net_investment(self) -> float:
        return self.inputs.investment - self.support.grant - self.support.tax_relief

    @property
    def simple_payback_years(self) -> float | None:
        if self.annual_savings <= 0:
            return None
        return self.net_investment / self.annual_savings


def deposit_hourly(export: np.ndarray, pricing: str) -> np.ndarray:
    """Value credited to the deposit in every hour: exported kWh * price * 1.23."""
    return export * export_price(pricing)[:, None] * NET_BILLING_COEFFICIENT


def net_billing(deposit_in: np.ndarray, grid_import: np.ndarray, consumption: np.ndarray, inp: Inputs) -> dict:
    """Annual bill under net-billing for an average weather year, for one or many configurations.

    Inputs are monthly values with shape (12,) or (12, configurations): deposit inflow [PLN],
    imported and consumed energy [kWh].
    The deposit from any month can pay energy bills of the following 12 months, so over a whole
    year it can cover at most the annual energy part of the bill. What is left is refunded up to
    30% of the deposited value; the rest is lost.
    """
    deposit_total = deposit_in.sum(axis=0)
    energy_cost = grid_import.sum(axis=0) * inp.energy_price
    distribution_cost = grid_import.sum(axis=0) * inp.distribution_price
    deposit_used = np.minimum(deposit_total, energy_cost)
    unused = deposit_total - deposit_used
    refund = np.minimum(unused, REFUND_CAP_SHARE * deposit_total)
    bill_before = consumption.sum() * inp.grid_price
    bill_after = distribution_cost + energy_cost - deposit_used - refund
    return {
        "deposit_total": deposit_total,
        "deposit_used": deposit_used,
        "deposit_refund": refund,
        "deposit_forfeited": unused - refund,
        "bill_before": bill_before,
        "bill_after": bill_after,
        "annual_savings": bill_before - bill_after,
    }


def finance(inp: Inputs, monthly: pd.DataFrame) -> Result:
    """Monthly money table plus the annual net-billing result for one configuration."""
    df = monthly.copy()  # must already contain "deposit_in" [PLN]
    df["rcem"] = RCEM_2025
    # Average value of one exported kWh in the month (RCE * 1.23 weighted by when it was exported).
    df["export_value_per_kwh"] = (df["deposit_in"] / df["export"].where(df["export"] > 0)).fillna(0.0)
    df["energy_cost"] = df["import"] * inp.energy_price
    df["distribution_cost"] = df["import"] * inp.distribution_price
    df["bill_before"] = df["consumption"] * inp.grid_price
    # Opportunity loss: an exported kWh earns its deposit value, a self-consumed one saves the full price.
    df["export_lost_value"] = df["export"] * inp.grid_price - df["deposit_in"]

    year = net_billing(df["deposit_in"].to_numpy(), df["import"].to_numpy(), df["consumption"].to_numpy(), inp)
    return Result(
        inputs=inp,
        monthly=df,
        savings_from_self_consumption=df["self_consumption"].sum() * inp.grid_price,
        support=support(inp),
        **{k: float(v) for k, v in year.items()},
    )


def _monthly_frame(flows: dict[str, np.ndarray], components: dict[str, np.ndarray], column: int) -> pd.DataFrame:
    data = {"month": MONTH_NAMES}
    for name, values in flows.items():
        data[name] = monthly_average(values[:, [column]])[:, 0]
    for name, values in components.items():
        data[f"load_{name}"] = monthly_average(values[:, None])[:, 0]
    frame = pd.DataFrame(data)
    frame["self_consumption"] = frame["direct_use"] + frame["from_battery"]
    frame["battery_losses"] = frame["battery_charge"] - frame["from_battery"] - frame["battery_to_grid"]
    return frame


def analyze(inp: Inputs) -> Result:
    components = load_components(inp)
    load = sum(components.values())
    flows = simulate_hourly(pv_per_kwp(inp.nw_share), load, np.array([inp.pv_kwp]),
                            np.array([inp.battery_kwh]), inp)
    monthly = _monthly_frame(flows, components, 0)
    checks = {p: monthly_average(deposit_hourly(flows["export"], p))[:, 0] for p in ("hourly", "rcem")}
    monthly["deposit_in"] = checks[inp.export_pricing]
    result = finance(inp, monthly)
    # Check: the same hourly exports valued both ways.
    return replace(result, deposit_at_rcem=float(checks["rcem"].sum()),
                   deposit_at_hourly_rce=float(checks["hourly"].sum()))


# ---------------------------------------------------------------- payback

@dataclass(frozen=True)
class PaybackOptions:
    price_growth: float = 0.0  # yearly growth of electricity prices (also RCEm), e.g. 0.03
    degradation: float = 0.0  # yearly loss of PV output, e.g. 0.005
    battery_life_years: int | None = None  # None = no battery replacement
    horizon_years: int = 25


def _cashflow(net_investment: float, annual_savings: float, battery_investment: float,
              options: PaybackOptions) -> pd.DataFrame:
    years = np.arange(1, options.horizon_years + 1)
    factor = ((1 + options.price_growth) * (1 - options.degradation)) ** (years - 1)
    savings = annual_savings * factor
    replacement = np.zeros_like(savings)
    if options.battery_life_years and battery_investment > 0:
        due = (years % options.battery_life_years == 0) & (years < options.horizon_years)
        replacement[due] = battery_investment
    return pd.DataFrame({
        "year": years,
        "savings": savings,
        "battery_replacement": replacement,
        "cumulative": -net_investment + np.cumsum(savings - replacement),
    })


def cashflow(result: Result, options: PaybackOptions) -> pd.DataFrame:
    """Cumulative cash flow over the horizon, starting from the net investment.

    Year t savings = first year savings * (1 + price growth)^(t-1) * (1 - degradation)^(t-1).
    Scaling savings by degradation is an approximation: it assumes savings fall in proportion
    to production. A battery replacement costs the same as the original battery (no grant).
    """
    return _cashflow(result.net_investment, result.annual_savings, result.inputs.battery_investment, options)


def _payback(net_investment: float, annual_savings: float, battery_investment: float,
             options: PaybackOptions) -> float | None:
    flow = _cashflow(net_investment, annual_savings, battery_investment, options)
    cumulative = flow["cumulative"].to_numpy()
    negative = np.nonzero(cumulative < 0)[0]
    if len(negative) == 0:
        first_net = flow["savings"].iloc[0] - flow["battery_replacement"].iloc[0]
        return net_investment / first_net if first_net > 0 else None
    last_negative = negative[-1]
    if last_negative == len(cumulative) - 1:
        return None
    before = cumulative[last_negative]
    step = cumulative[last_negative + 1] - before
    return float(last_negative + 1 + (-before / step))


def payback_years(result: Result, options: PaybackOptions) -> float | None:
    """Year in which cumulative cash flow becomes non-negative for good (interpolated).

    None when the investment does not pay back within the horizon.
    """
    return _payback(result.net_investment, result.annual_savings, result.inputs.battery_investment, options)


# ---------------------------------------------------------------- oversizing sweep

def sweep(base: Inputs, pv_values, battery_values, options: PaybackOptions) -> pd.DataFrame:
    """Full analysis for every PV size x battery size, simulated together in one pass."""
    grid = [(float(kwp), float(battery)) for battery in battery_values for kwp in pv_values]
    kwp_arr = np.array([g[0] for g in grid])
    bat_arr = np.array([g[1] for g in grid])
    components = load_components(base)
    flows = simulate_hourly(pv_per_kwp(base.nw_share), sum(components.values()), kwp_arr, bat_arr, base)
    m = {name: monthly_average(values) for name, values in flows.items()}
    deposit_in = monthly_average(deposit_hourly(flows["export"], base.export_pricing))
    money = net_billing(deposit_in, m["import"], m["consumption"][:, 0], base)
    summer = slice(3, 9)  # April - September
    lost = m["export"] * base.grid_price - deposit_in
    production = m["production"].sum(axis=0)

    rows = []
    for i, (kwp, battery) in enumerate(grid):
        inp = replace(base, pv_kwp=kwp, battery_kwh=battery)
        sup = support(inp)
        net_investment = inp.investment - sup.grant - sup.tax_relief
        savings = float(money["annual_savings"][i])
        rows.append({
            "pv_kwp": kwp,
            "battery_kwh": battery,
            "investment": inp.investment,
            "net_investment": net_investment,
            "grant": sup.grant,
            "annual_savings": savings,
            "payback": _payback(net_investment, savings, inp.battery_investment, options),
            "production": production[i],
            "self_consumption": (m["direct_use"][:, i] + m["from_battery"][:, i]).sum(),
            "export": m["export"][:, i].sum(),
            "export_share": m["export"][:, i].sum() / production[i],
            "battery_to_grid": m["battery_to_grid"][:, i].sum(),
            "summer_export_lost_value": lost[summer, i].sum(),
            "deposit_forfeited": float(money["deposit_forfeited"][i]),
        })
    return pd.DataFrame(rows)
