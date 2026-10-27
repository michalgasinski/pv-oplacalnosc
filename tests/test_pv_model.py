from dataclasses import replace

import numpy as np
import pytest

from pv_model import (
    DAYS_IN_MONTH,
    Inputs,
    PaybackOptions,
    analyze,
    load_daily_profile,
    monthly_consumption,
    monthly_production,
    payback_years,
    pv_daily_shape,
    simulate_energy,
    sweep,
)


def test_monthly_consumption_sums_to_annual_and_summer_is_low():
    cons = monthly_consumption(Inputs())
    assert cons.sum() == pytest.approx(7000)
    # June-August: household use only, 200-250 kWh
    assert all(200 <= c <= 250 for c in cons[5:8])
    assert cons[0] > 4 * cons[5]


def test_production_follows_yield():
    assert monthly_production(Inputs(pv_kwp=8)).sum() == pytest.approx(8000)


def test_daily_profiles_are_consistent():
    shape = pv_daily_shape()
    assert np.allclose(shape.sum(axis=1), 1)
    assert shape[0, 2] == 0 and shape[0, 22] == 0  # no sun at night in January
    inp = Inputs()
    assert np.allclose(load_daily_profile(inp).sum(axis=1) * DAYS_IN_MONTH, monthly_consumption(inp))


@pytest.mark.parametrize("battery", [0, 5, 20])
def test_energy_balance_holds(battery):
    df = simulate_energy(Inputs(battery_kwh=battery))
    # Every produced kWh is used directly, charged into the battery or exported.
    assert np.allclose(df["production"], df["direct_use"] + df["battery_charge"] + df["export"])
    # Every consumed kWh comes from PV directly, from the battery or from the grid.
    assert np.allclose(df["consumption"], df["direct_use"] + df["from_battery"] + df["import"])
    assert (df["battery_losses"] >= -1e-9).all()


def test_no_battery_means_no_battery_flows():
    df = simulate_energy(Inputs(battery_kwh=0))
    assert df["from_battery"].sum() == 0 and df["battery_charge"].sum() == 0


def test_battery_increases_self_consumption_and_reduces_export():
    without = simulate_energy(Inputs(battery_kwh=0))
    with_battery = simulate_energy(Inputs(battery_kwh=10))
    assert with_battery["self_consumption"].sum() > without["self_consumption"].sum()
    assert with_battery["export"].sum() < without["export"].sum()


def test_deposit_value_uses_rcem_and_coefficient():
    res = analyze(Inputs())
    m = res.monthly
    june = m.iloc[5]
    assert june["deposit_in"] == pytest.approx(june["export"] * 0.13630 * 1.23)


def test_savings_decomposition():
    # Savings = self-consumed kWh at full price + deposit used + refund.
    res = analyze(Inputs())
    expected = res.savings_from_self_consumption + res.deposit_used + res.deposit_refund
    assert res.annual_savings == pytest.approx(expected)
    assert res.bill_before == pytest.approx(7000 * 1.15)


def test_deposit_only_covers_energy_part_and_excess_is_capped():
    # Huge PV: deposit exceeds the energy part of the bill, distribution still has to be paid.
    res = analyze(Inputs(pv_kwp=15, battery_kwh=0, annual_consumption_kwh=3000, household_kwh=2800))
    m = res.monthly
    assert res.deposit_used == pytest.approx(m["energy_cost"].sum())
    assert res.bill_after == pytest.approx(m["distribution_cost"].sum() - res.deposit_refund)
    assert res.deposit_refund <= 0.2 * res.deposit_total + 1e-9
    assert res.deposit_forfeited > 0


def test_simple_payback_matches_cashflow_without_options():
    res = analyze(Inputs())
    assert payback_years(res, PaybackOptions()) == pytest.approx(res.simple_payback_years)


def test_price_growth_shortens_and_battery_replacement_lengthens_payback():
    res = analyze(Inputs())
    base = payback_years(res, PaybackOptions())
    assert payback_years(res, PaybackOptions(price_growth=0.05)) < base
    assert payback_years(res, PaybackOptions(battery_life_years=12)) > base


def test_payback_none_when_never_paid_back():
    res = analyze(Inputs(pv_cost_per_kwp=100_000))
    assert payback_years(res, PaybackOptions()) is None


def test_oversizing_raises_summer_export_losses():
    df = sweep(Inputs(), [6, 10, 14], [0], PaybackOptions())
    assert df["summer_export_lost_value"].is_monotonic_increasing
    assert df["export_share"].is_monotonic_increasing


def test_sweep_size():
    base = replace(Inputs(), battery_kwh=0)
    assert len(sweep(base, [4, 8], [0, 10], PaybackOptions())) == 4
