from dataclasses import replace

import numpy as np
import pytest

from pv_model import (
    Inputs,
    PaybackOptions,
    analyze,
    heat_pump_cop,
    load_components,
    monthly_average,
    payback_years,
    pme2_grant,
    pv_per_kwp,
    sweep,
    tax_relief,
    weather,
)


@pytest.fixture(scope="module")
def default_result():
    return analyze(Inputs())


def test_weather_data_covers_five_years():
    w = weather()
    assert sorted(w["year"].unique()) == [2019, 2020, 2021, 2022, 2023]
    assert len(w) > 5 * 8759


def test_pv_yield_is_plausible_for_bydgoszcz():
    se = monthly_average(pv_per_kwp(0.0)[:, None]).sum()
    nw = monthly_average(pv_per_kwp(1.0)[:, None]).sum()
    assert 850 < se < 950  # kWh per kWp per year on the south-east slope, 15°
    assert nw < se


def test_load_components_sum_to_inputs_in_average_year():
    inp = Inputs()
    comps = {k: monthly_average(v[:, None]).sum() for k, v in load_components(inp).items()}
    assert comps["household"] == pytest.approx(2600)
    assert comps["hot_water"] == pytest.approx(500)
    assert comps["heating"] == pytest.approx(7000 - 2600 - 500)


def test_heating_is_winter_and_summer_is_low(default_result):
    m = default_result.monthly
    assert m["load_heating"].iloc[6] == pytest.approx(0, abs=1)  # July
    assert m["load_heating"].iloc[0] > 600  # January
    assert all(200 <= c <= 260 for c in m["consumption"].iloc[5:8])  # summer: household + hot water


def test_cop_is_lower_in_frost():
    assert heat_pump_cop(np.array([-15.0]))[0] < heat_pump_cop(np.array([7.0]))[0]


@pytest.mark.parametrize("battery", [0, 5, 20])
def test_energy_balance_holds(battery):
    m = analyze(Inputs(battery_kwh=battery)).monthly
    assert np.allclose(m["production"], m["direct_use"] + m["battery_charge"] + m["export"])
    assert np.allclose(m["consumption"], m["direct_use"] + m["from_battery"] + m["import"])
    assert (m["battery_losses"] >= -1e-9).all()


def test_battery_increases_self_consumption_and_reduces_export():
    without = analyze(Inputs(battery_kwh=0)).monthly
    with_battery = analyze(Inputs(battery_kwh=10)).monthly
    assert with_battery["self_consumption"].sum() > without["self_consumption"].sum()
    assert with_battery["export"].sum() < without["export"].sum()


def test_deposit_value_uses_rcem_and_coefficient(default_result):
    june = default_result.monthly.iloc[5]
    assert june["deposit_in"] == pytest.approx(june["export"] * 0.13630 * 1.23)


def test_savings_decomposition(default_result):
    res = default_result
    expected = res.savings_from_self_consumption + res.deposit_used + res.deposit_refund
    assert res.annual_savings == pytest.approx(expected)
    assert res.bill_before == pytest.approx(7000 * res.inputs.grid_price)


def test_deposit_only_covers_energy_part_and_refund_is_capped_at_30_percent():
    res = analyze(Inputs(pv_kwp=15, battery_kwh=0, annual_consumption_kwh=3100, hot_water_kwh=0))
    m = res.monthly
    assert res.deposit_used == pytest.approx(m["energy_cost"].sum())
    assert res.bill_after == pytest.approx(m["distribution_cost"].sum() - res.deposit_refund)
    assert res.deposit_refund <= 0.3 * res.deposit_total + 1e-9


def test_hourly_rce_check_is_close_to_rcem(default_result):
    # RCEm is PV-weighted, so valuing the same exports at hourly prices gives a similar deposit.
    assert default_result.deposit_hourly_rce == pytest.approx(default_result.deposit_total, rel=0.15)


def test_pme2_grant_conditions_and_caps():
    on = Inputs(use_pme2_grant=True)
    assert pme2_grant(replace(on, pv_kwp=8, battery_kwh=10))[0] == 0  # needs >= 16 kWh
    assert pme2_grant(replace(on, pv_kwp=4, battery_kwh=8))[0] == 0  # needs >= 10 kWh
    assert pme2_grant(replace(on, pv_kwp=5, battery_kwh=10))[0] == pytest.approx(6000)  # 30% of 20 000
    assert pme2_grant(replace(on, pv_kwp=5, battery_kwh=10, battery_cost_per_kwh=4000))[0] == 8000  # 800/kWh
    assert pme2_grant(replace(on, pv_kwp=5, battery_kwh=25, battery_cost_per_kwh=4000))[0] == 16000  # cap
    assert pme2_grant(replace(on, use_pme2_grant=False, pv_kwp=5, battery_kwh=10))[0] == 0


def test_tax_relief_uses_higher_rate_first_and_excludes_grant():
    inp = Inputs(pv_kwp=10, battery_kwh=20)  # 35 000 + 40 000 = 75 000
    assert tax_relief(inp, grant=0) == pytest.approx(53000 * 0.32 + 22000 * 0.12)
    assert tax_relief(inp, grant=10000) == pytest.approx(53000 * 0.32 + 12000 * 0.12)
    assert tax_relief(replace(inp, use_tax_relief=False), grant=0) == 0


def test_net_investment_and_simple_payback(default_result):
    res = default_result
    assert res.net_investment == pytest.approx(res.inputs.investment - res.support.tax_relief)
    assert payback_years(res, PaybackOptions()) == pytest.approx(res.simple_payback_years)


def test_price_growth_shortens_and_battery_replacement_lengthens_payback(default_result):
    base = payback_years(default_result, PaybackOptions())
    assert payback_years(default_result, PaybackOptions(price_growth=0.05)) < base
    # Replacement before the payback year (here every 7 years) must push payback later.
    long = PaybackOptions(battery_life_years=7, horizon_years=40)
    assert payback_years(default_result, long) > base


def test_payback_none_when_never_paid_back():
    res = analyze(Inputs(pv_cost_per_kwp=100_000, use_tax_relief=False))
    assert payback_years(res, PaybackOptions()) is None


def test_sweep_matches_single_analysis():
    base = Inputs()
    row = sweep(base, [8], [10], PaybackOptions()).iloc[0]
    single = analyze(base)
    assert row["annual_savings"] == pytest.approx(single.annual_savings)
    assert row["payback"] == pytest.approx(payback_years(single, PaybackOptions()))


def test_oversizing_raises_summer_export_losses():
    df = sweep(Inputs(), [6, 10, 14], [0], PaybackOptions())
    assert df["summer_export_lost_value"].is_monotonic_increasing
    assert df["export_share"].is_monotonic_increasing
