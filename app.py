"""Streamlit UI for the PV + battery profitability analysis (calculations live in pv_model.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pv_model import (
    COP_AT_0C,
    COP_SLOPE,
    ENEA_DISTRIBUTION_PRICE_2026,
    ENEA_ENERGY_PRICE_2026,
    HEATING_BASE_TEMP_C,
    MONTH_NAMES,
    NET_BILLING_COEFFICIENT,
    PME2_CAP,
    PME2_PER_KWH,
    RCEM_2025,
    REFUND_CAP_SHARE,
    TAX_RELIEF_LIMIT,
    Inputs,
    PaybackOptions,
    analyze,
    cashflow,
    load_components,
    payback_years,
    pv_per_kwp,
    rce_2025,
    sweep,
    weather,
)

# Categorical colors in fixed order (colorblind-checked reference palette).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MUTED = "#8a8984"
SWEEP_BATTERIES = [0, 5, 10, 15, 20]
SWEEP_PV = np.arange(4, 15.01, 0.5)

st.set_page_config(page_title="Opłacalność PV + magazyn", page_icon="☀️", layout="wide")


def num(value: float, decimals: int = 0) -> str:
    """Polish number format: space as thousands separator, comma as decimal separator."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def pln(value: float, decimals: int = 0) -> str:
    return f"{num(value, decimals)} zł"


def kwh(value: float) -> str:
    return f"{num(value)} kWh"


def years_text(value: float | None, horizon: int) -> str:
    return f"> {horizon} lat" if value is None else f"{num(value, 1)} lat"


def style_figure(fig: go.Figure, y_title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        yaxis_title=y_title,
        bargap=0.25,
        bargroupgap=0.08,
        separators=", ",  # Polish number format in hover labels and axes
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.5)")
    return fig


@st.cache_data(show_spinner="Liczę warianty przewymiarowania…")
def cached_sweep(base: Inputs, batteries: tuple[float, ...], options: PaybackOptions) -> pd.DataFrame:
    return sweep(base, SWEEP_PV, batteries, options)


@st.cache_data(show_spinner="Symuluję 5 lat godzina po godzinie…")
def cached_analyze(inp: Inputs):
    return analyze(inp)


# ---------------------------------------------------------------- sidebar (inputs)
with st.sidebar:
    st.header("Parametry")
    st.subheader("Instalacja")
    pv_kwp = st.slider("Moc instalacji PV [kWp]", 4.0, 15.0, 8.0, 0.5)
    west_percent = st.slider(
        "Część paneli na połaci zachodniej [%]", 0, 100, 0, 10,
        help="Reszta stoi na połaci wschodniej. Dach 15°, połacie ok. 15° od osi wschód–zachód.",
    )
    battery_kwh = st.slider("Pojemność magazynu energii [kWh]", 0, 20, 10, 1)

    st.subheader("Zużycie energii")
    annual = st.number_input("Roczne zużycie razem [kWh]", 3000, 20000, 7000, 100)
    household = st.number_input(
        "w tym bytowe [kWh]", 1000, int(annual), min(2600, int(annual)), 100,
        help="2 osoby, jedna pracuje zdalnie: pralka, zmywarka, gotowanie, RTV, oświetlenie.",
    )
    heats_water = st.checkbox("Pompa ciepła grzeje też ciepłą wodę", value=True)
    hot_water = st.number_input("w tym ciepła woda [kWh]", 0, 3000, 500, 50, disabled=not heats_water) \
        if heats_water else 0
    st.caption(f"Ogrzewanie pompą ciepła: **{kwh(max(annual - household - hot_water, 0))}** (reszta)")

    st.subheader("Ceny prądu (Enea G11 2026)")
    energy_price = st.number_input("Energia [zł/kWh brutto]", 0.10, 2.00, ENEA_ENERGY_PRICE_2026, 0.01,
                                   format="%.4f", help="Tę część może pokryć depozyt prosumencki.")
    distribution_price = st.number_input("Dystrybucja zmienna [zł/kWh brutto]", 0.0, 2.0,
                                         ENEA_DISTRIBUTION_PRICE_2026, 0.01, format="%.4f",
                                         help="Płacona za każdą pobraną kWh, depozyt jej nie pokrywa.")
    st.caption(f"Razem za każdą kWh z sieci: **{pln(energy_price + distribution_price, 3)}**")

    st.subheader("Koszty inwestycji")
    pv_cost = st.number_input("Koszt PV [zł/kWp]", 1000, 10000, 3500, 100)
    battery_cost = st.number_input("Koszt magazynu [zł/kWh]", 500, 6000, 2000, 100)

    st.subheader("Dotacja i ulga")
    use_grant = st.checkbox("Dotacja PME 2 na magazyn", value=False,
                            help="30% kosztu magazynu, max 800 zł/kWh i 16 000 zł. Warunek: magazyn ≥ 10 kWh "
                                 "i ≥ 2 × moc PV.")
    use_relief = st.checkbox("Ulga termomodernizacyjna", value=True)
    rate_1 = st.selectbox("Stawka podatku – współwłaściciel 1", [32, 12], index=0, disabled=not use_relief)
    rate_2 = st.selectbox("Stawka podatku – współwłaściciel 2", [12, 32], index=0, disabled=not use_relief)
    with st.expander("Pozostały limit ulgi"):
        st.caption("53 000 zł na osobę na całe życie. Jeśli pompa ciepła była już odliczana, limit jest mniejszy.")
        limit_1 = st.number_input("Współwłaściciel 1 [zł]", 0, int(TAX_RELIEF_LIMIT), int(TAX_RELIEF_LIMIT), 1000)
        limit_2 = st.number_input("Współwłaściciel 2 [zł]", 0, int(TAX_RELIEF_LIMIT), int(TAX_RELIEF_LIMIT), 1000)

    st.subheader("Rozliczenie i praca magazynu")
    pricing_label = st.radio(
        "Wycena energii oddanej", ["Ceny godzinowe RCE (średnie 2025)", "RCEm (miesięczna)"], index=0,
        help="Ceny godzinowe: średnia cena 2025 dla każdego miesiąca i godziny. RCEm: jedna cena na miesiąc.",
    )
    hourly_pricing = pricing_label.startswith("Ceny godzinowe")
    evening_export = st.checkbox(
        "Magazyn sprzedaje nadwyżkę wieczorem", value=True, disabled=not hourly_pricing,
        help="Wieczorem oddaje do sieci to, czego dom nie potrzebuje do rana. Ma sens tylko przy cenach godzinowych.",
    )

    with st.expander("Zaawansowane: magazyn"):
        efficiency = st.slider("Sprawność magazynu (ładowanie + rozładowanie) [%]", 70, 100, 90) / 100
        dod = st.slider("Użyteczna pojemność magazynu (DoD) [%]", 50, 100, 90) / 100

    with st.expander("Czas zwrotu: opcje"):
        price_growth = st.slider("Wzrost cen prądu [%/rok]", 0.0, 10.0, 0.0, 0.5) / 100
        degradation = st.slider("Degradacja paneli [%/rok]", 0.0, 1.0, 0.0, 0.1) / 100
        replace_battery = st.checkbox("Uwzględnij wymianę magazynu")
        battery_life = st.slider("Żywotność magazynu [lata]", 8, 20, 15, disabled=not replace_battery)

inputs = Inputs(
    pv_kwp=pv_kwp,
    west_share=west_percent / 100,
    battery_kwh=float(battery_kwh),
    annual_consumption_kwh=float(annual),
    household_kwh=float(household),
    hot_water_kwh=float(hot_water),
    energy_price=float(energy_price),
    distribution_price=float(distribution_price),
    pv_cost_per_kwp=float(pv_cost),
    battery_cost_per_kwh=float(battery_cost),
    battery_round_trip_efficiency=efficiency,
    battery_depth_of_discharge=dod,
    export_pricing="hourly" if hourly_pricing else "rcem",
    evening_export=evening_export,
    use_pme2_grant=use_grant,
    use_tax_relief=use_relief,
    tax_rates=(rate_1 / 100, rate_2 / 100),
    relief_limits=(float(limit_1), float(limit_2)),
)
options = PaybackOptions(
    price_growth=price_growth,
    degradation=degradation,
    battery_life_years=battery_life if replace_battery else None,
)

result = cached_analyze(inputs)
m = result.monthly
payback = payback_years(result, options)
production = m["production"].sum()
self_consumption = m["self_consumption"].sum()
grid_price = inputs.grid_price

# ---------------------------------------------------------------- header and KPIs
st.title("☀️ Opłacalność fotowoltaiki i magazynu energii")
st.caption(
    "Dom z pompą ciepła pod Bydgoszczą, nowa instalacja w net-billingu (RCE × 1,23), taryfa Enea G11 2026. "
    "Średnia z pogody 2019–2023, liczona godzina po godzinie. Szacunek, nie oferta."
)

k1, k2, k4 = st.columns(3)
k3, k5, _ = st.columns(3)
k1.metric("Roczna oszczędność", pln(result.annual_savings))
k2.metric(
    "Rachunek (część zmienna)", pln(result.bill_after),
    delta=f"{pln(result.bill_after - result.bill_before)} vs bez PV", delta_color="inverse",
)
k3.metric("Nakład netto", pln(result.net_investment),
          help=f"Koszt {pln(inputs.investment)} − dotacja {pln(result.support.grant)} − ulga "
               f"{pln(result.support.tax_relief)}")
k4.metric("Czas zwrotu", years_text(payback, options.horizon_years))
k5.metric("Autokonsumpcja", f"{self_consumption / production:.0%}" if production else "–",
          help="Część produkcji zużyta w domu (od razu lub przez magazyn).")

tab_energy, tab_money, tab_oversize, tab_assumptions = st.tabs(
    ["Bilans energii", "Rachunek i zwrot", "Przewymiarowanie", "Założenia"]
)

# ---------------------------------------------------------------- energy balance
with tab_energy:
    st.subheader("Bilans miesięczny (średni rok)")
    fig = go.Figure()
    for i, (col, name) in enumerate([
        ("production", "Produkcja PV"),
        ("self_consumption", "Autokonsumpcja"),
        ("export", "Oddana do sieci"),
        ("import", "Pobrana z sieci"),
    ]):
        fig.add_bar(x=m["month"], y=m[col], name=name, marker_color=SERIES_COLORS[i],
                    hovertemplate="%{y:,.0f} kWh")
    fig.add_scatter(x=m["month"], y=m["consumption"], name="Zużycie domu", mode="lines+markers",
                    line=dict(color=MUTED, width=2, dash="dot"), marker=dict(size=8),
                    hovertemplate="%{y:,.0f} kWh")
    st.plotly_chart(style_figure(fig, "kWh"), width="stretch")

    st.subheader("Bilans roczny")
    yearly = pd.DataFrame({
        "Pozycja": ["Zużycie domu", "Produkcja PV", "Autokonsumpcja bezpośrednia", "Zużyte z magazynu",
                    "Straty w magazynie", "Oddana do sieci (razem)", "w tym sprzedana z magazynu wieczorem",
                    "Pobrana z sieci"],
        "kWh": [m["consumption"].sum(), production, m["direct_use"].sum(), m["from_battery"].sum(),
                m["battery_losses"].sum(), m["export"].sum(), m["battery_to_grid"].sum(), m["import"].sum()],
    })
    yearly["% produkcji"] = (yearly["kWh"] / production).where(yearly.index.isin([2, 3, 4, 5, 6]))
    st.dataframe(
        yearly, hide_index=True, width="stretch",
        column_config={"kWh": st.column_config.NumberColumn(format="%.0f"),
                       "% produkcji": st.column_config.NumberColumn(format="percent")},
    )
    st.caption(
        f"Pokrycie zużycia z własnej produkcji: {self_consumption / m['consumption'].sum():.0%}. "
        "Zimą (wysokie zużycie pompy ciepła, mało słońca) i tak większość energii pochodzi z sieci."
    )

# ---------------------------------------------------------------- bill and payback
with tab_money:
    st.subheader("Roczny rachunek za prąd: przed i po inwestycji")
    st.caption("Tylko opłaty zależne od zużycia. Opłaty stałe (sieciowa stała, abonament, mocowa) są takie "
               "same przed i po inwestycji, więc nie wpływają na oszczędność.")
    energy_after = m["energy_cost"].sum() - result.deposit_used
    fig = go.Figure()
    fig.add_bar(x=["Przed inwestycją", "Po inwestycji"],
                y=[m["consumption"].sum() * inputs.distribution_price, m["distribution_cost"].sum()],
                name="Dystrybucja", marker_color=SERIES_COLORS[0], hovertemplate="%{y:,.0f} zł")
    fig.add_bar(x=["Przed inwestycją", "Po inwestycji"],
                y=[m["consumption"].sum() * inputs.energy_price, energy_after],
                name="Energia (po odjęciu depozytu)", marker_color=SERIES_COLORS[1], hovertemplate="%{y:,.0f} zł")
    fig.update_layout(barmode="stack")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(style_figure(fig, "zł / rok", height=380), width="stretch")
    with c2:
        st.markdown(f"""
| | |
|---|---:|
| Rachunek przed | **{pln(result.bill_before)}** |
| Rachunek po | **{pln(result.bill_after)}** |
| **Oszczędność** | **{pln(result.annual_savings)}** |
| z tego: autokonsumpcja ({kwh(self_consumption)} × {pln(grid_price, 3)}) | {pln(result.savings_from_self_consumption)} |
| z tego: depozyt wykorzystany na energię | {pln(result.deposit_used)} |
| z tego: zwrot niewykorzystanego depozytu (max 30%) | {pln(result.deposit_refund)} |
| Depozyt przepadający | {pln(result.deposit_forfeited)} |
""")
        if result.deposit_forfeited > 1:
            st.warning(
                f"Depozyt ({pln(result.deposit_total)}) przewyższa koszt energii pobranej z sieci "
                f"({pln(m['energy_cost'].sum())}). Nadwyżka ponad 30% zwrotu przepada – "
                "instalacja jest za duża w stosunku do zużycia."
            )

    st.subheader("Nakład inwestycyjny")
    sup = result.support
    grant_label = "Dotacja PME 2 na magazyn"
    if inputs.use_pme2_grant and sup.grant_note:
        grant_label += f" – brak: {sup.grant_note}"
    st.markdown(f"""
| | |
|---|---:|
| Fotowoltaika ({num(inputs.pv_kwp, 1)} kWp × {pln(inputs.pv_cost_per_kwp)}) | {pln(inputs.pv_investment)} |
| Magazyn ({num(inputs.battery_kwh)} kWh × {pln(inputs.battery_cost_per_kwh)}) | {pln(inputs.battery_investment)} |
| {grant_label} | −{pln(sup.grant)} |
| Ulga termomodernizacyjna (zwrot podatku) | −{pln(sup.tax_relief)} |
| **Nakład netto** | **{pln(result.net_investment)}** |
""")

    st.subheader("Skumulowany przepływ pieniędzy")
    flow = cashflow(result, options)
    fig = go.Figure()
    fig.add_scatter(x=flow["year"], y=flow["cumulative"], mode="lines+markers", name="Bilans narastająco",
                    line=dict(color=SERIES_COLORS[0], width=2), marker=dict(size=8),
                    hovertemplate="Rok %{x}: %{y:,.0f} zł<extra></extra>")
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    if payback is not None:
        fig.add_vline(x=payback, line_color=SERIES_COLORS[1], line_dash="dash",
                      annotation_text=f"zwrot: {years_text(payback, options.horizon_years)}")
    fig.update_xaxes(title="rok")
    st.plotly_chart(style_figure(fig, "zł", height=360), width="stretch")
    st.caption("Start od nakładu netto (dotacja i ulga odjęte od razu). Prosty czas zwrotu = nakład netto / "
               "roczna oszczędność; opcje w panelu bocznym zmieniają oszczędności w kolejnych latach.")

    with st.expander("Rozliczenie miesięczne"):
        table = pd.DataFrame({
            "Miesiąc": m["month"],
            "RCEm [zł/kWh]": m["rcem"],
            "Średnia wartość 1 kWh oddanej (×1,23)": m["export_value_per_kwh"],
            "Oddana [kWh]": m["export"],
            "Wpływ do depozytu [zł]": m["deposit_in"],
            "Pobrana [kWh]": m["import"],
            "Koszt energii [zł]": m["energy_cost"],
            "Koszt dystrybucji [zł]": m["distribution_cost"],
        })
        st.dataframe(table.round(2), hide_index=True, width="stretch")
        st.caption(
            "Depozyt z danego miesiąca może opłacać energię przez kolejne 12 miesięcy, więc w ujęciu "
            "rocznym letnie nadwyżki opłacają zimowe rachunki. Dystrybucji depozyt nie pokrywa."
        )

# ---------------------------------------------------------------- oversizing analysis
with tab_oversize:
    st.subheader("Jak przewymiarowanie wpływa na czas zwrotu")
    june_rce = rce_2025()[5] * NET_BILLING_COEFFICIENT
    price_note = (f"**cenę z danej godziny × 1,23** – w czerwcu w południe średnio {pln(june_rce[12], 2)}, "
                  f"wieczorem (19–20) {pln(june_rce[19], 2)}" if hourly_pricing else
                  f"**RCEm × 1,23** – w czerwcu {pln(RCEM_2025[5] * NET_BILLING_COEFFICIENT, 2)}, "
                  f"w kwietniu {pln(RCEM_2025[3] * NET_BILLING_COEFFICIENT, 2)}")
    st.markdown(
        f"Każda kWh zużyta w domu oszczędza pełną cenę **{pln(grid_price, 2)}**. Każda kWh oddana do sieci "
        f"dostaje tylko {price_note}. Dokładanie paneli zwiększa głównie letnią nadwyżkę w południe, "
        "więc każdy kolejny kWp zarabia mniej. Magazyn przesuwa część tej nadwyżki na wieczór – dla domu "
        "albo do sprzedaży po wyższej cenie."
    )
    if inputs.use_pme2_grant:
        st.info("Dotacja PME 2 jest włączona: przysługuje tylko, gdy magazyn ma ≥ 10 kWh i ≥ 2 × moc PV, "
                "dlatego krzywe zwrotu mają skoki w miejscu, gdzie dotacja przestaje przysługiwać.")

    batteries = tuple(sorted(set(SWEEP_BATTERIES) | {battery_kwh}))
    sw = cached_sweep(inputs, batteries, options)

    def battery_lines(value_col: str, y_title: str, hover: str, height: int = 400) -> go.Figure:
        fig = go.Figure()
        for i, battery in enumerate(batteries):
            part = sw[sw["battery_kwh"] == battery]
            current = battery == battery_kwh
            fig.add_scatter(
                x=part["pv_kwp"], y=part[value_col], mode="lines",
                name=f"magazyn {battery} kWh" + (" (wybrany)" if current else ""),
                line=dict(color=SERIES_COLORS[i], width=3 if current else 2),
                hovertemplate=hover,
            )
        fig.add_vline(x=pv_kwp, line_color=MUTED, line_dash="dot")
        fig.update_xaxes(title="moc PV [kWp]")
        return style_figure(fig, y_title, height)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Czas zwrotu [lata]**")
        st.plotly_chart(battery_lines("payback", "lata", "%{y:.1f} lat"), width="stretch")
    with c2:
        st.markdown("**Utracona wartość letniego eksportu (kwiecień–wrzesień) [zł/rok]**")
        st.plotly_chart(battery_lines("summer_export_lost_value", "zł / rok", "%{y:,.0f} zł"), width="stretch")
    st.caption(
        "Utracona wartość = oddane kWh × cena zakupu − wpływ do depozytu, czyli ile więcej dałaby ta energia, "
        "gdyby udało się ją zużyć w domu. Linia kropkowana = wybrana moc PV. "
        f"Puste miejsca na wykresie zwrotu oznaczają brak zwrotu w {options.horizon_years} lat."
    )

    current = sw[sw["battery_kwh"] == battery_kwh].reset_index(drop=True)
    best = current.loc[current["payback"].idxmin()] if current["payback"].notna().any() else None

    # Marginal analysis: what each extra 0.5 kWp brings compared to what it costs.
    marginal_savings = current["annual_savings"].diff() / current["pv_kwp"].diff()
    marginal_cost = current["net_investment"].diff() / current["pv_kwp"].diff()
    marginal_payback = marginal_cost / marginal_savings

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Oszczędność z każdego kolejnego kWp (magazyn {battery_kwh} kWh) [zł/rok]**")
        fig = go.Figure()
        fig.add_bar(x=current["pv_kwp"], y=marginal_savings, name="Oszczędność z kolejnego kWp",
                    marker_color=SERIES_COLORS[0], customdata=marginal_payback,
                    hovertemplate="%{y:,.0f} zł/rok (zwrot tego kWp: %{customdata:.1f} lat)<extra></extra>")
        fig.update_xaxes(title="moc PV [kWp]")
        st.plotly_chart(style_figure(fig, "zł / rok na kWp", height=360), width="stretch")
    with c2:
        st.markdown("**Utracona wartość eksportu w miesiącach – obecna konfiguracja [zł]**")
        fig = go.Figure()
        fig.add_bar(x=m["month"], y=m["export_lost_value"], name="Utracona wartość",
                    marker_color=SERIES_COLORS[1],
                    customdata=np.stack([m["export"], m["export_value_per_kwh"]], axis=1),
                    hovertemplate="%{y:,.0f} zł (oddane %{customdata[0]:,.0f} kWh "
                                  "po %{customdata[1]:.2f} zł)<extra></extra>")
        st.plotly_chart(style_figure(fig, "zł", height=360), width="stretch")

    st.subheader("Wnioski dla wybranego magazynu")
    if best is not None:
        st.markdown(
            f"- Najkrótszy czas zwrotu przy magazynie {battery_kwh} kWh: **{num(best['pv_kwp'], 1)} kWp** "
            f"→ {years_text(best['payback'], options.horizon_years)}. Wybrane {num(pv_kwp, 1)} kWp → "
            f"{years_text(payback, options.horizon_years)}."
        )
    else:
        st.markdown(f"- Przy magazynie {battery_kwh} kWh żadna moc PV nie zwraca się w {options.horizon_years} lat.")
    share = m["export"].sum() / production
    st.markdown(
        f"- Przy {num(pv_kwp, 1)} kWp do sieci trafia **{share:.0%}** produkcji, a średnio 1 kWh oddana jest "
        f"warta {pln((result.deposit_used + result.deposit_refund) / max(m['export'].sum(), 1), 2)} "
        f"(wobec {pln(grid_price, 2)} za kWh zużytą w domu)."
    )
    last = marginal_payback.iloc[-1]
    st.markdown(
        f"- Ostatnie 0,5 kWp (z 14,5 na 15 kWp) zwraca się samo w sobie po **"
        f"{years_text(last if last > 0 else None, 99)}** – to pokazuje malejący zysk z przewymiarowania."
    )

# ---------------------------------------------------------------- assumptions
with tab_assumptions:
    w = weather()
    st.markdown("Wszystkie założenia modelu z wartościami i źródłami. Dane źródłowe są zapisane w folderze "
                "`data/` (skrypt `fetch_data.py`), więc każdą liczbę da się odtworzyć.")

    st.subheader("1. Lokalizacja i produkcja PV")
    per_year = w.groupby("year")[["pv_east", "pv_west"]].sum().round(0)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"""
| Założenie | Wartość |
|---|---|
| Lokalizacja | wieś pod Bydgoszczą, 53,12°N 18,00°E |
| Dach | dwuspadowy, nachylenie 15° |
| Połać główna | wschód-południowy wschód, azymut 75° na wschód od południa |
| Połać druga | zachód-północny zachód, azymut 105° na zachód od południa |
| Orientacja | odczytana ze zdjęcia satelitarnego (oś budynku obrócona ok. 15°), dokładność ok. ±10° |
| Udział połaci zachodniej | {west_percent}% (suwak) |
| Dane pogodowe | PVGIS 5.3, baza SARAH3, lata 2019–2023, godzinowe |
| Straty systemu | 14% (domyślne PVGIS: kable, falownik, zabrudzenie) |
| Średni uzysk – połać wschodnia | {num(per_year['pv_east'].mean())} kWh/kWp/rok |
| Średni uzysk – połać zachodnia | {num(per_year['pv_west'].mean())} kWh/kWp/rok |
""")
        st.dataframe(per_year.rename(columns={"pv_east": "Wschód [kWh/kWp]", "pv_west": "Zachód [kWh/kWp]"}),
                     width="stretch")
        st.caption("Różnice między latami to zmienność pogody – wyniki aplikacji to średnia z 5 lat.")
    with c2:
        monthly_pv = w.groupby(["year", "month"])[["pv_east", "pv_west"]].sum().groupby(level=1).mean()
        fig = go.Figure()
        fig.add_bar(x=MONTH_NAMES, y=monthly_pv["pv_east"], name="Połać wschodnia", marker_color=SERIES_COLORS[0],
                    hovertemplate="%{y:,.0f} kWh/kWp")
        fig.add_bar(x=MONTH_NAMES, y=monthly_pv["pv_west"], name="Połać zachodnia", marker_color=SERIES_COLORS[1],
                    hovertemplate="%{y:,.0f} kWh/kWp")
        st.plotly_chart(style_figure(fig, "kWh na 1 kWp", height=380), width="stretch")

    st.subheader("2. Zużycie energii")
    comps = load_components(inputs)
    comp_monthly = pd.DataFrame({
        "Miesiąc": MONTH_NAMES,
        "Bytowe": m["load_household"],
        "Ogrzewanie (pompa)": m["load_heating"],
        "Ciepła woda (pompa)": m["load_hot_water"],
    })
    comp_monthly["Razem"] = comp_monthly[["Bytowe", "Ogrzewanie (pompa)", "Ciepła woda (pompa)"]].sum(axis=1)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"""
| Składnik | Rocznie | Jak rozłożony w czasie |
|---|---:|---|
| Bytowe (2 osoby, praca zdalna) | {kwh(inputs.household_kwh)} | standardowy profil G11 Enea Operator 2026 (godzina po godzinie) |
| Ogrzewanie pompą ciepła | {kwh(inputs.heating_kwh)} | według temperatury: stopniodni poniżej {num(HEATING_BASE_TEMP_C)}°C ÷ COP |
| Ciepła woda | {kwh(inputs.hot_water_kwh)} | stała w roku, więcej rano i wieczorem |
| **Razem** | **{kwh(inputs.annual_consumption_kwh)}** | |
""")
        st.markdown(f"""
**Pompa ciepła (powietrze–woda, ogrzewanie podłogowe):**
- potrzeba ciepła w danym dniu ∝ (15°C − średnia temperatura dnia), zero gdy cieplej;
- prąd = ciepło ÷ COP, gdzie COP = {num(COP_AT_0C, 1)} + {num(COP_SLOPE, 2)} × temperatura
  (np. −10°C → {num(COP_AT_0C - 10 * COP_SLOPE, 1)}, 0°C → {num(COP_AT_0C, 1)}, 7°C → {num(COP_AT_0C + 7 * COP_SLOPE, 1)});
- roczna suma skalowana tak, by średni rok dał {kwh(inputs.heating_kwh)} – zimniejsze lata zużywają więcej.

Wartości roczne to **szacunki** (2600 kWh bytowe dla 2 osób z pracą zdalną, 500 kWh na ciepłą wodę).
Gdy będą dane z licznika, wystarczy wpisać rzeczywiste zużycie roczne.
""")
    with c2:
        fig = go.Figure()
        for i, col in enumerate(["Bytowe", "Ogrzewanie (pompa)", "Ciepła woda (pompa)"]):
            fig.add_bar(x=MONTH_NAMES, y=comp_monthly[col], name=col, marker_color=SERIES_COLORS[i],
                        hovertemplate="%{y:,.0f} kWh")
        fig.update_layout(barmode="stack", bargap=0.25)
        st.plotly_chart(style_figure(fig, "kWh", height=380), width="stretch")

    st.markdown("**Średnia doba: zużycie a produkcja PV** – pokazuje, ile nadwyżki może przejąć magazyn.")
    month_pick = st.selectbox("Miesiąc", MONTH_NAMES, index=5)
    month_no = MONTH_NAMES.index(month_pick) + 1
    hourly = pd.DataFrame({
        "month": w["month"], "hour": w["hour"],
        "load": sum(comps.values()), "pv": pv_per_kwp(inputs.west_share) * inputs.pv_kwp,
    })
    day = hourly[hourly["month"] == month_no].groupby("hour")[["load", "pv"]].mean()
    fig = go.Figure()
    fig.add_bar(x=day.index, y=day["pv"], name=f"Produkcja PV ({num(inputs.pv_kwp, 1)} kWp)",
                marker_color=SERIES_COLORS[3], hovertemplate="%{y:.2f} kWh")
    fig.add_scatter(x=day.index, y=day["load"], name="Zużycie domu", mode="lines+markers",
                    line=dict(color=SERIES_COLORS[0], width=2), marker=dict(size=8),
                    hovertemplate="%{y:.2f} kWh")
    fig.update_xaxes(title="godzina", dtick=2)
    st.plotly_chart(style_figure(fig, "kWh w godzinie", height=340), width="stretch")

    st.subheader("3. Ceny prądu – Enea, taryfa G11, 2026")
    st.markdown(f"""
Ceny dla gospodarstw domowych zatwierdza **Prezes URE** – osobno taryfę sprzedawcy (Enea S.A.) i taryfę
dystrybucyjną (Enea Operator). Obie decyzje z 17.12.2025, obowiązują w 2026 r.

| Składnik | netto [zł/kWh] | brutto [zł/kWh] | Czy pokrywa go depozyt? |
|---|---:|---:|---|
| Energia czynna (Enea S.A.) | 0,4980 + akcyza | **{num(ENEA_ENERGY_PRICE_2026, 4)}** | tak |
| Stawka sieciowa zmienna | 0,2456 | | nie |
| Stawka jakościowa | 0,0331 | | nie |
| Opłata OZE | 0,0073 | | nie |
| Opłata kogeneracyjna | 0,0030 | | nie |
| **Dystrybucja zmienna razem** | 0,2890 | **{num(ENEA_DISTRIBUTION_PRICE_2026, 4)}** | nie |
| **Razem za 1 kWh z sieci** | | **{num(ENEA_ENERGY_PRICE_2026 + ENEA_DISTRIBUTION_PRICE_2026, 4)}** | |

Opłaty stałe (sieciowa stała 7,45 / 10,41 zł netto/mies., abonamentowa 3,84 zł, mocowa 24,05 zł/mies.
przy zużyciu powyżej 2800 kWh) nie zmieniają się po inwestycji, więc ich nie liczymy.
Dla porównania: w 2025 r. taryfa URE wynosiła 622,8 zł/MWh netto, ale gospodarstwa domowe płaciły
ustawowo zamrożone 500 zł/MWh netto. Od 2026 r. mrożenia nie ma. Stawka jakościowa zmieniła się
od 1.02.2026 – różnica rzędu groszy.
""")

    st.subheader("4. Net-billing")
    at_rcem, at_hourly = result.deposit_at_rcem or 0, result.deposit_at_hourly_rce or 0
    pricing_text = ("**RCE godzinowe × 1,23** – średnia cena z 2025 r. dla danego miesiąca i godziny"
                    if hourly_pricing else "**RCEm 2025 × 1,23** – jedna cena na miesiąc")
    st.markdown(f"""
| Założenie | Wartość |
|---|---|
| System | net-billing, nowa instalacja (przyłączenie po 1.07.2024) – rozliczenie po RCE godzinowym |
| Cena energii oddanej | {pricing_text} |
| Współczynnik korekcyjny | {num(NET_BILLING_COEFFICIENT, 2)} (dla rozliczenia po RCE) |
| Ujemne ceny | liczone jako 0 (depozyt nie maleje) |
| Zwrot niewykorzystanego depozytu | do **{REFUND_CAP_SHARE:.0%}** wartości, po 12 miesiącach |
| Co pokrywa depozyt | tylko energię czynną ({pln(inputs.energy_price, 4)}/kWh), nie dystrybucję |
| Bilansowanie | godzinowe – w jednej godzinie pobór i oddanie się znoszą |

**Ceny godzinowe czy RCEm?** Nowa instalacja formalnie rozlicza się po godzinowej cenie RCE. Model używa
**średnich** RCE z 2025 r. dla każdej pary miesiąc × godzina (288 liczb), a nie cen z konkretnych dni –
to „typowy profil dnia”, który lepiej nadaje się na prognozę niż przypadkowy przebieg jednego roku.
RCEm to średnia RCE ważona produkcją fotowoltaiki w Polsce (czerwiec 2025: zwykła średnia ≈ 346 zł/MWh,
ważona produkcją PV ≈ 128, RCEm = 136), więc dobrze wycenia zwykłe oddawanie z paneli, ale nie widzi,
**o której** godzinie oddajemy – a to jest kluczowe dla wieczornej sprzedaży z magazynu.

**Porównanie dla obecnej konfiguracji** (te same godzinowe przepływy energii): depozyt wyceniony cenami
godzinowymi **{pln(at_hourly)}**, wyceniony RCEm **{pln(at_rcem)}** rocznie
(różnica {(at_hourly / at_rcem - 1) if at_rcem else 0:+.0%}). Ceny godzinowe dają więcej, bo wschodnia
połać produkuje rano, gdy prąd jest droższy, a magazyn może sprzedawać wieczorem.
""")
    rce = rce_2025()
    rcem_table = pd.DataFrame({
        "Miesiąc": MONTH_NAMES,
        "RCEm 2025 [zł/kWh]": RCEM_2025,
        "RCEm × 1,23": RCEM_2025 * NET_BILLING_COEFFICIENT,
        "RCE rano (8–9)": rce[:, 8],
        "RCE w południe (12–13)": rce[:, 12],
        "RCE wieczorem (19–20)": rce[:, 19],
    })
    st.caption("Ceny w zł/kWh netto, średnie z 2025 r. (PSE). Do depozytu trafia cena × 1,23.")
    st.dataframe(rcem_table.round(3), hide_index=True, width="stretch")

    st.subheader("5. Dotacja i ulga")
    st.markdown(f"""
**Dotacja PME 2 – Przydomowe Magazyny Energii** (NFOŚiGW, nabór od 20.10.2026 do 1.10.2027
lub wyczerpania budżetu 1 mld zł):
- 30% kosztów kwalifikowanych magazynu, max {num(PME2_PER_KWH)} zł za kWh i max {num(PME2_CAP)} zł (net-billing);
- warunki: magazyn **≥ 10 kWh** i **≥ 2 × moc PV**, system zarządzania energią (EMS), praca wyspowa;
- koszty poniesione od 1.11.2025. Dotacji na samą fotowoltaikę obecnie nie ma.

**Ulga termomodernizacyjna** (PIT):
- koszt PV i magazynu odlicza się od dochodu, do **53 000 zł na każdego współwłaściciela** (limit na całe życie,
  wspólny z innymi wydatkami, np. na pompę ciepła);
- zysk = odliczona kwota × stawka podatku. Aplikacja odlicza najpierw u osoby z wyższą stawką;
- części pokrytej dotacją nie można odliczyć; magazyn można (interpretacja KIS z 2024 r.);
- **uwaga:** 32% dotyczy tylko dochodu powyżej 120 000 zł. Jeśli odliczenie zejdzie poniżej progu,
  ta część da tylko 12%. Odliczenie można rozłożyć na kolejne lata.
""")

    st.subheader("6. Magazyn i obliczenia")
    st.markdown(f"""
| Założenie | Wartość |
|---|---|
| Sprawność magazynu (cykl) | {efficiency:.0%}, straty po połowie przy ładowaniu i rozładowaniu |
| Użyteczna pojemność | {dod:.0%} pojemności nominalnej |
| Moc ładowania/rozładowania | 0,5 × pojemność na godzinę |
| Ładowanie | tylko z nadwyżki PV (bez ładowania z sieci) |
| Kolejność | PV → dom → magazyn → sieć; brak energii: magazyn → sieć |
| Wieczorna sprzedaż | {"włączona" if inputs.evening_export and hourly_pricing else "wyłączona"} – opis poniżej |
| Okres symulacji | 5 lat × 8760 godzin, wynik = średni rok |

**Strategia wieczornej sprzedaży (wymaga EMS, który pozwala oddawać z magazynu do sieci):**
1. **Ładowanie w tanich godzinach** – magazyn ładuje się nadwyżką tylko wtedy, gdy typowa cena w tej godzinie
   jest niska (≤ 1,3 × średnia z 10–15) albo od 14:00. Dzięki temu poranna nadwyżka (droższa) idzie do sieci,
   a do magazynu trafia najtańsza energia z południa.
2. **Sprzedaż wieczorem (17–23)** – tylko w godzinach, w których cena po stratach w magazynie jest wyższa
   niż w południe. Magazyn oddaje do sieci tylko to, co przekracza **rezerwę na noc** = średnie zużycie domu
   w danym miesiącu od następnej godziny do 7:00.
3. Reguły opierają się na typowym profilu cen i zużycia, a nie na znajomości przyszłości – realny EMS
   z prognozą pogody i cen może zrobić to trochę lepiej albo trochę gorzej.

W tej konfiguracji magazyn sprzedaje wieczorem **{kwh(m["battery_to_grid"].sum())}** rocznie, a dla domu
oddaje **{kwh(m["from_battery"].sum())}**.

**Uproszczenia:** brak opłat stałych (takie same przed i po), dotacja i ulga odjęte od nakładu od razu,
wzrost cen dotyczy jednakowo zakupu i cen energii oddanej, degradacja zmniejsza oszczędność proporcjonalnie,
ceny godzinowe to średnie 2025 (bez zależności między pogodą a ceną w danym dniu), profil bytowy to średni
profil G11, a nie Twój licznik, model pompy ciepła jest przybliżony.
""")

    st.subheader("7. Źródła")
    st.markdown("""
- PVGIS 5.3 (Komisja Europejska, JRC) – [re.jrc.ec.europa.eu/pvg_tools](https://re.jrc.ec.europa.eu/pvg_tools/en/)
- Standardowe profile zużycia Enea Operator 2026 (IRiESD, zał. 4) – [operator.enea.pl](https://www.operator.enea.pl/uslugi-dystrybucyjne/iriesd)
- Wyciąg z taryfy Enea Operator 2026 – [PDF](https://www.operator.enea.pl/media/3265/3-umowy-i-rozliczenia/enea-operatorwyciag-z-taryfy01012026-31122026pdf.pdf)
- Taryfa Enea S.A. dla grup G 2026 – [PDF](https://www.enea.pl/media/6667/taryfa-g-2026.pdf)
- Ceny RCE (PSE) – [raporty.pse.pl](https://raporty.pse.pl/)
- Program PME 2 – [przydomowemagazyny.gov.pl](https://przydomowemagazyny.gov.pl/o-programie/)
- Net-billing, RCE/RCEm, współczynnik 1,23 – [Gramwzielone](https://www.gramwzielone.pl/energia-sloneczna/20378973/rozliczenia-prosumentow-net-billing-net-metering)
- Ulga a magazyn energii – [Gramwzielone](https://www.gramwzielone.pl/magazynowanie-energii/20196572/ulga-termomodernizacyjna-a-magazyn-energii-co-moze-odliczyc-prosument)
""")
