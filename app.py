"""Streamlit UI for the PV + battery profitability analysis (calculations live in pv_model.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pv_model import (
    MONTH_NAMES,
    NET_BILLING_COEFFICIENT,
    PV_MONTHLY_SHARE,
    RCEM_2025,
    Inputs,
    PaybackOptions,
    analyze,
    cashflow,
    monthly_consumption,
    payback_years,
    sweep,
)

# Categorical colors in fixed order (colorblind-checked reference palette).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MUTED = "#8a8984"
SWEEP_BATTERIES = [0, 5, 10, 15, 20]
SWEEP_PV = np.arange(4, 15.01, 0.5)

st.set_page_config(page_title="Opłacalność PV + magazyn", page_icon="☀️", layout="wide")


def pln(value: float, decimals: int = 0) -> str:
    """Polish number format: space as thousands separator, comma as decimal separator."""
    text = f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return f"{text} zł"


def kwh(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") + " kWh"


def years_text(value: float | None, horizon: int) -> str:
    return f"> {horizon} lat" if value is None else f"{value:.1f}".replace(".", ",") + " lat"


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


# ---------------------------------------------------------------- sidebar (inputs)
with st.sidebar:
    st.header("Parametry")
    st.subheader("Instalacja")
    pv_kwp = st.slider("Moc instalacji PV [kWp]", 4.0, 15.0, 8.0, 0.5)
    battery_kwh = st.slider("Pojemność magazynu energii [kWh]", 0, 20, 10, 1)

    st.subheader("Zużycie energii")
    annual = st.number_input("Roczne zużycie [kWh]", 2000, 20000, 7000, 100)
    household = st.number_input(
        "w tym zużycie bytowe (bez ogrzewania) [kWh]", 1000, int(annual), min(2800, int(annual)), 100,
        help="Reszta to pompa ciepła, rozłożona na sezon grzewczy (październik–kwiecień).",
    )

    st.subheader("Ceny prądu")
    grid_price = st.number_input("Cena zakupu z sieci [zł/kWh brutto z dystrybucją]", 0.30, 3.00, 1.15, 0.01)
    energy_part = st.number_input(
        "w tym część za energię [zł/kWh brutto]", 0.0, float(grid_price), min(0.62, float(grid_price)), 0.01,
        help="Tylko tę część może pokryć depozyt prosumencki. Dystrybucję płacisz za każdą pobraną kWh.",
    )

    st.subheader("Koszty inwestycji")
    pv_cost = st.number_input("Koszt PV [zł/kWp]", 1000, 10000, 3500, 100)
    battery_cost = st.number_input("Koszt magazynu [zł/kWh]", 500, 6000, 2000, 100)

    with st.expander("Zaawansowane: produkcja i magazyn"):
        pv_yield = st.number_input("Uzysk roczny [kWh/kWp]", 700, 1300, 1000, 10)
        efficiency = st.slider("Sprawność magazynu (ładowanie + rozładowanie) [%]", 70, 100, 90) / 100
        dod = st.slider("Użyteczna pojemność magazynu (DoD) [%]", 50, 100, 90) / 100

    with st.expander("Czas zwrotu: opcje"):
        price_growth = st.slider("Wzrost cen prądu [%/rok]", 0.0, 10.0, 0.0, 0.5) / 100
        degradation = st.slider("Degradacja paneli [%/rok]", 0.0, 1.0, 0.0, 0.1) / 100
        replace_battery = st.checkbox("Uwzględnij wymianę magazynu")
        battery_life = st.slider("Żywotność magazynu [lata]", 8, 20, 15, disabled=not replace_battery)

inputs = Inputs(
    pv_kwp=pv_kwp,
    battery_kwh=float(battery_kwh),
    annual_consumption_kwh=float(annual),
    household_kwh=float(household),
    pv_yield_kwh_per_kwp=float(pv_yield),
    grid_price=float(grid_price),
    energy_price_part=float(energy_part),
    pv_cost_per_kwp=float(pv_cost),
    battery_cost_per_kwh=float(battery_cost),
    battery_round_trip_efficiency=efficiency,
    battery_depth_of_discharge=dod,
)
options = PaybackOptions(
    price_growth=price_growth,
    degradation=degradation,
    battery_life_years=battery_life if replace_battery else None,
)

result = analyze(inputs)
m = result.monthly
payback = payback_years(result, options)
production = m["production"].sum()
self_consumption = m["self_consumption"].sum()

# ---------------------------------------------------------------- header and KPIs
st.title("☀️ Opłacalność fotowoltaiki i magazynu energii")
st.caption(
    "Dom jednorodzinny z pompą ciepła, rozliczenie net-billing, stawki RCEm 2025 × 1,23. "
    "Wyniki to szacunek na podstawie typowych profili, nie oferta."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Roczna oszczędność", pln(result.annual_savings))
k2.metric(
    "Rachunek za prąd (rok)", pln(result.bill_after),
    delta=f"{pln(result.bill_after - result.bill_before)} vs bez PV", delta_color="inverse",
)
k3.metric("Czas zwrotu", years_text(payback, options.horizon_years), help=f"Nakład: {pln(inputs.investment)}")
k4.metric(
    "Autokonsumpcja", f"{self_consumption / production:.0%}" if production else "–",
    help="Część produkcji zużyta w domu (od razu lub przez magazyn).",
)

tab_energy, tab_money, tab_oversize, tab_method = st.tabs(
    ["Bilans energii", "Rachunek i zwrot", "Przewymiarowanie", "Założenia i metoda"]
)

# ---------------------------------------------------------------- energy balance
with tab_energy:
    st.subheader("Bilans miesięczny")
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
                    "Straty w magazynie", "Oddana do sieci", "Pobrana z sieci"],
        "kWh": [m["consumption"].sum(), production, m["direct_use"].sum(), m["from_battery"].sum(),
                m["battery_losses"].sum(), m["export"].sum(), m["import"].sum()],
    })
    yearly["% produkcji"] = (yearly["kWh"] / production).where(yearly.index.isin([2, 3, 4, 5]))
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
    energy_after = m["energy_cost"].sum() - result.deposit_used
    fig = go.Figure()
    fig.add_bar(x=["Przed inwestycją", "Po inwestycji"],
                y=[m["consumption"].sum() * inputs.distribution_price_part, m["distribution_cost"].sum()],
                name="Dystrybucja", marker_color=SERIES_COLORS[0], hovertemplate="%{y:,.0f} zł")
    fig.add_bar(x=["Przed inwestycją", "Po inwestycji"],
                y=[m["consumption"].sum() * inputs.energy_price_part, energy_after],
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
| z tego: autokonsumpcja ({kwh(self_consumption)} × {pln(grid_price, 2)}) | {pln(result.savings_from_self_consumption)} |
| z tego: depozyt wykorzystany na energię | {pln(result.deposit_used)} |
| z tego: zwrot niewykorzystanego depozytu (max 20%) | {pln(result.deposit_refund)} |
| Depozyt przepadający | {pln(result.deposit_forfeited)} |
""")
        if result.deposit_forfeited > 1:
            st.warning(
                f"Depozyt ({pln(result.deposit_total)}) przewyższa koszt energii pobranej z sieci "
                f"({pln(m['energy_cost'].sum())}). Nadwyżka ponad 20% zwrotu przepada – "
                "instalacja jest za duża w stosunku do zużycia."
            )

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
    st.caption(
        f"Nakład: {pln(inputs.investment)} (PV {pln(inputs.pv_kwp * inputs.pv_cost_per_kwp)}, "
        f"magazyn {pln(inputs.battery_investment)}). Prosty czas zwrotu = nakład / roczna oszczędność; "
        "opcje w panelu bocznym zmieniają oszczędności w kolejnych latach."
    )

    with st.expander("Rozliczenie miesięczne"):
        table = pd.DataFrame({
            "Miesiąc": m["month"],
            "RCEm [zł/kWh]": m["rcem"],
            "Wartość 1 kWh oddanej (×1,23)": m["export_value_per_kwh"],
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
    st.markdown(
        "Każda kWh zużyta w domu oszczędza pełną cenę **"
        f"{pln(grid_price, 2)}**. Każda kWh oddana do sieci jest warta tylko **RCEm × 1,23** – "
        f"w czerwcu {pln(RCEM_2025[5] * NET_BILLING_COEFFICIENT, 2)}, w kwietniu "
        f"{pln(RCEM_2025[3] * NET_BILLING_COEFFICIENT, 2)}. Dokładanie paneli zwiększa głównie letnią "
        "nadwyżkę, więc każdy kolejny kWp zarabia mniej. Magazyn przesuwa część tej nadwyżki na wieczór."
    )

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
        st.plotly_chart(battery_lines("summer_export_lost_value", "zł / rok", "%{y:,.0f} zł"),
                        width="stretch")
    st.caption(
        "Utracona wartość = oddane kWh × (cena zakupu − RCEm × 1,23), czyli ile więcej dałaby ta energia, "
        "gdyby udało się ją zużyć w domu. Linia kropkowana = wybrana moc PV. "
        f"Puste miejsca na wykresie zwrotu oznaczają brak zwrotu w {options.horizon_years} lat."
    )

    current = sw[sw["battery_kwh"] == battery_kwh].reset_index(drop=True)
    best = current.loc[current["payback"].idxmin()] if current["payback"].notna().any() else None

    # Marginal analysis: what each extra 0.5 kWp brings compared to what it costs.
    marginal_savings = current["annual_savings"].diff() / current["pv_kwp"].diff()
    marginal_payback = inputs.pv_cost_per_kwp / marginal_savings

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Oszczędność z każdego kolejnego kWp (magazyn {battery_kwh} kWh) [zł/rok]**")
        fig = go.Figure()
        fig.add_bar(x=current["pv_kwp"], y=marginal_savings, name="Oszczędność z kolejnego kWp",
                    marker_color=SERIES_COLORS[0],
                    customdata=marginal_payback,
                    hovertemplate="%{y:,.0f} zł/rok (zwrot tego kWp: %{customdata:.1f} lat)<extra></extra>")
        fig.update_xaxes(title="moc PV [kWp]")
        st.plotly_chart(style_figure(fig, "zł / rok na kWp", height=360), width="stretch")
    with c2:
        st.markdown(f"**Utracona wartość eksportu w miesiącach – obecna konfiguracja [zł]**")
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
            f"- Najkrótszy czas zwrotu przy magazynie {battery_kwh} kWh: **{best['pv_kwp']:g} kWp** "
            f"→ {years_text(best['payback'], options.horizon_years)}. Wybrane {pv_kwp:g} kWp → "
            f"{years_text(payback, options.horizon_years)}."
        )
    else:
        st.markdown(f"- Przy magazynie {battery_kwh} kWh żadna moc PV nie zwraca się w {options.horizon_years} lat.")
    share = result.monthly["export"].sum() / production
    st.markdown(
        f"- Przy {pv_kwp:g} kWp do sieci trafia **{share:.0%}** produkcji, a średnio 1 kWh oddana jest "
        f"warta {pln((result.deposit_used + result.deposit_refund) / max(m['export'].sum(), 1), 2)} "
        f"(wobec {pln(grid_price, 2)} za kWh zużytą w domu)."
    )
    last = marginal_payback.iloc[-1]
    st.markdown(
        f"- Ostatnie 0,5 kWp (z 14,5 na 15 kWp) zwraca się samo w sobie po **"
        f"{years_text(last if last > 0 else None, 99)}** – to pokazuje malejący zysk z przewymiarowania."
    )

# ---------------------------------------------------------------- assumptions
with tab_method:
    st.subheader("Profile miesięczne")
    profile = pd.DataFrame({
        "Miesiąc": MONTH_NAMES,
        "RCEm 2025 [zł/kWh]": RCEM_2025,
        "Wartość 1 kWh oddanej [zł]": RCEM_2025 * NET_BILLING_COEFFICIENT,
        "Produkcja na 1 kWp [kWh]": PV_MONTHLY_SHARE * inputs.pv_yield_kwh_per_kwp,
        "Zużycie domu [kWh]": monthly_consumption(inputs),
    })
    st.dataframe(profile.round(3), hide_index=True, width="stretch")

    st.subheader("Jak liczę")
    st.markdown(f"""
**1. Produkcja i zużycie godzina po godzinie.** Roczny uzysk ({pv_yield} kWh/kWp) dzielę na miesiące,
a każdy miesiąc na typową dobę: produkcja ma kształt łuku od wschodu do zachodu słońca. Są dwa typy dni
o równej częstości – słoneczny (1,5× średniej) i pochmurny (0,5× średniej), żeby magazyn nie był
liczony tak, jakby każdego dnia miał idealnie tyle samo nadwyżki. Zużycie to część bytowa
(szczyty rano i wieczorem) plus pompa ciepła (pracuje całą dobę, głównie od października do kwietnia).

**2. Każda godzina:** PV zasila najpierw dom → nadwyżka ładuje magazyn (do {dod:.0%} pojemności,
moc do 0,5 × pojemność na godzinę) → reszta idzie do sieci. Brak energii pokrywa najpierw magazyn,
potem sieć. Sprawność magazynu {efficiency:.0%} (straty po połowie przy ładowaniu i rozładowaniu).

**3. Net-billing.** Energia oddana w miesiącu trafia do depozytu po cenie **RCEm × 1,23**.
Depozyt opłaca tylko część energetyczną rachunku ({pln(energy_part, 2)}/kWh); dystrybucję
({pln(inputs.distribution_price_part, 2)}/kWh) płacisz za każdą pobraną kWh. Depozyt jest ważny 12 miesięcy,
więc liczę rok jako stan ustalony: depozyt może pokryć cały roczny koszt energii. Nadwyżka jest zwracana
do 20% wartości oddanej energii, reszta przepada.

**4. Oszczędność** = (zużycie − pobrane z sieci) × cena zakupu + wykorzystany depozyt + zwrot.
**Czas zwrotu** = pierwszy rok, w którym suma oszczędności przekracza nakład.

**Uproszczenia:** brak opłat stałych (są takie same przed i po), magazyn nie ładuje się z sieci,
wzrost cen dotyczy jednakowo zakupu i RCEm, degradacja zmniejsza oszczędność proporcjonalnie,
brak dotacji (np. Mój Prąd) i ulgi termomodernizacyjnej, profile są typowe, a nie z Twojego licznika.
""")
