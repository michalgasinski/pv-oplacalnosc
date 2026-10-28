"""Download source data once and store it as small CSV files in data/.

Run: .venv\\Scripts\\python.exe fetch_data.py

Sources:
- PVGIS 5.3 (EU JRC) hourly PV output and air temperature for a site near Bydgoszcz.
- Enea Operator standard G11 load profile for 2026 (IRiESD, annex 4).
- PSE hourly market prices RCE for 2025, averaged per month and hour of day.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import openpyxl
import pandas as pd

DATA = Path(__file__).parent / "data"
LAT, LON = 53.12, 18.00  # village near Bydgoszcz
TILT = 15
ROOF_AZIMUTHS = {"pv_east": -75, "pv_west": 105}  # PVGIS convention: 0 = south, -90 = east, 90 = west
WEATHER_YEARS = (2019, 2023)
ENEA_PROFILE_URL = "https://www.operator.enea.pl/media/3253/2025/zalacznik4iriesdprofile2026xlsx.xlsx"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def fetch_pvgis() -> pd.DataFrame:
    """Hourly PV output [kW per 1 kWp] for both roof slopes plus air temperature, in local time."""
    frames = {}
    for column, aspect in ROOF_AZIMUTHS.items():
        url = (
            "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?"
            f"lat={LAT}&lon={LON}&startyear={WEATHER_YEARS[0]}&endyear={WEATHER_YEARS[1]}"
            f"&pvcalculation=1&peakpower=1&loss=14&angle={TILT}&aspect={aspect}&outputformat=json"
        )
        hourly = pd.DataFrame(json.loads(get(url))["outputs"]["hourly"])
        frames[column] = hourly
    base = frames["pv_east"]
    # PVGIS time "YYYYMMDD:HHMM" is UTC; values are averages for that hour.
    utc = pd.to_datetime(base["time"], format="%Y%m%d:%H%M", utc=True).dt.floor("h")
    local = utc.dt.tz_convert("Europe/Warsaw")
    frame = pd.DataFrame({
        "time_utc": utc.dt.strftime("%Y-%m-%d %H:%M"),
        "year": local.dt.year,
        "month": local.dt.month,
        "day_of_year": local.dt.day_of_year,
        "hour": local.dt.hour,
        "pv_east": frames["pv_east"]["P"] / 1000,
        "pv_west": frames["pv_west"]["P"] / 1000,
        "temp_c": base["T2m"],
    })
    # The last UTC hours of the final year fall into the next year in local time; drop them.
    return frame[frame["year"].between(*WEATHER_YEARS)]


def fetch_enea_profile() -> pd.DataFrame:
    """Standard G11 profile: share of annual household consumption in each hour of the year."""
    workbook = openpyxl.load_workbook(io.BytesIO(get(ENEA_PROFILE_URL)), read_only=True, data_only=True)
    sheet = next(ws for ws in workbook.worksheets if ws.title.strip().startswith("G11"))
    rows = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if row[1] is None:
            continue
        day = row[1]
        for hour in range(24):  # column "1" = 00:00-01:00 local time
            rows.append({"day_of_year": day.timetuple().tm_yday, "hour": hour, "value": float(row[2 + hour] or 0)})
    profile = pd.DataFrame(rows)
    profile["share"] = profile["value"] / profile["value"].sum()
    return profile[["day_of_year", "hour", "share"]]


def fetch_rce_2025() -> pd.DataFrame:
    """Average hourly RCE price [PLN/MWh] for each month and hour of day in 2025.

    Negative prices are set to zero first, the same way the prosumer deposit treats them.
    """
    records = []
    for month in range(1, 13):
        start = f"2025-{month:02d}-01"
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        url = (
            "https://api.raporty.pse.pl/api/rce-pln?$filter=business_date%20ge%20'"
            f"{start}'%20and%20business_date%20le%20'{end}'&$first=5000"
        )
        records += json.loads(get(url))["value"]
    rce = pd.DataFrame(records)
    rce["month"] = pd.to_datetime(rce["business_date"]).dt.month
    rce["hour"] = rce["period"].str[:2].astype(int)
    rce["price"] = rce["rce_pln"].clip(lower=0)
    return rce.groupby(["month", "hour"], as_index=False)["price"].mean()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    fetch_pvgis().to_csv(DATA / "pvgis_bydgoszcz.csv", index=False)
    fetch_enea_profile().to_csv(DATA / "enea_g11_profile_2026.csv", index=False)
    fetch_rce_2025().to_csv(DATA / "rce_2025_month_hour.csv", index=False)
    for path in sorted(DATA.glob("*.csv")):
        print(path.name, len(pd.read_csv(path)), "rows")


if __name__ == "__main__":
    main()
