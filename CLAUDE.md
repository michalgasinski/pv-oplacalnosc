# pv-oplacalnosc

Analiza opłacalności fotowoltaiki (z przewymiarowaniem) i magazynu energii dla domu jednorodzinnego
z pompą ciepła w Polsce, rozliczenie net-billing.

## Stack
Python 3.11 + Streamlit + Plotly. Wybrany, bo użytkownik chciał aplikację w Pythonie/Streamlit:
suwaki i wykresy bez pisania frontendu. Środowisko w `.venv/` (nie commitujemy).

## Komendy (PowerShell, w folderze aplikacji)
- instalacja: `python -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt`
- uruchomienie: `.venv\Scripts\python.exe -m streamlit run app.py` (podgląd: wpis `pv-oplacalnosc`
  w `../.claude/launch.json`, port 8501)
- testy: `.venv\Scripts\python.exe -m pytest`

## Struktura
- `pv_model.py` – wszystkie obliczenia (bez UI), stałe: RCEm 2025, profile miesięczne i godzinowe
- `app.py` – interfejs Streamlit (panel boczny, zakładki, wykresy)
- `tests/test_pv_model.py` – testy modelu (bilans energii, depozyt, czas zwrotu)

## Ustalenia
- Symulacja godzinowa: typowa doba dla każdego miesiąca, 2 typy dni (słoneczny 1,5×, pochmurny 0,5×).
- Depozyt = oddane kWh × RCEm × 1,23; pokrywa tylko część energetyczną ceny (domyślnie 0,62 zł z 1,15 zł),
  dystrybucję płaci się zawsze. Rok liczony jako stan ustalony (depozyt ważny 12 mies.),
  nadwyżka zwracana do 20% wartości oddanej energii, reszta przepada.
- Czas zwrotu: prosty + opcje (wzrost cen, degradacja, wymiana magazynu), domyślnie wyłączone.
- Zużycie domyślne 7000 kWh, w tym 2800 kWh bytowe (lato ~200–250 kWh/mies.).
- Magazyn ładuje się tylko z PV, moc 0,5 × pojemność, sprawność 90%, DoD 90%.
- Logika w `pv_model.py`, UI w `app.py` (odstępstwo od pierwotnego pomysłu „jeden plik” – świadomie).
