# pv-oplacalnosc

Analiza opłacalności fotowoltaiki (z przewymiarowaniem) i magazynu energii dla domu jednorodzinnego
z pompą ciepła pod Bydgoszczą, nowy prosument w net-billingu, Enea G11.

## Stack
Python 3.11 + Streamlit + Plotly. Wybrany, bo użytkownik chciał aplikację w Pythonie/Streamlit:
suwaki i wykresy bez pisania frontendu. Środowisko w `.venv/` (nie commitujemy).

## Komendy (PowerShell, w folderze aplikacji)
- instalacja: `python -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt`
- uruchomienie: `.venv\Scripts\python.exe -m streamlit run app.py` (podgląd: wpis `pv-oplacalnosc`
  w `../.claude/launch.json`, port 8501)
- testy: `.venv\Scripts\python.exe -m pytest`
- odświeżenie danych źródłowych: `.venv\Scripts\python.exe fetch_data.py` (PVGIS, Enea, PSE)

## Struktura
- `pv_model.py` – wszystkie obliczenia (bez UI): symulacja godzinowa, net-billing, dotacja, ulga, zwrot
- `app.py` – interfejs Streamlit (panel boczny, zakładki: bilans, rachunek, przewymiarowanie, założenia)
- `fetch_data.py` – jednorazowe pobranie danych do `data/` (commitujemy CSV, ~2,5 MB)
- `data/pvgis_bydgoszcz.csv` – PVGIS 5.3, 2019–2023, produkcja na 1 kWp (połać wschodnia i zachodnia) + temperatura
- `data/enea_g11_profile_2026.csv` – standardowy profil G11 Enea Operator 2026 (udział w rocznym zużyciu)
- `data/rce_2025_month_hour.csv` – średnie RCE 2025 per miesiąc × godzina (ujemne = 0) – wycena eksportu
- `tests/test_pv_model.py` – testy modelu

## Ustalenia (z użytkownikiem)
- Dom: wieś pod Bydgoszczą (53,12°N 18,00°E), dach dwuspadowy 15°. Orientacja ze zdjęcia satelitarnego
  (oś budynku obrócona ~15°): połać wschodnia azymut −75° (ESE), zachodnia +105° (WNW), ±10°.
  Domyślnie panele na wschodniej, suwak udziału zachodniej.
- Zużycie (szacunek, brak danych z licznika): 7000 kWh razem, 2600 bytowe (2 dorosłych, jedno pracuje
  zdalnie), 500 ciepła woda z pompy (opcja), reszta ogrzewanie. Pompa: stopniodni poniżej 15°C ÷ COP,
  COP = 3,6 + 0,12·T (1,5–5).
- Wycena eksportu: domyślnie **RCE godzinowe × 1,23** = średnie RCE 2025 dla miesiąc × godzina
  (ujemne = 0); RCEm 2025 × 1,23 jako opcja. Użytkownik chciał ceny godzinowe, żeby widzieć efekt
  wieczornej sprzedaży z magazynu (przy RCEm pora oddania nie ma znaczenia).
- Magazyn (opcja, domyślnie wł., tylko przy cenach godzinowych): ładuje się w tanich godzinach
  (cena ≤ 1,3 × średnia 10–15 albo od 14:00), wieczorem 17–23 sprzedaje to, co ponad rezerwę na noc
  (średnie zużycie miesiąca do 7:00), i tylko gdy cena × sprawność > cena w południe.
- Nowa instalacja → zwrot niewykorzystanego depozytu do 30%. Depozyt pokrywa tylko energię czynną.
- Ceny: taryfy Enea 2026 zatwierdzone przez Prezesa URE 17.12.2025: energia 0,6187 zł brutto,
  dystrybucja zmienna 0,2890 netto = 0,3555 brutto. Opłaty stałe pomijamy (bez zmian po inwestycji).
- Dotacja PME 2 (opcja, domyślnie wyłączona): 30% kosztu magazynu, max 800 zł/kWh i 16 000 zł;
  warunek magazyn ≥ 10 kWh i ≥ 2 × kWp.
- Ulga termomodernizacyjna: 2 współwłaścicieli (małżeństwo), stawki 32% i 12%, limit 53 000 zł/os.;
  odliczenie najpierw u osoby z wyższą stawką; dotacja nie podlega odliczeniu.
- Czas zwrotu od nakładu netto (koszt − dotacja − ulga), opcje: wzrost cen, degradacja, wymiana magazynu.
- Magazyn ładuje się tylko z PV, moc 0,5 × pojemność, sprawność 90%, DoD 90%.
