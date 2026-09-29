# Prognozowanie wyników meczów piłkarskich 1X2 — kod i wyniki pracy magisterskiej

**Temat:** Modelowanie statystyczne i uczenie maszynowe w prognozowaniu wyników meczów piłkarskich z wykorzystaniem danych historycznych

**Autor:** Michał Olejnik (nr albumu 148210)

## Opis projektu

System probabilistycznego prognozowania wyników meczów w ujęciu 1X2 (wygrana gospodarzy / remis / wygrana gości) budowany na danych historycznych. Praca porównuje modele statystyczne (regresja Poissona, korekta Dixona-Colesa) z modelami uczenia maszynowego (regresja logistyczna, Random Forest, ExtraTrees, XGBoost, CatBoost, LightGBM) i odnosi wyniki do benchmarku rynkowego (prawdopodobieństwa implikowane z kursów bukmacherskich po usunięciu marży).

Ewaluacja prowadzona jest w dwóch torach:

- **Track A** — wyłącznie cechy sportowe (średnie kroczące goli i xG, lambda, dni odpoczynku, rating Elo),
- **Track B** — cechy sportowe + cechy rynkowe (prawdopodobieństwa implikowane, marża, entropia, różnica faworyta).

Protokół: walidacja *rolling-origin* (3 ligi × 3 foldy = 9 podziałów), model uczy się wyłącznie na meczach rozegranych przed okresem testowym. Metryki: *log loss* (główna), *Brier score* (pomocnicza) oraz testy istotności (paired bootstrap, sign-flip randomization test).

## Dane

| Źródło | Zakres | Zawartość |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | 2014/15–2020/21 | wyniki meczów, kursy bukmacherskie (Bet365, Pinnacle) |
| [Understat](https://understat.com) | 2014/15–2020/21 | dane strzałowe z miarą xG |

Ligi: angielska Premier League, hiszpańska LaLiga, niemiecka Bundesliga. Po integracji obu źródeł ok. 7,5 tys. meczów. Pliki w `datasets/`: `{liga}_{sezon}.csv` (wyniki i kursy) oraz `shots_{liga}_{sezon}.csv` (dane strzałowe). Dane wykorzystywane są wyłącznie w celach badawczych; prawa do danych należą do ich dostawców.

## Struktura repozytorium

| Ścieżka | Opis |
|---|---|
| `datasets/` | wyniki meczów, kursy i dane strzałowe (xG) |
| `outputs/` | artefakty etapowe oraz finalne tabele i testy istotności |
| `experiment_utils.py` | wspólne narzędzia (ładowanie danych, cechy, metryki) |
| `pipeline.py` | silnik cech i predykcji używany przez aplikację demo |
| `experiment_stage*.py`, `experiment_week*.py` | skrypty etapów badawczych (rolling-origin) |
| `build_day*.py`, `run_final_pipeline.py` | warstwa raportowa (finalne tabele) |
| `app_final.py` | aplikacja demonstracyjna (Streamlit) |

## Wymagania i instalacja

Python 3.9 lub nowszy:

```bash
pip install -r requirements.txt
```

## Odtworzenie wyników

Finalne tabele i testy istotności (warstwa raportowa nie trenuje modeli — korzysta z artefaktów w `outputs/`):

```bash
python run_final_pipeline.py --mode both
```

Kontrola spójności i odtwarzalności:

```bash
python build_day7_freeze.py
```

Ponowne wytrenowanie wybranego etapu (przykład):

```bash
python experiment_stageI5_modelzoo_reentry.py
```

## Wyniki finalne (log loss, średnia z 9 podziałów)

| Tor | Model | Log loss | Brier |
|---|---|---|---|
| A | Poisson (cechy objaśniające) | 1,0040 | 0,5966 |
| A | ExtraTrees | 1,0041 | 0,5993 |
| A | model bazowy (częstościowy) | 1,0752 | 0,6510 |
| B | **Bookmaker implied (benchmark)** | **0,9755** | **0,5789** |
| B | ExtraTrees + cechy rynkowe | 0,9901 | 0,5888 |
| B | Poisson + Dixon-Coles + cechy rynkowe | 0,9926 | 0,5890 |

Różnica między finalistami Track A nie jest istotna statystycznie (p = 0,98), a przewaga benchmarku rynkowego nad modelami autorskimi w Track B jest istotna (p = 0,012 oraz p = 0,008).

## Aplikacja demonstracyjna

```bash
streamlit run app_final.py
```

Zakładka predykcji ma charakter poglądowy — używa uproszczonego modelu opartego na xG, a nie finalnych modeli opisanych w pracy. Wyniki naukowe znajdują się w tabelach `outputs/final_*.csv`.
