# 🏁 F1 Podium Predictor

[🇧🇷 Português](README.pt-br.md) · **🇬🇧 English**

A reproducible, end-to-end Formula 1 podium prediction pipeline:
**API collection → transformation → ML model → presentation layer**.

A portfolio piece focused on **honest data engineering**: no data hardcoded in
the source, weights learned by the model (never hand-tuned), and **temporal**
validation (train on the past, test on the present — no data leakage).

---

## The idea in one sentence

Given a race's **starting grid** (set in Saturday qualifying) and the **history**
of previous seasons, the model estimates each driver's probability of finishing
on the podium and returns the **predicted top 3** — *before* the race happens.

---

## Pipeline

```
   jolpi.ca/Ergast API
          │
          ▼
   collect.py ───────────► race results + qualifying grid
          │
          ▼
   store.py ─────────────► data/raw/season_YYYY.parquet  (history)
          │
          ▼
   features.py ──────────► driver/team form, circuit history,
          │                championship points  (all via shift → no leakage)
          ▼
   train.py ─────────────► XGBoost + temporal validation → models/podium_xgb.joblib
          │
          ▼
   predict.py ───────────► predicted top 3 (from a saved race OR grid-only)
          │
          ▼
   app.py (Streamlit) ───► "F1 telemetry" interface
```

---

## Modules (`src/`)

| File | Responsibility |
|---|---|
| **`collect.py`** | API collection. `get_race_results` (one race's result), `get_season_results` (a full season, respecting the rate limit), `get_qualifying_grid` (starting grid from qualifying — the input for predicting before a race). |
| **`store.py`** | Parquet persistence (`data/raw/`). Reads/writes seasons and dynamically discovers which years are on disk (`listar_temporadas`). |
| **`collect_seasons.py`** | The "conductor" that orchestrates `collect` + `store` to download several seasons at once. |
| **`features.py`** | Leakage-free feature engineering (`shift(1)` before any average/cumulative). `construir_features_df` applies the engineering to any DataFrame — reused by both training and grid-only prediction. |
| **`train.py`** | Trains XGBoost with **temporal validation** (`TimeSeriesSplit` + holdout on the most recent season). Reports honest accuracy and the learned feature importances. |
| **`predict.py`** | `prever_corrida` (a saved race) and `prever_pelo_grid` (grid-only, simulating Saturday) + `conferir_previsao` (compares against the real result). |
| **`app.py`** | Streamlit interface with an "F1 telemetry" theme — podium, probability grid, and a "why this prediction?" panel. Presentation only; no model logic lives here. |

---

## Model features

All computed using **pre-race information only** (`shift(1)`):

- `grid` — starting position (from qualifying, known before the race)
- `driver_form_pos` / `driver_form_points` — driver's recent form (last 3 races)
- `constructor_form` — team's recent form
- `circuit_best_pos` — driver's best past finish at that circuit
- `driver_season_points` / `constructor_season_points` — championship points up to race eve

The target (`podium`) is `position <= 3`. Each feature's **weight** is learned by
XGBoost — never hand-tuned.

---

## How to run

> Environment: Windows + PowerShell. The original Ergast API was discontinued;
> we use the mirror `https://api.jolpi.ca/ergast/f1`.

```powershell
# 1. Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Collect the history (creates data/raw/season_YYYY.parquet)
python src/collect_seasons.py     # edit the year range in ANOS if you want

# 3. Train the model (creates models/podium_xgb.joblib)
python src/train.py

# 4. Predict a race from the qualifying grid and check against the real result
python src/predict.py

# 5. Launch the interface
streamlit run src/app.py
```

Every module has an `if __name__ == "__main__"` block for isolated testing —
run `python src/<module>.py` to exercise just that piece.

---

## "Live" prediction: grid-only, from Saturday

The pipeline's real use case. On Saturday, after qualifying, only the grid
exists — not the result. `prever_pelo_grid` injects that grid as the "next race",
computes features from the past, and ranks the podium. After the race,
`conferir_previsao` measures accuracy against the real result.

Baseline measured on the **2026** season (used by the model as a holdout — never
seen during training), predicting each race using **only the qualifying grid**:

| r | race | predicted top 3 | hits |
|---|---|---|---|
| 1 | Australian | ANT RUS LEC | 3/3 |
| 2 | Chinese | RUS ANT HAM | 3/3 |
| 3 | Japanese | ANT RUS LEC | 2/3 |
| 4 | Miami | ANT LEC VER | 1/3 |
| 5 | Canadian | RUS ANT NOR | 1/3 |
| 6 | Monaco | ANT VER HAM | 2/3 |
| 7 | Barcelona | HAM ANT RUS | 2/3 |
| 8 | Austrian | HAM RUS LEC | 1/3 |

**Average: 1.88/3 per race — 15 of 24 podium slots (62%).**

The number is what it is: the model leans on grid + form, so it nails the
"obvious" early-season podium and misses more in races with heavy position
changes on Sunday. **Honest accuracy > a pretty number.**

---

## Project principles

- **Data always from the API.** Never hardcode times/positions in the source.
- **Learned weights**, not guessed. No `0.38 * scoreA + ...`.
- **Temporal validation.** Train on past seasons, test on the current one.
  Never shuffle (avoids leakage).
- **Fail loudly.** `raise_for_status`, empty-return checks, no silent errors.
- **One module, one responsibility.** Each piece runs and validates on its own.

---

## Stack

`requests` · `fastf1` · `pandas` · `numpy` · `xgboost` · `scikit-learn` ·
`streamlit` · `python-dotenv` · `matplotlib`

## Structure

```
src/
  collect.py          # API collection (results + qualifying grid)
  collect_seasons.py  # orchestrates multi-season collection
  store.py            # Parquet persistence
  features.py         # leakage-free feature engineering
  train.py            # training + temporal validation (XGBoost)
  predict.py          # prediction (from a saved race or grid-only)
  app.py              # Streamlit interface
data/raw/             # seasons in Parquet (gitignored, reproducible)
models/               # trained model (gitignored, reproducible)
```

> `data/` and `models/` are kept out of Git because they are **reproducible**
> by the pipeline — just rerun collection and training.
