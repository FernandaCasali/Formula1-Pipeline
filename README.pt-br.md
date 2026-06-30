# 🏁 F1 Podium Predictor

**🇧🇷 Português** · [🇬🇧 English](README.md)

Pipeline reprodutível de previsão de pódios da Fórmula 1, ponta a ponta:
**coleta via API → transformação → modelo de ML → camada de apresentação**.

Peça de portfólio focada em **engenharia de dados honesta**: nada de dados
chumbados no código, pesos aprendidos pelo modelo (não chutados) e validação
**temporal** (treinar no passado, testar no presente — sem data leakage).

---

## A ideia em uma frase

Dado o **grid de largada** de uma corrida (definido no quali de sábado) e o
**histórico** das temporadas anteriores, o modelo estima a probabilidade de
cada piloto subir no pódio e devolve o **top 3 previsto** — *antes* da corrida
acontecer.

---

## Pipeline

```
   API jolpi.ca/Ergast
          │
          ▼
   collect.py ───────────► resultados de corrida + grid do quali
          │
          ▼
   store.py ─────────────► data/raw/season_AAAA.parquet  (histórico)
          │
          ▼
   features.py ──────────► forma do piloto/equipe, histórico no circuito,
          │                pontos no campeonato  (tudo via shift → sem leakage)
          ▼
   train.py ─────────────► XGBoost + validação temporal → models/podium_xgb.joblib
          │
          ▼
   predict.py ───────────► top 3 previsto (por corrida salva OU só pelo grid)
          │
          ▼
   app.py (Streamlit) ───► interface "telemetria F1"
```

---

## Módulos (`src/`)

| Arquivo | Responsabilidade |
|---|---|
| **`collect.py`** | Coleta da API. `get_race_results` (resultado de uma corrida), `get_season_results` (temporada inteira, respeitando rate limit), `get_qualifying_grid` (grid de largada a partir do quali — insumo p/ prever antes da corrida). |
| **`store.py`** | Persistência em Parquet (`data/raw/`). Lê/grava temporadas e descobre dinamicamente quais anos estão no disco (`listar_temporadas`). |
| **`collect_seasons.py`** | "Maestro" que orquestra `collect` + `store` para baixar várias temporadas de uma vez. |
| **`features.py`** | Engenharia de features sem leakage (`shift(1)` antes de qualquer média/acumulado). `construir_features_df` aplica a engenharia sobre um DataFrame qualquer — reusado pelo treino e pela previsão pelo grid. |
| **`train.py`** | Treina o XGBoost com **validação temporal** (`TimeSeriesSplit` + holdout na temporada mais recente). Reporta acurácia honesta e a importância aprendida das features. |
| **`predict.py`** | `prever_corrida` (corrida já salva) e `prever_pelo_grid` (só com o grid do quali, simulando o sábado) + `conferir_previsao` (compara com o resultado real). |
| **`app.py`** | Interface Streamlit com tema "telemetria F1" — pódio, grid de probabilidade e painel "por que essa previsão?". Só apresenta; nenhuma regra de modelo vive aqui. |

---

## Features do modelo

Todas calculadas **apenas com informação anterior à largada** (`shift(1)`):

- `grid` — posição de largada (vem do quali, conhecida antes da corrida)
- `driver_form_pos` / `driver_form_points` — forma recente do piloto (últimas 3 corridas)
- `constructor_form` — forma recente da equipe
- `circuit_best_pos` — melhor posição do piloto naquele circuito no passado
- `driver_season_points` / `constructor_season_points` — pontos no campeonato até a véspera

O alvo (`podium`) é `position <= 3`. Os **pesos** de cada feature são aprendidos
pelo XGBoost — nunca chutados à mão.

---

## Como rodar

> Ambiente: Windows + PowerShell. A API original Ergast foi descontinuada;
> usamos o mirror `https://api.jolpi.ca/ergast/f1`.

```powershell
# 1. Ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Coletar o histórico (gera data/raw/season_AAAA.parquet)
python src/collect_seasons.py     # edite o range de anos em ANOS, se quiser

# 3. Treinar o modelo (gera models/podium_xgb.joblib)
python src/train.py

# 4. Prever uma corrida pelo grid do quali e conferir com o real
python src/predict.py

# 5. Abrir a interface
streamlit run src/app.py
```

Cada módulo tem um bloco `if __name__ == "__main__"` para teste isolado —
rode `python src/<modulo>.py` para exercitar só aquela peça.

---

## Previsão "ao vivo": só com o grid de sábado

O caso de uso real do pipeline. No sábado, depois do quali, só existe o grid —
não o resultado. `prever_pelo_grid` injeta esse grid como a "próxima corrida",
calcula as features a partir do passado e ranqueia o pódio. Depois da corrida,
`conferir_previsao` mede o acerto contra o resultado real.

Baseline medido na temporada **2026** (que o modelo usou como holdout — nunca
viu no treino), prevendo cada corrida **apenas com o grid do quali**:

| r | corrida | top-3 previsto | acertos |
|---|---|---|---|
| 1 | Australian | ANT RUS LEC | 3/3 |
| 2 | Chinese | RUS ANT HAM | 3/3 |
| 3 | Japanese | ANT RUS LEC | 2/3 |
| 4 | Miami | ANT LEC VER | 1/3 |
| 5 | Canadian | RUS ANT NOR | 1/3 |
| 6 | Monaco | ANT VER HAM | 2/3 |
| 7 | Barcelona | HAM ANT RUS | 2/3 |
| 8 | Austrian | HAM RUS LEC | 1/3 |

**Média: 1.88/3 por corrida — 15 de 24 vagas de pódio (62%).**

O número é o que é: o modelo se apoia em grid + forma, então acerta bem o pódio
"óbvio" do começo de temporada e erra mais nas corridas com muita troca de
posição no domingo. **Acurácia honesta > número bonito.**

---

## Princípios do projeto

- **Dados sempre da API.** Nunca inserir tempos/posições à mão no código.
- **Pesos aprendidos**, não chutados. Nada de `0.38 * scoreA + ...`.
- **Validação temporal.** Treinar em temporadas passadas, testar na atual.
  Nunca embaralhar (evita leakage).
- **Falha explícita.** `raise_for_status`, checagem de retornos vazios, sem erro
  silencioso.
- **Um módulo, uma responsabilidade.** Cada peça roda e se valida sozinha.

---

## Stack

`requests` · `fastf1` · `pandas` · `numpy` · `xgboost` · `scikit-learn` ·
`streamlit` · `python-dotenv` · `matplotlib`

## Estrutura

```
src/
  collect.py          # coleta da API (resultados + grid do quali)
  collect_seasons.py  # orquestra a coleta de várias temporadas
  store.py            # persistência em Parquet
  features.py         # engenharia de features sem leakage
  train.py            # treino + validação temporal (XGBoost)
  predict.py          # previsão (por corrida salva ou só pelo grid)
  app.py              # interface Streamlit
data/raw/             # temporadas em Parquet (gitignored, recriáveis)
models/               # modelo treinado (gitignored, recriável)
```

> `data/` e `models/` ficam fora do Git porque são **recriáveis** pelo
> pipeline — é só rodar a coleta e o treino de novo.
