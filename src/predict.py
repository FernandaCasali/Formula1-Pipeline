"""
predict.py — Usa o modelo treinado para prever o pódio de uma corrida.

Carrega o modelo salvo (models/podium_xgb.joblib), reconstrói as features da
corrida-alvo (só com informação anterior à largada — sem leakage) e devolve
os pilotos ranqueados pela probabilidade de subir no pódio.

O top 3 dessa fila é o "pódio previsto" — porque pódio tem exatamente 3 lugares,
ranquear e cortar no 3 respeita a regra real da corrida.
"""

from pathlib import Path

import joblib
import pandas as pd

from collect import get_qualifying_grid, get_race_results  # grid do quali e resultado real
from features import construir_features, construir_features_df, carregar_temporadas
from store import listar_temporadas      # temporadas disponíveis no disco
from train import COLUNAS_FEATURE        # reusa a MESMA lista de features do treino

# Caminho do modelo treinado salvo pelo train.py.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELO_PATH = MODELS_DIR / "podium_xgb.joblib"


def carregar_modelo():
    """
    Carrega o pipeline treinado do disco.

    Returns:
        O modelo (Pipeline Imputer + XGBoost) pronto para prever.
    """
    # Falha explícita e útil se o modelo ainda não foi treinado
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"{MODELO_PATH} não existe. Rode o treino primeiro: python src/train.py"
        )

    # joblib.load reconstrói o objeto exatamente como foi salvo
    return joblib.load(MODELO_PATH)


def prever_corrida(year: int, round_number: int, anos_contexto=None) -> pd.DataFrame:
    """
    Prevê a probabilidade de pódio de cada piloto numa corrida específica.

    Args:
        year: ano da corrida a prever (ex: 2024).
        round_number: rodada no calendário (ex: 22).
        anos_contexto: anos usados para reconstruir o histórico das features.
            Precisa incluir o ano-alvo e os anteriores. Padrão: todas as
            temporadas salvas no disco.

    Returns:
        DataFrame com os pilotos ordenados por probabilidade de pódio (maior
        primeiro), já marcando o top 3 previsto e o resultado real p/ conferência.
    """
    # Por padrão usamos todas as temporadas no disco — o histórico que
    # alimenta as features (descoberto dinamicamente, sem chumbar anos).
    if anos_contexto is None:
        anos_contexto = listar_temporadas()

    # Carrega o modelo treinado
    modelo = carregar_modelo()

    # Reconstrói as features de TODAS as corridas do contexto (mesma lógica do treino).
    # A corrida-alvo terá suas features calculadas só com o passado (shift).
    df = construir_features(anos_contexto)

    # Filtra só a corrida que queremos prever
    corrida = df[(df["year"] == year) & (df["round"] == round_number)].copy()

    # Falha explícita se a corrida não existe no dataset
    if corrida.empty:
        raise ValueError(f"Corrida não encontrada: {year}, rodada {round_number}")

    # Separa as features de entrada (as MESMAS colunas do treino)
    X = corrida[COLUNAS_FEATURE]

    # predict_proba devolve [P(não-pódio), P(pódio)]; pegamos a coluna 1 (pódio).
    corrida["prob_podium"] = modelo.predict_proba(X)[:, 1]

    # Ordena os pilotos do mais provável para o menos provável de subir no pódio
    corrida = corrida.sort_values("prob_podium", ascending=False).reset_index(drop=True)

    # Marca o top 3 da fila como o pódio PREVISTO (3 lugares, exatamente).
    # O índice já está 0,1,2,... após o reset -> os 3 primeiros recebem True.
    corrida["pódio_previsto"] = corrida.index < 3

    # 'podium' (real) já veio do features.py -> usamos só p/ conferir os acertos.
    return corrida


def prever_pelo_grid(year: int, round_number: int, anos_contexto=None) -> pd.DataFrame:
    """
    Prevê o pódio de uma corrida usando SÓ o grid do quali — antes da corrida.

    Diferença para prever_corrida: aquela função só funciona para corridas que
    JÁ rodaram e estão salvas (com resultado) no disco. Esta aqui simula o
    cenário real de previsão: sábado à noite temos o grid do quali e nada do
    resultado. Ela:

      1. Carrega só o HISTÓRICO (corridas anteriores à corrida-alvo).
      2. Busca o grid da corrida-alvo no quali (get_qualifying_grid).
      3. Encaixa esse grid como a "próxima corrida" e calcula as features dela
         a partir do passado (mesma engenharia do treino, via shift -> sem leakage).
      4. Ranqueia os pilotos por probabilidade de pódio.

    Garantia anti-leakage: removemos qualquer linha da corrida-alvo que por
    acaso já esteja salva (resultado real) ANTES de injetar o grid. Assim, mesmo
    que a temporada tenha sido recoletada com a corrida já disputada, a previsão
    enxerga apenas o que existia no sábado.

    Args:
        year: ano da corrida-alvo (ex: 2026).
        round_number: rodada da corrida-alvo (ex: 8).
        anos_contexto: anos do histórico. Padrão: todas as temporadas no disco.

    Returns:
        DataFrame dos pilotos ordenados por probabilidade de pódio, com o top 3
        previsto marcado. NÃO traz resultado real (a corrida "ainda não rodou").
    """
    if anos_contexto is None:
        anos_contexto = listar_temporadas()

    modelo = carregar_modelo()

    # Histórico cru de todas as temporadas do contexto, já ordenado no tempo.
    historico = carregar_temporadas(anos_contexto)

    # Anti-leakage: descarta a própria corrida-alvo se ela já estiver salva.
    # Queremos prever como se fosse sábado — sem o resultado do domingo.
    historico = historico[
        ~((historico["year"] == year) & (historico["round"] == round_number))
    ]

    # Grid de largada da corrida-alvo (posição do quali). É o único dado "novo".
    grid = get_qualifying_grid(year, round_number)

    # O grid não tem resultado: position/points ficam NaN (desconhecidos). Como
    # as features usam shift(1) e a corrida-alvo é a ÚLTIMA da fila, esses NaN
    # nunca entram no cálculo das próprias features dela — só o passado entra.
    # Usamos float("nan") (e não pd.NA) p/ a coluna nascer float64 e casar com
    # o dtype do histórico no concat, sem o FutureWarning de coluna all-NA.
    grid["position"] = float("nan")
    grid["points"] = float("nan")

    # Empilha histórico + corrida-alvo e reordena no tempo (a alvo vai pro fim).
    df = pd.concat([historico, grid], ignore_index=True)
    df = df.sort_values(["year", "round"]).reset_index(drop=True)

    # Mesma engenharia de features do treino, agora com a corrida-alvo incluída.
    df = construir_features_df(df)

    # Isola a corrida-alvo já com as features calculadas a partir do passado.
    corrida = df[(df["year"] == year) & (df["round"] == round_number)].copy()
    if corrida.empty:
        raise ValueError(f"Corrida-alvo não encontrada: {year}, rodada {round_number}")

    # Probabilidade de pódio (coluna 1 = classe positiva).
    corrida["prob_podium"] = modelo.predict_proba(corrida[COLUNAS_FEATURE])[:, 1]

    # Ranqueia e marca o top 3 como o pódio previsto (3 vagas, exatamente).
    corrida = corrida.sort_values("prob_podium", ascending=False).reset_index(drop=True)
    corrida["pódio_previsto"] = corrida.index < 3

    return corrida


def conferir_previsao(year: int, round_number: int, previsao: pd.DataFrame) -> pd.DataFrame:
    """
    Compara a previsão feita pelo grid com o RESULTADO REAL da corrida.

    Só faz sentido rodar DEPOIS da corrida: busca o pódio real na API e marca,
    para cada piloto previsto no top 3, se ele realmente subiu.

    Args:
        year, round_number: a corrida-alvo.
        previsao: saída de prever_pelo_grid.

    Returns:
        O top 3 previsto com a coluna 'subiu_no_podio' (posição real <= 3).
    """
    # Resultado real da corrida (já disputada). Traz position por piloto.
    real = get_race_results(year, round_number)

    # Posição final real de cada piloto, p/ casar com a previsão pelo código.
    posicao_real = real.set_index("driver_code")["position"]

    top3 = previsao[previsao["pódio_previsto"]].copy()
    # Posição real do piloto (NaN se ele nem largou/consta no resultado).
    top3["posicao_real"] = top3["driver_code"].map(posicao_real)
    # Subiu no pódio de verdade? (top 3 real)
    top3["subiu_no_podio"] = top3["posicao_real"] <= 3

    return top3


# Teste rápido: prevê a corrida-alvo SÓ com o grid do quali e depois confere.
# Esse é o uso real do pipeline — no sábado só existe o grid; o resultado do
# domingo serve apenas p/ medir o acerto depois.
if __name__ == "__main__":
    ANO, RODADA = 2026, 8  # GP da Áustria 2026 (corrida-alvo)

    # 1) Previsão usando apenas o grid de sábado (sem olhar o resultado).
    previsao = prever_pelo_grid(ANO, RODADA)
    nome_corrida = previsao["race_name"].iloc[0]
    print(f"\nPrevisão de pódio (só com o grid) — {nome_corrida} ({ANO}, rodada {RODADA})\n")
    print(
        previsao[[
            "driver_code",     # piloto
            "grid",            # largada (vinda do quali)
            "prob_podium",     # probabilidade prevista de pódio
            "pódio_previsto",  # entrou no top 3 previsto?
        ]].head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    # 2) Conferência: agora que a corrida rodou, quantos dos 3 previstos subiram?
    conferencia = conferir_previsao(ANO, RODADA, previsao)
    print("\nConferência contra o resultado real:")
    print(
        conferencia[[
            "driver_code",     # piloto previsto no pódio
            "grid",            # de onde largou
            "posicao_real",    # onde terminou de verdade
            "subiu_no_podio",  # bateu o pódio? (posição real <= 3)
        ]].to_string(index=False, float_format=lambda x: f"{x:.0f}")
    )
    acertos = int(conferencia["subiu_no_podio"].sum())
    print(f"\nAcertos no pódio previsto: {acertos}/3")
