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

from features import construir_features  # mesma pipeline de features do treino
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
            Precisa incluir o ano-alvo e os anteriores. Padrão: 2021-2024.

    Returns:
        DataFrame com os pilotos ordenados por probabilidade de pódio (maior
        primeiro), já marcando o top 3 previsto e o resultado real p/ conferência.
    """
    # Por padrão usamos as 4 temporadas — o histórico que alimenta as features.
    if anos_contexto is None:
        anos_contexto = [2021, 2022, 2023, 2024]

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


# Teste rápido: prevê uma corrida e compara com o resultado real.
if __name__ == "__main__":
    # Vamos prever uma corrida de 2024 (temporada que o modelo NÃO viu no treino).
    ANO, RODADA = 2024, 10  # GP da Espanha 2024

    resultado = prever_corrida(ANO, RODADA)

    # Nome da corrida p/ o cabeçalho
    nome_corrida = resultado["race_name"].iloc[0]
    print(f"\nPrevisão de pódio — {nome_corrida} ({ANO}, rodada {RODADA})\n")

    # Mostra a fila completa: probabilidade, previsão e resultado real lado a lado
    print(
        resultado[[
            "driver_code",       # piloto
            "grid",              # largada
            "prob_podium",       # probabilidade prevista de pódio
            "pódio_previsto",    # entrou no top 3 previsto?
            "position",          # posição final real
            "podium",            # subiu no pódio de verdade? (1/0)
        ]].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    # Resumo dos acertos: dos 3 previstos, quantos realmente subiram?
    previstos = resultado[resultado["pódio_previsto"]]          # top 3 previsto
    acertos = int(previstos["podium"].sum())                    # quantos subiram
    print(f"\nAcertos no pódio previsto: {acertos}/3")
