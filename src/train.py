"""
train.py — Treina o modelo de previsão de pódio (XGBoost).

Princípios (do CLAUDE.md):
  - Pesos APRENDIDOS pelo XGBoost, nunca chutados à mão.
  - Validação TEMPORAL: treinar no passado, testar no futuro. Nunca embaralhar.
  - Acurácia honesta > número bonito.

Duas avaliações:
  1) TimeSeriesSplit  -> validação cruzada temporal (intervalo de desempenho).
  2) Holdout final    -> treina nas temporadas passadas, testa na mais
                          recente do dataset (cenário real).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import construir_features  # monta target + features sem leakage
from store import listar_temporadas      # descobre as temporadas salvas no disco

# As colunas que o modelo usa como entrada. NÃO inclui 'position'/'points'/
# 'status' da corrida atual — isso é resultado, viraria leakage.
COLUNAS_FEATURE = [
    "grid",                # posição de largada (vem do quali, conhecida antes)
    "driver_form_pos",     # média de posição do piloto nas últimas corridas
    "driver_form_points",  # média de pontos do piloto nas últimas corridas
    "constructor_form",    # média de pontos da equipe nas últimas corridas
    "circuit_best_pos",    # melhor posição do piloto naquele circuito no passado
    "driver_season_points",       # pontos do piloto no campeonato até a corrida
    "constructor_season_points",  # pontos da equipe no campeonato até a corrida
]

# Coluna alvo: 1 = subiu no pódio, 0 = não.
COLUNA_ALVO = "podium"

# Onde o modelo treinado será salvo. Fica fora do git (gitignored, recriável).
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def montar_modelo(scale_pos_weight: float) -> Pipeline:
    """
    Monta o pipeline Imputer -> XGBoost.

    O Pipeline garante que o Imputer aprenda a média SÓ do treino de cada
    dobra — sem isso, a média do teste vazaria para o treino (data leakage).

    Args:
        scale_pos_weight: razão não-pódio/pódio, p/ compensar o desbalanceamento.

    Returns:
        Pipeline pronto para .fit().
    """
    modelo = Pipeline(steps=[
        # Passo 1: preenche os NaN (estreias sem histórico) com a média da coluna.
        ("imputer", SimpleImputer(strategy="mean")),

        # Passo 2: o classificador. Os hiperparâmetros abaixo são CONFIGURAÇÃO
        # do treino (não pesos do modelo). Os PESOS de fato — quanto cada
        # feature importa — o XGBoost aprende sozinho no .fit().
        ("xgb", XGBClassifier(
            n_estimators=200,            # nº de árvores
            max_depth=4,                 # profundidade — raso evita overfit
            learning_rate=0.05,          # passo do aprendizado
            subsample=0.9,               # amostra de linhas por árvore
            colsample_bytree=0.9,        # amostra de colunas por árvore
            scale_pos_weight=scale_pos_weight,  # compensa pódio ser raro (~15%)
            eval_metric="logloss",       # métrica interna de otimização
            random_state=42,             # reprodutibilidade
        )),
    ])
    return modelo


def calcular_peso_classe(y: pd.Series) -> float:
    """
    Razão não-pódio / pódio, usada no scale_pos_weight.

    NÃO é número mágico: é a proporção real das classes nos dados. Sem ela, o
    modelo "ganha" prevendo que ninguém sobe no pódio (acerta ~85% e é inútil).

    Args:
        y: série do alvo (0/1).

    Returns:
        Razão entre exemplos negativos e positivos.
    """
    n_positivos = (y == 1).sum()  # quantas linhas são pódio
    n_negativos = (y == 0).sum()  # quantas não são
    return n_negativos / n_positivos


def avaliar_temporal(df: pd.DataFrame) -> None:
    """
    Validação cruzada temporal com TimeSeriesSplit.

    Faz cortes cronológicos crescentes (treina passado, testa o trecho seguinte)
    e reporta o desempenho em cada dobra. Dá um intervalo honesto, não um número
    de sorte.

    Args:
        df: DataFrame já com features e alvo, ordenado no tempo.
    """
    # X = features, y = alvo. Já estão na ordem cronológica vinda do features.py.
    X = df[COLUNAS_FEATURE]
    y = df[COLUNA_ALVO]

    # 5 cortes temporais. Cada dobra treina no que veio antes e testa no próximo.
    tscv = TimeSeriesSplit(n_splits=5)

    print("\n=== Validação cruzada temporal (TimeSeriesSplit) ===")

    # enumerate p/ numerar as dobras a partir de 1
    for i, (idx_treino, idx_teste) in enumerate(tscv.split(X), start=1):
        # Separa as linhas de treino e teste por POSIÇÃO (iloc), mantendo a ordem
        X_treino, X_teste = X.iloc[idx_treino], X.iloc[idx_teste]
        y_treino, y_teste = y.iloc[idx_treino], y.iloc[idx_teste]

        # Peso da classe calculado SÓ no treino desta dobra (sem olhar o teste)
        peso = calcular_peso_classe(y_treino)

        # Monta e treina o modelo nesta dobra
        modelo = montar_modelo(peso)
        modelo.fit(X_treino, y_treino)

        # Prevê o teste e mede o recall do pódio (dos que subiram, quantos pegamos)
        y_prev = modelo.predict(X_teste)

        # report como dict p/ extrair só os números da classe pódio ("1")
        report = classification_report(
            y_teste, y_prev, output_dict=True, zero_division=0
        )
        podio = report["1"]  # métricas da classe positiva (pódio)

        print(
            f"Dobra {i}: "
            f"precision={podio['precision']:.2f}  "
            f"recall={podio['recall']:.2f}  "
            f"f1={podio['f1-score']:.2f}  "
            f"(treino={len(idx_treino)}, teste={len(idx_teste)} linhas)"
        )


def avaliar_acertos_podio(modelo: Pipeline, teste: pd.DataFrame) -> None:
    """
    Métrica HONESTA do produto: acertos médios no pódio previsto por corrida.

    O classification_report usa corte de 0.5 por piloto — mas não é assim que o
    pódio é escolhido. Na vida real, ranqueamos os pilotos e cortamos no TOP 3
    (o pódio tem exatamente 3 lugares). Esta função reproduz essa regra corrida
    a corrida e mede quantos dos 3 previstos realmente subiram.

    É a linha de base contra a qual vamos comparar mudanças futuras no modelo:
    se uma feature nova não mexe NESTE número, ela não ajudou o produto.

    Args:
        modelo: pipeline já treinado.
        teste: DataFrame do ano de teste, com 'year', 'round', 'podium' e as
            COLUNAS_FEATURE (precisa das colunas de contexto, não só as features).
    """
    # Trabalhamos numa cópia p/ não sujar o DataFrame original do chamador.
    teste = teste.copy()

    # Probabilidade de pódio de cada piloto (coluna 1 = classe positiva).
    teste["prob_podium"] = modelo.predict_proba(teste[COLUNAS_FEATURE])[:, 1]

    acertos_por_corrida = []  # guarda quantos acertos (0..3) cada corrida teve

    # Cada 'round' é uma corrida independente; agrupamos p/ ranquear dentro dela.
    for _, corrida in teste.groupby("round"):
        # Top 3 pilotos mais prováveis = o pódio que o modelo "apostaria".
        top3 = corrida.nlargest(3, "prob_podium")

        # Dos 3 previstos, quantos realmente subiram (podium == 1).
        acertos = int(top3["podium"].sum())
        acertos_por_corrida.append(acertos)

    n_corridas = len(acertos_por_corrida)          # total de corridas avaliadas
    total_acertos = sum(acertos_por_corrida)       # acertos somados na temporada
    media = total_acertos / n_corridas             # acertos médios por corrida (0..3)

    # Acerto percentual: do total de 3 vagas por corrida, quantas pegamos.
    pct = total_acertos / (3 * n_corridas)

    print("\n=== Acertos no pódio previsto (TOP 3 por corrida) ===")
    print(f"Corridas avaliadas: {n_corridas}")
    print(f"Acerto médio: {media:.2f}/3 por corrida  ({pct:.0%} das vagas de pódio)")
    # Distribuição: em quantas corridas acertamos 3, 2, 1, 0 — mostra a consistência.
    for k in (3, 2, 1, 0):
        qtd = acertos_por_corrida.count(k)
        print(f"  {k}/3 em {qtd:2d} corridas")


def treinar_final(df: pd.DataFrame) -> Pipeline:
    """
    Treino final no cenário real: treina nos anos passados, testa no mais recente.

    O ano de teste é descoberto do próprio dataset (o maior ano presente), não
    chumbado — assim o holdout acompanha sozinho a chegada de temporadas novas.

    Args:
        df: DataFrame já com features e alvo.

    Returns:
        O modelo treinado (pipeline), pronto para ser salvo/usado.
    """
    # Ano de teste = a temporada mais recente do dataset (a "atual").
    ano_teste = int(df["year"].max())

    # Máscara temporal: treino = todos os anos anteriores, teste = o ano atual.
    # Split por ANO (não aleatório) — é o coração da validação temporal.
    treino = df[df["year"] < ano_teste]   # todas as temporadas anteriores
    teste = df[df["year"] == ano_teste]   # a "temporada atual" deste dataset

    # Separa features (X) e alvo (y) de cada conjunto
    X_treino, y_treino = treino[COLUNAS_FEATURE], treino[COLUNA_ALVO]
    X_teste, y_teste = teste[COLUNAS_FEATURE], teste[COLUNA_ALVO]

    # Peso da classe calculado só no treino (não pode olhar 2024)
    peso = calcular_peso_classe(y_treino)

    # Monta e treina o modelo
    modelo = montar_modelo(peso)
    modelo.fit(X_treino, y_treino)

    # Avalia em 2024 — dados que o modelo NUNCA viu no treino
    y_prev = modelo.predict(X_teste)

    print(f"\n=== Holdout final: treino < {ano_teste}, teste {ano_teste} ===")
    print(f"Linhas de treino: {len(X_treino)} | teste: {len(X_teste)}")
    # Relatório completo: precision/recall/f1 das duas classes
    print(classification_report(y_teste, y_prev, zero_division=0))

    # Pesos APRENDIDOS pelo XGBoost: o quanto cada feature pesou na decisão.
    # Acessamos o passo "xgb" dentro do pipeline p/ pegar as importâncias.
    importancias = modelo.named_steps["xgb"].feature_importances_

    # Casa cada importância com o nome da feature e ordena da maior p/ menor
    pares = sorted(zip(COLUNAS_FEATURE, importancias), key=lambda x: x[1], reverse=True)

    print("Importância das features (aprendida pelo XGBoost):")
    for nome, imp in pares:
        print(f"  {nome:20s} {imp:.3f}")

    # Métrica honesta do produto: ranqueia e corta no top 3 corrida a corrida.
    # 'teste' ainda tem year/round/podium (foi fatiado antes de separar X/y).
    avaliar_acertos_podio(modelo, teste)

    return modelo


# Roda o treino completo: python src/train.py
if __name__ == "__main__":
    # 1) Constrói as features de TODAS as temporadas salvas no disco
    df = construir_features(listar_temporadas())

    # 2) Validação cruzada temporal — intervalo honesto de desempenho
    avaliar_temporal(df)

    # 3) Treino final no cenário real e avaliação em 2024
    modelo = treinar_final(df)

    # 4) Salva o modelo treinado em models/ (cria a pasta se não existir)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = MODELS_DIR / "podium_xgb.joblib"
    joblib.dump(modelo, caminho)
    print(f"\nModelo salvo em: {caminho}")
