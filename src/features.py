"""
features.py — Transforma resultados brutos em features para o modelo.

Pega os DataFrames crus (uma linha por piloto-corrida) e cria:
  - o alvo (target): pódio sim/não
  - features que só usam informação do PASSADO (sem data leakage)

Regra de ouro contra leakage: ordenar por (ano, rodada), aplicar .shift(1)
para excluir a corrida atual, e SÓ ENTÃO calcular médias/históricos. Nenhuma
feature pode enxergar o resultado da corrida que estamos tentando prever.
"""

import pandas as pd

from store import load_season  # lê cada temporada salva em data/raw/

# Quantas corridas anteriores entram no cálculo de "forma recente" do piloto.
# 3 é um meio-termo: curto o bastante p/ refletir o momento atual, longo o
# bastante p/ não oscilar a cada corrida isolada.
JANELA_FORMA = 3


def carregar_temporadas(anos) -> pd.DataFrame:
    """
    Carrega várias temporadas do disco e junta tudo num só DataFrame.

    Args:
        anos: iterável de anos (ex: [2021, 2022, 2023, 2024]).

    Returns:
        DataFrame com todas as temporadas concatenadas e ordenadas no tempo.
    """
    # Lê cada temporada do Parquet e guarda numa lista de DataFrames
    frames = [load_season(ano) for ano in anos]

    # Junta todas as temporadas empilhando as linhas (uma embaixo da outra)
    df = pd.concat(frames, ignore_index=True)

    # Ordena no tempo: primeiro por ano, depois por rodada dentro do ano.
    # Essencial — todo o cálculo anti-leakage depende dessa ordem cronológica.
    df = df.sort_values(["year", "round"]).reset_index(drop=True)

    return df


def adicionar_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna alvo: pódio (1) ou não (0).

    Args:
        df: DataFrame com a coluna 'position' (posição final).

    Returns:
        O mesmo DataFrame com a coluna 'podium' adicionada.
    """
    # position <= 3 vira True/False; .astype(int) converte para 1/0.
    # Esse é o que o modelo vai prever — não pode virar feature de si mesmo.
    df["podium"] = (df["position"] <= 3).astype(int)
    return df


def adicionar_forma_piloto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forma recente do piloto: média de posição e de pontos nas últimas corridas.

    Args:
        df: DataFrame já ordenado no tempo.

    Returns:
        DataFrame com 'driver_form_pos' e 'driver_form_points'.
    """
    # Agrupa por piloto: cada piloto tem sua própria sequência temporal.
    grupo = df.groupby("driver_code")

    # Média da POSIÇÃO final nas últimas JANELA_FORMA corridas.
    # shift(1) -> joga a série uma corrida pra frente, removendo a atual.
    # rolling(JANELA) -> janela móvel das corridas anteriores.
    # min_periods=1 -> aceita calcular mesmo com menos de 3 corridas no início.
    df["driver_form_pos"] = grupo["position"].transform(
        lambda s: s.shift(1).rolling(JANELA_FORMA, min_periods=1).mean()
    )

    # Mesma ideia, mas com PONTOS: capta quão bem o piloto vinha pontuando.
    df["driver_form_points"] = grupo["points"].transform(
        lambda s: s.shift(1).rolling(JANELA_FORMA, min_periods=1).mean()
    )

    return df


def adicionar_forma_equipe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forma recente da equipe: média de pontos da equipe por corrida (últimas 3).

    A equipe tem 2 pilotos por corrida, então primeiro somamos os pontos da
    equipe em cada corrida (nível de corrida) e só depois fazemos a média móvel.

    Args:
        df: DataFrame já ordenado no tempo.

    Returns:
        DataFrame com a coluna 'constructor_form'.
    """
    # Soma os pontos dos 2 carros da equipe em cada corrida -> 1 valor por
    # (ano, rodada, equipe). reset_index transforma o resultado de volta em tabela.
    pontos_corrida = (
        df.groupby(["year", "round", "constructor"])["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "constructor_race_points"})
    )

    # Ordena essa tabela de equipe também no tempo, p/ a janela móvel fazer sentido
    pontos_corrida = pontos_corrida.sort_values(["year", "round"])

    # Média móvel dos pontos da equipe nas últimas JANELA_FORMA corridas.
    # De novo shift(1) p/ não incluir a corrida atual (evita leakage).
    pontos_corrida["constructor_form"] = (
        pontos_corrida.groupby("constructor")["constructor_race_points"]
        .transform(lambda s: s.shift(1).rolling(JANELA_FORMA, min_periods=1).mean())
    )

    # Traz a feature de volta p/ a tabela principal casando por (ano, rodada, equipe).
    # how="left" mantém todas as linhas originais de piloto-corrida.
    df = df.merge(
        pontos_corrida[["year", "round", "constructor", "constructor_form"]],
        on=["year", "round", "constructor"],
        how="left",
    )

    return df


def adicionar_historico_circuito(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melhor posição que o piloto já fez NAQUELE circuito, em corridas passadas.

    Args:
        df: DataFrame já ordenado no tempo.

    Returns:
        DataFrame com a coluna 'circuit_best_pos'.
    """
    # Agrupa por piloto E circuito: queremos só as visitas anteriores do piloto
    # àquela pista específica.
    grupo = df.groupby(["driver_code", "circuit"])

    # expanding().min() -> melhor (menor) posição acumulada até ali.
    # shift(1) -> só corridas ANTERIORES naquele circuito (não a atual).
    df["circuit_best_pos"] = grupo["position"].transform(
        lambda s: s.shift(1).expanding().min()
    )

    return df


def construir_features(anos) -> pd.DataFrame:
    """
    Pipeline completo de features: carrega, cria target e todas as features.

    Args:
        anos: iterável de anos a incluir (ex: [2021, 2022, 2023, 2024]).

    Returns:
        DataFrame pronto para o modelo, com target + features.
    """
    df = carregar_temporadas(anos)        # junta as temporadas no tempo
    df = adicionar_target(df)             # cria a coluna 'podium' (alvo)
    df = adicionar_forma_piloto(df)       # forma recente do piloto
    df = adicionar_forma_equipe(df)       # forma recente da equipe
    df = adicionar_historico_circuito(df) # histórico no circuito
    return df


# Teste rápido do módulo: python src/features.py
if __name__ == "__main__":
    # Constrói as features para as 4 temporadas que estão no disco
    df = construir_features([2021, 2022, 2023, 2024])

    # As colunas de feature que acabamos de criar (+ contexto p/ leitura)
    colunas_feature = [
        "driver_form_pos",
        "driver_form_points",
        "constructor_form",
        "circuit_best_pos",
        "grid",
    ]

    print(f"\nTotal de linhas: {len(df)}")  # deve bater com a soma das temporadas

    # Mostra um exemplo: as últimas 10 linhas com features + alvo.
    # Usamos as últimas porque já têm histórico (as primeiras vêm cheias de NaN).
    print("\nExemplo (últimas 10 linhas):")
    print(
        df[["year", "round", "driver_code"] + colunas_feature + ["position", "podium"]]
        .tail(10)
        .to_string(index=False)
    )

    # Quantos NaN sobraram em cada feature — esperado nas primeiras corridas de
    # cada piloto/circuito. O SimpleImputer trata isso no módulo de treino.
    print("\nValores faltando (NaN) por feature:")
    print(df[colunas_feature].isna().sum().to_string())
