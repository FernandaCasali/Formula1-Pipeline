"""
collect.py — Coleta de dados da API Ergast (mirror jolpi.ca)

Objetivo deste primeiro passo:
    Buscar o resultado de UMA corrida e devolver como DataFrame.
    Nada de features ou modelo ainda — só garantir que dados reais chegam.
"""

import time

import requests
import pandas as pd

# URL base do mirror da Ergast (a API original foi descontinuada em 2024)
BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Pausa entre chamadas ao buscar uma temporada inteira.
# O jolpi.ca limita ~4 req/s (e 500/hora). 0.3s nos mantém folgados
# abaixo do teto de burst e evita levar bloqueio no meio da coleta.
REQUEST_DELAY_SECONDS = 0.3


def get_race_results(year: int, round_number: int) -> pd.DataFrame:
    """
    Busca os resultados de uma corrida específica.

    Args:
        year: ano da temporada (ex: 2024)
        round_number: número da rodada no calendário (ex: 1 = primeira corrida)

    Returns:
        DataFrame com uma linha por piloto.
    """
    # Monta a URL final. Ex: .../f1/2024/1/results.json
    url = f"{BASE_URL}/{year}/{round_number}/results.json"

    # Faz a chamada HTTP GET
    response = requests.get(url, timeout=10)

    # Se o status não for 200 (OK), levanta um erro — falha cedo e explícito
    response.raise_for_status()

    # Converte a resposta JSON em um dicionário Python
    data = response.json()

    # A estrutura da Ergast é bem aninhada. Navegamos até a lista de corridas.
    races = data["MRData"]["RaceTable"]["Races"]

    # Se a lista vier vazia, essa corrida não existe (ano/rodada inválidos)
    if not races:
        raise ValueError(f"Nenhuma corrida encontrada para {year}, rodada {round_number}")

    race = races[0]
    results = race["Results"]  # lista de pilotos com seus resultados

    # Extraímos só os campos que interessam de cada piloto
    rows = []
    for r in results:
        rows.append({
            "year": year,
            "round": round_number,
            "race_name": race["raceName"],
            "circuit": race["Circuit"]["circuitName"],
            "driver_code": r["Driver"].get("code", r["Driver"]["driverId"]),
            "driver_name": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
            "constructor": r["Constructor"]["name"],
            "grid": int(r["grid"]),            # posição de largada
            "position": int(r["position"]),    # posição final
            "status": r["status"],             # "Finished", "+1 Lap", "Accident"...
            "points": float(r["points"]),
        })

    # Transforma a lista de dicionários em um DataFrame
    return pd.DataFrame(rows)


def get_qualifying_grid(year: int, round_number: int) -> pd.DataFrame:
    """
    Busca o GRID DE LARGADA de uma corrida a partir do qualifying.

    Por que existe esta função (e não dá p/ usar get_race_results):
        get_race_results só funciona DEPOIS da corrida — o campo 'grid' vem
        junto dos resultados finais. Para prever uma corrida que ainda NÃO
        rodou, a gente só tem a informação que existe na véspera: a ordem do
        grid, definida no quali de sábado. Este é justamente o insumo que o
        modelo precisa (a feature 'grid') p/ ranquear o pódio antes da largada.

    Detalhe honesto: usamos a posição do QUALI como grid. O grid oficial pode
    mudar por punições (perda de posições, troca de motor). Mas no sábado, logo
    após o quali, essa é a melhor informação disponível e sem leakage — nada
    aqui enxerga o resultado da corrida.

    Args:
        year: ano da temporada (ex: 2026).
        round_number: número da rodada-alvo no calendário (ex: 8).

    Returns:
        DataFrame com uma linha por piloto, na ordem de largada. Traz as mesmas
        colunas de identificação de get_race_results (year, round, race_name,
        circuit, driver_code, driver_name, constructor, grid) — mas SEM
        position/points/status, porque a corrida ainda não aconteceu.
    """
    # Endpoint de qualifying. Ex: .../f1/2026/8/qualifying.json
    url = f"{BASE_URL}/{year}/{round_number}/qualifying.json"

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    races = data["MRData"]["RaceTable"]["Races"]

    # Lista vazia = o quali ainda não aconteceu (ou ano/rodada inválidos).
    # Falha explícita: melhor avisar "o quali ainda não saiu" do que devolver
    # um DataFrame vazio que quebraria silenciosamente lá na frente.
    if not races:
        raise ValueError(
            f"Nenhum qualifying encontrado para {year}, rodada {round_number}. "
            "O quali já aconteceu? (sai no sábado da corrida-alvo)"
        )

    race = races[0]
    quali = race["QualifyingResults"]  # lista de pilotos na ordem do quali

    rows = []
    for q in quali:
        rows.append({
            "year": year,
            "round": round_number,
            "race_name": race["raceName"],
            "circuit": race["Circuit"]["circuitName"],
            "driver_code": q["Driver"].get("code", q["Driver"]["driverId"]),
            "driver_name": f"{q['Driver']['givenName']} {q['Driver']['familyName']}",
            "constructor": q["Constructor"]["name"],
            # posição no quali = posição de largada (nosso 'grid' p/ a previsão)
            "grid": int(q["position"]),
        })

    return pd.DataFrame(rows)


def get_season_rounds(year: int) -> int:
    """
    Descobre quantas rodadas uma temporada teve, perguntando à API.

    Em vez de chutar (cada ano tem um número diferente de corridas),
    consultamos o calendário oficial. Isso também evita buscar rodadas
    que ainda não aconteceram numa temporada em andamento.

    Args:
        year: ano da temporada (ex: 2024)

    Returns:
        Número de rodadas no calendário daquele ano.
    """
    # Endpoint do calendário. Ex: .../f1/2024.json
    url = f"{BASE_URL}/{year}.json"

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    races = data["MRData"]["RaceTable"]["Races"]

    # Sem corridas no calendário = ano inválido ou ainda não publicado
    if not races:
        raise ValueError(f"Nenhuma corrida encontrada no calendário de {year}")

    # Cada corrida traz seu próprio "round"; o maior é o total de rodadas.
    return max(int(race["round"]) for race in races)


def get_season_results(year: int) -> pd.DataFrame:
    """
    Coleta os resultados de TODAS as corridas de uma temporada.

    Faz um loop pelas rodadas reaproveitando get_race_results, com uma
    pausa entre chamadas para respeitar o rate limit da API.

    Decisão de resiliência: se uma rodada específica falhar, avisamos no
    terminal e seguimos em frente — uma corrida problemática não deve
    jogar fora as outras 20+ que vieram bem. (Sem erro silencioso: a
    falha aparece, mas não derruba a coleta toda.)

    Args:
        year: ano da temporada (ex: 2024)

    Returns:
        DataFrame com uma linha por piloto-corrida, todas as rodadas
        concatenadas.
    """
    total_rounds = get_season_rounds(year)
    print(f"Temporada {year}: {total_rounds} rodadas no calendário.")

    season_frames = []
    for round_number in range(1, total_rounds + 1):
        try:
            df = get_race_results(year, round_number)
            season_frames.append(df)
            print(f"  rodada {round_number:2d}/{total_rounds} — "
                  f"{df['race_name'].iloc[0]} ({len(df)} pilotos)")
        except (requests.RequestException, ValueError, KeyError) as erro:
            # Logamos e continuamos — não interrompemos a temporada inteira
            print(f"  rodada {round_number:2d}/{total_rounds} — FALHOU: {erro}")

        # Educados com a API: pausa entre chamadas
        time.sleep(REQUEST_DELAY_SECONDS)

    # Se NADA foi coletado, algo está muito errado — falha explícita
    if not season_frames:
        raise RuntimeError(f"Nenhuma corrida coletada para a temporada {year}")

    # ignore_index=True para ter um índice limpo e contínuo no resultado final
    return pd.concat(season_frames, ignore_index=True)


# Esse bloco só roda quando você executa "python src/collect.py" diretamente.
# Serve como um teste rápido do módulo.
if __name__ == "__main__":
    # 1) Resultados de uma temporada já disputada (vira histórico p/ o modelo)
    season = get_season_results(2024)

    print(f"\nTotal: {len(season)} linhas (piloto-corrida) coletadas.\n")
    # Resumo rápido: quantas corridas e pilotos distintos vieram
    print(f"Corridas distintas: {season['round'].nunique()}")
    print(f"Pilotos distintos:  {season['driver_code'].nunique()}")

    # 2) Grid de largada da corrida-alvo (insumo p/ prever ANTES da corrida).
    #    Usamos uma corrida cujo quali já saiu só p/ exercitar a função.
    print("\nGrid de largada (a partir do quali) — exemplo:")
    grid = get_qualifying_grid(2024, 1)
    print(grid[["grid", "driver_code", "constructor"]].head(5).to_string(index=False))