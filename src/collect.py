"""
collect.py — Coleta de dados da API Ergast (mirror jolpi.ca)

Objetivo deste primeiro passo:
    Buscar o resultado de UMA corrida e devolver como DataFrame.
    Nada de features ou modelo ainda — só garantir que dados reais chegam.
"""

import requests
import pandas as pd

# URL base do mirror da Ergast (a API original foi descontinuada em 2024)
BASE_URL = "https://api.jolpi.ca/ergast/f1"


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


# Esse bloco só roda quando você executa "python src/collect.py" diretamente.
# Serve como um teste rápido do módulo.
if __name__ == "__main__":
    # Vamos buscar a primeira corrida de 2024 como teste
    df = get_race_results(2024, 1)

    print(f"\n{len(df)} pilotos encontrados:\n")
    # Mostra só as colunas mais relevantes para não poluir o terminal
    print(df[["position", "grid", "driver_code", "constructor", "status"]].to_string(index=False))