"""
test_collect.py — Testa o parsing da coleta SEM chamar a API de verdade.

Técnica: "mock". Substituímos requests.get por uma função falsa que devolve
um JSON controlado. Assim testamos só a SUA lógica de parsing, de forma
rápida e offline — a internet nunca é tocada.
"""

import collect


class RespostaFalsa:
    """Imita o objeto que requests.get devolve, com só o que a função usa."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        # Fingimos status 200 (OK): não faz nada, não levanta erro.
        pass

    def json(self):
        # Devolve o JSON controlado que passamos no teste.
        return self._payload


def test_get_race_results_faz_o_parsing(monkeypatch):
    # JSON mínimo no formato aninhado da Ergast, com 1 piloto.
    payload = {
        "MRData": {
            "RaceTable": {
                "Races": [
                    {
                        "raceName": "GP Teste",
                        "Circuit": {"circuitName": "Circuito Teste"},
                        "Results": [
                            {
                                "Driver": {
                                    "code": "VER",
                                    "driverId": "max_verstappen",
                                    "givenName": "Max",
                                    "familyName": "Verstappen",
                                },
                                "Constructor": {"name": "Red Bull"},
                                "grid": "1",
                                "position": "1",
                                "status": "Finished",
                                "points": "25",
                            }
                        ],
                    }
                ]
            }
        }
    }

    # Troca requests.get por uma função que ignora os argumentos e devolve
    # nossa resposta falsa. O monkeypatch desfaz isso sozinho no fim do teste.
    monkeypatch.setattr(collect.requests, "get", lambda *args, **kwargs: RespostaFalsa(payload))

    # Chama a função real — ela "acha" que falou com a API, mas é tudo fake.
    df = collect.get_race_results(2024, 1)

    # Verifica que o parsing extraiu os campos certos.
    assert len(df) == 1
    assert df.loc[0, "driver_code"] == "VER"
    assert df.loc[0, "position"] == 1  # veio como int (o código converte)
    assert df.loc[0, "constructor"] == "Red Bull"
