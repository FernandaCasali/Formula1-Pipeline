"""
test_features.py — Testa a engenharia de features (o módulo mais crítico).

Dois testes:
  1) O alvo (podium) é marcado corretamente.
  2) NÃO há data leakage: a feature de forma do piloto na corrida N só pode
     enxergar corridas ANTERIORES — nunca o resultado da corrida N.
"""

import pandas as pd

# Graças ao pythonpath=["src"] no pyproject.toml, o import por nome funciona.
from features import adicionar_target, adicionar_forma_piloto


def test_target_marca_podio_ate_posicao_3():
    # Monta um DataFrame mínimo só com a coluna que a função usa (position).
    df = pd.DataFrame({"position": [1, 3, 4, 10]})

    # Aplica a função sob teste.
    out = adicionar_target(df)

    # Posições 1 e 3 são pódio (1); 4 e 10 não são (0).
    # É a regra de negócio pura: podium = (position <= 3).
    assert list(out["podium"]) == [1, 1, 0, 0]


def test_forma_piloto_nao_tem_leakage():
    # Um único piloto, 4 corridas EM ORDEM cronológica, posições conhecidas.
    # Montamos de propósito uma sequência limpa (1,2,3,4) p/ a conta ser óbvia.
    df = pd.DataFrame({
        "driver_code": ["VER", "VER", "VER", "VER"],
        "year":        [2024,  2024,  2024,  2024],
        "round":       [1,     2,     3,     4],
        "position":    [1,     2,     3,     4],
        "points":      [25,    18,    15,    12],
    })

    # Calcula a forma recente do piloto (média móvel das posições anteriores).
    out = adicionar_forma_piloto(df)

    # Corrida 1: não existe passado -> a média vem NaN (shift(1) zera o início).
    # Se aqui viesse um número, seria leakage: estaria usando a própria corrida.
    assert pd.isna(out.loc[0, "driver_form_pos"])

    # Corrida 4: a forma é a média das posições das corridas 1,2,3 = (1+2+3)/3 = 2.0.
    # O PULO DO GATO: a posição da corrida 4 (que é 4) NÃO entra na conta.
    # Se o resultado fosse (1+2+3+4)/4 = 2.5, teríamos leakage. Tem que dar 2.0.
    assert out.loc[3, "driver_form_pos"] == 2.0