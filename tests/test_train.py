"""
test_train.py — Testa a lógica pura do treino (sem treinar modelo de verdade).
"""

import pandas as pd

from train import calcular_peso_classe


def test_peso_classe_e_a_razao_negativo_positivo():
    # 3 linhas "não-pódio" (0) e 1 linha "pódio" (1).
    y = pd.Series([0, 0, 0, 1])

    # O peso deve ser negativos/positivos = 3/1 = 3.0.
    # Esse número compensa o pódio ser raro — e é CALCULADO dos dados,
    # nunca chutado (coerente com a filosofia do projeto).
    assert calcular_peso_classe(y) == 3.0
