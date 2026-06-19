"""
app.py — Camada de apresentação (Streamlit) do previsor de pódio.

Responsabilidade única: interface. Não treina nem calcula features — só
embrulha prever_corrida() do predict.py numa página interativa.

Rodar com:  streamlit run src/app.py
"""

import matplotlib.pyplot as plt
import streamlit as st

from features import construir_features  # p/ montar o catálogo de corridas
from predict import prever_corrida       # o motor de previsão (reusado, não copiado)

# Anos disponíveis no dataset — o mesmo contexto usado no treino/previsão.
ANOS = [2021, 2022, 2023, 2024]


@st.cache_data
def carregar_catalogo():
    """
    Lista as corridas disponíveis (ano, rodada, nome) para os dropdowns.

    @st.cache_data -> roda uma vez e guarda o resultado; cliques seguintes
    reaproveitam, deixando a interface fluida.

    Returns:
        DataFrame com colunas year, round, race_name (sem duplicatas).
    """
    # Reconstrói as features só p/ extrair a lista de corridas existentes
    df = construir_features(ANOS)

    # Uma linha por corrida (tira as 20 linhas de piloto), ordenada no tempo
    catalogo = (
        df[["year", "round", "race_name"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .reset_index(drop=True)
    )
    return catalogo


@st.cache_data
def prever(year: int, round_number: int):
    """
    Previsão de uma corrida, com cache por (ano, rodada).

    Args:
        year: ano da corrida.
        round_number: rodada no calendário.

    Returns:
        DataFrame de prever_corrida (pilotos ranqueados por prob. de pódio).
    """
    # Delega 100% ao predict.py — a app não tem lógica de modelo própria
    return prever_corrida(year, round_number)


# ---------------------------------------------------------------------------
# Layout da página
# ---------------------------------------------------------------------------

# Configuração geral da aba do navegador
st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏁", layout="wide")

st.title("🏁 F1 Podium Predictor")
st.caption("Previsão de pódio via XGBoost — validação temporal, sem data leakage.")

# Carrega o catálogo de corridas (cacheado)
catalogo = carregar_catalogo()

# --- Sidebar: seleção da corrida ---
st.sidebar.header("Escolha a corrida")

# Dropdown de ano
ano = st.sidebar.selectbox("Temporada", ANOS, index=len(ANOS) - 1)

# Corridas só do ano escolhido, p/ o segundo dropdown
corridas_do_ano = catalogo[catalogo["year"] == ano]

# Dropdown de corrida: mostra o nome, mas guardamos o número da rodada.
# format_func converte o número da rodada no nome legível do GP.
rodada = st.sidebar.selectbox(
    "Grande Prêmio",
    options=corridas_do_ano["round"].tolist(),
    format_func=lambda r: corridas_do_ano.loc[
        corridas_do_ano["round"] == r, "race_name"
    ].iloc[0],
)

# --- Previsão ---
resultado = prever(ano, rodada)              # roda o modelo (cacheado)
nome_corrida = resultado["race_name"].iloc[0]  # nome p/ os títulos

st.subheader(f"Pódio previsto — {nome_corrida} ({ano})")

# Os 3 pilotos mais prováveis = pódio previsto
top3 = resultado.head(3)

# Medalhas p/ cada posição do pódio
medalhas = ["🥇", "🥈", "🥉"]

# Três colunas lado a lado, uma por posição do pódio
colunas = st.columns(3)
for coluna, medalha, (_, piloto) in zip(colunas, medalhas, top3.iterrows()):
    # st.metric mostra o piloto em destaque e a probabilidade como "delta"
    coluna.metric(
        label=f"{medalha} {piloto['driver_name']}",
        value=f"{piloto['prob_podium'] * 100:.0f}%",   # probabilidade de pódio
        delta=f"Largada P{piloto['grid']}",            # de onde larga
        delta_color="off",
    )

# --- Placar de acerto (validação contra o resultado real) ---
acertos = int(top3["podium"].sum())  # dos 3 previstos, quantos subiram de fato
st.info(f"**Acertos:** {acertos}/3 pilotos do pódio previsto subiram de verdade.")

# --- Gráfico de barras: probabilidade de todos os pilotos ---
st.subheader("Probabilidade de pódio — todos os pilotos")

# Ordena do menor p/ o maior só para a barra ficar crescente de baixo p/ cima
grafico_df = resultado.sort_values("prob_podium")

fig, ax = plt.subplots(figsize=(8, 6))
# Top 3 em destaque (dourado), o resto em cinza
cores = ["#d4af37" if p else "#cccccc" for p in grafico_df["pódio_previsto"]]
ax.barh(grafico_df["driver_code"], grafico_df["prob_podium"], color=cores)
ax.set_xlabel("Probabilidade de pódio")
ax.set_xlim(0, 1)
st.pyplot(fig)

# --- Tabela completa: previsto vs. real ---
st.subheader("Detalhe por piloto")
st.dataframe(
    resultado[[
        "driver_code", "driver_name", "constructor", "grid",
        "prob_podium", "pódio_previsto", "position", "podium",
    ]],
    use_container_width=True,
    hide_index=True,
)
