"""
app.py — Camada de apresentação (Streamlit) do previsor de pódio.

Responsabilidade única: interface. Não treina nem calcula features — só
embrulha prever_corrida() do predict.py numa página interativa, agora com
um tema visual "telemetria F1" (preto + vermelho vivo + verde-limão).

Toda a UI rica é HTML/CSS injetado com st.markdown(unsafe_allow_html=True),
montado a partir do DataFrame que prever_corrida() já devolve. Nenhuma regra
de modelo vive aqui.

Rodar com:  streamlit run src/app.py
"""

import streamlit as st

from features import construir_features        # p/ montar o catálogo de corridas
from predict import prever_corrida, carregar_modelo  # motor de previsão + modelo salvo
from store import listar_temporadas            # temporadas disponíveis no disco
from train import COLUNAS_FEATURE              # mesma lista de features do treino

# Anos disponíveis no dataset — o mesmo contexto usado no treino/previsão.
# Descobertos do disco p/ não chumbar a lista (entra ano novo sozinho).
ANOS = listar_temporadas()

# ---------------------------------------------------------------------------
# Paleta F1 (constantes de tema, não dados)
# ---------------------------------------------------------------------------
VERMELHO = "#E10600"   # vermelho corrida (acento primário)
LIMAO = "#C6FF00"      # verde-limão (acerto / sucesso)
FUNDO = "#0A0A0B"      # preto carbono
GRAFITE = "#121214"    # superfície dos cards

# Cores oficiais aproximadas das equipes, casadas por trecho do nome.
# Mantém a identidade visual das escuderias nas barras e cards.
CORES_EQUIPE = {
    "red bull": "#3671C6",
    "mclaren": "#FF8000",
    "mercedes": "#27F4D2",
    "ferrari": "#E8002D",
    "alpine": "#0093CC",
    "aston": "#229971",
    "williams": "#64C4FF",
    "rb ": "#6692FF",
    "alphatauri": "#6692FF",
    "haas": "#B6BABD",
    "sauber": "#52E252",
}


def cor_equipe(nome: str) -> str:
    """Cor da escuderia a partir do nome do construtor (com fallback cinza)."""
    chave = (nome or "").lower()
    for trecho, cor in CORES_EQUIPE.items():
        if trecho in chave:
            return cor
    return "#8A8A92"


def nome_curto(nome_completo: str) -> str:
    """'Max Verstappen' -> 'M. Verstappen' (formato do mockup)."""
    partes = (nome_completo or "").split()
    if len(partes) < 2:
        return nome_completo
    return f"{partes[0][0]}. {' '.join(partes[1:])}"


@st.cache_data
def carregar_catalogo():
    """Lista as corridas disponíveis (ano, rodada, nome) para os dropdowns."""
    df = construir_features(ANOS)
    catalogo = (
        df[["year", "round", "race_name"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .reset_index(drop=True)
    )
    return catalogo


@st.cache_data
def prever(year: int, round_number: int):
    """Previsão de uma corrida, com cache por (ano, rodada)."""
    return prever_corrida(year, round_number)


@st.cache_data
def importancia_features():
    """
    Importância aprendida pelo XGBoost (para o painel 'Por que essa previsão?').

    Lê feature_importances_ do passo 'xgb' dentro do pipeline salvo e devolve
    pares (label_amigável, valor) ordenados do maior p/ o menor.
    """
    rotulos = {
        "grid": "Largada (grid)",
        "driver_form_pos": "Forma do piloto",
        "driver_form_points": "Pontos recentes",
        "constructor_form": "Forma da equipe",
        "circuit_best_pos": "Histórico no circuito",
    }
    try:
        modelo = carregar_modelo()
        imp = modelo.named_steps["xgb"].feature_importances_
        pares = sorted(
            zip(COLUNAS_FEATURE, imp), key=lambda x: x[1], reverse=True
        )
        return [(rotulos.get(c, c), float(v)) for c, v in pares]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# CSS — tema "telemetria F1". String pura (sem f-string) p/ as chaves do CSS
# permanecerem literais.
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Archivo:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Fundo geral da app */
.stApp {
    background:
        radial-gradient(900px 500px at 80% -10%, rgba(225,6,0,0.10), transparent 60%),
        #0A0A0B;
    color: #F5F5F5;
    font-family: 'Archivo', sans-serif;
}
/* Esconde chrome padrão do Streamlit */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }

/* Selectboxes (controles nativos) com cara dark */
div[data-baseweb="select"] > div {
    background: #16161A !important;
    border: 1px solid #26262C !important;
    border-radius: 8px !important;
    color: #F5F5F5 !important;
}
.stSelectbox label { color: #8A8A92 !important; font-size: 11px !important;
    letter-spacing: 1.5px; text-transform: uppercase; font-weight: 700; }

/* Componentes do mockup */
.f1-flag { width:42px; height:42px; border-radius:8px;
    background: conic-gradient(#fff 0 25%, #0A0A0B 0 50%, #fff 0 75%, #0A0A0B 0 100%);
    background-size:14px 14px; border:1px solid #2c2c33; }
.f1-mono { font-family:'JetBrains Mono', monospace; }
.f1-oswald { font-family:'Oswald', sans-serif; }
.f1-card { background:#121214; border:1px solid #26262C; border-radius:12px; }
</style>
"""


# ---------------------------------------------------------------------------
# Builders de HTML (recebem o DataFrame e devolvem strings de markup)
# ---------------------------------------------------------------------------
def html_podio(top3) -> str:
    """3 degraus do pódio (ordem visual P2 · P1 · P3) a partir do top 3."""
    linhas = list(top3.iterrows())  # já vem ordenado por prob desc (0=P1,1=P2,2=P3)
    p1 = linhas[0][1]
    p2 = linhas[1][1]
    p3 = linhas[2][1]

    def degrau(piloto, medalha, posicao, altura, fonte_nome, cor_num, brilho,
               cor_pct, num_fonte):
        cor = cor_equipe(piloto["constructor"])
        pct = f"{piloto['prob_podium'] * 100:.0f}%"
        return f"""
        <div>
          <div style="text-align:center; margin-bottom:14px;">
            <div style="display:inline-flex; align-items:center; gap:6px; padding:5px 12px;
                border-radius:20px; background:{brilho['chip_bg']};
                border:1px solid {brilho['chip_bd']}; font-family:'JetBrains Mono',monospace;
                font-size:12px; color:{brilho['chip_fg']}; font-weight:700;">
                {medalha} {posicao} · {pct}</div>
            <div class="f1-oswald" style="font-weight:700; font-size:{fonte_nome}px;
                margin-top:10px; text-transform:uppercase;">{nome_curto(piloto['driver_name'])}</div>
            <div style="display:flex; align-items:center; justify-content:center; gap:7px; margin-top:4px;">
              <span style="width:9px;height:9px;border-radius:50%;background:{cor};"></span>
              <span style="font-size:12px; color:#8A8A92;">{piloto['constructor']}</span>
            </div>
            <div class="f1-mono" style="font-size:11px; color:#6c6c74; margin-top:6px;">Largada P{int(piloto['grid'])}</div>
          </div>
          <div style="height:{altura}px; border-radius:10px 10px 0 0; background:{brilho['step_bg']};
              border:1px solid {brilho['step_bd']}; border-bottom:none; display:flex;
              align-items:flex-start; justify-content:center; padding-top:16px;
              box-shadow:{brilho['step_sh']};">
            <span class="f1-oswald" style="font-weight:700; font-size:{num_fonte}px; color:{cor_num};">{posicao[1:]}</span>
          </div>
        </div>"""

    estilo_p1 = {
        "chip_bg": "rgba(225,6,0,.14)", "chip_bd": "rgba(225,6,0,.5)", "chip_fg": "#ff5b54",
        "step_bg": "linear-gradient(180deg,#2a1413,#180d0d)", "step_bd": "rgba(225,6,0,.4)",
        "step_sh": "inset 0 3px 0 #E10600, 0 0 50px rgba(225,6,0,.22)",
    }
    estilo_prata = {
        "chip_bg": "#1c1c20", "chip_bd": "#2c2c33", "chip_fg": "#C0C0C8",
        "step_bg": "linear-gradient(180deg,#1f1f24,#17171b)", "step_bd": "#2c2c33",
        "step_sh": "inset 0 3px 0 #C0C0C8",
    }
    estilo_bronze = {
        "chip_bg": "#1c1c20", "chip_bd": "#2c2c33", "chip_fg": "#cd8b5a",
        "step_bg": "linear-gradient(180deg,#1f1f24,#17171b)", "step_bd": "#2c2c33",
        "step_sh": "inset 0 3px 0 #cd8b5a",
    }

    d2 = degrau(p2, "🥈", "P2", 150, 24, "#3a3a42", estilo_prata, "#fff", 54)
    d1 = degrau(p1, "🥇", "P1", 238, 30, VERMELHO, estilo_p1, "#fff", 78)
    d3 = degrau(p3, "🥉", "P3", 120, 24, "#3a3a42", estilo_bronze, "#fff", 48)

    return f"""
    <section style="margin-top:30px; background:linear-gradient(180deg,#121214,#0d0d0f);
        border:1px solid #26262C; border-radius:16px; padding:34px 28px 0; position:relative; overflow:hidden;">
      <div style="position:absolute; top:0; left:0; right:0; height:6px;
          background:conic-gradient(#fff 0 25%, #0A0A0B 0 50%, #fff 0 75%, #0A0A0B 0 100%);
          background-size:18px 6px; opacity:.18;"></div>
      <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:20px; align-items:end;
          max-width:none; margin:0 auto;">
        {d2}{d1}{d3}
      </div>
    </section>"""


def html_grid_prob(resultado) -> str:
    """Lista de todos os pilotos com barra de probabilidade (grid de probabilidade)."""
    linhas = ""
    for i, (_, d) in enumerate(resultado.iterrows()):
        previsto = bool(d["pódio_previsto"])
        subiu = int(d["podium"]) == 1
        cor = cor_equipe(d["constructor"])
        pct_val = d["prob_podium"] * 100
        rank = f"P{i+1}" if previsto else str(i + 1)
        rank_cor = VERMELHO if previsto else "#6c6c74"
        row_bg = "rgba(225,6,0,0.06)" if previsto else "#16161A"
        row_bd = "rgba(225,6,0,0.28)" if previsto else "#1f1f24"
        pct_cor = "#fff" if previsto else "#8A8A92"
        resultado_icone = "✅" if subiu else "·"
        barra_bg = (f"linear-gradient(90deg,{cor},{VERMELHO})" if previsto else cor)
        barra_op = "1" if previsto else "0.5"
        linhas += f"""
        <div style="display:grid; grid-template-columns:30px 150px 1fr 58px 40px; align-items:center;
            gap:12px; padding:8px 10px; border-radius:8px; background:{row_bg}; border:1px solid {row_bd};">
          <span class="f1-oswald" style="font-weight:700; font-size:16px; color:{rank_cor}; text-align:center;">{rank}</span>
          <div style="display:flex; align-items:center; gap:9px; min-width:0;">
            <span style="width:8px;height:8px;border-radius:50%;flex:none;background:{cor};"></span>
            <span class="f1-mono" style="font-weight:700; font-size:13px;">{d['driver_code']}</span>
            <span style="font-size:12px; color:#8A8A92; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{d['driver_name'].split()[-1]}</span>
          </div>
          <div style="height:10px; border-radius:6px; background:#1c1c20; overflow:hidden;">
            <div style="height:100%; width:{pct_val:.0f}%; border-radius:6px; background:{barra_bg}; opacity:{barra_op};"></div>
          </div>
          <span class="f1-mono" style="font-weight:700; font-size:13px; text-align:right; color:{pct_cor};">{pct_val:.0f}%</span>
          <span style="text-align:center; font-size:14px;">{resultado_icone}</span>
        </div>"""

    return f"""
    <section class="f1-card" style="padding:22px 24px; margin-top:34px;">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;">
        <h2 class="f1-oswald" style="font-weight:600; font-size:17px; letter-spacing:1px; text-transform:uppercase; margin:0;">Grid de probabilidade</h2>
        <span class="f1-mono" style="font-size:10px; color:#6c6c74;">PREVISTO ✓ · REAL</span>
      </div>
      <div style="display:flex; flex-direction:column; gap:5px;">{linhas}</div>
    </section>"""


def html_importancia(features, acertos, nomes_previstos) -> str:
    """Painel 'Por que essa previsão?' + bloco Previsto × Real."""
    barras = ""
    if features:
        maximo = max(v for _, v in features) or 1.0
        for label, v in features:
            largura = v / maximo * 100
            barras += f"""
            <div>
              <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">
                <span style="font-size:13px; font-weight:600;">{label}</span>
                <span class="f1-mono" style="font-size:12px; color:{LIMAO}; font-weight:700;">{v*100:.0f}%</span>
              </div>
              <div style="height:5px; border-radius:5px; background:#1c1c20; overflow:hidden;">
                <div style="height:100%; width:{largura:.0f}%; border-radius:5px; background:linear-gradient(90deg,#7a9e00,{LIMAO});"></div>
              </div>
            </div>"""
    else:
        barras = '<div style="color:#8A8A92; font-size:12px;">Modelo não carregado.</div>'

    ordem = " → ".join(nomes_previstos)
    # Ordem: 'Previsto × Real' EM CIMA, 'Por que essa previsão?' embaixo.
    # Fontes em tamanho normal; só os ESPAÇOS (padding/margin/gap/line-height)
    # foram apertados p/ o rodapé alinhar com a base do pódio.
    return f"""
    <div style="display:flex; flex-direction:column; gap:10px; margin-top:30px;">
      <section style="background:linear-gradient(180deg,rgba(198,255,0,.06),#121214); border:1px solid #2c3a14; border-radius:16px; padding:10px 20px;">
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:4px;">
          <span style="width:7px; height:7px; border-radius:50%; background:{LIMAO}; box-shadow:0 0 10px {LIMAO};"></span>
          <h3 class="f1-oswald" style="font-weight:600; font-size:16px; letter-spacing:1px; text-transform:uppercase; margin:0;">Previsto × Real</h3>
        </div>
        <p style="font-size:13px; color:#cfd8c0; margin:0; line-height:1.35;">
          <strong style="color:{LIMAO};">{acertos} de 3</strong> pilotos do pódio previsto subiram de verdade.
          Ordem prevista: <strong style="color:#fff;">{ordem}</strong>.</p>
      </section>
      <section class="f1-card" style="padding:10px 20px;">
        <h3 class="f1-oswald" style="font-weight:600; font-size:16px; letter-spacing:1px; text-transform:uppercase; margin:0 0 2px;">Por que essa previsão?</h3>
        <p style="font-size:13px; color:#8A8A92; margin:0 0 10px; line-height:1.35;">O que mais pesou na decisão do modelo (importância aprendida pelo XGBoost).</p>
        <div style="display:flex; flex-direction:column; gap:7px;">{barras}</div>
      </section>
    </div>"""


def html_strip_e_medidores(nome_corrida, ano, rodada, acertos, confianca) -> str:
    """Faixa do GP + medidores de acertos e confiança (lado direito)."""
    # Barras do medidor de confiança (4 níveis acesos conforme a confiança)
    niveis = {"Alta": 4, "Média": 3, "Baixa": 2}.get(confianca, 2)
    barras_conf = ""
    alturas = ["40%", "65%", "85%", "100%", "100%"]
    for idx, h in enumerate(alturas):
        cor = VERMELHO if idx < niveis else "#2c2c33"
        barras_conf += f'<span style="width:7px; height:{h}; background:{cor}; border-radius:2px;"></span>'

    # Três luzes de acerto (acesas = acertos)
    luzes = ""
    for idx in range(3):
        if idx < acertos:
            luzes += f'<span style="width:10px; height:22px; border-radius:3px; background:{LIMAO}; box-shadow:0 0 10px rgba(198,255,0,.5);"></span>'
        else:
            luzes += '<span style="width:10px; height:22px; border-radius:3px; background:#2c2c33;"></span>'

    return f"""
    <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:24px; margin-top:26px; flex-wrap:wrap;">
      <div style="position:relative; padding-left:18px;">
        <div style="position:absolute; left:0; top:4px; bottom:4px; width:5px; background:{VERMELHO}; border-radius:3px;"></div>
        <div style="font-size:11px; letter-spacing:2px; color:#8A8A92; text-transform:uppercase; font-weight:700;">Pódio previsto</div>
        <h1 class="f1-oswald" style="font-weight:700; font-size:46px; line-height:.95; margin:6px 0 0; text-transform:uppercase; letter-spacing:.5px;">{nome_corrida}</h1>
        <div class="f1-mono" style="font-size:12px; color:#8A8A92; margin-top:8px;">{ano} · Rodada {rodada}</div>
      </div>
      <div style="display:flex; gap:14px;">
        <div class="f1-card" style="padding:14px 18px; min-width:148px;">
          <div style="font-size:10px; letter-spacing:1.5px; color:#8A8A92; text-transform:uppercase; font-weight:700; margin-bottom:10px;">Acertos do pódio</div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="f1-oswald" style="font-weight:700; font-size:32px; line-height:1; color:{LIMAO};">{acertos}</span>
            <span class="f1-oswald" style="font-weight:500; font-size:18px; color:#6c6c74;">/3</span>
            <div style="display:flex; gap:5px; margin-left:6px;">{luzes}</div>
          </div>
        </div>
        <div class="f1-card" style="padding:14px 18px; min-width:148px;">
          <div style="font-size:10px; letter-spacing:1.5px; color:#8A8A92; text-transform:uppercase; font-weight:700; margin-bottom:10px;">Confiança do modelo</div>
          <div style="display:flex; align-items:center; gap:10px;">
            <span class="f1-oswald" style="font-weight:700; font-size:24px; line-height:1; text-transform:uppercase;">{confianca}</span>
            <div style="display:flex; gap:3px; align-items:flex-end; height:22px;">{barras_conf}</div>
          </div>
        </div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Injeção de HTML
# ---------------------------------------------------------------------------
def _limpar_html(markup: str) -> str:
    """
    Remove indentação e linhas em branco do HTML antes de injetar.

    PORQUÊ: o markdown do Streamlit trata linhas indentadas com 4+ espaços
    como BLOCO DE CÓDIGO — e linhas em branco encerram o bloco HTML, fazendo
    o resto virar texto cru na tela (era o bug do pódio aparecendo como código).
    Tiramos o espaço de cada linha e juntamos tudo, sem indentação p/ confundir.
    """
    linhas = [linha.strip() for linha in markup.splitlines()]  # tira a indentação
    return " ".join(linha for linha in linhas if linha)        # descarta linhas vazias


def render(markup: str) -> None:
    """Limpa o HTML e injeta na página (atalho usado em todos os blocos)."""
    st.markdown(_limpar_html(markup), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏁", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# --- Cabeçalho + filtros na MESMA linha: título à esquerda, dropdowns à direita ---
catalogo = carregar_catalogo()
# Colunas: título largo + 2 filtros estreitos encostados na direita
h_titulo, h_ano, h_gp = st.columns([2.6, 0.8, 1.2], gap="small")

with h_titulo:
    # Logo + título (mesmo HTML de antes, agora dentro da 1ª coluna)
    render(
        f"""
        <div style="display:flex; align-items:center; gap:16px; padding-bottom:6px;">
          <div class="f1-flag"></div>
          <div>
            <div class="f1-oswald" style="font-weight:700; font-size:26px; letter-spacing:1px; line-height:1; text-transform:uppercase;">
              F1 Podium <span style="color:{VERMELHO};">Predictor</span></div>
            <div class="f1-mono" style="font-size:11px; color:#8A8A92; letter-spacing:.5px; margin-top:5px;">
              XGBoost · validação temporal · sem data leakage</div>
          </div>
        </div>
        """
    )

with h_ano:
    # Filtro de temporada (canto superior direito)
    ano = st.selectbox("Temporada", ANOS, index=len(ANOS) - 1)

# Corridas só do ano escolhido, p/ alimentar o 2º dropdown
corridas_do_ano = catalogo[catalogo["year"] == ano]

with h_gp:
    # Filtro de Grande Prêmio (ao lado do de temporada)
    rodada = st.selectbox(
        "Grande Prêmio",
        options=corridas_do_ano["round"].tolist(),
        format_func=lambda r: corridas_do_ano.loc[
            corridas_do_ano["round"] == r, "race_name"
        ].iloc[0],
    )

# --- Previsão (cacheada) ---
resultado = prever(ano, rodada)
nome_corrida = resultado["race_name"].iloc[0]
top3 = resultado.head(3)
acertos = int(top3["podium"].sum())

# Confiança = gap de probabilidade entre o 3º e o 4º colocado da fila.
probs = resultado["prob_podium"].tolist()
gap = (probs[2] - probs[3]) if len(probs) > 3 else 0.0
confianca = "Alta" if gap >= 0.10 else ("Média" if gap >= 0.05 else "Baixa")

nomes_previstos = [nome_curto(n) for n in top3["driver_name"].tolist()]
features = importancia_features()

# --- Faixa do GP + medidores (full width no topo) ---
render(html_strip_e_medidores(nome_corrida, ano, rodada, acertos, confianca))

# --- Pódio (esq) ao lado dos painéis 'por que' / 'previsto×real' (dir) ---
col_podio, col_lado = st.columns([1, 0.42], gap="small")
with col_podio:
    render(html_podio(top3))
with col_lado:
    render(html_importancia(features, acertos, nomes_previstos))

# --- Grid de probabilidade sozinho embaixo (margem aplicada no próprio section) ---
render(html_grid_prob(resultado))

render(
    '<div style="text-align:center; margin-top:26px;" class="f1-mono">'
    '<span style="font-size:10px; color:#4a4a52; letter-spacing:.5px;">'
    'F1 PODIUM PREDICTOR · CAMADA DE APRESENTAÇÃO</span></div>'
)
