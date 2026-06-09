"""
app.py — Aplicação Streamlit do GAIE (Generative AI For Engineering).

O GAIE estima a probabilidade/risco de trabalho forçado por região brasileira,
cruzando a detecção de fornos (carvoarias/olarias) via satélite com indicadores
socioeconômicos — uma aplicação do tema "Economia Espacial" ao combate ao
trabalho análogo à escravidão.

Esta interface consome os artefatos produzidos pelo pipeline de ML:
  * o dataset sintético (preprocessing.load_raw);
  * os modelos treinados (bundles joblib de classificação e regressão);
  * os relatórios de métricas (reports/metrics.json);
  * a importância de atributos via SHAP (reports/shap_importance.json);
  * as figuras geradas por train.py e explain.py.

O app é tolerante a falhas: se os modelos/relatórios ainda não existirem, as
seções que dependem deles exibem um aviso amigável em vez de quebrar. Basta o
dataset existir para o app abrir.

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Garante que a raiz do projeto esteja no sys.path para importar o pacote `src`.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config, preprocessing  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GAIE — Risco de Trabalho Forçado",
    layout="wide",
)

# Mensagem padrão quando faltam artefatos de modelagem.
_MSG_SEM_MODELOS = "Modelos não encontrados. Rode: python run_pipeline.py"


# ===========================================================================
# Carregamento de dados / artefatos (com cache e checagem de existência)
# ===========================================================================
@st.cache_data(show_spinner=False)
def carregar_dataset() -> pd.DataFrame | None:
    """Carrega o dataset bruto. Retorna None se o CSV não existir."""
    if not Path(config.DATA_RAW).exists():
        return None
    return preprocessing.load_raw()


@st.cache_data(show_spinner=False)
def carregar_json(caminho_str: str) -> dict | None:
    """Carrega um arquivo JSON de relatório. Retorna None se não existir."""
    caminho = Path(caminho_str)
    if not caminho.exists():
        return None
    with open(caminho, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource(show_spinner=False)
def carregar_bundle(caminho_str: str) -> dict | None:
    """Carrega um bundle de modelo (joblib). Retorna None se não existir."""
    caminho = Path(caminho_str)
    if not caminho.exists():
        return None
    return joblib.load(caminho)


def _categoria_colorida(categoria: str) -> str:
    """Retorna um HTML <span> com a categoria pintada na cor do config."""
    cor = config.RISK_COLORS.get(categoria, "#444444")
    return (
        f"<span style='color:{cor};font-weight:700;font-size:1.6rem'>"
        f"{categoria}</span>"
    )


def _hex_para_rgba(hex_cor: str, alpha: float = 0.2) -> str:
    """Converte '#RRGGBB' em 'rgba(r,g,b,alpha)' (Plotly não aceita hex com alpha)."""
    h = hex_cor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ===========================================================================
# Cabeçalho
# ===========================================================================
def render_cabecalho() -> None:
    """Renderiza o título e o subtítulo explicativo do app."""
    st.title("GAIE — Risco de Trabalho Forçado")
    st.markdown(
        "**Generative AI For Engineering** · Estima a **probabilidade/risco de "
        "trabalho forçado** por região, cruzando a **detecção de fornos "
        "(carvoarias/olarias) por satélite** com **indicadores socioeconômicos** "
        "— uma aplicação do tema **Economia Espacial** ao combate ao trabalho "
        "análogo à escravidão."
    )


# ===========================================================================
# Seção 1 — Mapa de Risco
# ===========================================================================
def render_mapa(df: pd.DataFrame) -> None:
    """Mapa interativo das regiões com filtros e métricas-resumo."""
    st.header("Mapa de Risco")

    # ----------------------------- Filtros --------------------------------
    with st.expander("Filtros", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            ufs_sel = st.multiselect(
                "UF",
                options=sorted(df["uf"].unique().tolist()),
                default=[],
                help="Vazio = todas as UFs.",
            )
            macros_sel = st.multiselect(
                "Macrorregião",
                options=sorted(df["macrorregiao"].unique().tolist()),
                default=[],
                help="Vazio = todas as macrorregiões.",
            )
        with col_b:
            cats_sel = st.multiselect(
                "Categoria de risco",
                options=config.RISK_ORDER,
                default=config.RISK_ORDER,
            )
            score_min, score_max = st.slider(
                "Faixa de score de risco (0-100)",
                min_value=0,
                max_value=100,
                value=(0, 100),
            )
            fornos_min = st.slider(
                "Nº mínimo de fornos detectados",
                min_value=0,
                max_value=int(df["fornos_detectados"].max()),
                value=0,
            )

    # --------------------------- Aplicação ---------------------------------
    filtro = df.copy()
    if ufs_sel:
        filtro = filtro[filtro["uf"].isin(ufs_sel)]
    if macros_sel:
        filtro = filtro[filtro["macrorregiao"].isin(macros_sel)]
    if cats_sel:
        filtro = filtro[filtro["categoria_risco"].astype(str).isin(cats_sel)]
    filtro = filtro[
        (filtro["score_risco"] >= score_min)
        & (filtro["score_risco"] <= score_max)
        & (filtro["fornos_detectados"] >= fornos_min)
    ]

    # --------------------------- Métricas ----------------------------------
    n_reg = len(filtro)
    if n_reg == 0:
        st.warning("Nenhuma região corresponde aos filtros selecionados.")
        return

    pct_alto = (
        100.0
        * (filtro["categoria_risco"].astype(str) == "Alto Risco").sum()
        / n_reg
    )
    total_pessoas = int(filtro["pessoas_em_risco_estimadas"].sum())
    media_score = float(filtro["score_risco"].mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Regiões filtradas", f"{n_reg:,}".replace(",", "."))
    m2.metric("% Alto Risco", f"{pct_alto:.1f}%")
    m3.metric("Pessoas em risco", f"{total_pessoas:,}".replace(",", "."))
    m4.metric("Score médio", f"{media_score:.1f}")

    # ----------------------------- Mapa ------------------------------------
    # px.scatter_map (API MapLibre) substitui o antigo scatter_mapbox (depreciado).
    fig = px.scatter_map(
        filtro,
        lat="latitude",
        lon="longitude",
        color="categoria_risco",
        color_discrete_map=config.RISK_COLORS,
        category_orders={"categoria_risco": config.RISK_ORDER},
        size="pessoas_em_risco_estimadas",
        size_max=22,
        zoom=3,
        hover_name="municipio",
        hover_data={
            "uf": True,
            "score_risco": ":.1f",
            "fornos_detectados": True,
            "idh": ":.3f",
            "pessoas_em_risco_estimadas": True,
            "latitude": False,
            "longitude": False,
            "categoria_risco": False,
        },
        height=560,
    )
    fig.update_layout(
        map_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        legend_title_text="Categoria de risco",
    )
    st.plotly_chart(fig, width="stretch")

    # ----------------------- Tabela + download -----------------------------
    st.subheader("Regiões filtradas")
    colunas_tabela = [
        "regiao_id",
        "municipio",
        "uf",
        "macrorregiao",
        "score_risco",
        "categoria_risco",
        "fornos_detectados",
        "idh",
        "indice_pobreza",
        "pessoas_em_risco_estimadas",
    ]
    colunas_tabela = [c for c in colunas_tabela if c in filtro.columns]
    tabela = filtro[colunas_tabela].sort_values("score_risco", ascending=False)
    st.dataframe(tabela, width="stretch", hide_index=True)

    csv = tabela.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar CSV filtrado",
        data=csv,
        file_name="gaie_regioes_filtradas.csv",
        mime="text/csv",
    )


# ===========================================================================
# Seção 2 — Simulador de Risco
# ===========================================================================
# Rótulos amigáveis e configuração de incremento por atributo base.
_LABELS_BASE = {
    "fornos_detectados": "Fornos detectados (satélite)",
    "densidade_populacional": "Densidade populacional (hab/km²)",
    "pib_per_capita": "PIB per capita (R$/ano)",
    "distancia_centro_urbano_km": "Distância ao centro urbano (km)",
    "indice_pobreza": "Índice de pobreza (0-100)",
    "idh": "IDH (0-1)",
    "taxa_informalidade": "Taxa de informalidade (%)",
    "casos_oit_historicos": "Casos históricos OIT",
    "cobertura_fiscalizacao": "Cobertura de fiscalização (%)",
    "area_desmatada_km2": "Área desmatada (km²)",
    "acesso_educacao": "Acesso à educação (0-100)",
    "populacao_total": "População total",
}


def render_simulador(
    df: pd.DataFrame,
    bundle_clf: dict | None,
    bundle_reg: dict | None,
    shap_data: dict | None,
) -> None:
    """Simulador interativo: entra atributos de uma região e prevê o risco."""
    st.header("Simulador de Risco")

    if bundle_clf is None and bundle_reg is None:
        st.warning(_MSG_SEM_MODELOS)
        return

    st.caption(
        "Ajuste os atributos de uma região hipotética e clique em **Calcular "
        "risco**. As faixas dos controles derivam do próprio dataset."
    )

    # ---------------------- Entradas: atributos base -----------------------
    uf = st.selectbox("UF", options=config.UFS, index=config.UFS.index("PA"))
    macrorregiao = config.UF_INFO[uf]["macrorregiao"]
    st.caption(f"Macrorregião definida automaticamente: **{macrorregiao}**")

    entradas: dict[str, float] = {}
    colunas = st.columns(3)
    for i, feat in enumerate(config.NUMERIC_BASE):
        serie = pd.to_numeric(df[feat], errors="coerce").dropna()
        v_min = float(serie.min())
        v_max = float(serie.max())
        v_med = float(serie.median())
        rotulo = _LABELS_BASE.get(feat, feat)
        with colunas[i % 3]:
            if feat in {
                "fornos_detectados",
                "casos_oit_historicos",
                "populacao_total",
            }:
                # Atributos inteiros (contagens / população).
                entradas[feat] = float(
                    st.number_input(
                        rotulo,
                        min_value=int(v_min),
                        max_value=int(v_max),
                        value=int(round(v_med)),
                        step=1,
                    )
                )
            elif feat == "idh":
                entradas[feat] = float(
                    st.slider(
                        rotulo,
                        min_value=round(v_min, 3),
                        max_value=round(v_max, 3),
                        value=round(v_med, 3),
                        step=0.001,
                    )
                )
            else:
                entradas[feat] = float(
                    st.slider(
                        rotulo,
                        min_value=round(v_min, 2),
                        max_value=round(v_max, 2),
                        value=round(v_med, 2),
                    )
                )

    calcular = st.button("Calcular risco", type="primary")

    if not calcular:
        return

    # ------------------- Montagem do DataFrame de 1 linha ------------------
    linha = {feat: entradas[feat] for feat in config.NUMERIC_BASE}
    linha["uf"] = uf
    linha["macrorregiao"] = macrorregiao
    df1 = pd.DataFrame([linha])
    df1 = preprocessing.add_engineered_features(df1)
    X = df1[config.MODEL_FEATURES]

    col_esq, col_dir = st.columns(2)

    # --------------------------- Classificação -----------------------------
    score_estimado: float | None = None
    with col_esq:
        st.subheader("Classificação de risco")
        if bundle_clf is None:
            st.info("Modelo de classificação indisponível.")
        else:
            pipe = bundle_clf["pipeline"]
            classes = bundle_clf.get("classes") or list(pipe.classes_)
            categoria = str(pipe.predict(X)[0])
            st.markdown(
                _categoria_colorida(categoria), unsafe_allow_html=True
            )

            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X)[0]
                ordem_classes = [str(c) for c in pipe.classes_]
                prob_df = pd.DataFrame(
                    {"Categoria": ordem_classes, "Probabilidade": proba}
                )
                # Reordena conforme RISK_ORDER quando possível.
                prob_df["__ord"] = prob_df["Categoria"].apply(
                    lambda c: config.RISK_ORDER.index(c)
                    if c in config.RISK_ORDER
                    else 99
                )
                prob_df = prob_df.sort_values("__ord")
                fig_prob = px.bar(
                    prob_df,
                    x="Categoria",
                    y="Probabilidade",
                    color="Categoria",
                    color_discrete_map=config.RISK_COLORS,
                    category_orders={"Categoria": config.RISK_ORDER},
                    text_auto=".1%",
                )
                fig_prob.update_layout(
                    showlegend=False,
                    yaxis_tickformat=".0%",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=320,
                )
                st.plotly_chart(fig_prob, width="stretch")

    # ----------------------------- Regressão -------------------------------
    with col_dir:
        st.subheader("Score de risco (0-100)")
        if bundle_reg is None:
            st.info("Modelo de regressão indisponível.")
        else:
            score_estimado = float(bundle_reg["pipeline"].predict(X)[0])
            score_estimado = max(0.0, min(100.0, score_estimado))
            categoria_score = config.score_to_category(score_estimado)
            cor_gauge = config.RISK_COLORS.get(categoria_score, "#444444")
            low, high = config.RISK_THRESHOLDS
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score_estimado,
                    number={"suffix": " / 100"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": cor_gauge},
                        "steps": [
                            {
                                "range": [0, low],
                                "color": _hex_para_rgba(
                                    config.RISK_COLORS["Baixo Risco"]
                                ),
                            },
                            {
                                "range": [low, high],
                                "color": _hex_para_rgba(
                                    config.RISK_COLORS["Médio Risco"]
                                ),
                            },
                            {
                                "range": [high, 100],
                                "color": _hex_para_rgba(
                                    config.RISK_COLORS["Alto Risco"]
                                ),
                            },
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_gauge, width="stretch")
            st.markdown(
                f"Categoria pelo score: {_categoria_colorida(categoria_score)}",
                unsafe_allow_html=True,
            )

    # --------------------- Pessoas em risco estimadas ----------------------
    if score_estimado is not None:
        pop = entradas["populacao_total"]
        pessoas = int(round(pop * (score_estimado / 100.0)))
        st.metric(
            "Pessoas em risco estimadas (pop × score/100)",
            f"{pessoas:,}".replace(",", "."),
        )

    # ----------------- Principais fatores do modelo (SHAP) -----------------
    if shap_data and shap_data.get("classification"):
        importancias = shap_data["classification"].get("feature_importance", [])
        if importancias:
            st.subheader("Principais fatores do modelo (classificação)")
            top5 = importancias[:5]
            nomes = ", ".join(
                f"**{item['feature']}**" for item in top5
            )
            st.markdown(
                "Os atributos de maior peso médio (|SHAP|) na decisão são: "
                f"{nomes}."
            )


# ===========================================================================
# Seção 3 — Comparação de Modelos
# ===========================================================================
def _estilo_destaque(df_tab: pd.DataFrame, best: str | None):
    """Destaca a linha do melhor modelo (índice == best)."""

    def _highlight(row):
        if best is not None and row.name == best:
            return ["background-color: #d9f0d9"] * len(row)
        return [""] * len(row)

    return df_tab.style.apply(_highlight, axis=1).format(precision=4)


def render_comparacao(metrics: dict | None) -> None:
    """Tabelas e figuras de comparação dos modelos (clf e reg)."""
    st.header("Comparação de Modelos")

    if metrics is None:
        st.warning(_MSG_SEM_MODELOS)
        return

    # --------------------------- Classificação -----------------------------
    clf = metrics.get("classification", {})
    if clf.get("models"):
        st.subheader("Classificação")
        best_clf = clf.get("best_model")
        st.caption(f"Alvo: `{clf.get('target')}` · Melhor modelo: **{best_clf}**")
        tab_clf = pd.DataFrame(clf["models"]).T
        st.dataframe(
            _estilo_destaque(tab_clf, best_clf), width="stretch"
        )

    # ----------------------------- Regressão -------------------------------
    reg = metrics.get("regression", {})
    if reg.get("models"):
        st.subheader("Regressão")
        best_reg = reg.get("best_model")
        st.caption(f"Alvo: `{reg.get('target')}` · Melhor modelo: **{best_reg}**")
        tab_reg = pd.DataFrame(reg["models"]).T
        st.dataframe(
            _estilo_destaque(tab_reg, best_reg), width="stretch"
        )

    # ------------------------------ Figuras --------------------------------
    st.subheader("Figuras de avaliação")
    figuras = [
        (config.FIG_COMPARACAO_CLF, "Comparação — Classificação"),
        (config.FIG_COMPARACAO_REG, "Comparação — Regressão"),
        (config.FIG_MATRIZ_CONFUSAO, "Matriz de confusão"),
        (config.FIG_REG_PRED_VS_REAL, "Regressão: predito vs. real"),
    ]
    cols = st.columns(2)
    idx = 0
    for caminho, legenda in figuras:
        if Path(caminho).exists():
            with cols[idx % 2]:
                st.image(str(caminho), caption=legenda, width="stretch")
            idx += 1
    if idx == 0:
        st.info("Nenhuma figura de avaliação encontrada ainda.")

    # ------------------- Distribuição de classes ---------------------------
    dataset = metrics.get("dataset", {})
    dist = dataset.get("class_distribution")
    if dist:
        st.subheader("Distribuição de classes (dataset)")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write(
                f"Linhas: **{dataset.get('n_rows', '—')}** · "
                f"Atributos: **{dataset.get('n_features', '—')}**"
            )
        dist_df = pd.DataFrame(
            {
                "Categoria": list(dist.keys()),
                "Regiões": list(dist.values()),
            }
        )
        fig_dist = px.bar(
            dist_df,
            x="Categoria",
            y="Regiões",
            color="Categoria",
            color_discrete_map=config.RISK_COLORS,
            category_orders={"Categoria": config.RISK_ORDER},
            text_auto=True,
        )
        fig_dist.update_layout(
            showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=320
        )
        with c2:
            st.plotly_chart(fig_dist, width="stretch")


# ===========================================================================
# Seção 4 — Interpretabilidade (SHAP)
# ===========================================================================
def _grafico_shap(importancias: list[dict], titulo: str) -> None:
    """Renderiza o Top-15 de importância SHAP como barras horizontais."""
    if not importancias:
        st.info("Sem dados de importância para este modelo.")
        return
    top15 = importancias[:15]
    df_imp = pd.DataFrame(top15)
    fig = px.bar(
        df_imp,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title=titulo,
        labels={"mean_abs_shap": "Importância média |SHAP|", "feature": "Atributo"},
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
    )
    st.plotly_chart(fig, width="stretch")


def render_interpretabilidade(shap_data: dict | None) -> None:
    """Importância de atributos via SHAP (classificação e regressão)."""
    st.header("Interpretabilidade (SHAP)")

    if shap_data is None:
        st.warning(_MSG_SEM_MODELOS)
        return

    aba_clf, aba_reg = st.tabs(["Classificação", "Regressão"])

    with aba_clf:
        bloco = shap_data.get("classification", {})
        imps = bloco.get("feature_importance", [])
        if bloco.get("model"):
            st.caption(f"Modelo: **{bloco['model']}**")
        _grafico_shap(imps, "Top-15 atributos — Classificação")
        if imps:
            tops = ", ".join(f"**{i['feature']}**" for i in imps[:3])
            st.markdown(
                "Os fatores de maior peso confirmam a hipótese do problema: "
                f"{tops} aparecem entre os mais influentes — refletindo a "
                "combinação de **detecção de fornos**, **baixo IDH** e "
                "**isolamento geográfico** associada ao trabalho forçado."
            )
        if Path(config.FIG_SHAP_BAR_CLF).exists():
            st.image(
                str(config.FIG_SHAP_BAR_CLF),
                caption="SHAP — barras (classificação)",
                width="stretch",
            )
        if Path(config.FIG_SHAP_SUMMARY_CLF).exists():
            st.image(
                str(config.FIG_SHAP_SUMMARY_CLF),
                caption="SHAP — summary (classificação)",
                width="stretch",
            )

    with aba_reg:
        bloco = shap_data.get("regression", {})
        imps = bloco.get("feature_importance", [])
        if bloco.get("model"):
            st.caption(f"Modelo: **{bloco['model']}**")
        _grafico_shap(imps, "Top-15 atributos — Regressão")
        if imps:
            tops = ", ".join(f"**{i['feature']}**" for i in imps[:3])
            st.markdown(
                "No modelo de score contínuo, os fatores de maior peso são "
                f"{tops}."
            )
        if Path(config.FIG_SHAP_BAR_REG).exists():
            st.image(
                str(config.FIG_SHAP_BAR_REG),
                caption="SHAP — barras (regressão)",
                width="stretch",
            )
        if Path(config.FIG_SHAP_SUMMARY_REG).exists():
            st.image(
                str(config.FIG_SHAP_SUMMARY_REG),
                caption="SHAP — summary (regressão)",
                width="stretch",
            )


# ===========================================================================
# Seção 5 — Sobre
# ===========================================================================
def render_sobre(df: pd.DataFrame | None) -> None:
    """Contexto do problema, dataset e metodologia."""
    st.header("Sobre o projeto")

    st.markdown(
        """
### O problema
O **trabalho forçado** (análogo à escravidão) persiste em regiões isoladas,
frequentemente ligado à produção de **carvão vegetal** em carvoarias e olarias.
O **GAIE** propõe combater esse crime com **Inteligência Artificial** aplicada a
**dados geoespaciais/satélite**: a detecção de **fornos** por satélite, somada a
**indicadores socioeconômicos** (IDH, pobreza, informalidade, fiscalização),
permite **priorizar regiões** para fiscalização e ação preventiva — uma
aplicação concreta do tema **Economia Espacial**.

### O dataset (sintético)
Um **modelo gerador causal** produz regiões fictícias com geolocalização,
indicadores socioeconômicos/ambientais e a detecção de fornos por satélite. Um
**risco latente** combina esses fatores — com **maiores pesos** para
(1) nº de fornos, (2) baixo IDH e (3) isolamento geográfico — dando origem às
três variáveis-alvo:

- `score_risco` (0-100) — alvo do **modelo de regressão**;
- `categoria_risco` (Baixo / Médio / Alto) — alvo do **modelo de classificação**;
- `pessoas_em_risco_estimadas` — derivado, exibido no app.

### Metodologia (pipeline de ML)
1. **Geração** do dataset sintético (≥ 1.000 linhas).
2. **Pré-processamento** com `ColumnTransformer` (imputação + escala + one-hot)
   encapsulado em `Pipeline`, evitando vazamento de dados.
3. **Treino e comparação** de modelos (classificação e regressão) com validação
   cruzada e métricas reportadas.
4. **Interpretabilidade** com **SHAP**, agregando as colunas one-hot de volta aos
   atributos originais.
5. **Visualização** nesta aplicação **Streamlit**.
"""
    )

    if df is not None:
        st.subheader("Resumo do dataset carregado")
        c1, c2, c3 = st.columns(3)
        c1.metric("Linhas", f"{len(df):,}".replace(",", "."))
        c2.metric("Colunas", f"{df.shape[1]}")
        c3.metric("UFs", f"{df['uf'].nunique()}")

    st.warning(
        "**Aviso:** os dados utilizados são **SINTÉTICOS**, gerados "
        "programaticamente para fins **acadêmicos**. Não representam pessoas, "
        "municípios ou ocorrências reais."
    )


# ===========================================================================
# Aplicação principal
# ===========================================================================
def main() -> None:
    render_cabecalho()

    # Carrega artefatos (todos tolerantes à ausência).
    df = carregar_dataset()
    metrics = carregar_json(str(config.METRICS_JSON))
    shap_data = carregar_json(str(config.SHAP_IMPORTANCE_JSON))
    bundle_clf = carregar_bundle(str(config.CLF_MODEL_PATH))
    bundle_reg = carregar_bundle(str(config.REG_MODEL_PATH))

    # Navegação.
    st.sidebar.title("Navegação")
    secao = st.sidebar.radio(
        "Seção",
        options=[
            "Mapa de Risco",
            "Simulador de Risco",
            "Comparação de Modelos",
            "Interpretabilidade (SHAP)",
            "Sobre",
        ],
        label_visibility="collapsed",
    )

    # Status dos artefatos na barra lateral.
    st.sidebar.divider()
    st.sidebar.caption("Status dos artefatos")
    st.sidebar.write("Dataset:", "OK" if df is not None else "ausente")
    st.sidebar.write("Modelo classificação:", "OK" if bundle_clf else "ausente")
    st.sidebar.write("Modelo regressão:", "OK" if bundle_reg else "ausente")
    st.sidebar.write("Métricas:", "OK" if metrics else "ausente")
    st.sidebar.write("SHAP:", "OK" if shap_data else "ausente")

    # O dataset é o mínimo necessário para a maioria das seções.
    if df is None and secao in {"Mapa de Risco", "Simulador de Risco"}:
        st.error(
            "Dataset não encontrado. Rode: `python run_pipeline.py` "
            "(ou gere os dados com `python -m src.data_generation`)."
        )
        return

    # Roteamento.
    if secao == "Mapa de Risco":
        render_mapa(df)
    elif secao == "Simulador de Risco":
        render_simulador(df, bundle_clf, bundle_reg, shap_data)
    elif secao == "Comparação de Modelos":
        render_comparacao(metrics)
    elif secao == "Interpretabilidade (SHAP)":
        render_interpretabilidade(shap_data)
    else:
        render_sobre(df)


if __name__ == "__main__":
    main()
