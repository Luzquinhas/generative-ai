"""
config.py — Contrato compartilhado do projeto GAIE (Generative AI For Engineering).

Este módulo é a ÚNICA fonte de verdade para:
  * caminhos de arquivos (dados, modelos, relatórios);
  * nomes de colunas e listas de atributos (features) usadas na modelagem;
  * limiares de risco e rótulos de classe;
  * metadados geográficos das Unidades Federativas (UFs) do Brasil.

Todos os demais scripts (data_generation, preprocessing, train, explain, app)
importam daqui para garantir consistência de ponta a ponta.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
N_ROWS: int = 2000  # >= 1.000 linhas exigidas pelo edital

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "gaie_dataset.csv"

MODELS_DIR = ROOT / "models"
CLF_MODEL_PATH = MODELS_DIR / "modelo_classificacao.joblib"
REG_MODEL_PATH = MODELS_DIR / "modelo_regressao.joblib"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_JSON = REPORTS_DIR / "metrics.json"
SHAP_IMPORTANCE_JSON = REPORTS_DIR / "shap_importance.json"

# Figuras geradas (nomes fixos, compartilhados entre train/explain e o app).
FIG_COMPARACAO_CLF = FIGURES_DIR / "comparacao_modelos_classificacao.png"
FIG_COMPARACAO_REG = FIGURES_DIR / "comparacao_modelos_regressao.png"
FIG_MATRIZ_CONFUSAO = FIGURES_DIR / "matriz_confusao.png"
FIG_REG_PRED_VS_REAL = FIGURES_DIR / "regressao_pred_vs_real.png"
FIG_SHAP_SUMMARY_CLF = FIGURES_DIR / "shap_summary_classificacao.png"
FIG_SHAP_BAR_CLF = FIGURES_DIR / "shap_bar_classificacao.png"
FIG_SHAP_SUMMARY_REG = FIGURES_DIR / "shap_summary_regressao.png"
FIG_SHAP_BAR_REG = FIGURES_DIR / "shap_bar_regressao.png"


def ensure_dirs() -> None:
    """Cria os diretórios de saída caso ainda não existam."""
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Colunas do dataset
# ---------------------------------------------------------------------------
# Identificadores e geolocalização (NÃO entram no modelo; geo é usado no mapa).
ID_COLS = ["regiao_id", "municipio"]
GEO_COLS = ["latitude", "longitude"]

# Atributos numéricos "base" presentes no dataset bruto.
NUMERIC_BASE = [
    "fornos_detectados",          # nº de fornos (carvoarias/olarias) via satélite
    "densidade_populacional",     # hab/km²
    "pib_per_capita",             # R$ por habitante/ano
    "distancia_centro_urbano_km", # isolamento geográfico
    "indice_pobreza",             # 0-100 (maior = mais pobreza)
    "idh",                        # 0-1 (Índice de Desenvolvimento Humano)
    "taxa_informalidade",         # % de trabalho informal
    "casos_oit_historicos",       # casos históricos OIT/resgates na região
    "cobertura_fiscalizacao",     # % de cobertura de fiscalização trabalhista
    "area_desmatada_km2",         # área desmatada (proxy de pressão por carvão)
    "acesso_educacao",            # índice 0-100 de acesso à educação
    "populacao_total",            # nº de habitantes
]

# Atributos numéricos criados em preprocessing.add_engineered_features().
ENGINEERED_FEATURES = [
    "fornos_por_1000hab",      # densidade de fornos por mil habitantes
    "vulnerabilidade_composta",# indice_pobreza * (1 - idh)
    "isolamento_fornos",       # distancia_centro_urbano_km * log1p(fornos)
    "pressao_carvao",          # fornos_detectados * log1p(area_desmatada_km2)
    "log_populacao",           # log1p(populacao_total)
    "deficit_fiscalizacao",    # (100 - cobertura_fiscalizacao) * log1p(fornos)
]

# Atributos categóricos.
CATEGORICAL_FEATURES = ["uf", "macrorregiao"]

# Conjunto completo de atributos usado pelos modelos (classificação e regressão).
NUMERIC_FEATURES = NUMERIC_BASE + ENGINEERED_FEATURES
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Variáveis-alvo (targets)
# ---------------------------------------------------------------------------
TARGET_REG = "score_risco"            # contínuo 0-100 (Modelo de Regressão)
TARGET_CLF = "categoria_risco"        # Baixo/Médio/Alto Risco (Modelo de Classificação)
TARGET_EXTRA = "pessoas_em_risco_estimadas"  # derivado; exibido no app, NÃO é feature

# Colunas que jamais devem entrar como feature (evita vazamento de informação).
LEAKAGE_COLS = [TARGET_REG, TARGET_CLF, TARGET_EXTRA]

# Limiares de risco aplicados sobre score_risco (0-100).
RISK_THRESHOLDS = (40.0, 70.0)  # < 40 Baixo; 40-70 Médio; >= 70 Alto
RISK_LABELS = ["Baixo Risco", "Médio Risco", "Alto Risco"]
RISK_ORDER = RISK_LABELS  # ordem categórica (Baixo < Médio < Alto)

# Mapa de cores consistente para o app e os gráficos.
RISK_COLORS = {
    "Baixo Risco": "#2ca02c",   # verde
    "Médio Risco": "#ff7f0e",   # laranja
    "Alto Risco": "#d62728",    # vermelho
}


def score_to_category(score: float) -> str:
    """Converte um score contínuo (0-100) na categoria de risco correspondente."""
    low, high = RISK_THRESHOLDS
    if score < low:
        return RISK_LABELS[0]
    if score < high:
        return RISK_LABELS[1]
    return RISK_LABELS[2]


# ---------------------------------------------------------------------------
# Metadados geográficos das UFs (centróides aproximados + peso de risco)
# ---------------------------------------------------------------------------
# peso_risco (0-1): prior de risco por UF, calibrado por incidência histórica de
# trabalho forçado/escravo (OIT e "Cadastro de Empregadores" - Lista Suja, MTE).
# Estados com maior histórico (PA, MA, MT, TO, BA, MG, GO, PI) recebem peso maior.
UF_INFO = {
    "AC": {"lat": -8.77,  "lon": -70.55, "macrorregiao": "Norte",        "peso_risco": 0.50},
    "AL": {"lat": -9.62,  "lon": -36.82, "macrorregiao": "Nordeste",     "peso_risco": 0.55},
    "AP": {"lat": 1.41,   "lon": -51.77, "macrorregiao": "Norte",        "peso_risco": 0.45},
    "AM": {"lat": -3.47,  "lon": -65.10, "macrorregiao": "Norte",        "peso_risco": 0.55},
    "BA": {"lat": -12.96, "lon": -41.70, "macrorregiao": "Nordeste",     "peso_risco": 0.80},
    "CE": {"lat": -5.20,  "lon": -39.53, "macrorregiao": "Nordeste",     "peso_risco": 0.65},
    "DF": {"lat": -15.83, "lon": -47.86, "macrorregiao": "Centro-Oeste", "peso_risco": 0.15},
    "ES": {"lat": -19.19, "lon": -40.34, "macrorregiao": "Sudeste",      "peso_risco": 0.40},
    "GO": {"lat": -15.98, "lon": -49.86, "macrorregiao": "Centro-Oeste", "peso_risco": 0.70},
    "MA": {"lat": -5.42,  "lon": -45.44, "macrorregiao": "Nordeste",     "peso_risco": 0.88},
    "MT": {"lat": -12.64, "lon": -55.42, "macrorregiao": "Centro-Oeste", "peso_risco": 0.85},
    "MS": {"lat": -20.51, "lon": -54.54, "macrorregiao": "Centro-Oeste", "peso_risco": 0.55},
    "MG": {"lat": -18.10, "lon": -44.38, "macrorregiao": "Sudeste",      "peso_risco": 0.78},
    "PA": {"lat": -3.79,  "lon": -52.48, "macrorregiao": "Norte",        "peso_risco": 0.90},
    "PB": {"lat": -7.28,  "lon": -36.72, "macrorregiao": "Nordeste",     "peso_risco": 0.55},
    "PR": {"lat": -24.89, "lon": -51.55, "macrorregiao": "Sul",          "peso_risco": 0.30},
    "PE": {"lat": -8.38,  "lon": -37.86, "macrorregiao": "Nordeste",     "peso_risco": 0.62},
    "PI": {"lat": -7.72,  "lon": -42.73, "macrorregiao": "Nordeste",     "peso_risco": 0.72},
    "RJ": {"lat": -22.25, "lon": -42.66, "macrorregiao": "Sudeste",      "peso_risco": 0.30},
    "RN": {"lat": -5.81,  "lon": -36.59, "macrorregiao": "Nordeste",     "peso_risco": 0.50},
    "RS": {"lat": -29.70, "lon": -53.20, "macrorregiao": "Sul",          "peso_risco": 0.25},
    "RO": {"lat": -10.83, "lon": -63.34, "macrorregiao": "Norte",        "peso_risco": 0.68},
    "RR": {"lat": 2.05,   "lon": -61.40, "macrorregiao": "Norte",        "peso_risco": 0.50},
    "SC": {"lat": -27.45, "lon": -50.95, "macrorregiao": "Sul",          "peso_risco": 0.20},
    "SP": {"lat": -22.19, "lon": -48.79, "macrorregiao": "Sudeste",      "peso_risco": 0.35},
    "SE": {"lat": -10.57, "lon": -37.45, "macrorregiao": "Nordeste",     "peso_risco": 0.48},
    "TO": {"lat": -10.17, "lon": -48.30, "macrorregiao": "Norte",        "peso_risco": 0.75},
}

UFS = list(UF_INFO.keys())
