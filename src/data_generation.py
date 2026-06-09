"""
data_generation.py — Geração do dataset sintético do GAIE.

Cumpre a "Opção B" do edital: usar geração programática (IA generativa /
modelo gerador estatístico) para produzir um conjunto de dados sintético com,
no mínimo, 1.000 linhas e 10 colunas.

Estratégia (modelo gerador causal):
  1. Cada região pertence a uma UF brasileira (centróide + prior de risco).
  2. Um fator latente de "desenvolvimento" gera, de forma correlacionada, os
     indicadores socioeconômicos (IDH, pobreza, PIB, educação, informalidade,
     fiscalização).
  3. Indicadores geográficos/ambientais (isolamento, desmatamento) e a
     detecção de fornos por satélite são gerados com dependência do
     desenvolvimento, da região e do prior de risco da UF.
  4. Um "risco latente" combina linearmente os fatores padronizados, com os
     MAIORES pesos atribuídos a (a) nº de fornos, (b) baixo IDH e (c)
     isolamento geográfico — exatamente os fatores que a literatura da OIT
     associa a trabalho forçado. O risco vira `score_risco` (0-100) via função
     logística, do qual derivam `categoria_risco` e `pessoas_em_risco_estimadas`.

O resultado é um dataset com sinal aprendível e interpretável: os modelos
conseguem prever o risco e o SHAP confirma os fatores esperados como mais
influentes.

Uso:
    python -m src.data_generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # permite execução como módulo (`python -m src.data_generation`) ou script
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _zscore(x: np.ndarray) -> np.ndarray:
    """Padroniza um vetor (média 0, desvio 1), robusto a desvio nulo."""
    std = x.std()
    return (x - x.mean()) / (std if std > 1e-9 else 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# Nomes sintéticos para compor municípios fictícios.
_PREFIXOS = [
    "Vale", "Serra", "Porto", "Campo", "Rio", "Alto", "Lago", "Boa", "São",
    "Santa", "Nova", "Vila", "Bela", "Monte", "Cruz", "Santo",
]
_NUCLEOS = [
    "Verde", "do Norte", "Esperança", "das Flores", "Dourado", "Grande",
    "do Sertão", "da Mata", "Azul", "do Carvão", "do Cedro", "Limpo",
    "do Amparo", "Feliz", "do Forno", "da Serra", "Vermelho", "do Vale",
]


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------
def generate_dataset(n: int = config.N_ROWS, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Gera o dataset sintético do GAIE com `n` regiões.

    Retorna um DataFrame com identificadores, geolocalização, indicadores
    socioeconômicos/geográficos e as três variáveis-alvo.
    """
    rng = np.random.default_rng(seed)

    ufs = np.array(config.UFS)
    # Amostragem de UFs: leve ênfase em estados maiores/críticos para volume.
    pesos_uf = np.array([1.0 + 1.5 * config.UF_INFO[u]["peso_risco"] for u in ufs])
    pesos_uf = pesos_uf / pesos_uf.sum()
    uf = rng.choice(ufs, size=n, p=pesos_uf)

    peso_risco = np.array([config.UF_INFO[u]["peso_risco"] for u in uf])
    macrorregiao = np.array([config.UF_INFO[u]["macrorregiao"] for u in uf])
    lat0 = np.array([config.UF_INFO[u]["lat"] for u in uf])
    lon0 = np.array([config.UF_INFO[u]["lon"] for u in uf])

    # Geolocalização: jitter gaussiano ao redor do centróide da UF.
    latitude = np.clip(lat0 + rng.normal(0, 1.4, n), -33.5, 4.5)
    longitude = np.clip(lon0 + rng.normal(0, 1.4, n), -73.0, -34.0)

    # --- Fator latente de desenvolvimento (0=pobre/vulnerável, 1=desenvolvido) ---
    # Quanto maior o prior de risco da UF, menor o desenvolvimento médio.
    mu_desenv = 0.62 - 0.45 * (peso_risco - 0.5)
    desenv = np.clip(mu_desenv + rng.normal(0, 0.16, n), 0.02, 0.98)

    # --- Indicadores socioeconômicos (correlacionados com `desenv`) ---
    idh = np.clip(0.46 + 0.46 * desenv + rng.normal(0, 0.035, n), 0.45, 0.96)
    indice_pobreza = np.clip(72 - 58 * desenv + rng.normal(0, 6, n), 2, 82)
    pib_per_capita = np.clip(
        7000 + 46000 * desenv + rng.normal(0, 5000, n), 3200, 95000
    )
    acesso_educacao = np.clip(34 + 56 * desenv + rng.normal(0, 7, n), 8, 99)
    taxa_informalidade = np.clip(72 - 46 * desenv + rng.normal(0, 7, n), 8, 88)
    cobertura_fiscalizacao = np.clip(
        18 + 56 * desenv + rng.normal(0, 9, n), 2, 96
    )

    # --- Demografia ---
    populacao_total = np.clip(
        np.exp(rng.normal(10.4, 1.25, n)).round(), 1200, 4_500_000
    ).astype(int)
    densidade_populacional = np.clip(
        np.exp(rng.normal(2.6 + 1.1 * desenv, 1.0, n)), 0.3, 9000
    ).round(2)

    # --- Geografia / ambiente ---
    # Isolamento: maior em regiões menos desenvolvidas e no Norte/Centro-Oeste.
    base_isolamento = np.where(
        np.isin(macrorregiao, ["Norte", "Centro-Oeste"]), 140, 60
    )
    distancia_centro_urbano_km = np.clip(
        base_isolamento + 220 * (1 - desenv) + rng.normal(0, 35, n), 2, 850
    ).round(1)

    # Desmatamento: maior no Norte/Centro-Oeste; cresce com isolamento.
    base_desmat = np.where(
        np.isin(macrorregiao, ["Norte", "Centro-Oeste"]), 1.6, 0.4
    )
    area_desmatada_km2 = np.clip(
        np.exp(rng.normal(base_desmat + 0.9 * (1 - desenv), 0.8, n)), 0.05, 4500
    ).round(2)

    # --- Fornos detectados por satélite (variável-chave) ---
    # Intensidade (Poisson) cresce com risco da UF, baixo desenvolvimento,
    # desmatamento e isolamento.
    lambda_fornos = np.exp(
        -0.4
        + 1.3 * (peso_risco - 0.5)
        + 1.1 * (1 - desenv)
        + 0.35 * _zscore(np.log1p(area_desmatada_km2))
        + 0.25 * _zscore(distancia_centro_urbano_km)
    )
    fornos_detectados = rng.poisson(np.clip(lambda_fornos, 0.05, 60)).astype(int)

    # --- Casos históricos OIT (contagem rara, ligada a risco/fornos) ---
    lambda_oit = np.exp(
        -1.8
        + 1.1 * (peso_risco - 0.5)
        + 0.5 * _zscore(np.log1p(fornos_detectados))
        + 0.4 * (1 - desenv)
    )
    casos_oit_historicos = rng.poisson(np.clip(lambda_oit, 0.02, 25)).astype(int)

    # --- Risco latente (combinação linear padronizada) ---
    idh_scaled = (idh - 0.45) / 0.51  # ~0..1
    risk_lin = (
        1.45 * _zscore(np.log1p(fornos_detectados))   # (1) fornos — maior peso
        + 1.25 * _zscore(1 - idh_scaled)              # (2) baixo IDH
        + 1.10 * _zscore(distancia_centro_urbano_km)  # (3) isolamento geográfico
        + 0.95 * _zscore(indice_pobreza)
        + 0.80 * _zscore(taxa_informalidade)
        + 0.70 * _zscore(np.log1p(casos_oit_historicos))
        + 0.55 * _zscore(np.log1p(area_desmatada_km2))
        - 0.80 * _zscore(np.log1p(pib_per_capita))
        - 0.70 * _zscore(cobertura_fiscalizacao)
        - 0.45 * _zscore(acesso_educacao)
        + 1.05 * (peso_risco - 0.5)
        + rng.normal(0, 0.55, n)                      # ruído irredutível
    )

    # Mapeia o risco linear para 0-100 via logística aplicada ao risco
    # padronizado (z-score). A inclinação (~1.0) é calibrada para boa
    # dispersão e equilíbrio entre as três classes de risco.
    score_risco = (100.0 * _sigmoid(1.0 * _zscore(risk_lin))).round(1)

    categoria_risco = pd.Categorical(
        [config.score_to_category(s) for s in score_risco],
        categories=config.RISK_ORDER,
        ordered=True,
    )

    # Pessoas estimadas em situação de risco: fração da população que escala
    # com o score (não-linear) e com a informalidade.
    frac_risco = (score_risco / 100.0) ** 1.6 * (0.015 + 0.07 * taxa_informalidade / 100.0)
    pessoas_em_risco_estimadas = np.clip(
        (populacao_total * frac_risco).round(), 0, None
    ).astype(int)

    # Identificadores sintéticos.
    regiao_id = np.array([f"BR-{i:05d}" for i in range(1, n + 1)])
    municipio = np.array(
        [
            f"{rng.choice(_PREFIXOS)} {rng.choice(_NUCLEOS)}"
            for _ in range(n)
        ]
    )

    df = pd.DataFrame(
        {
            "regiao_id": regiao_id,
            "municipio": municipio,
            "uf": uf,
            "macrorregiao": macrorregiao,
            "latitude": latitude.round(4),
            "longitude": longitude.round(4),
            "fornos_detectados": fornos_detectados,
            "densidade_populacional": densidade_populacional,
            "pib_per_capita": pib_per_capita.round(2),
            "distancia_centro_urbano_km": distancia_centro_urbano_km,
            "indice_pobreza": indice_pobreza.round(2),
            "idh": idh.round(3),
            "taxa_informalidade": taxa_informalidade.round(2),
            "casos_oit_historicos": casos_oit_historicos,
            "cobertura_fiscalizacao": cobertura_fiscalizacao.round(2),
            "area_desmatada_km2": area_desmatada_km2,
            "acesso_educacao": acesso_educacao.round(2),
            "populacao_total": populacao_total,
            "score_risco": score_risco,
            "categoria_risco": categoria_risco,
            "pessoas_em_risco_estimadas": pessoas_em_risco_estimadas,
        }
    )

    # --- Injeção controlada de valores ausentes (para exercitar o pré-proc.) ---
    # Apenas em colunas de atributo (nunca nos alvos/identificadores/geo).
    df = _inject_missing(df, rng)

    return df


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Insere NaNs realistas em algumas colunas de atributo."""
    n = len(df)
    plano = {
        "pib_per_capita": 0.04,
        "cobertura_fiscalizacao": 0.05,
        "idh": 0.03,
        "acesso_educacao": 0.03,
    }
    for col, frac in plano.items():
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


def main() -> None:
    config.ensure_dirs()
    df = generate_dataset()
    df.to_csv(config.DATA_RAW, index=False, encoding="utf-8")

    print(f"Dataset gerado: {config.DATA_RAW}")
    print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("\nDistribuição de categoria_risco:")
    print(df["categoria_risco"].value_counts().reindex(config.RISK_ORDER))
    print("\nResumo de score_risco:")
    print(df["score_risco"].describe().round(2))
    print(f"\nValores ausentes por coluna (total {int(df.isna().sum().sum())}):")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()
