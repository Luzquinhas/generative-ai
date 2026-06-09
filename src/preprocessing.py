"""
preprocessing.py — Pré-processamento e engenharia de atributos do GAIE.

Responsabilidades:
  * carregar o dataset bruto (`load_raw`);
  * criar atributos derivados (`add_engineered_features`);
  * montar o `ColumnTransformer` (imputação + escala + one-hot) usado dentro
    dos pipelines de modelagem (`build_preprocessor`);
  * separar matriz de atributos X e alvo y para cada tarefa (`prepare_xy`).

O pré-processamento é encapsulado num `ColumnTransformer` e SEMPRE entra dentro
de um `Pipeline` do scikit-learn junto com o estimador. Assim, imputação e
escala são ajustadas apenas no conjunto de treino (evitando vazamento) e
reaplicadas automaticamente em validação, teste e inferência no app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_raw(path=None) -> pd.DataFrame:
    """Carrega o dataset bruto e normaliza o tipo da coluna de categoria."""
    path = path or config.DATA_RAW
    df = pd.read_csv(path)
    if config.TARGET_CLF in df.columns:
        df[config.TARGET_CLF] = pd.Categorical(
            df[config.TARGET_CLF], categories=config.RISK_ORDER, ordered=True
        )
    return df


# ---------------------------------------------------------------------------
# Engenharia de atributos
# ---------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona os atributos derivados definidos em `config.ENGINEERED_FEATURES`.

    Os derivados combinam os fatores que a literatura associa ao trabalho
    forçado (fornos, baixo IDH, isolamento, fiscalização precária). Valores
    ausentes nas colunas base propagam-se como NaN e são tratados depois pelo
    imputador do `ColumnTransformer`.
    """
    df = df.copy()

    df["fornos_por_1000hab"] = df["fornos_detectados"] / (df["populacao_total"] / 1000.0)
    df["vulnerabilidade_composta"] = df["indice_pobreza"] * (1.0 - df["idh"])
    df["isolamento_fornos"] = df["distancia_centro_urbano_km"] * np.log1p(df["fornos_detectados"])
    df["pressao_carvao"] = df["fornos_detectados"] * np.log1p(df["area_desmatada_km2"])
    df["log_populacao"] = np.log1p(df["populacao_total"])
    df["deficit_fiscalizacao"] = (100.0 - df["cobertura_fiscalizacao"]) * np.log1p(df["fornos_detectados"])

    # Substitui eventuais infinitos por NaN (serão imputados adiante).
    df[config.ENGINEERED_FEATURES] = df[config.ENGINEERED_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    return df


# ---------------------------------------------------------------------------
# Transformador de colunas
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Constrói o ColumnTransformer (numérico + categórico).

    Numérico: imputação pela mediana + padronização (StandardScaler).
    Categórico: imputação pela moda + One-Hot Encoding (ignora categorias
    desconhecidas, permitindo simular UFs arbitrárias no app).
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Nomes das colunas após a transformação (úteis para SHAP)."""
    return list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# Separação X / y
# ---------------------------------------------------------------------------
def prepare_xy(df: pd.DataFrame, task: str):
    """Retorna (X, y) já com atributos derivados, para a tarefa indicada.

    task: "classification" -> y = categoria_risco
          "regression"     -> y = score_risco

    X contém somente `config.MODEL_FEATURES` (sem identificadores, geo ou
    qualquer coluna-alvo — prevenindo vazamento de informação).
    """
    if task not in {"classification", "regression"}:
        raise ValueError(f"task inválida: {task!r}")

    df_feat = add_engineered_features(df)
    X = df_feat[config.MODEL_FEATURES].copy()

    if task == "classification":
        y = df_feat[config.TARGET_CLF].astype("object")
    else:
        y = df_feat[config.TARGET_REG].astype(float)
    return X, y


if __name__ == "__main__":
    # Sanity check rápido do pré-processamento.
    df = load_raw()
    X, y = prepare_xy(df, "classification")
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    print(f"X bruto: {X.shape} -> X transformado: {Xt.shape}")
    print(f"Atributos de saída ({len(get_output_feature_names(pre))}):")
    print(get_output_feature_names(pre))
    print(f"NaNs restantes após transformação: {int(np.isnan(Xt).sum())}")
