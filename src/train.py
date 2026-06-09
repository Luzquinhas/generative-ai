"""
train.py — Treinamento, comparação e seleção de modelos do GAIE.

Este script:
  1. Carrega o dataset bruto e aplica UM ÚNICO split treino/teste (estratificado
     pela categoria de risco), reaproveitado para classificação E regressão —
     garantindo que ambas as tarefas usem exatamente as mesmas regiões.
  2. CLASSIFICAÇÃO (alvo `categoria_risco`): compara 3 técnicas distintas via
     validação cruzada estratificada (f1_macro) no treino, refina o vencedor com
     um GridSearchCV pequeno e o avalia no teste.
  3. REGRESSÃO (alvo `score_risco`): compara 4 técnicas via validação cruzada
     (KFold, r2) no treino, refina o vencedor e o avalia no teste.
  4. Salva os bundles dos vencedores (joblib), o `reports/metrics.json` e as
     figuras de comparação.

Todo o pré-processamento vive DENTRO de cada Pipeline (passos nomeados
"preprocessor" e "model"), de modo que imputação/escala/one-hot são ajustados
somente no treino, evitando vazamento de informação.

Uso:
    python -m src.train
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # backend sem display (deve vir antes do pyplot)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

try:
    from . import config, preprocessing
except ImportError:  # pragma: no cover (execução como script solto)
    import config  # type: ignore
    import preprocessing  # type: ignore

RS = config.RANDOM_STATE
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Conversão numpy -> tipos nativos (para serialização JSON)
# ---------------------------------------------------------------------------
def _to_native(obj):
    """Converte recursivamente escalares/contêineres numpy em tipos nativos."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj


def _make_pipeline(estimator) -> Pipeline:
    """Monta o Pipeline padrão: preprocessor + model (nomes exigidos no contrato)."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            ("model", estimator),
        ]
    )


# ---------------------------------------------------------------------------
# Classificação
# ---------------------------------------------------------------------------
def treinar_classificacao(X_train, y_train, X_test, y_test):
    """Compara 3 classificadores, refina o vencedor e o avalia no teste.

    Retorna (bundle, metrics_dict) onde metrics_dict segue o schema do contrato
    para a seção "classification".
    """
    candidatos = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RS,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RS),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    modelos_metrics: dict[str, dict] = {}
    cv_means: dict[str, float] = {}

    # --- Validação cruzada no treino (seleção do vencedor) ---
    for nome, est in candidatos.items():
        pipe = _make_pipeline(est)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        modelos_metrics[nome] = {
            "cv_f1_macro_mean": float(scores.mean()),
            "cv_f1_macro_std": float(scores.std()),
        }
        cv_means[nome] = float(scores.mean())
        print(
            f"  [CLF] {nome:<20} cv_f1_macro = "
            f"{scores.mean():.4f} (+/- {scores.std():.4f})"
        )

    best_name = max(cv_means, key=cv_means.get)
    print(f"  [CLF] Vencedor da validação cruzada: {best_name}")

    # --- GridSearchCV pequeno só no vencedor ---
    param_grids = {
        "LogisticRegression": {"model__C": [0.5, 1.0, 2.0]},
        "RandomForest": {"model__max_depth": [None, 12, 20]},
        "GradientBoosting": {"model__learning_rate": [0.05, 0.1]},
    }
    pipe_best = _make_pipeline(candidatos[best_name])
    grid = GridSearchCV(
        pipe_best,
        param_grid=param_grids[best_name],
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_pipe = grid.best_estimator_  # já refit no treino completo
    print(f"  [CLF] Melhores hiperparâmetros: {grid.best_params_}")

    # --- Avaliação no teste ---
    labels = config.RISK_ORDER
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro", labels=labels))
    precm = float(
        precision_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
    )
    recm = float(
        recall_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
    )
    # As colunas de predict_proba seguem a ordem de best_pipe.classes_ (ordem
    # lexicográfica do estimador), que NÃO é a de RISK_ORDER. Para o roc_auc_ovr,
    # `labels` precisa refletir a ordem das COLUNAS de y_proba.
    proba_labels = list(best_pipe.classes_)
    roc = float(
        roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro", labels=proba_labels
        )
    )

    # Garante que TODOS os modelos exponham as 7 métricas exigidas no schema.
    # Para os não-vencedores (sem fit/teste), preenchemos as de teste como NaN.
    for nome in modelos_metrics:
        modelos_metrics[nome].setdefault("accuracy", float("nan"))
        modelos_metrics[nome].setdefault("f1_macro", float("nan"))
        modelos_metrics[nome].setdefault("precision_macro", float("nan"))
        modelos_metrics[nome].setdefault("recall_macro", float("nan"))
        modelos_metrics[nome].setdefault("roc_auc_ovr", float("nan"))

    modelos_metrics[best_name].update(
        {
            "accuracy": acc,
            "f1_macro": f1m,
            "precision_macro": precm,
            "recall_macro": recm,
            "roc_auc_ovr": roc,
        }
    )

    # Avalia também os demais candidatos no teste (refit no treino) para uma
    # comparação justa de f1_macro no gráfico e no JSON.
    for nome, est in candidatos.items():
        if nome == best_name:
            continue
        pipe = _make_pipeline(est)
        pipe.fit(X_train, y_train)
        yp = pipe.predict(X_test)
        ypr = pipe.predict_proba(X_test)
        modelos_metrics[nome].update(
            {
                "accuracy": float(accuracy_score(y_test, yp)),
                "f1_macro": float(
                    f1_score(y_test, yp, average="macro", labels=labels)
                ),
                "precision_macro": float(
                    precision_score(
                        y_test, yp, average="macro", labels=labels, zero_division=0
                    )
                ),
                "recall_macro": float(
                    recall_score(
                        y_test, yp, average="macro", labels=labels, zero_division=0
                    )
                ),
                "roc_auc_ovr": float(
                    roc_auc_score(
                        y_test,
                        ypr,
                        multi_class="ovr",
                        average="macro",
                        labels=list(pipe.classes_),
                    )
                ),
            }
        )

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    metrics = {
        "target": config.TARGET_CLF,
        "classes": list(labels),
        "best_model": best_name,
        "models": modelos_metrics,
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": _to_native(report),
    }

    bundle = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "task": "classification",
        "feature_columns": config.MODEL_FEATURES,
        "classes": list(config.RISK_ORDER),
    }

    return bundle, metrics, cm


# ---------------------------------------------------------------------------
# Regressão
# ---------------------------------------------------------------------------
def treinar_regressao(X_train, y_train, X_test, y_test):
    """Compara 4 regressores, refina o vencedor e o avalia no teste.

    Retorna (bundle, metrics_dict, y_pred_teste) com metrics_dict no schema do
    contrato para a seção "regression".
    """
    candidatos = {
        "Ridge": Ridge(alpha=1.0, random_state=RS),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=RS, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RS),
    }

    # XGBoost é opcional: se indisponível, registra log e segue sem ele.
    try:
        from xgboost import XGBRegressor

        candidatos["XGBoost"] = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RS,
            n_jobs=-1,
        )
    except ImportError:
        print("  [REG] xgboost indisponível — modelo XGBoost ignorado.")

    cv = KFold(n_splits=5, shuffle=True, random_state=RS)
    modelos_metrics: dict[str, dict] = {}
    cv_means: dict[str, float] = {}

    # --- Validação cruzada no treino (seleção do vencedor) ---
    for nome, est in candidatos.items():
        pipe = _make_pipeline(est)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1
        )
        modelos_metrics[nome] = {
            "cv_r2_mean": float(scores.mean()),
            "cv_r2_std": float(scores.std()),
        }
        cv_means[nome] = float(scores.mean())
        print(
            f"  [REG] {nome:<20} cv_r2 = "
            f"{scores.mean():.4f} (+/- {scores.std():.4f})"
        )

    best_name = max(cv_means, key=cv_means.get)
    print(f"  [REG] Vencedor da validação cruzada: {best_name}")

    # --- GridSearchCV pequeno só no vencedor ---
    param_grids = {
        "Ridge": {"model__alpha": [0.5, 1.0, 5.0]},
        "RandomForest": {"model__max_depth": [None, 14, 22]},
        "GradientBoosting": {"model__learning_rate": [0.05, 0.1]},
        "XGBoost": {"model__max_depth": [4, 5, 6]},
    }
    pipe_best = _make_pipeline(candidatos[best_name])
    grid = GridSearchCV(
        pipe_best,
        param_grid=param_grids[best_name],
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_pipe = grid.best_estimator_  # já refit no treino completo
    print(f"  [REG] Melhores hiperparâmetros: {grid.best_params_}")

    def _avalia(pipe):
        yp = pipe.predict(X_test)
        r2 = float(r2_score(y_test, yp))
        rmse = float(np.sqrt(mean_squared_error(y_test, yp)))
        mae = float(mean_absolute_error(y_test, yp))
        return yp, r2, rmse, mae

    # --- Avaliação no teste (vencedor) ---
    y_pred, r2, rmse, mae = _avalia(best_pipe)
    modelos_metrics[best_name].update({"r2": r2, "rmse": rmse, "mae": mae})

    # --- Avaliação no teste dos demais candidatos (para o gráfico/JSON) ---
    for nome, est in candidatos.items():
        if nome == best_name:
            continue
        pipe = _make_pipeline(est)
        pipe.fit(X_train, y_train)
        _, r2_i, rmse_i, mae_i = _avalia(pipe)
        modelos_metrics[nome].update({"r2": r2_i, "rmse": rmse_i, "mae": mae_i})

    metrics = {
        "target": config.TARGET_REG,
        "best_model": best_name,
        "models": modelos_metrics,
    }

    bundle = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "task": "regression",
        "feature_columns": config.MODEL_FEATURES,
        "classes": None,
    }

    return bundle, metrics, y_pred


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _fig_comparacao(modelos_metrics: dict, chave: str, titulo: str, ylabel: str, caminho):
    """Gráfico de barras comparando uma métrica de teste entre modelos."""
    nomes = list(modelos_metrics.keys())
    valores = [modelos_metrics[n].get(chave, float("nan")) for n in nomes]

    fig, ax = plt.subplots(figsize=(8, 5))
    cores = sns.color_palette("viridis", len(nomes))
    barras = ax.bar(nomes, valores, color=cores)
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Modelo")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for barra, val in zip(barras, valores):
        if not np.isnan(val):
            ax.annotate(
                f"{val:.3f}",
                (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.xticks(rotation=15, ha="right")
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_matriz_confusao(cm: np.ndarray, caminho):
    """Heatmap anotado da matriz de confusão (linhas=verdadeiro, colunas=predito)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.RISK_ORDER,
        yticklabels=config.RISK_ORDER,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title("Matriz de Confusão — Classificação (teste)")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_pred_vs_real(y_true, y_pred, caminho):
    """Dispersão score real vs. predito (teste) com linha de identidade y=x."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolor="none", s=25, color="#1f77b4")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel("Score de risco real")
    ax.set_ylabel("Score de risco predito")
    ax.set_title("Regressão — Predito vs. Real (teste)")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Resumo no console
# ---------------------------------------------------------------------------
def _imprime_resumo(metrics: dict):
    """Imprime tabelas legíveis de comparação para classificação e regressão."""
    print("\n" + "=" * 70)
    print("RESUMO — CLASSIFICAÇÃO (alvo: categoria_risco)")
    print("=" * 70)
    clf = metrics["classification"]
    print(
        f"{'Modelo':<20}{'f1_macro':>10}{'acc':>9}{'roc_auc':>10}"
        f"{'cv_f1':>9}{'cv_std':>9}"
    )
    for nome, m in clf["models"].items():
        print(
            f"{nome:<20}{m['f1_macro']:>10.4f}{m['accuracy']:>9.4f}"
            f"{m['roc_auc_ovr']:>10.4f}{m['cv_f1_macro_mean']:>9.4f}"
            f"{m['cv_f1_macro_std']:>9.4f}"
        )
    print(f"  -> Melhor classificador: {clf['best_model']}")

    print("\n" + "=" * 70)
    print("RESUMO — REGRESSÃO (alvo: score_risco)")
    print("=" * 70)
    reg = metrics["regression"]
    print(f"{'Modelo':<20}{'r2':>10}{'rmse':>10}{'mae':>10}{'cv_r2':>10}{'cv_std':>10}")
    for nome, m in reg["models"].items():
        print(
            f"{nome:<20}{m['r2']:>10.4f}{m['rmse']:>10.4f}{m['mae']:>10.4f}"
            f"{m['cv_r2_mean']:>10.4f}{m['cv_r2_std']:>10.4f}"
        )
    print(f"  -> Melhor regressor: {reg['best_model']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def main() -> None:
    config.ensure_dirs()
    df = preprocessing.load_raw()
    print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # --- SPLIT COMPARTILHADO (mesmos índices para clf e reg) ---
    idx = df.index.to_numpy()
    estratos = df[config.TARGET_CLF]
    idx_train, idx_test = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=RS,
        stratify=estratos,
    )
    df_train = df.loc[idx_train]
    df_test = df.loc[idx_test]
    print(f"Split: treino={len(idx_train)} | teste={len(idx_test)}")

    # Deriva X/y de cada subconjunto, por tarefa.
    Xc_tr, yc_tr = preprocessing.prepare_xy(df_train, "classification")
    Xc_te, yc_te = preprocessing.prepare_xy(df_test, "classification")
    Xr_tr, yr_tr = preprocessing.prepare_xy(df_train, "regression")
    Xr_te, yr_te = preprocessing.prepare_xy(df_test, "regression")

    # --- CLASSIFICAÇÃO ---
    print("\n>>> Treinando modelos de CLASSIFICAÇÃO...")
    bundle_clf, metrics_clf, cm = treinar_classificacao(Xc_tr, yc_tr, Xc_te, yc_te)
    dump(bundle_clf, config.CLF_MODEL_PATH)
    print(f"  [CLF] Bundle salvo em: {config.CLF_MODEL_PATH}")

    # --- REGRESSÃO ---
    print("\n>>> Treinando modelos de REGRESSÃO...")
    bundle_reg, metrics_reg, yr_pred = treinar_regressao(Xr_tr, yr_tr, Xr_te, yr_te)
    dump(bundle_reg, config.REG_MODEL_PATH)
    print(f"  [REG] Bundle salvo em: {config.REG_MODEL_PATH}")

    # --- FIGURAS ---
    print("\n>>> Gerando figuras...")
    _fig_comparacao(
        metrics_clf["models"],
        chave="f1_macro",
        titulo="Comparação de Modelos — Classificação (F1-macro no teste)",
        ylabel="F1-macro",
        caminho=config.FIG_COMPARACAO_CLF,
    )
    _fig_comparacao(
        metrics_reg["models"],
        chave="r2",
        titulo="Comparação de Modelos — Regressão (R² no teste)",
        ylabel="R²",
        caminho=config.FIG_COMPARACAO_REG,
    )
    _fig_matriz_confusao(cm, config.FIG_MATRIZ_CONFUSAO)
    _fig_pred_vs_real(yr_te, yr_pred, config.FIG_REG_PRED_VS_REAL)
    print("  Figuras salvas em:", config.FIGURES_DIR)

    # --- metrics.json (schema do contrato) ---
    dist = (
        df[config.TARGET_CLF]
        .value_counts()
        .reindex(config.RISK_ORDER)
        .fillna(0)
        .astype(int)
    )
    metrics = {
        "classification": metrics_clf,
        "regression": metrics_reg,
        "dataset": {
            "n_rows": int(df.shape[0]),
            "n_features": int(len(config.MODEL_FEATURES)),
            "class_distribution": {k: int(v) for k, v in dist.items()},
        },
        "random_state": int(RS),
        "test_size": float(TEST_SIZE),
    }
    metrics = _to_native(metrics)
    with open(config.METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f"\nMétricas salvas em: {config.METRICS_JSON}")

    # --- Resumo no console ---
    _imprime_resumo(metrics)


if __name__ == "__main__":
    main()
