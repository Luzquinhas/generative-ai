"""
explain.py — Interpretabilidade do GAIE via SHAP.

Carrega os melhores modelos salvos (classificação e regressão), calcula os
valores SHAP sobre o MESMO conjunto de teste usado no treino (split
compartilhado do contrato), gera as figuras SHAP (beeswarm + barra) e escreve
``reports/shap_importance.json`` no schema do contrato.

A importância reportada agrega as colunas one-hot de "uf"/"macrorregiao" de
volta para as features ORIGINAIS de ``config.MODEL_FEATURES``.

Uso:
    python -m src.explain
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # backend sem display, deve vir antes do pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
import joblib  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

try:  # permite execução como módulo (`python -m src.explain`) ou script
    from . import config, preprocessing
except ImportError:  # pragma: no cover
    from src import config, preprocessing  # type: ignore


# Tamanho máximo da amostra de teste usada para explicar (velocidade do SHAP).
MAX_AMOSTRA = 500

# Features-chave que a literatura da OIT associa ao trabalho forçado e que
# esperamos ver entre as mais influentes (confirmação no final do relatório).
FEATURES_ESPERADAS = [
    "fornos_detectados",
    "idh",
    "distancia_centro_urbano_km",
    "vulnerabilidade_composta",
    "isolamento_fornos",
]


# ---------------------------------------------------------------------------
# Split compartilhado (idêntico ao de train.py)
# ---------------------------------------------------------------------------
def _split_indices(df):
    """Reproduz o split único do contrato e devolve os índices de teste.

    Um único ``train_test_split`` sobre os índices do df, estratificado por
    ``config.TARGET_CLF``. Os mesmos índices valem para classificação e
    regressão (ambas as tarefas explicam as mesmas regiões).
    """
    idx = np.arange(len(df))
    _, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET_CLF],
    )
    return test_idx


# ---------------------------------------------------------------------------
# Robustez SHAP
# ---------------------------------------------------------------------------
def _build_explainer(model, Xt):
    """Escolhe o explainer SHAP adequado ao tipo de estimador.

    TreeExplainer cobre RandomForest/GradientBoosting/XGBoost. Para modelos
    lineares (LogisticRegression/Ridge/LinearRegression) usa-se LinearExplainer;
    qualquer outro caso recai no ``shap.Explainer`` genérico.
    """
    nome_classe = type(model).__name__.lower()

    modelos_arvore = (
        "randomforest",
        "gradientboosting",
        "histgradientboosting",
        "extratrees",
        "decisiontree",
        "xgb",
        "lgbm",
        "catboost",
    )
    if any(tag in nome_classe for tag in modelos_arvore):
        return shap.TreeExplainer(model)

    modelos_lineares = ("logistic", "ridge", "linear", "lasso", "elasticnet", "sgd")
    if any(tag in nome_classe for tag in modelos_lineares):
        return shap.LinearExplainer(model, Xt)

    # Fallback genérico (pode ser mais lento, mas é robusto).
    return shap.Explainer(model, Xt)


def _extrair_shap_values(explainer, Xt):
    """Extrai os SHAP values normalizando para ``np.ndarray``.

    Retorna um array com um dos formatos:
      * regressão / binário: ``(n_amostras, n_features)``;
      * multiclasse:         ``(n_amostras, n_features, n_classes)``.

    Trata as variações de formato do shap 0.51: objeto ``Explanation``,
    lista de arrays (um por classe) e arrays empilhados.
    """
    try:
        valores = explainer.shap_values(Xt)
    except Exception:
        # Alguns explainers expõem apenas a interface de chamada (__call__).
        explicacao = explainer(Xt)
        valores = getattr(explicacao, "values", explicacao)

    # Lista de arrays (um por classe) -> empilha no eixo das classes.
    if isinstance(valores, list):
        valores = np.stack(valores, axis=-1)

    valores = np.asarray(valores)

    # Caso binário em que o TreeExplainer devolve (n, f, 2): mantemos como está
    # (será tratado como multiclasse pela função de importância). Para shape
    # (n, f) ou (n, f, c), nenhuma alteração adicional é necessária.
    return valores


def _shap_classe(shap_values: np.ndarray, n_features: int, classe_idx: int) -> np.ndarray:
    """Devolve os SHAP de UMA classe no formato ``(n_amostras, n_features)``.

    Aceita tanto o formato multiclasse ``(n, f, c)`` quanto o já bidimensional
    ``(n, f)`` (nesse caso o índice de classe é ignorado).
    """
    if shap_values.ndim == 3:
        return shap_values[:, :, classe_idx]
    return shap_values


def _importancia_por_coluna(shap_values: np.ndarray) -> np.ndarray:
    """Calcula ``mean(|SHAP|)`` por coluna transformada.

    * Formato ``(n, f)``: média do |SHAP| sobre as amostras.
    * Formato ``(n, f, c)``: média sobre amostras E classes (agregação total).
    """
    abs_vals = np.abs(shap_values)
    if abs_vals.ndim == 3:
        # média sobre amostras (eixo 0) e classes (eixo 2) -> vetor de tamanho f
        return abs_vals.mean(axis=(0, 2))
    return abs_vals.mean(axis=0)


# ---------------------------------------------------------------------------
# Agregação para features originais
# ---------------------------------------------------------------------------
def _agregar_para_originais(nomes: list[str], importancias: np.ndarray) -> list[dict]:
    """Agrega a importância das colunas transformadas nas features originais.

    Colunas one-hot iniciadas por "uf_" somam em "uf"; as iniciadas por
    "macrorregiao_" somam em "macrorregiao"; as demais mantêm o próprio nome.
    O resultado é ordenado por importância decrescente.
    """
    agregado: dict[str, float] = {}
    for nome, valor in zip(nomes, importancias):
        if nome.startswith("uf_"):
            chave = "uf"
        elif nome.startswith("macrorregiao_"):
            chave = "macrorregiao"
        else:
            chave = nome
        agregado[chave] = agregado.get(chave, 0.0) + float(valor)

    itens = [
        {"feature": feat, "mean_abs_shap": float(val)}
        for feat, val in agregado.items()
    ]
    itens.sort(key=lambda d: d["mean_abs_shap"], reverse=True)
    return itens


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _plot_summary(shap_values_2d: np.ndarray, Xt, nomes: list[str], titulo: str, caminho):
    """Gera o beeswarm summary do SHAP e salva em ``caminho``."""
    plt.figure()
    shap.summary_plot(shap_values_2d, Xt, feature_names=nomes, show=False)
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close("all")


def _plot_barra(itens: list[dict], titulo: str, caminho):
    """Gera a barra horizontal das 15 features ORIGINAIS mais importantes."""
    top = itens[:15]
    feats = [d["feature"] for d in top][::-1]      # invertido p/ maior no topo
    vals = [d["mean_abs_shap"] for d in top][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(feats, vals, color="#1f77b4")
    plt.xlabel("Importância média |SHAP|")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# Explicação por tarefa
# ---------------------------------------------------------------------------
def _explicar_tarefa(bundle, df_teste, task: str):
    """Calcula SHAP para uma tarefa, gera figuras e retorna o bloco do JSON.

    Retorna uma tupla ``(bloco_json, itens_agregados)`` em que ``bloco_json``
    segue o schema do contrato e ``itens_agregados`` é a lista ordenada de
    importâncias por feature original (usada também para impressão no console).
    """
    pipeline = bundle["pipeline"]
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    model_name = bundle.get("model_name", type(model).__name__)

    # X já engenheirado e restrito a MODEL_FEATURES (prepare_xy faz isso).
    X_sample, _ = preprocessing.prepare_xy(df_teste, task)
    Xt = pre.transform(X_sample)
    nomes = preprocessing.get_output_feature_names(pre)

    explainer = _build_explainer(model, Xt)
    shap_values = _extrair_shap_values(explainer, Xt)
    n_features = Xt.shape[1]

    if task == "regression":
        # Regressão: shap_values esperado (n, f). Se vier 3D (n, f, 1), achata.
        shap_2d = _shap_classe(shap_values, n_features, classe_idx=0)
        importancias = _importancia_por_coluna(shap_2d)

        _plot_summary(
            shap_2d, Xt, nomes,
            "SHAP — Regressão (score de risco)",
            config.FIG_SHAP_SUMMARY_REG,
        )
        itens = _agregar_para_originais(nomes, importancias)
        _plot_barra(
            itens,
            "Top 15 atributos — Regressão (mean |SHAP|)",
            config.FIG_SHAP_BAR_REG,
        )
    else:
        # Classificação multiclasse: beeswarm da classe "Alto Risco".
        # IMPORTANTE: o eixo de classes do array SHAP (n, f, c) segue a ordem de
        # model.classes_ (lexicográfica no estimador), que NÃO é a de RISK_ORDER.
        # Derivamos o índice da ordem REAL do estimador para não plotar a classe errada.
        classe_alvo = "Alto Risco"
        classes_modelo = list(getattr(model, "classes_", config.RISK_ORDER))
        classe_idx = (
            classes_modelo.index(classe_alvo)
            if classe_alvo in classes_modelo
            else 0
        )
        shap_2d = _shap_classe(shap_values, n_features, classe_idx=classe_idx)

        # Importância p/ barra/JSON: agregada sobre amostras E classes.
        importancias = _importancia_por_coluna(shap_values)

        _plot_summary(
            shap_2d, Xt, nomes,
            f'SHAP — Classificação (classe "{classe_alvo}")',
            config.FIG_SHAP_SUMMARY_CLF,
        )
        itens = _agregar_para_originais(nomes, importancias)
        _plot_barra(
            itens,
            "Top 15 atributos — Classificação (mean |SHAP|)",
            config.FIG_SHAP_BAR_CLF,
        )

    bloco = {"model": model_name, "feature_importance": itens}
    return bloco, itens


# ---------------------------------------------------------------------------
# Relatório no console
# ---------------------------------------------------------------------------
def _imprimir_top10(rotulo: str, itens: list[dict]) -> None:
    """Imprime o Top-10 de features por importância para a tarefa indicada."""
    print(f"\nTop-10 atributos por importância ({rotulo}):")
    for i, d in enumerate(itens[:10], start=1):
        print(f"  {i:2d}. {d['feature']:<28} {d['mean_abs_shap']:.5f}")


def _confirmar_features_chave(rotulo: str, itens: list[dict]) -> None:
    """Confirma se as features-chave esperadas estão entre as mais influentes."""
    top_nomes = [d["feature"] for d in itens]
    posicoes = {f: (top_nomes.index(f) + 1) for f in FEATURES_ESPERADAS if f in top_nomes}
    presentes_top10 = [f for f, p in posicoes.items() if p <= 10]

    if presentes_top10:
        detalhe = ", ".join(f"{f} (#{posicoes[f]})" for f in presentes_top10)
        print(
            f"Confirmação ({rotulo}): os fatores-chave esperados aparecem entre "
            f"os mais influentes — {detalhe}."
        )
    else:
        print(
            f"Atenção ({rotulo}): nenhum dos fatores-chave esperados "
            f"(fornos_detectados, idh, distancia_centro_urbano_km e derivados) "
            f"ficou no Top-10."
        )


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def main() -> None:
    """Carrega modelos, calcula SHAP, gera figuras e grava o JSON de importância."""
    config.ensure_dirs()

    # Verifica a presença dos modelos antes de qualquer cálculo.
    faltando = [
        p for p in (config.CLF_MODEL_PATH, config.REG_MODEL_PATH) if not p.exists()
    ]
    if faltando:
        print("Modelos não encontrados:")
        for p in faltando:
            print(f"  - {p}")
        print("Rode primeiro o treino para gerá-los:  python -m src.train")
        return

    df = preprocessing.load_raw()
    test_idx = _split_indices(df)
    df_teste = df.iloc[test_idx].copy()

    # Amostra de até MAX_AMOSTRA linhas do teste (velocidade do SHAP).
    if len(df_teste) > MAX_AMOSTRA:
        df_teste = df_teste.sample(
            n=MAX_AMOSTRA, random_state=config.RANDOM_STATE
        )

    print(
        f"Explicando sobre {len(df_teste)} regiões do conjunto de teste "
        f"(split estratificado, random_state={config.RANDOM_STATE})."
    )

    bundle_clf = joblib.load(config.CLF_MODEL_PATH)
    bundle_reg = joblib.load(config.REG_MODEL_PATH)

    print("\nCalculando SHAP — classificação...")
    bloco_clf, itens_clf = _explicar_tarefa(bundle_clf, df_teste, "classification")

    print("Calculando SHAP — regressão...")
    bloco_reg, itens_reg = _explicar_tarefa(bundle_reg, df_teste, "regression")

    # --- Escrita do JSON no schema do contrato ---
    saida = {"classification": bloco_clf, "regression": bloco_reg}
    with open(config.SHAP_IMPORTANCE_JSON, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)

    # --- Relatório no console ---
    _imprimir_top10("classificação", itens_clf)
    _imprimir_top10("regressão", itens_reg)

    print()
    _confirmar_features_chave("classificação", itens_clf)
    _confirmar_features_chave("regressão", itens_reg)

    print(f"\nFiguras SHAP salvas em: {config.FIGURES_DIR}")
    print(f"Importâncias SHAP salvas em: {config.SHAP_IMPORTANCE_JSON}")


if __name__ == "__main__":
    main()
