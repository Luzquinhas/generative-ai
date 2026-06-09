"""
run_pipeline.py — Orquestrador do pipeline completo do GAIE.

Executa, em ordem, as etapas de Machine Learning:

    1. Geração dos dados sintéticos   (src.data_generation)
    2. Treino, validação e comparação (src.train)
    3. Interpretabilidade com SHAP     (src.explain)

Após a execução, os artefatos ficam disponíveis em:
    data/gaie_dataset.csv          -> dataset
    models/*.joblib                -> melhores modelos (classificação e regressão)
    reports/metrics.json           -> métricas e comparação de modelos
    reports/shap_importance.json   -> importância de atributos (SHAP)
    reports/figures/*.png          -> gráficos (comparação, matriz de confusão, SHAP)

Em seguida, suba a aplicação web:
    streamlit run app.py

Uso:
    python run_pipeline.py
    python run_pipeline.py --skip-data   # reaproveita o dataset já gerado
"""

from __future__ import annotations

import argparse
import time

from src import config, data_generation, explain, train


def _step(titulo: str) -> None:
    print("\n" + "=" * 72)
    print(f">>> {titulo}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo do GAIE.")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Não regenera o dataset (usa data/gaie_dataset.csv existente).",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    inicio = time.perf_counter()

    if args.skip_data and config.DATA_RAW.exists():
        _step("1/3 — Dados (reaproveitando dataset existente)")
        print(f"Usando: {config.DATA_RAW}")
    else:
        _step("1/3 — Geração do dataset sintético")
        data_generation.main()

    _step("2/3 — Treino, validação e comparação de modelos")
    train.main()

    _step("3/3 — Interpretabilidade com SHAP")
    explain.main()

    dur = time.perf_counter() - inicio
    _step(f"Pipeline concluído em {dur:.1f}s")
    print("Artefatos em models/ e reports/. Para a aplicação: streamlit run app.py")


if __name__ == "__main__":
    main()
