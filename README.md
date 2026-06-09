# 🛰️ GAIE — Generative AI For Engineering

> **Combate ao trabalho forçado com IA + dados geoespaciais.**
> Um pipeline completo de Machine Learning que estima a **probabilidade/risco de
> trabalho forçado por região**, cruzando a **detecção de fornos** (carvoarias e
> olarias, via imagens de satélite) com **indicadores socioeconômicos** locais.

Projeto prático da disciplina de **Inteligência Artificial e Machine Learning na
Economia Espacial**. Demonstra, ponta a ponta, geração de dados, pré-processamento,
engenharia de atributos, treino e comparação de modelos, validação, interpretabilidade
com **SHAP** e **deploy** de uma aplicação web interativa em **Streamlit**.

**Integrantes:** 
- Lucas Rodrigues da Silva | RM: 98344
- Juan Pinheiro de França  | RM: 552202 
- Kaiky Alvaro de Miranda  | RM: 98118

---

## 1. Contexto do problema

O **trabalho análogo à escravidão** persiste em regiões isoladas do Brasil,
frequentemente associado à produção de **carvão vegetal** (carvoarias) e cerâmica
(olarias). Esses fornos são detectáveis por **sensoriamento remoto/satélite**, o que
abre uma oportunidade de **Economia Espacial**: combinar a contagem de fornos por
região com indicadores socioeconômicos (IDH, pobreza, informalidade, isolamento
geográfico, fiscalização) para **priorizar regiões** de fiscalização e ação preventiva.

O **GAIE** ataca esse problema sob duas óticas complementares:

| Ótica | Tarefa | Variável-alvo |
| :--- | :--- | :--- |
| **Onde agir primeiro?** | **Classificação** | `categoria_risco` ∈ {Baixo, Médio, Alto Risco} |
| **Qual a intensidade do risco?** | **Regressão** | `score_risco` contínuo (0–100) |

Como subproduto, estima-se `pessoas_em_risco_estimadas` por região (exibido no app).

> ⚠️ **Aviso ético:** os dados são **100% sintéticos**, gerados programaticamente
> para fins **acadêmicos**. Não representam pessoas, municípios ou ocorrências reais.

---

## 2. Fonte dos dados

Seguindo a **Opção B** do edital (IA generativa para criar dados sintéticos), o
projeto usa um **modelo gerador causal** ([src/data_generation.py](src/data_generation.py))
que produz um dataset com **2.000 regiões (linhas) × 21 colunas** (acima do mínimo de
1.000 × 10). O gerador embute relações causais realistas para que o sinal seja
**aprendível e interpretável**:

1. Cada região pertence a uma das **27 UFs** (centróide geográfico + prior de risco
   calibrado pela incidência histórica de trabalho forçado — ex.: PA, MA, MT, TO, BA, MG).
2. Um **fator latente de desenvolvimento** gera, de forma correlacionada, os
   indicadores socioeconômicos (IDH, pobreza, PIB per capita, educação, informalidade,
   cobertura de fiscalização).
3. Indicadores ambientais/geográficos (**isolamento**, **desmatamento**) e a
   **detecção de fornos** (processo de Poisson) dependem do desenvolvimento e da UF.
4. Um **risco latente** combina os fatores padronizados — com **maior peso** para
   (1) nº de fornos, (2) baixo IDH e (3) isolamento geográfico — e vira `score_risco`
   via função logística, do qual derivam `categoria_risco` e `pessoas_em_risco_estimadas`.

São injetados **valores ausentes** (~3–5%) em algumas colunas para exercitar o
pré-processamento de forma realista.

<details>
<summary><b>Dicionário de dados (clique para expandir)</b></summary>

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `regiao_id`, `municipio` | id | Identificadores sintéticos |
| `uf`, `macrorregiao` | categórico | Unidade Federativa e macrorregião |
| `latitude`, `longitude` | geo | Coordenadas (usadas no mapa) |
| `fornos_detectados` | int | Nº de fornos por satélite (**variável-chave**) |
| `densidade_populacional` | float | hab/km² |
| `pib_per_capita` | float | R$/habitante/ano |
| `distancia_centro_urbano_km` | float | Isolamento geográfico |
| `indice_pobreza` | float | 0–100 |
| `idh` | float | 0–1 |
| `taxa_informalidade` | float | % de trabalho informal |
| `casos_oit_historicos` | int | Casos/resgates históricos (OIT) |
| `cobertura_fiscalizacao` | float | % de cobertura de fiscalização |
| `area_desmatada_km2` | float | Proxy de pressão por carvão |
| `acesso_educacao` | float | Índice 0–100 |
| `populacao_total` | int | Habitantes |
| `score_risco` | float | **Alvo (regressão)** 0–100 |
| `categoria_risco` | categórico | **Alvo (classificação)** |
| `pessoas_em_risco_estimadas` | int | Derivado |

</details>

> **Extensível por API:** a arquitetura permite complementar/substituir o dataset
> sintético por dados reais (IBGE, Atlas do Desenvolvimento Humano, MapBiomas para
> desmatamento, Radar/SAR para detecção de fornos) mantendo o mesmo contrato de colunas
> em [src/config.py](src/config.py).

---

## 3. Metodologia (pipeline de ML)

```
Geração → Pré-processamento → Feature Engineering → Treino → Validação
        → Comparação → Escolha do melhor modelo → Interpretabilidade (SHAP) → Deploy
```

- **Pré-processamento** ([src/preprocessing.py](src/preprocessing.py)): todo o
  tratamento vive **dentro de um `Pipeline`/`ColumnTransformer`** do scikit-learn —
  imputação (mediana p/ numéricas, moda p/ categóricas), **padronização**
  (`StandardScaler`) e **One-Hot Encoding** (com `handle_unknown="ignore"`). Como o
  pré-processador é ajustado **somente no treino** dentro do pipeline, **não há
  vazamento de dados** (data leakage) para validação/teste.
- **Engenharia de atributos** (6 derivados): `fornos_por_1000hab`,
  `vulnerabilidade_composta` (= pobreza × (1−IDH)), `isolamento_fornos`
  (= distância × log(1+fornos)), `pressao_carvao`, `log_populacao`,
  `deficit_fiscalizacao`. Após o One-Hot, o modelo recebe **50 colunas**.
- **Prevenção de vazamento:** identificadores, geolocalização e as três colunas-alvo
  nunca entram como atributo.
- **Split único e estratificado** (80/20, `random_state=42`, estratificado por
  `categoria_risco`), **reutilizado** por classificação e regressão — garantindo
  comparabilidade entre as duas tarefas e com o SHAP.
- **Validação cruzada** 5-fold no treino (StratifiedKFold p/ classificação, KFold p/
  regressão) para selecionar o vencedor; **GridSearchCV** refina os hiperparâmetros do
  vencedor; o modelo final é avaliado no **conjunto de teste** retido.

---

## 4. Modelos testados

| Tarefa | Técnicas comparadas | Métrica de seleção |
| :--- | :--- | :--- |
| **Classificação** | Regressão Logística · Random Forest · Gradient Boosting | F1-macro (CV) |
| **Regressão** | Ridge · Random Forest · Gradient Boosting · **XGBoost** | R² (CV) |

(Atende e supera o requisito de "pelo menos duas técnicas diferentes".)

---

## 5. Resultados obtidos

### Classificação — alvo `categoria_risco` (conjunto de teste, n = 400)

| Modelo | F1-macro | Acurácia | Precisão (macro) | Revocação (macro) | ROC-AUC (OvR) | CV F1-macro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** ✅ | **0,936** | **0,933** | 0,931 | 0,942 | **0,993** | 0,930 ± 0,013 |
| Gradient Boosting | 0,922 | 0,918 | 0,925 | 0,919 | 0,987 | 0,916 ± 0,011 |
| Random Forest | 0,905 | 0,903 | 0,913 | 0,898 | 0,982 | 0,921 ± 0,013 |

**Melhor modelo: Regressão Logística** — F1-macro por classe: Baixo 0,933 · Médio 0,924 · Alto **0,950**.
A matriz de confusão é fortemente diagonal (ver [reports/figures/](reports/figures/)).

### Regressão — alvo `score_risco` (conjunto de teste, n = 400)

| Modelo | R² | RMSE | MAE | CV R² |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** ✅ | **0,985** | **2,53** | **1,95** | 0,979 ± 0,002 |
| Gradient Boosting | 0,980 | 2,98 | 2,32 | 0,975 ± 0,001 |
| Ridge | 0,972 | 3,51 | 2,78 | 0,966 ± 0,003 |
| Random Forest | 0,971 | 3,58 | 2,78 | 0,962 ± 0,004 |

**Melhor modelo: XGBoost** — erro médio absoluto de ~2 pontos num score de 0 a 100.

> Que a **Regressão Logística** vença na classificação e o **XGBoost** na regressão é
> coerente: a fronteira entre classes é bem separável no espaço de atributos engenheirados
> (favorecendo um modelo linear robusto), enquanto o score contínuo se beneficia das
> interações não-lineares capturadas pelo boosting.

Métricas completas: [reports/metrics.json](reports/metrics.json).

---

## 6. Interpretação com SHAP

Usamos **SHAP** ([src/explain.py](src/explain.py)) para auditar as decisões dos modelos.
As contribuições das colunas One-Hot são **agregadas de volta** aos atributos originais.

**Top atributos por importância média |SHAP| (classificação):**

1. `distancia_centro_urbano_km` (isolamento geográfico)
2. `isolamento_fornos` (**distância × nº de fornos**)
3. `idh` (baixo IDH)
4. `pib_per_capita`
5. `taxa_informalidade`

✅ **Confirma a hipótese do projeto.** Os fatores de maior peso são exatamente a tríade
esperada — **detecção de fornos + baixo IDH + isolamento geográfico**. Note que o atributo
de maior peso **`isolamento_fornos`** combina literalmente *fornos* e *isolamento*, e a
`vulnerabilidade_composta` (pobreza × baixo IDH) lidera a regressão. As figuras
`shap_summary_*.png` e `shap_bar_*.png` em [reports/figures/](reports/figures/) detalham
direção e magnitude do efeito de cada atributo.

---

## 7. Como executar

### Pré-requisitos
- Python **3.11** (testado), `pip`.

### Instalação
```bash
# 1) (opcional) ambiente virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate    |  Linux/Mac:  source .venv/bin/activate

# 2) dependências
pip install -r requirements.txt
```

### Reproduzir o pipeline completo (dados → modelos → SHAP)
```bash
python run_pipeline.py
# ou, etapa a etapa:
python -m src.data_generation   # gera data/gaie_dataset.csv
python -m src.train             # treina, compara e salva models/ + reports/metrics.json
python -m src.explain           # calcula SHAP -> reports/shap_importance.json + figuras
```
> O pipeline é **determinístico** (`random_state=42`): reexecutar reproduz os mesmos
> resultados.

### Subir a aplicação web
```bash
streamlit run app.py
```
A aplicação abre em `http://localhost:8501` e funciona mesmo sem os modelos treinados
(as seções dependentes exibem um aviso amigável). Para a experiência completa, rode o
pipeline antes.

---

## 8. A aplicação (Streamlit)

🔗 **Aplicação em funcionamento:** 
(ver [DEPLOY](https://generative-ai-nemesis.streamlit.app/)).

Cinco seções:

- **🗺️ Mapa de Risco** — mapa interativo (Plotly) das regiões, colorido por categoria
  de risco, com filtros (UF, macrorregião, categoria, faixa de score, nº mínimo de
  fornos), métricas-resumo, tabela e **download do CSV filtrado**.
- **🎯 Simulador de Risco** — informe os atributos de uma região hipotética e obtenha
  **categoria prevista + probabilidades** (classificação) e **score 0–100** num medidor
  (regressão), além das pessoas em risco estimadas.
- **📊 Comparação de Modelos** — tabelas de métricas (destacando o vencedor), figuras de
  avaliação e a distribuição de classes.
- **🧠 Interpretabilidade (SHAP)** — ranking de importância e gráficos SHAP.
- **ℹ️ Sobre** — contexto, dados e metodologia.

---

## 9. Deploy (opcional)

**Streamlit Community Cloud** (gratuito):
1. Suba o repositório para o GitHub.
2. Em [share.streamlit.io](https://share.streamlit.io), conecte o repositório e aponte
   para `app.py`.
3. Importante: o `.gitignore` ignora `models/*.joblib` e `reports/figures/*.png` por
   serem **regeneráveis**. Para o deploy, **comente essas linhas** no
   [.gitignore](.gitignore) e versione os artefatos, **ou** rode `python run_pipeline.py`
   como passo de build. O app gera o dataset/avisos automaticamente caso falte algo.

---

## 10. Estrutura do repositório

```
.
├── app.py                      # Aplicação Streamlit (deploy)
├── run_pipeline.py             # Orquestra dados → treino → SHAP
├── requirements.txt
├── src/
│   ├── config.py               # Contrato compartilhado (paths, features, alvos, UFs)
│   ├── data_generation.py      # Gerador sintético causal (≥ 1.000 × 10)
│   ├── preprocessing.py         # Feature engineering + ColumnTransformer
│   ├── train.py                # Treino, comparação, seleção e métricas
│   └── explain.py              # Interpretabilidade com SHAP
├── data/                       # gaie_dataset.csv (gerado)
├── models/                     # *.joblib (gerados)
└── reports/
    ├── metrics.json            # Métricas e comparação
    ├── shap_importance.json    # Importância de atributos (SHAP)
    └── figures/                # Gráficos (comparação, confusão, SHAP)
```

---

## 11. Mapa para os critérios de avaliação

| Critério (edital) | Onde está |
| :--- | :--- |
| 1. Definição do problema e dados | Seções 1–2 · [src/data_generation.py](src/data_generation.py) |
| 2. Pré-processamento e atributos | Seção 3 · [src/preprocessing.py](src/preprocessing.py) |
| 3. Aplicação e comparação de modelos | Seções 4–5 · [src/train.py](src/train.py) |
| 4. Validação e análise de métricas | Seção 5 · CV + teste · [reports/metrics.json](reports/metrics.json) |
| 5. Interpretabilidade com SHAP | Seção 6 · [src/explain.py](src/explain.py) |
| 6. Deploy da aplicação | Seções 8–9 · [app.py](app.py) |
| 7. Organização e GitHub | Este README · estrutura modular · `random_state` fixo |

---

_Tecnologias: Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Plotly ·
Matplotlib/Seaborn · Streamlit._
