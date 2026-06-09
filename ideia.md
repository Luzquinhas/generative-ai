# GAIE — Generative AI For Engineering

## Visão Geral do Projeto
O projeto **GAIE (Generative AI For Engineering)** visa combater o trabalho forçado por meio da inteligência artificial, utilizando dados geoespaciais e indicadores socioeconômicos para identificar regiões de vulnerabilidade e criticidade.

---

## O Que Entregar
Um modelo preditivo que estima a **probabilidade de ocorrência de trabalho forçado por região**, realizando o cruzamento de dados de detecção de fornos (por exemplo, de carvoarias ou olarias via imagens de satélite) com indicadores socioeconômicos locais.

---

## Passo a Passo da Implementação

### 1. Dados (Dataset Sintético e APIs)
* **Abordagem:** Geração de um dataset sintético com, no mínimo, **1.000 linhas × 10 colunas** para atender aos requisitos regulamentares/edital, com possibilidade de complementação via consumo de APIs onde houver dados disponíveis.
* **Variáveis/Colunas do Dataset:**
  * Número de fornos detectados na região
  * Densidade populacional
  * PIB local
  * Distância de centros urbanos (isolamento geográfico)
  * Índice de pobreza
  * Índice de Desenvolvimento Humano (IDH)
  * Históricos/Dados da OIT (Organização Internacional do Trabalho) sobre trabalho forçado
  * *Outras métricas socioeconômicas e geográficas relevantes.*

### 2. Modelagem Preditiva
O problema será abordado sob duas óticas complementares:
* **Modelo 1 (Classificação):** Segmentação da região em categorias de risco: `"Alto Risco"`, `"Médio Risco"` ou `"Baixo Risco"` de trabalho forçado.
* **Modelo 2 (Regressão):** Geração de um *score* contínuo de risco ou estimativa numérica do número de pessoas em situação de vulnerabilidade/risco na região.

### 3. Pipeline Completo de Machine Learning
O fluxo de desenvolvimento seguirá as melhores práticas de Engenharia de Aprendizado de Máquina:
$$	ext{Pré-processamento} \longrightarrow 	ext{Feature Engineering} \longrightarrow 	ext{Treino} \longrightarrow 	ext{Validação} \longrightarrow 	ext{Comparação} \longrightarrow 	ext{Escolha do Melhor Modelo} \longrightarrow 	ext{Deploy}$$

### 4. Explicabilidade e Transparência (SHAP)
* Utilização de valores **SHAP (SHapley Additive exPlanations)** para garantir a interpretabilidade e auditoria do modelo.
* **Insights Esperados:** Demonstração científica (alinhada com a literatura da área) de que fatores como `"número de fornos ativos + baixo IDH + isolamento geográfico"` são os atributos com maior peso (*feature importance*) na composição do risco.

### 5. Deploy e Interface do Usuário
* **Tecnologia:** Desenvolvimento da aplicação web utilizando **Streamlit** ou **Gradio**.
* **Funcionalidade Principal:** Um **mapa de risco interativo** que permite aos órgãos fiscalizadores e tomadores de decisão visualizar graficamente as áreas de atenção e filtrar os dados por região, nível de risco e indicadores associados.
