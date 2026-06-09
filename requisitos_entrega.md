# Diretrizes do Projeto Prático: Inteligência Artificial e Machine Learning na Economia Espacial

## 1. Visão Geral do Projeto
O objetivo deste projeto é que cada equipe conceba, desenvolva, treine e documente um pipeline completo de Inteligência Artificial e Machine Learning aplicado a um problema real relacionado à **Economia Espacial**. O trabalho deve demonstrar, de forma prática, os principais conceitos e técnicas estudados ao longo deste semestre.

Cada equipe deverá escolher um problema relevante da Economia Espacial e desenvolver uma solução baseada em IA/ML que inclua, obrigatoriamente, as etapas descritas a seguir.

---

## 2. Escopo do Projeto e Etapas Obrigatórias

### 1. Obtenção ou Geração dos Dados
* **Opção A:** Utilizar uma API para coletar dados reais relacionados ao tema escolhido.
* **Opção B:** Utilizar uma IA generativa para criar um conjunto de dados sintético contendo, no mínimo, **1.000 linhas e 10 colunas**.

### 2. Desenvolvimento de Modelos Preditivos
* Aplicar pelo menos **duas técnicas diferentes** estudadas durante o semestre.
* O problema pode ser de **Regressão, Classificação ou Clusterização**.

### 3. Construção do Pipeline de Machine Learning
O projeto deve contemplar um fluxo de trabalho completo contendo:
* Pré-processamento de dados
* Engenharia e seleção de atributos (feature engineering)
* Treinamento dos modelos
* Validação e comparação de desempenho
* Escolha do melhor modelo
* **Deploy da solução** utilizando ferramentas como Gradio, Streamlit, Flask, FastAPI, Ngrok ou similares.

### 4. Interpretabilidade do Modelo
* Utilizar **SHAP (SHapley Additive exPlanations)** para analisar como o modelo toma decisões.
* Identificar e explicar quais variáveis tiveram maior influência nas previsões.

### 5. Documentação e Reprodutibilidade
* Disponibilizar todo o código-fonte em Python em um repositório no **GitHub**.
* Elaborar um **README detalhado** contendo:
    * Contexto do problema
    * Fonte dos dados
    * Metodologia utilizada
    * Modelos testados
    * Resultados obtidos
    * Interpretação com SHAP
    * Instruções para execução do projeto
    * Link para a aplicação em funcionamento

---

## 3. Critérios de Avaliação e Distribuição de Notas

| Critério de Avaliação | Descrição | Peso / Valor |
| :--- | :--- | :---: |
| **1. Definição do problema e dados** | Clareza na escolha do problema da Economia Espacial e qualidade/estrutura dos dados obtidos ou gerados. | 15 pts |
| **2. Pré-processamento e atributos** | Qualidade do tratamento de dados e relevância das técnicas de engenharia de atributos aplicadas. | 20 pts |
| **3. Aplicação e comparação de modelos**| Correção técnica na implementação e teste de pelo menos duas abordagens distintas de algoritmos. | 20 pts |
| **4. Validação e análise de métricas** | Rigor na escolha das métricas de avaliação e na análise comparativa de performance. | 15 pts |
| **5. Interpretabilidade com SHAP** | Capacidade de explicar o comportamento do modelo e o impacto das variáveis usando SHAP. | 10 pts |
| **6. Deploy da aplicação** | Funcionamento, usabilidade e acessibilidade da interface de usuário criada para o modelo. | 10 pts |
| **7. Organização e GitHub** | Estrutura clara do repositório, boas práticas de código e qualidade do arquivo README. | 10 pts |
| **Total** | | **100 pts** |

---

## 4. Formato e Instruções de Entrega

Cada equipe deverá submeter os seguintes itens:
1.  **Link do repositório no GitHub** contendo todo o código-fonte, scripts, arquivos de configuração e a documentação completa.
2.  **Link da aplicação em funcionamento** (URL pública do deploy da solução).
