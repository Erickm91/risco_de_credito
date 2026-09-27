# 💳 Credit Risk Classification

Modelo de Machine Learning para prever a probabilidade de **inadimplência** (*loan default*) de clientes, a partir de dados de perfil pessoal e características do empréstimo — com foco em boas práticas de modelagem (pipeline sem vazamento de dados, validação cruzada e interpretabilidade).

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-purple)

## 🎯 Objetivo

Classificar se um cliente vai ou não inadimplir (`loan_status`: 0 = adimplente, 1 = inadimplente), comparando diferentes algoritmos de classificação e otimizando o modelo final para apoiar decisões de concessão de crédito.

## 📊 Sobre os Dados

O dataset contém informações de **32.581 clientes**, com as seguintes variáveis:

| Coluna | Descrição |
|---|---|
| `person_age` | Idade do cliente |
| `person_income` | Renda anual |
| `person_home_ownership` | Tipo de moradia |
| `person_emp_length` | Tempo de emprego (anos) |
| `loan_intent` | Finalidade do empréstimo |
| `loan_grade` | Grau do empréstimo (A a G) |
| `loan_amnt` | Valor do empréstimo |
| `loan_int_rate` | Taxa de juros |
| `loan_status` | Variável alvo (0 = adimplente, 1 = inadimplente) |
| `loan_percent_income` | Percentual da renda comprometida |
| `cb_person_default_on_file` | Histórico de inadimplência |
| `cb_person_cred_hist_length` | Tempo de histórico de crédito |

## 🔍 Etapas do Projeto

1. **Análise Exploratória (EDA)** — distribuição das variáveis e relação entre inadimplência, tipo de moradia, finalidade do empréstimo, grau do empréstimo e renda.
2. **Tratamento de Dados** — remoção de duplicatas, tratamento de outliers (idade, renda, tempo de emprego) e imputação de valores ausentes.
3. **Pipeline sem vazamento de dados** — toda imputação e transformação é ajustada (`fit`) apenas no conjunto de treino, dentro de um `Pipeline`, evitando que estatísticas de validação/teste influenciem o modelo.
4. **Pré-processamento** — padronização de variáveis numéricas, encoding ordinal para `loan_grade` (respeitando a ordem A→G) e one-hot encoding para as demais variáveis categóricas.
5. **Comparação de Modelos** — Logistic Regression, Decision Tree, Random Forest, KNN e XGBoost, todos avaliados com balanceamento de classes via **SMOTE** dentro do pipeline (`imblearn.Pipeline`).
6. **Otimização de Hiperparâmetros** — `RandomizedSearchCV` com validação cruzada estratificada (`StratifiedKFold`), otimizando F1-score para equilibrar precision e recall na classe de inadimplentes.
7. **Análise de Threshold** — avaliação do trade-off entre precision e recall em diferentes pontos de corte, permitindo ajustar a decisão conforme a política de risco do negócio.
8. **Interpretabilidade** — importância de features (nativa do XGBoost) e valores SHAP, para entender quais variáveis mais influenciam as previsões do modelo.

## 🏆 Resultado

O **XGBoost** foi o modelo com melhor desempenho geral:

| Modelo              | Precision | Recall | F1-Score | Acurácia |
| ------------------- | --------- | ------ | -------- | -------- |
| Logistic Regression | 0.72      | 0.79   | 0.74     | 0.7965   |
| Decision Tree        | 0.81      | 0.84   | 0.82     | 0.8741   |
| Random Forest        | 0.94      | 0.86   | 0.89     | 0.9329   |
| KNN                  | 0.73      | 0.79   | 0.75     | 0.8051   |
| **XGBoost**          | **0.94**  | **0.86** | **0.90** | **0.9350** |

Após o tuning de hiperparâmetros, o modelo final atingiu **ROC-AUC de ~0.95** no conjunto de validação.

## 🛠️ Tecnologias

- Python
- Pandas / NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- SHAP
- Matplotlib / Seaborn
