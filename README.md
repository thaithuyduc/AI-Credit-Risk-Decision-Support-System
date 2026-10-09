# Credit Risk Decision Support System  
### Machine Learning + Explainable AI + AI Agent

A Decision Support System for credit risk assessment that combines Machine Learning, Explainable AI (SHAP), and an AI Agent with Retrieval-Augmented Generation (RAG) to support loan officers in evaluating individual credit applications.

---

## 📌 Problem

Lending is one of the primary revenue sources for banks and financial institutions, yet it carries significant **Credit Risk** — the risk that a customer will default on their obligations.

Traditional approaches (scorecards and manual underwriting) have clear limitations in accuracy and consistency. Modern Machine Learning models can predict Probability of Default (PD) more accurately, but they are often treated as **black-box** systems. In a highly regulated industry like banking, the inability to explain model decisions creates a major barrier to adoption.

In addition, credit officers still spend considerable time manually compiling prediction results, looking up internal credit policies, and writing assessment reports. There is a clear need for a system that can not only predict risk accurately, but also **explain** the prediction and **recommend** actions grounded in business rules.

---

## 🎯 Objective

The goal of this project is to build a **Credit Risk Decision Support System** that assists (rather than replaces) credit officers by providing:

- Accurate prediction of late payment / default risk
- Transparent explanations of the key drivers behind each prediction
- Automated generation of structured decision-support reports that combine model outputs with internal credit policies

The system integrates three core components:
1. Supervised Machine Learning models for risk prediction
2. Explainable AI (SHAP) for local and global interpretability
3. An AI Agent powered by RAG and tool-calling to produce actionable recommendations

---

## 🏆 Main Results

- Successfully built an end-to-end **Data Pipeline** that handles multi-table credit data (application, bureau, previous applications, etc.), prevents data leakage, and produces model-ready features.
- Evaluated and selected **LightGBM** (with class weighting + sigmoid calibration) as the final model:
  - ROC-AUC: **0.771** (Validation) → **0.775** (Test)
  - Optimal decision threshold (cost-sensitive): **0.08645**
  - Recall ≈ **0.690**, F-beta ≈ **0.440**, KS Statistic ≈ **0.412**
- Applied **SHAP** to provide both local (individual customer) and global feature importance explanations, solving the black-box problem.
- Designed and implemented an **AI Credit Risk Decision Support Agent** that:
  - Calls the ML model and SHAP
  - Retrieves relevant credit policies via RAG
  - Generates structured reports (customer summary, risk explanation, and business recommendations)
- Demonstrated the full workflow on real use cases covering True Positive, False Positive, True Negative, and False Negative scenarios.

## Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/33489d5e-691d-45b1-bb46-919d872b5da5" alt="Architecture Overview" width="639">
</p>
<p align="center"><em>Data Pipeline</em></p>

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c23f3fb8-34df-41cb-ae61-ddf923fcd057" alt="Architecture Detail" width="631">
</p>
<p align="center"><em>System Architecture</em></p>

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/884262e6-2b5e-4a5f-89e6-4b9f9e8267b6" alt="Architecture Component" width="524">
</p>
<p align="center"><em>Agentic Workflow inspired by ReAct Framework</em></p>

---

## Demo

<img width="650" height="619" alt="image" src="https://github.com/user-attachments/assets/def9ba8c-b6b4-446e-9a12-7b4c45c6e1a5" />

---

## 💡 Main Contributions

1. **Methodological Integration**  
   Combined Machine Learning, SHAP-based Explainable AI, and an Agentic workflow with RAG into a unified Decision Support System for credit risk.

2. **Bridging Quantitative Explanations and Business Rules**  
   Converted SHAP values into structured, human-readable risk explanations and linked them with internal credit policies through RAG, addressing a common gap in existing XAI research.

3. **Business-Oriented Model Optimization**  
   Moved beyond pure machine learning metrics by optimizing the decision threshold using a cost-sensitive approach that reflects the asymmetric cost of False Negatives in credit risk.

4. **Human-in-the-Loop Design**  
   The system is explicitly designed as a **Decision Support System** rather than a fully automated underwriting engine, ensuring credit officers retain final decision authority.

5. **Practical End-to-End Pipeline**  
   Delivered a complete workflow from raw multi-source data → feature engineering → model prediction → explanation → policy-grounded recommendation → final report generation.

---

## 🛠️ Tech Stack

- **Machine Learning**: LightGBM, XGBoost, Logistic Regression, Scikit-learn  
- **Explainable AI**: SHAP  
- **AI Agent & RAG**: LangChain / LangGraph, Vector Database, Embedding models  
- **Data Processing**: Pandas, NumPy  
- **Evaluation**: ROC-AUC, PR-AUC, KS Statistic, Calibration metrics, Cost-sensitive threshold optimization

---

## 📂 Project Structure

```text
├── data/                   # Raw and processed datasets
├── notebooks/              # EDA, model training, SHAP analysis
├── src/                    # include all source code in this project
├── reports/                # Sample generated decision reports
└── README.md
└── main.py                 # Used to run application
