# 🛒 AI-Powered Customer Segmentation & Next-Best-Offer Recommendation System

> *Personalization and Next-Best-Offer Recommendation: A Machine Learning and Business Intelligence Framework for Data-Driven Marketing Decisions*

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow/Keras-LSTM-orange?style=flat-square&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means%20%7C%20RFM-f7931e?style=flat-square&logo=scikit-learn)
![mlxtend](https://img.shields.io/badge/mlxtend-Apriori-green?style=flat-square)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-blue?style=flat-square&logo=tableau)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
![Timeline](https://img.shields.io/badge/Timeline-Sep%202025%20–%20Jan%202026-lightgrey?style=flat-square)

---

## 📌 Overview

This project builds and evaluates an end-to-end **customer segmentation and Next-Best-Offer (NBO) recommendation framework** using the real-world **Instacart Market Basket dataset** (3.4 million grocery orders, ~206,000 users, 49,688 products). The system addresses a critical gap in retail analytics: bridging interpretable classical segmentation with the predictive power of modern deep learning.

Two complementary models are constructed and rigorously compared:

- **Model 1 — Apriori Association Rule Mining**: A rule-based baseline using frequent itemset mining to identify co-purchasing patterns (e.g., {A, B} → {C}) with support, confidence, and lift thresholds
- **Model 2 — Attention-based LSTM**: A sequential deep learning model that learns temporal dependencies and repeat-purchase behaviour across users' chronological order histories

Results were validated with statistical significance testing (paired t-test: *t* = 4.759, *p* < 0.001) confirming the LSTM model significantly outperforms the rule-based baseline across all metrics.

---

## 🎯 Research Objectives

1. Critically review literature on customer segmentation, RFM frameworks, and ML/DL analytics in e-commerce and CRM
2. Develop a data-driven segmentation strategy using behavioral metrics and K-Means clustering in Python
3. Integrate and evaluate advanced sequential models (LSTM) for capturing temporal and dynamic customer behaviour
4. Determine managerial implications for customer targeting, retention, and personalized marketing strategies

---

## 🏗️ System Architecture

```
Instacart Dataset (3.4M orders | 206K users | 49,688 products)
        │
        ▼
┌─────────────────────────────────────────┐
│        Data Pipeline                    │
│  Relational Joins (3NF Schema)          │
│  Feature Engineering | RFM Scoring      │
│  Train / Validation / Test Split (80/10/10) │
└─────────────────────────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌──────────────────┐        ┌──────────────────────────┐
│ MODEL 1          │        │ MODEL 2                  │
│ RFM + K-Means    │        │ Attention-based LSTM     │
│ Segmentation     │        │ Sequential Recommender   │
│ +                │        │                          │
│ Apriori ARM      │        │ Embedding → LSTM →       │
│ (Baseline)       │        │ Dense Softmax Output     │
└──────────────────┘        └──────────────────────────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
           ┌───────────────────────┐
           │  Evaluation Framework │
           │  Accuracy | P@5 | R@5 │
           │  F1 | Top-K Hit Ratio │
           └───────────────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │ Statistical Testing   │
           │ Paired t-test         │
           │ t=4.759, p < 0.001    │
           └───────────────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │   Tableau Dashboard   │
           │   Business Insights   │
           └───────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.9+ |
| **Deep Learning** | TensorFlow / Keras (LSTM, Embedding, Softmax) |
| **Association Rule Mining** | mlxtend (Apriori, TransactionEncoder) |
| **Clustering & Segmentation** | Scikit-learn (K-Means, Elbow Method, Silhouette Score) |
| **Data Processing** | Pandas, NumPy |
| **Statistical Testing** | SciPy (paired two-tailed t-test) |
| **Visualization & BI** | Tableau, Matplotlib, Seaborn |
| **Notebooks** | Jupyter Notebook |

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                          # Instacart source tables (see Data Note below)
│   └── processed/                    # Merged, cleaned & feature-engineered datasets
│
├── notebooks/
│   ├── data_preprocessing.ipynb      # Relational joins, schema normalization, RFM features
│   ├── eda.ipynb                     # Exploratory Data Analysis (full report)
│   ├── model1_apriori.ipynb          # Apriori ARM: frequent itemsets & association rules
│   ├── model2_lstm.ipynb             # Attention-LSTM: architecture, training & inference
│   └── model_evaluation.ipynb        # Comparative evaluation, metrics & statistical tests
│
├── src/
│   ├── data_pipeline.py              # Data ingestion, relational joins & cleaning
│   ├── rfm_segmentation.py           # RFM scoring, min-max scaling, K-Means clustering
│   ├── apriori_model.py              # Apriori implementation with support/confidence/lift thresholds
│   ├── lstm_model.py                 # LSTM architecture (Embedding → LSTM → Softmax)
│   └── model_evaluation.py           # Accuracy, Precision@K, Recall@K, F1, t-test utilities
│
├── tableau/
│   └── dashboard_screenshots/        # Tableau NBO dashboard visuals
│
├── reports/
│   └── EDA_Report/                   # Full Exploratory Data Analysis documentation
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The **Instacart Market Basket Analysis** dataset contains real-world grocery orders from an anonymized user base.

| Statistic | Value |
|---|---|
| Total orders | **3.4 million** |
| Unique users | **~206,000** |
| Unique products | **49,688** |
| Aisles / Departments | 134 / 21 |
| Orders per user | 4 – 100 (avg ~16–17) |
| Overall reorder rate | **59.9%** |

The dataset follows a **3rd Normal Form (3NF)** relational schema. Key tables joined via primary/foreign keys: `orders`, `order_products_prior`, `order_products_train`, `products`, `aisles`, `departments`.


---

## 📈 Key EDA Findings

- **Top products**: Dominated by fresh organic produce — Bananas rank #1 most ordered
- **Basket sizes**: Right-skewed; median ~5 items, average ~10 items per order
- **Reorder rate**: 59.9% of items are repeat purchases; 88% of carts mix new and known items
- **High-reorder departments**: Dairy, eggs, produce, beverages — ideal for NBO targeting
- **Temporal patterns**: Peak ordering on Sundays/Mondays; busiest hours 10 AM – 3 PM
- **Reorder intervals**: Strong weekly (7-day) and monthly (30-day) cycles — key sequential signal for LSTM

> 📄 Full EDA documentation available in `/reports/EDA_Report/`

---

## 🧠 Models & Methodology

### Model 1 — Apriori Association Rule Mining (Baseline)

- Transformed transactions into binary basket format using `mlxtend`'s `TransactionEncoder`
- Applied Apriori with `min_support = 0.001`, `min_confidence = 0.2`, `lift > 1`
- Generated rules of the form **{A, B} → {C}**; top-k filtered by confidence and lift
- Limitation: treats all baskets as independent events; no sequential/temporal awareness

### Model 2 — Attention-based LSTM (Main Model)

**Architecture:**
1. **Input Layer** — Chronological purchase sequence per user (ordered basket history)
2. **Embedding Layer** — Maps discrete product IDs → dense vectors (V × d lookup table); learned jointly with the model
3. **LSTM Layer** — Processes embedded sequences; uses forget/input/output gates to model both short- and long-term dependencies; addresses vanishing gradient via cell state **c_t**
4. **Dense Softmax Output** — Produces probability distribution over all N products; top-K items selected as NBO recommendations

**Training setup:** Vocabulary restricted to top-N most frequent products; temporally coherent 80/10/10 train/val/test split; hyperparameters tuned on validation set.

---

## 📊 Results

| Metric | Apriori (Baseline) | LSTM |
|---|---|---|
| **Top-5 Accuracy** | 12.0% | **19.2%** |
| **Precision@5** | 0.022 | **0.090** |
| **Recall@5** | 0.045 | **0.080** |
| **F1 Score** | 0.029 | **0.075** |

**Statistical Significance:** Paired two-tailed t-test: *t-statistic* = **4.759**, *df* = 999, ***p* < 0.001** → LSTM significantly outperforms Apriori across all metrics.

These results align with prior literature: Yu et al. (2016) reported Recall@5 = 0.076 for a GRU/LSTM model (DREAM) vs. 0.049 for a popularity baseline on the same Instacart dataset.

---

## 💼 Business & Managerial Implications

| Model | Best Use Case |
|---|---|
| **Apriori** | Static cross-selling rules, bundle deals, raising average basket size |
| **LSTM** | Personalized "Recommended for You", reorder reminders, churn prevention |
| **Combined** | Comprehensive data-driven personalization — covering both new product discovery and habitual repurchase |

**RFM Segments Identified** (K-Means via Elbow + Silhouette): Champions, Loyal Customers, At-Risk, Occasional Shoppers, Lost Customers — each targetable with tailored promotions and loyalty strategies.

**Projected uplift**: Advanced sequential modeling combined with behavioral segmentation supports a targeted **~30% improvement in marketing ROI** through reduced budget waste and higher conversion rates.

---

## ⚖️ Ethical Considerations

This project addresses key ethical responsibilities in algorithmic recommendation:
- **Fairness**: Awareness of potential algorithmic bias and over-targeting risks
- **Transparency**: Model explainability considerations for managerial adoption
- **Data Privacy**: GDPR compliance; user anonymity preserved; no PII used
- **Responsible AI**: Measures to prevent harmful recommendation loops or exclusionary patterns

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Pipeline (Step-by-Step)
```bash
# 1. Download Instacart data → /data/raw/

# 2. Data preprocessing & RFM segmentation
jupyter notebook notebooks/data_preprocessing.ipynb

# 3. Exploratory Data Analysis
jupyter notebook notebooks/eda.ipynb

# 4. Train and evaluate Apriori model
jupyter notebook notebooks/model1_apriori.ipynb

# 5. Train and evaluate LSTM model
jupyter notebook notebooks/model2_lstm.ipynb

# 6. Comparative evaluation & statistical testing
jupyter notebook notebooks/model_evaluation.ipynb
```

---

## 📚 Key References

- Cao & Lam (2020) — Ensemble RFM + LSTM for customer profitability
- Yu et al. (2016) — DREAM: RNN-based next-basket recommendation on Instacart
- El-Shaer et al. (2024) — LSTM + Poisson-Gamma time-dynamic models
- Hochreiter & Schmidhuber (1997) — Original LSTM architecture
- Agrawal et al. (1993) — Apriori algorithm for association rule mining

---

## 👤 Author

**Aakriti Goyal**  
📧 aakritigoyal1705@gmail.com  
