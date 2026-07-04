
# Online Shoppers Purchasing Intention Prediction

A machine learning classification project that predicts whether an online shopper will complete a purchase, based on their browsing session behavior — with a deployed, interactive web app.

**Live demo:** https://online-shoppers-prediction-vlcatzpj6zyqwzb7rpjmlv.streamlit.app/

## Overview

E-commerce platforms generate large volumes of user interaction data, but turning that data into actionable insight is a real challenge. This project builds and compares three classification models to predict purchasing intention from session-level browsing behavior, then deploys the best-performing model as a live Streamlit application.

**Team:** Leena Alsaif, Lama Alabdulkarim, Layan Almarshud, Amjaad Abaalkhail
*(Course project — DS323: Machine Learning)*

## Dataset

- **Source:** [Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention) (UCI Machine Learning Repository / Kaggle)
- **Size:** 12,330 sessions, 18 features
- **Target variable:** `Revenue` (True = purchase completed, False = no purchase)
- **Key features:** Administrative/Informational/ProductRelated page counts and durations, BounceRates, ExitRates, PageValues, VisitorType, Weekend, Month
- **Class balance:** Only ~15.5% of sessions resulted in a purchase — a meaningfully imbalanced classification problem

## Methodology (CRISP-DM)

1. **Data understanding** — explored structure, types, and key features (no missing values found across all 12,330 records)
2. **EDA** — target distribution, histograms, box plots for outlier detection, correlation heatmap
3. **Data preparation** — encoded categorical variables (VisitorType, Weekend), applied `StandardScaler` to numerical features; outliers were **retained** since they reflect real user behavior (e.g. highly engaged browsers)
4. **Modeling** — trained and compared three classifiers: **K-Nearest Neighbors**, **Decision Tree**, and **Random Forest** (70/30 stratified train/test split)
5. **Hyperparameter tuning** — `GridSearchCV` used to tune all three models
6. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrices, with explicit over/underfitting analysis
7. **Deployment** — best model serialized and deployed via a Streamlit web app

## Key Findings

| Model | Test Accuracy (default) | Test Accuracy (tuned) | Recall (purchase class) | Notes |
|---|---|---|---|---|
| KNN | 0.8567 | 0.8586 | 0.29 | Slight underfitting; stable but weak on minority class |
| Decision Tree | 0.8470 | 0.8907 | 0.52 | Overfit initially (train acc. 1.00); tuning fixed this |
| **Random Forest** | **0.8962** | 0.8951 | **0.53** | Best overall balance of accuracy and generalization |

- **Random Forest was selected as the final model** — highest test accuracy, best F1-score (0.61) on the minority (purchase) class, and minimal overfitting despite near-perfect training accuracy
- `PageValues` showed the strongest positive correlation with purchase completion; `BounceRates` and `ExitRates` were negatively correlated — sessions where users leave quickly rarely convert
- The dataset's class imbalance (only 15.5% purchases) was treated as an explicit AI ethics consideration; recall on the minority class, not just overall accuracy, was used to judge model quality

## Tech Stack

Python, pandas, scikit-learn (KNN, Decision Tree, Random Forest, GridSearchCV, StandardScaler), seaborn/matplotlib, Streamlit

## Project Structure

```
├── ML_Project.ipynb             # Full analysis: EDA, preprocessing, modeling, tuning
├── online_shoppers_intention.csv # Dataset
├── streamlit_app.py             # Streamlit web app for live predictions
├── random_forest_model.pkl       # Trained Random Forest model
├── scaler.pkl                    # Fitted StandardScaler
├── training_columns.pkl          # Column order used at training time
├── requirements.txt              # Python dependencies
└── README.md
```

## How to Run

**Explore the analysis:**
Open `ML_Project.ipynb` in Jupyter, JupyterLab, VS Code, or Google Colab and run all cells in order.

**Run the app locally:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Or just try the live version:** https://online-shoppers-prediction-vlcatzpj6zyqwzb7rpjmlv.streamlit.app/

## Limitations

- The dataset reflects a single e-commerce context; behavior patterns (and the model's accuracy) may not generalize to other platforms or industries
- Class imbalance limits recall on the purchase class — even the best model only identifies about half of actual purchasers, so false negatives remain a meaningful risk in this deployment
- Outliers were intentionally retained rather than removed, which improves realism but leaves the model somewhat sensitive to extreme sessions
- Feature scaling and encoding choices were tuned for these three model types specifically; other algorithms (e.g. gradient boosting) were not explored

## Authors

Leena Alsaif, Lama Alabdulkarim, Layan Almarshud, Amjaad Abaalkhail — Data Science Students
Developed as coursework for DS323: Machine Learning, 2nd Semester 2025-2026.
