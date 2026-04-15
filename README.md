 Overview
This project focuses on predicting customer churn using machine learning. It includes exploratory data analysis (EDA), model development, hyperparameter tuning, and deployment with FastAPI. The goal is to identify customers likely to churn and provide actionable insights for retention strategies.
🗂 Repository Structure
- coster_churn_model.ipynb → Jupyter notebook with model training, evaluation, and EDA.
- rf_dep.ipynb → Notebook for tuned Random Forest deployment and evaluation.
- customer_churn_pipeline.pkl → Serialized pipeline containing preprocessing and trained model.
- preprocessing.py → Script for data preprocessing steps (scaling, encoding, feature engineering).
- fast_api.py → FastAPI application for serving the churn prediction model.
- requirements.txt → Python dependencies required to run the project.
- runtime.txt → Runtime environment specification for deployment.
🔍 Exploratory Data Analysis (EDA)
EDA was performed to understand customer behavior, identify churn drivers, and highlight key features influencing churn. Visualizations and statistical summaries are included in the notebooks to support business insights.
⚙️ Model Development
- Multiple models were tested (Logistic Regression, Random Forest, XGBoost, Stacking).
- Recall was prioritized during hyperparameter tuning to minimize missed churn cases.
- Tuned Random Forest achieved balanced performance with accuracy ≈0.83 and ROC‑AUC ≈0.90.
- Stacking improved recall but reduced precision, offering an alternative depending on business needs.
🚀 Deployment
The final pipeline is served via FastAPI, enabling real‑time churn predictions through an API endpoint.
🎯 Objective
To build a reliable churn prediction system that balances recall and accuracy, supporting proactive customer retention strategies.

