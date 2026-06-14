# 🔮 Customer Retention AI Dashboard (Telco Customer Churn Prediction)

### 🔗 **<a href="https://end-to-end-customer-churn-prediction-system-1.streamlit.app/" target="_blank">Launch Live Dashboard 🚀</a>**

---

## 🎯 Project Overview & Objective
Customer churn is a critical metric for telecommunications companies, as retaining an existing customer is significantly more cost-effective than acquiring a new one. The goal of this project is to build an end-to-end machine learning pipeline and interactive dashboard that predicts whether a customer is likely to churn (leave the company) based on their demographics, account information, and service usage patterns.

By presenting predictions in clear, non-technical business language alongside real-time metrics and dynamic retention strategies, the system serves as an early-warning system to proactively protect customer loyalty. 📈📉

---

## 🚀 Key Features of the Live App

- **📊 Live Churn Risk Profiler:** Simulates customer profiles dynamically (tenure, billing details, contract type, and internet service).
- **🔌 Conditional Input Logic:** Dynamically shows/hides technical properties (e.g., online security, tech support subscription, backup, protection, and streaming services) depending on whether an active internet connection (DSL or Fiber optic) is selected, fallback-defaulting to `"No internet service"` in the backend.
- **🤖 Dual AI Model Predictions:**
  - **AI Model A (Trend Analyzer - Logistic Regression):** Focuses on broad customer trends (contract type, internet packages) and delivers ~80% accuracy.
  - **AI Model B (Similar-Customer Matcher - KNN):** Predicts risk based on mathematical similarities with historical profiles, delivering ~76% accuracy.
- **💡 Smart Business Insights & Actions:** Automatically analyzes agreement/disagreement between both models to yield dynamic business summaries and color-coded recommendation banners (Low, Medium, High risk) suggesting custom retention actions.
- **📈 Comprehensive Model Evaluation Tab:**
  - Interactive performance metrics grid highlighting the "winning" model per metric.
  - Responsive visual bar charts comparing Accuracy, Precision, Recall, and F1-Score.
  - Beautiful, customized confusion matrix heatmaps showing detailed classification performance (TN, FP, FN, TP) for both models.
- **📱 Fully Responsive Design:** Automatically adjusts content layouts, text contrast via CSS theme variables, and stretches matplotlib charts dynamically across all device viewports (from mobile phones up to 24-inch screens).

---

## 📊 Dataset & Preprocessing Pipeline
This project uses the **IBM Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
Preprocessing steps executed in the background include:
1. **TotalCharges Imputation:** Cleaned and filled empty text strings for new subscribers (tenure = 0) with a baseline value of `0`.
2. **Feature Dropping:** Dropped `customerID` as it holds no predictive weight.
3. **Categorical Encoding:** Standardized categories (e.g., DSL, contract structures) using custom `LabelEncoder` pipelines.
4. **Standard Scaling:** Scaled numerical metrics (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to ensure mathematical consistency.

---

## 🤖 Model Evaluation Results

| Metric    | AI Model A (Trend Analyzer) 🏆 | AI Model B (Similar-Customer Matcher) 🥈 |
|-----------|------------------------------|------------------------------------------|
| **Accuracy**  | 79.99%              | 76.01%                    |
| **Precision** | 64.47%              | 55.14%                    |
| **Recall**    | 54.81%              | 51.60%                    |
| **F1-Score**  | 59.25%              | 53.31%                    |

*Logistic Regression (AI Model A) proved to be the more stable, accurate, and reliable model for this business problem.*

---

## 💻 Running the App Locally

### 1. Clone & Set Up Environment
```bash
# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```
