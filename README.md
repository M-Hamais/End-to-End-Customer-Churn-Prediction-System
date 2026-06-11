# 📞 Telco Customer Churn Prediction

## 🎯 Problem Statement
Customer churn is a critical metric for telecommunications companies, as retaining an existing customer is significantly more cost-effective than acquiring a new one. The goal of this project is to build a machine learning pipeline that predicts whether a customer is likely to churn (leave the company) based on their demographics, account information, and service usage patterns. By accurately identifying high-risk customers, the business can proactively offer targeted incentives to improve retention rates. 📉🚀

## 📊 Dataset
This project uses the **IBM Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). It contains 7,043 customer records with 21 features, including the target variable, `Churn`. 📁

## 🧹 Data Cleaning & Feature Engineering
To prepare the raw data for machine learning models, the following preprocessing steps were performed:
1. **Handling Blank Entries 🈳:** The `TotalCharges` column contained blank string entries for new customers (tenure = 0). These were converted to `NaN` and then filled with `0` to reflect that no charges had been accumulated yet.
2. **Dropping Irrelevant Features 🗑️:** The `customerID` column was removed as it acts as a unique identifier and holds no predictive power.
3. **Encoding Categorical Variables 🔠:** All object/string columns (such as gender, Partner, InternetService, etc.) were encoded into numerical formats using `LabelEncoder`.
4. **Feature Scaling ⚖️:** The numerical features (`tenure`, `MonthlyCharges`, and `TotalCharges`) operate on vastly different mathematical scales. They were standardized to have a mean of 0 and a standard deviation of 1 using `StandardScaler` to ensure optimal performance for distance-based and gradient-descent-based algorithms.

## 🤖 Model Evaluation
Two classification models were trained and evaluated on an 80/20 stratified split of the data: **Logistic Regression** and **K-Nearest Neighbors (KNN)**. 

### 📈 Summary of Results
| Metric    | Logistic Regression 🏆 | K-Nearest Neighbors (K=5) 🥈 |
|-----------|------------------------|------------------------------|
| **Accuracy**  | 79.99%              | 76.01%                    |
| **Precision** | 64.47%              | 55.14%                    |
| **Recall**    | 54.81%              | 51.60%                    |
| **F1-Score**  | 59.25%              | 53.31%                    |

## ✅ Conclusion
**Logistic Regression** emerged as the superior model for this problem across all evaluation metrics. 🎉

While K-Nearest Neighbors struggles with the high dimensionality of the encoded categorical features (due to the curse of dimensionality affecting its distance calculations), Logistic Regression successfully captures the linear relationships between the features and the log-odds of churning. With an overall accuracy of nearly 80% and a significantly better balance of Precision and Recall (F1-Score of 59.25% compared to KNN's 53.31%), Logistic Regression provides a more reliable and interpretable foundation for identifying at-risk customers in this dataset. 💡📉
