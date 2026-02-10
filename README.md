# Assignment 2 — ML Model Comparison

## Problem Statement

Financial institutions often conduct marketing campaigns to promote long-term investment products such as term deposits. However, contacting every customer is costly and inefficient, making it important to identify individuals who are most likely to subscribe.

This assignment addresses a binary classification problem using the UCI Bank Marketing Dataset. The objective is to develop and evaluate multiple machine learning models to predict whether a customer will subscribe to a term deposit based on demographic attributes, financial indicators, and prior campaign interaction data.

By leveraging historical campaign data containing both numerical and categorical features, the task aims to compare the predictive performance of six machine learning algorithms — Logistic Regression, Decision Tree, k-Nearest Neighbors, Naive Bayes, Random Forest, and XGBoost — using standard evaluation metrics including Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

The outcomes of this analysis will provide insight into which modeling approaches are most effective for marketing response prediction, supporting data-driven decision making in targeted customer outreach.

---

## Dataset Description
Provide details about the dataset.

- **Dataset Name:** Bank Marketing 
- **Source:**  UCI (https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Number of Samples:**  45,211
- **Number of Features:**  17
- **Target Variable:**  y (binary: "yes","no")
- **Class Distribution:**  yes: 11.7%, no: 88.3%
- **Preprocessing Steps:**  
  - Handling missing values:  Since missing values are only present in categorical features (job, education, contact, poutcomes), I have represented them as "unknown", and will treat them as a separate category.
  - Encoding:  One-hot encoding for categorical features.
  - Scaling/Normalization:  Standardization for numerical features.
  - Train/Test Split:  80% training, 20% testing.

---


## Model Performance Comparison Table

| Model                  | Accuracy | AUC     | Precision | Recall  | F1     | MCC     |
|------------------------|----------|---------|-----------|---------|--------|---------|
| Logistic Regression    | 0.9013   | 0.9056  | 0.6445    | 0.3478  | 0.4518 | 0.4261  |
| Decision Tree          | 0.8746   | 0.7015  | 0.4649    | 0.4754  | 0.4701 | 0.3990  |
| K-Nearest Neighbors    | 0.8962   | 0.8277  | 0.5990    | 0.3403  | 0.4340 | 0.4001  |
| Naive Bayes            | 0.8548   | 0.8101  | 0.4059    | 0.5198  | 0.4559 | 0.3774  |
| Random Forest          | 0.9045   | 0.9263  | 0.6506    | 0.3960  | 0.4924 | 0.4597  |
| XGBoost                | 0.9055   | 0.9287  | 0.6267    | 0.4745  | 0.5401 | 0.4944  |

---

## Observations on Model Performance

| ML Model | Observation |
|---------|-------------|
| Logistic Regression | High accuracy and AUC but low recall, indicating difficulty in identifying positive subscription cases; serves as a strong linear baseline with limited non-linear modeling capability. |
| Decision Tree | Moderate performance with lower AUC and MCC, suggesting overfitting and instability despite reasonable recall. |
| K-Nearest Neighbors (KNN) | Good accuracy but low recall, likely affected by high dimensionality after one-hot encoding. |
| Naive Bayes | Higher recall indicating better detection of positive cases, but lower precision due to strong feature independence assumptions. |
| Random Forest | Balanced and robust performance with high accuracy, AUC, and MCC due to ensemble learning. |
| XGBoost | Best overall performance with highest AUC, F1-score, and MCC, effectively handling class imbalance and complex feature interactions. |


---



