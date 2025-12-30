
# Adey Innovations Fraud Detection System

## Overview

This repository contains an end-to-end fraud detection system developed for Adey Innovations Inc. The project covers the full machine learning lifecycle, from business understanding and exploratory data analysis to model deployment via a FastAPI service, explainability, testing.

Two fraud scenarios are addressed:

* **E-commerce transaction fraud** using behavioral, temporal, and geolocation features.
* **Bank credit card fraud** using transaction amount and timing patterns.

The system emphasizes not only predictive performance, but also interpretability, robustness, and real-world business usability.

---

## Phase 0: Business Understanding and Success Definition

### Business Context

Adey Innovations Inc operates in the financial technology domain, providing solutions for e-commerce platforms and banking institutions. Fraudulent transactions directly translate into financial losses, regulatory risk, and erosion of customer trust. At the same time, overly aggressive fraud detection systems can negatively impact user experience by blocking legitimate transactions.

This project focuses on building fraud detection models that balance these competing objectives by accurately identifying high-risk transactions while minimizing unnecessary disruption to legitimate users.

The project addresses two distinct but related fraud scenarios:

* **E-commerce transaction fraud**, where behavioral signals such as account age, transaction velocity, device usage, and geolocation inconsistencies are critical.
* **Bank credit card fraud**, where transaction timing patterns and amount-based anomalies dominate.

Although both datasets represent fraud detection problems, they differ significantly in feature structure and operational context. Models and evaluation strategies are therefore tailored per dataset while following a unified decision philosophy.

---

### Business Objective

The primary objective is to develop robust and explainable machine learning models that detect fraudulent transactions with high recall for fraud cases, while maintaining acceptable precision to avoid excessive false alarms.

Beyond predictive accuracy, the system is designed to:

* Support **risk-based decision making** rather than binary blocking.
* Provide **interpretable signals** that fraud analysts and business stakeholders can trust.
* Enable future **near real-time integration** into operational monitoring pipelines.

---

### Cost of Errors

Fraud detection is an asymmetric cost problem.

* **False Negatives (Missed Fraud)**

  * Direct financial loss
  * Regulatory exposure
  * Reputational damage

* **False Positives (Legitimate Transactions Flagged)**

  * Poor customer experience
  * Reduced transaction completion rates
  * Increased operational overhead due to manual reviews

Because of this imbalance, overall accuracy is not a meaningful success metric.

---

### Definition of Success

A successful model in this project is defined by:

* Strong performance on metrics suited for highly imbalanced data
* Stable behavior across cross-validation folds
* Clear interpretability using explainability techniques
* Actionable insights that inform fraud prevention strategies

Model selection prioritizes **business utility and interpretability** alongside predictive performance.

---

### Evaluation Metrics Rationale

The following metrics are used:

* **AUC-PR (Area Under Precision-Recall Curve)** as the primary metric due to extreme class imbalance
* **F1-score** to balance precision and recall at operational thresholds
* **Confusion matrix analysis** to explicitly examine false positives and false negatives

All metrics are evaluated using **stratified cross-validation** to ensure robustness.

---

## High-Level Solution Architecture

The implemented fraud detection pipeline follows these stages:

1. Data cleaning and validation
2. Exploratory data analysis focused on fraud behavior
3. Feature engineering informed by transaction dynamics and geolocation
4. Handling class imbalance
5. Model training and evaluation
6. Model explainability using SHAP
7. API deployment, testing

This structured approach ensures a systematic transition from raw data to actionable intelligence.

---

## Task 1: Exploratory Data Analysis and Feature Engineering

### Key Activities

* Class distribution analysis and confirmation of extreme imbalance
* Temporal analysis of transaction behavior
* User behavior profiling (account age, device usage)
* Geolocation integration using IP-to-country mapping

### Geolocation Integration

* IP addresses were converted to integer format
* Transactions were merged with IP-to-country mapping using range-based joins
* Fraud rates were analyzed by country

This revealed meaningful geographic disparities in fraud risk, validating geolocation as a high-impact feature.

---

## Task 2: Model Building and Evaluation

### Data Preparation

* Stratified train-test split to preserve class distribution
* Feature-target separation
* Numeric feature standardization where required

### Baseline Model

* Logistic Regression trained as an interpretable baseline
* Evaluated using AUC-PR, F1-score, and confusion matrix

### Ensemble Model

* Random Forest classifier trained as the primary model
* Basic hyperparameter tuning applied
* Evaluated using the same metrics as the baseline

### Cross-Validation

* Stratified 5-fold cross-validation
* Mean and standard deviation of AUC-PR and F1 reported

### Model Selection

Random Forest was selected as the final model due to:

* Substantially higher AUC-PR
* Stable cross-validation performance
* Strong recall for fraud cases
* Compatibility with SHAP-based explainability

---

## Task 3: Model Explainability and Business Insights

### 3.1 Global Model Interpretation (SHAP)

SHAP analysis was applied to the Random Forest model to identify globally important fraud drivers.

Key observations:

* Fraud risk is driven by a small subset of behavioral and transactional features
* Abnormal transaction behavior relative to user history strongly increases fraud probability
* The dominance of a limited number of features suggests targeted monitoring is effective

This confirms the model learns meaningful and intuitive fraud signals.

---

### 3.2 Local Model Interpretation (Individual Predictions)

SHAP waterfall plots were generated for:

* True Positive (correctly detected fraud)
* False Positive (legitimate transaction flagged)
* False Negative (missed fraud)

**True Positive:** Multiple high-risk features jointly pushed the prediction above the fraud threshold.

**False Positive:** A small number of features resembled fraud behavior, highlighting the need for secondary verification.

**False Negative:** Subtle fraud cases lacked strong signals across key features, reinforcing the need for layered defenses.

---

### 3.3 Business Recommendations Derived from SHAP

1. **Risk-Based Monitoring**
   Focus monitoring on top SHAP-identified features to reduce computational cost.

2. **Adaptive Thresholds**
   Use different fraud score thresholds for high-risk and low-risk users.

3. **Secondary Verification**
   Apply OTP or step-up authentication for borderline cases.

4. **Fraud Pattern Intelligence**
   Aggregate SHAP explanations over time to detect emerging fraud strategies.

5. **Explainability for Compliance**
   Use SHAP explanations for auditability, regulatory compliance, and dispute resolution.

---

## Project Structure

```
├── dvc/
├── data/
├── models/
├── notebooks/
│   ├── eda_fraud_data.ipynb
│   ├── feature-engineering.ipynb
│   ├── smote_modeling.ipynb
│   └── eda_creditcard.ipynb
├── reports/
├── src/
│   ├── api/
│   ├── train_model.py
│   └── explain_model.py
├── tests/test_api.py
├── requirements.txt
└── README.md
```

## API Deployment

A FastAPI service exposes the trained model for real-time fraud prediction.

### Features

* JSON-based prediction endpoint
* Fraud probability and risk-level output
* Swagger UI documentation

### Run the API

```
uvicorn src.api.serve_model:app --reload
```

---

## Testing

Unit tests are implemented using FastAPI TestClient to validate:

* API availability
* Valid prediction requests
* Graceful handling of malformed input

Run tests using:

```
pytest
```

---

## Conclusion

This project delivers a production-ready fraud detection system that balances predictive performance, interpretability, and business relevance. By combining robust modeling with explainability and API deployment, the solution moves beyond experimentation into practical, operational intelligence.





