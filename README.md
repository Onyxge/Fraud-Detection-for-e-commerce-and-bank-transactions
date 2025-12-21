## Phase 0: Business Understanding and Success Definition

### Business Context

Adey Innovations Inc operates in the financial technology domain, providing solutions for both e-commerce platforms and banking institutions. Fraudulent transactions directly translate into financial losses, regulatory risk, and erosion of customer trust. At the same time, overly aggressive fraud detection systems can negatively impact user experience by blocking legitimate transactions.

This project focuses on building fraud detection models that balance these competing objectives by identifying high-risk transactions accurately while minimizing unnecessary disruption to legitimate users.

The project addresses two distinct but related fraud scenarios:

* E-commerce transaction fraud, where behavioral signals such as account age, transaction velocity, and geolocation inconsistencies are critical.
* Bank credit card fraud, where transaction timing patterns and amount-based anomalies dominate.

Although both datasets represent fraud detection problems, they differ significantly in feature structure, data generation process, and operational context. As a result, models and evaluation strategies are tailored per dataset while following a unified decision philosophy.

---

### Business Objective

The primary objective of this project is to develop robust and explainable machine learning models that can detect fraudulent transactions with high recall for fraud cases, while maintaining acceptable precision to avoid excessive false alarms.

Beyond raw predictive performance, the system is designed to:

* Support risk-based decision making rather than binary blocking.
* Provide interpretable signals that fraud analysts and business stakeholders can trust.
* Enable future integration into near real-time monitoring pipelines.

---

### Cost of Errors

Fraud detection is an asymmetric cost problem.

* **False Negatives (Missed Fraud):**
  Result in direct financial loss, potential regulatory exposure, and reputational damage.

* **False Positives (Legitimate Transactions Flagged as Fraud):**
  Degrade customer experience, reduce transaction completion rates, and increase operational overhead for manual reviews.

Given this imbalance, overall accuracy is not a meaningful metric for success. Instead, evaluation focuses on metrics that reflect the business cost trade-offs inherent in fraud detection.

---

### Definition of Success

A successful model in this project is defined by the following criteria:

* Strong performance on metrics suited for highly imbalanced data.
* Stable behavior across cross-validation folds, indicating robustness rather than overfitting.
* Clear interpretability of predictions using model explainability techniques.
* Actionable insights that can inform fraud prevention policies and operational rules.

Model selection prioritizes business utility and explainability alongside predictive performance.

---

### Evaluation Metrics Rationale

The primary evaluation metrics used in this project are:

* **Area Under the Precision-Recall Curve (AUC-PR):**
  Chosen due to extreme class imbalance, where ROC-AUC can be misleading.

* **F1-Score:**
  Used to balance precision and recall when selecting operational thresholds.

* **Confusion Matrix Analysis:**
  Used to directly inspect false positive and false negative behavior under different decision thresholds.

Metrics are evaluated using stratified cross-validation to ensure reliable performance estimates.

---

### High-Level Modeling Pipeline

The fraud detection workflow implemented in this project follows these stages:

1. Data cleaning and validation
2. Exploratory data analysis focused on fraud behavior
3. Feature engineering informed by transaction dynamics and geolocation signals
4. Handling class imbalance using data-level and algorithm-level strategies
5. Model training and evaluation with appropriate metrics
6. Model explainability and interpretation using SHAP
7. Translation of model insights into business recommendations

This structured approach ensures a systematic progression from raw data to actionable intelligence.
