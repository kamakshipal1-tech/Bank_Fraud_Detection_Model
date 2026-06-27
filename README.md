# Bank Fraud Detection using Machine Learning

## Project Overview

Financial fraud causes billions of dollars in losses every year. Traditional rule-based systems often fail to identify sophisticated fraud patterns hidden inside large transaction datasets.

This project develops an end-to-end fraud detection pipeline using machine learning techniques to identify potentially fraudulent banking transactions. The workflow covers data exploration, feature engineering, imbalance handling, model training, threshold optimization, and model deployment preparation.

The goal is not only to maximize prediction accuracy but also to minimize missed fraud cases while maintaining a practical false-alarm rate.

---

## Business Problem

Banks process millions of transactions every day.

Only a very small percentage of these transactions are fraudulent, making fraud detection a highly imbalanced classification problem.

A missed fraud transaction can directly result in financial loss, customer dissatisfaction, and reputational damage.

The objective is to build a system that can:

* Detect fraudulent transactions early
* Reduce financial losses
* Support risk investigation teams
* Improve monitoring efficiency

---

## Dataset Description

The dataset contains transactional information including:

* Transaction amount
* Transaction type
* Sender information
* Receiver information
* Account balances before and after transactions
* Fraud labels

Target Variable:

`isFraud`

* 0 = Legitimate transaction
* 1 = Fraudulent transaction

---

## Exploratory Data Analysis

### Fraud Rate by Transaction Amount

Transactions were divided into quantile-based buckets using:

```python
df['amount_bin'] = pd.qcut(df['amount'], q=10)
```

Purpose:

* Understand how fraud risk changes across transaction sizes
* Identify high-risk transaction ranges

Observation:

Higher-value transactions exhibit significantly higher fraud rates.

---

### Fraud Rate by Transaction Type

Transaction categories:

* TRANSFER
* CASH_OUT
* CASH_IN
* PAYMENT
* DEBIT

Observation:

TRANSFER transactions show the highest fraud concentration.

Business Interpretation:

Fraudsters prefer transaction methods that quickly move funds between accounts.

---

### Hourly Fraud Analysis

A transaction hour was extracted from the transaction step variable.

```python
df['hour'] = df['step'] % 24
```

Purpose:

* Detect temporal fraud patterns
* Compare day and night fraud behavior

Observation:

Certain time windows show elevated fraud activity.

Business Interpretation:

Fraud monitoring systems should increase sensitivity during high-risk periods.

---

## Feature Engineering

### 1. Log Amount Transformation

```python
df['log_amount'] = np.log1p(df['amount'])
```

Why?

Transaction amounts are highly skewed.

Benefits:

* Reduces outlier impact
* Improves model stability

---

### 2. High Value Transaction Flag

```python
p99 = df['amount'].quantile(0.99)
df['is_high_amount'] = (df['amount'] > p99).astype(int)
```

Why?

Large transactions often have elevated fraud risk.

---

### 3. Balance Difference Features

```python
df['balance_diff_orig'] =
df['oldbalanceOrg'] - df['newbalanceOrig']
```

```python
df['balance_diff_dest'] =
df['newbalanceDest'] - df['oldbalanceDest']
```

Why?

Fraud frequently creates unusual balance movement patterns.

Business Value:

Captures transactional anomalies not visible from raw balances.

---

### 4. Label Encoding

```python
df['type_enc'] =
LabelEncoder().fit_transform(df['type'])
```

Why?

Machine learning models require numerical inputs.

---

## Data Preparation

### Missing Value Handling

```python
df.fillna(0, inplace=True)
```

Purpose:

Ensure model compatibility and stable training.

---

### Feature Scaling

```python
RobustScaler()
```

Why RobustScaler?

It is less sensitive to extreme transaction values than standard scaling techniques.

---

## Machine Learning Models

### Logistic Regression

Purpose:

Create a fast and interpretable baseline model.

Advantages:

* Easy to explain
* Computationally efficient

---

### Random Forest

Purpose:

Capture nonlinear fraud patterns.

Advantages:

* Handles complex interactions
* Robust against overfitting

---

### XGBoost

Purpose:

Maximize fraud detection performance.

Advantages:

* Excellent performance on tabular datasets
* Handles nonlinear relationships
* Effective for imbalanced classification

---

## Class Imbalance Strategy

Fraud transactions represent only a tiny fraction of the dataset.

To address this challenge:

### Logistic Regression

```python
class_weight='balanced'
```

### Random Forest

```python
class_weight='balanced'
```

### XGBoost

```python
scale_pos_weight =
(non_fraud_count / fraud_count)
```

This prevents the model from ignoring minority fraud cases.

---

## Model Optimization and Hyperparameter Tuning

Fraud detection presents two major machine learning challenges:

1. Extreme class imbalance, where fraudulent transactions represent only a small fraction of total transactions.
2. High risk of overfitting due to complex transaction patterns and engineered features.

To address these challenges, hyperparameter tuning was performed on the XGBoost model.

### Objectives of Tuning

The optimization process focused on:

* Improving fraud detection capability
* Reducing model overfitting
* Handling class imbalance effectively
* Improving generalization on unseen transactions
* Reducing unnecessary false-positive alerts
* Maintaining high recall for fraudulent transactions

### Key Hyperparameters

#### max_depth

Controls the maximum depth of decision trees.

Business impact:
Shallow trees may miss complex fraud patterns, while very deep trees may memorize historical transactions and fail on future fraud cases.

Tuning this parameter helped balance model complexity and generalization.

---

#### min_child_weight

Controls the minimum number of observations required before a node can split.

Business impact:
Prevents the model from creating rules based on a very small number of transactions, reducing sensitivity to noise and improving robustness.

---

#### subsample

Randomly samples a percentage of training records for each tree.

Business impact:
Reduces overfitting and improves model stability by preventing dependence on specific transactions.

---

#### colsample_bytree

Randomly samples features during tree construction.

Business impact:
Encourages diversity among trees and reduces variance, improving performance on unseen data.

---

#### reg_alpha and reg_lambda

L1 and L2 regularization parameters.

Business impact:
Penalize excessive model complexity, helping the model focus on meaningful fraud signals instead of memorizing training data.

---

#### scale_pos_weight

Used specifically to address severe class imbalance.

Business impact:
Assigns greater importance to fraudulent transactions during training, ensuring that the minority fraud class receives sufficient attention from the model.

Without this adjustment, the model could achieve high accuracy while failing to detect many fraud cases.

### Business Outcome

The tuned model achieved a better balance between fraud detection capability and operational efficiency.

Rather than optimizing only for accuracy, the tuning process prioritized:

* High recall to capture fraudulent transactions
* Strong precision to limit unnecessary investigations
* Reduced overfitting for improved real-world deployment performance

This approach produces a more practical fraud detection system that aligns with real banking and financial risk management requirements.

## Evaluation Metrics

Accuracy is not sufficient for fraud detection.

This project evaluates models using:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* Confusion Matrix

Special emphasis is placed on Recall because missing fraudulent transactions is significantly more costly than generating additional alerts.

---

## Threshold Optimization

Instead of using the default 0.5 threshold, the model searches for a threshold that maintains:

* Recall ≥ 90%
* Maximum possible precision

This aligns the system with real-world fraud monitoring requirements.

---

## Model Deployment Preparation

The best-performing model is exported as:

```python
best_fraud_model_tuned.pkl
```

Stored Components:

* Trained model
* Optimized threshold
* Feature list

This allows future inference without retraining.

---

## Key Business Insights

1. TRANSFER transactions show the highest fraud concentration.

2. Large-value transactions exhibit increased fraud risk.

3. Fraud occurs in only a tiny fraction of transactions, making imbalance handling critical.

4. Balance-difference features are highly informative fraud indicators.

5. Time-of-day patterns can improve fraud monitoring effectiveness.

6. Threshold tuning significantly improves operational fraud detection performance.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Joblib

---

## Deployment

The application has been deployed using **Streamlit Community Cloud** for easy access and interactive predictions.

### Live Demo

https://bankfrauddetectionmodel-zxwbhderhdbyizjtw3tdk4.streamlit.app/

---

## Future Improvements

* Real-time fraud scoring
* Model explainability using SHAP
* Deep learning approaches
* Streaming fraud detection pipelines
* Drift monitoring and automated retraining
