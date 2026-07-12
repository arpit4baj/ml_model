# Netflix Customer Churn Prediction

A machine learning project designed to estimate the cancellation probability of Netflix customers. The predictive pipeline integrates feature engineering, standard normalization, data validation, and classification algorithms.

---

## Problem Statement
The cost of acquiring a new customer is significantly higher than retaining an existing one. Predicting when a subscriber is likely to cancel (churn) allows streaming services to proactively target customers with personalized messaging, pricing interventions, or targeted recommendations. This project builds a complete end-to-end classification system using demographics, billing, and system usage metrics.

---

## Dataset Description
The model trains on a 5,000-row tabular dataset (`data/netflix_customer_churn.csv`) with the following fields:

* `customer_id`: Unique identifier (removed before training)
* `age`: Age of customer
* `gender`: Biological sex
* `subscription_type`: Basic, Standard, or Premium membership tier
* `watch_hours`: Monthly platform consumption in hours
* `last_login_days`: Days elapsed since last app login
* `region`: Geographic location of account
* `device`: Primary hardware platform used for streaming
* `monthly_fee`: Subscription cost billed per month
* `churned`: Churn status (1 = Canceled, 0 = Active, Target variable)
* `payment_method`: Bill processing medium
* `number_of_profiles`: Profiles hosted on the account
* `avg_watch_time_per_day`: Daily usage average in hours
* `favorite_genre`: Favorite genre category

---

## Project Workflow
1. **Load and Validate**: Verify structural issues, verify dimensions, report null ratios, and balance of target class values.
2. **Clean**: Impute numerical columns using median values, impute object columns using modes, strip whitespace, and drop ID attributes.
3. **Exploratory Data Analysis (EDA)**: Generate categorical plots, correlation heatmaps, and distribution plots, saving results to `outputs/charts/`.
4. **Feature Engineering**: Compute interactions to feed raw signals into clean categorical and physical groupings.
5. **Preprocessing Pipeline**: Use Scikit-learn's `ColumnTransformer` to combine imputations, standard scale numerics, and run one-hot encoding on categoricals.
6. **Model Training**: Compare Logistic Regression, Random Forest, and Gradient Boosting Classifier models using 5-Fold Cross-Validation.
7. **Best Model Selection**: Select the best pipeline model based on ROC-AUC and Recall.
8. **Export Pipeline**: Save the preprocessing and model steps together inside `models/best_netflix_churn_model.pkl` using Joblib.
9. **Interactive Dashboard**: Construct an interactive Streamlit application enabling real-time risk simulation and prescription strategies.

---

## Feature Engineering Breakdown
To extract the maximum predictive signal from the dataset, we construct the following engineered features:
* **`engagement_score`**: (`watch_hours * avg_watch_time_per_day`) - Represents total active usage.
* **`is_inactive`**: (1 if `last_login_days` > 30, else 0) - Signals severe disengagement.
* **`is_low_watch_time`**: (1 if `watch_hours` is below the 25th percentile, else 0) - Flags users at risk of cancellation due to low usage.
* **`fee_per_profile`**: (`monthly_fee / number_of_profiles`) - Measures individual cost efficiency.
* **`age_group`**: Binned age classes (`18-25`, `26-35`, `36-50`, `51+`).
* **`login_category`**: Binned inactivity duration (`Active`, `Less Active`, `Inactive`).

---

## Installation & Execution Instructions

### 1. Set Up Environment
Ensure you have Python 3.8+ installed. Install the project requirements:
```bash
pip install -r requirements.txt
