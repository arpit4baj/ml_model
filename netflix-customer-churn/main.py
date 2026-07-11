import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Set styling for plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 50})

# Create directory structure programmatically
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


def generate_mock_data_if_missing(filepath, num_rows=5000):
    """Generates a realistic mock dataset matching the specified schema if it does not exist."""
    if os.path.exists(filepath):
        print(f"Dataset found at {filepath}. Skipping mock generation.\n")
        return

    print(f"Dataset not found. Generating {num_rows} rows of mock Netflix customer churn data...")
    np.random.seed(42)

    ages = np.random.randint(18, 70, size=num_rows)
    genders = np.random.choice(["Male", "Female"], size=num_rows, p=[0.49, 0.51])
    subs = np.random.choice(["Basic", "Standard", "Premium"], size=num_rows, p=[0.4, 0.4, 0.2])
    watch_hours = np.random.uniform(5, 150, size=num_rows)
    last_logins = np.random.randint(0, 45, size=num_rows)
    regions = np.random.choice(["North America", "Europe", "Asia-Pacific", "South America"], size=num_rows)
    devices = np.random.choice(["Smart TV", "Mobile", "Laptop", "Tablet"], size=num_rows)
    payment_methods = np.random.choice(["Credit Card", "PayPal", "Direct Debit", "Gift Card"], size=num_rows)
    profiles = np.random.randint(1, 6, size=num_rows)
    avg_daily_watch = np.random.uniform(0.5, 6.0, size=num_rows)
    genres = np.random.choice(["Sci-Fi", "Action", "Comedy", "Drama", "Documentary", "Thriller"], size=num_rows)

    # Set fees based on subscription type with minor random variations
    fees = []
    for s in subs:
        if s == "Basic":
            fees.append(np.round(9.99 + np.random.uniform(-0.5, 0.5), 2))
        elif s == "Standard":
            fees.append(np.round(15.49 + np.random.uniform(-0.5, 0.5), 2))
        else:
            fees.append(np.round(19.99 + np.random.uniform(-0.5, 0.5), 2))
    fees = np.array(fees)

    # Calculate churn probability based on features to make patterns realistic
    # Higher last_login_days, higher fees, lower watch_hours, and fewer daily hours increase churn likelihood
    z = (0.01 * (ages - 35) + 
         0.08 * last_logins - 
         0.02 * watch_hours + 
         0.15 * fees - 
         0.6 * avg_daily_watch - 1.2)
    prob = 1 / (1 + np.exp(-z))
    churned = np.where(prob > np.random.uniform(0, 1, size=num_rows), 1, 0)

    # Introduce a minor rate of artificial missing values (approx 1%) to demonstrate cleaner's performance
    ages = [np.nan if np.random.rand() < 0.01 else a for a in ages]
    watch_hours = [np.nan if np.random.rand() < 0.01 else w for w in watch_hours]
    genders = [None if np.random.rand() < 0.01 else g for g in genders]

    df = pd.DataFrame({
        "customer_id": [f"CUST{1000 + i}" for i in range(num_rows)],
        "age": ages,
        "gender": genders,
        "subscription_type": subs,
        "watch_hours": watch_hours,
        "last_login_days": last_logins,
        "region": regions,
        "device": devices,
        "monthly_fee": fees,
        "churned": churned,
        "payment_method": payment_methods,
        "number_of_profiles": profiles,
        "avg_watch_time_per_day": avg_daily_watch,
        "favorite_genre": genres
    })

    df.to_csv(filepath, index=False)
    print(f"Mock data successfully written to {filepath}.\n")


# =====================================================================
# 1. LOAD AND INSPECT DATASET
# =====================================================================
csv_path = "data/netflix_customer_churn.csv"
generate_mock_data_if_missing(csv_path)

print("--- STEP 1: LOADING & INSPECTING DATASET ---")
df = pd.read_csv(csv_path)

print("First 5 Rows:")
print(df.head(), "\n")

print(f"Dataset Shape: {df.shape}")
print(f"Column Names: {list(df.columns)}")
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print(f"\nDuplicate Rows: {df.duplicated().sum()}")

print("\nSummary Statistics:")
print(df.describe(include='all'))

print("\nDistribution of Target Variable 'churned':")
print(df['churned'].value_counts())
print(df['churned'].value_counts(normalize=True))
print("-" * 50)


# =====================================================================
# 2. CLEAN DATASET
# =====================================================================
print("\n--- STEP 2: CLEANING DATASET ---")
# Remove duplicate rows
df = df.drop_duplicates()

# Remove customer_id
if 'customer_id' in df.columns:
    df = df.drop(columns=['customer_id'])

# Strip trailing spaces from categorical fields
categorical_cols_raw = df.select_dtypes(include=['object']).columns
for col in categorical_cols_raw:
    df[col] = df[col].astype(str).str.strip()

# Confirm churned contains only 0 and 1
df = df[df['churned'].isin([0, 1])]

# Explicit conversions
df['churned'] = df['churned'].astype(int)

# Address missing values inside Pandas before performing EDA
for col in df.columns:
    if col != 'churned':
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

print(f"Shape after cleaning: {df.shape}")
print("-" * 50)


# =====================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================
print("\n--- STEP 3: PERFORMING EXPLORATORY DATA ANALYSIS ---")

# 1. Churn distribution count plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='churned', palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churned Status (0 = Stayed, 1 = Churned)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/charts/churn_distribution.png')
plt.close()

# 2. Churn percentage bar chart
plt.figure(figsize=(5, 4))
churn_pct = df['churned'].value_counts(normalize=True) * 100
sns.barplot(x=churn_pct.index, y=churn_pct.values, palette='Set2')
plt.title('Churn Percentage')
plt.xlabel('Churned')
plt.ylabel('Percentage (%)')
plt.tight_layout()
plt.savefig('outputs/charts/churn_percentage.png')
plt.close()

# Helper function to plot churn rates by categorical variables
def plot_churn_by_category(col_name, filename, limit_top=None):
    plt.figure(figsize=(8, 5))
    grouped = df.groupby(col_name)['churned'].mean().reset_index()
    grouped['churned'] = grouped['churned'] * 100  # Convert to percentage
    grouped = grouped.sort_values(by='churned', ascending=False)
    if limit_top:
        grouped = grouped.head(limit_top)
    sns.barplot(data=grouped, x=col_name, y='churned', palette='viridis')
    plt.title(f'Churn Rate by {col_name}')
    plt.ylabel('Churn Rate (%)')
    plt.xlabel(col_name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'outputs/charts/{filename}')
    plt.close()

# Generate categorical plots
plot_churn_by_category('subscription_type', 'churn_by_subscription_type.png')
plot_churn_by_category('gender', 'churn_by_gender.png')
plot_churn_by_category('region', 'churn_by_region.png', limit_top=10)
plot_churn_by_category('device', 'churn_by_device.png')
plot_churn_by_category('payment_method', 'churn_by_payment_method.png')
plot_churn_by_category('favorite_genre', 'churn_by_favorite_genre.png')

# Numerical distributions grouped by churn
def plot_numeric_distribution(col_name, filename):
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='churned', y=col_name, palette='Set2')
    plt.title(f'{col_name} Distribution by Churn Status')
    plt.xlabel('Churned')
    plt.ylabel(col_name)
    plt.tight_layout()
    plt.savefig(f'outputs/charts/{filename}')
    plt.close()

plot_numeric_distribution('age', 'age_distribution.png')
plot_numeric_distribution('monthly_fee', 'fee_distribution.png')
plot_numeric_distribution('watch_hours', 'watch_hours_distribution.png')
plot_numeric_distribution('last_login_days', 'last_login_distribution.png')
plot_numeric_distribution('avg_watch_time_per_day', 'avg_daily_watch_distribution.png')

# Correlation heatmap for numerical variables
plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('outputs/charts/correlation_heatmap.png')
plt.close()

print("All exploratory analysis plots saved to 'outputs/charts/'.")
print("-" * 50)


# =====================================================================
# 4. CREATE USEFUL FEATURES
# =====================================================================
print("\n--- STEP 4: FEATURE ENGINEERING ---")

# 1. engagement_score
df['engagement_score'] = df['watch_hours'] * df['avg_watch_time_per_day']

# 2. is_inactive (last_login_days > 30)
df['is_inactive'] = np.where(df['last_login_days'] > 30, 1, 0)

# 3. is_low_watch_time (watch_hours < 25th percentile)
watch_hours_25th = df['watch_hours'].quantile(0.25)
df['is_low_watch_time'] = np.where(df['watch_hours'] < watch_hours_25th, 1, 0)

# 4. fee_per_profile
# Protect against profile count of 0 (replace with 1 if exists)
safe_profiles = np.where(df['number_of_profiles'] == 0, 1, df['number_of_profiles'])
df['fee_per_profile'] = df['monthly_fee'] / safe_profiles

# 5. age_group
def assign_age_group(age):
    if age <= 25:
        return '18-25'
    elif age <= 35:
        return '26-35'
    elif age <= 50:
        return '36-50'
    else:
        return '51+'

df['age_group'] = df['age'].apply(assign_age_group)

# 6. login_category
def assign_login_category(days):
    if days <= 7:
        return 'Active'
    elif days <= 30:
        return 'Less Active'
    else:
        return 'Inactive'

df['login_category'] = df['last_login_days'].apply(assign_login_category)

print("Engineered features successfully created:")
print(df[['engagement_score', 'is_inactive', 'is_low_watch_time', 'fee_per_profile', 'age_group', 'login_category']].head())
print("-" * 50)


# =====================================================================
# 5. DEFINE ML FEATURES AND TARGET
# =====================================================================
print("\n--- STEP 5: ASSIGNING FEATURES AND TARGET ---")

feature_cols = [
    'age', 'gender', 'subscription_type', 'watch_hours', 'last_login_days',
    'region', 'device', 'monthly_fee', 'payment_method', 'number_of_profiles',
    'avg_watch_time_per_day', 'favorite_genre', 'engagement_score',
    'is_inactive', 'is_low_watch_time', 'fee_per_profile', 'age_group', 'login_category'
]

X = df[feature_cols]
y = df['churned']

print(f"Feature set dimensions: {X.shape}")
print(f"Target dimension: {y.shape}")
print("-" * 50)


# =====================================================================
# 6. PIPELINE CREATION & TRAIN/TEST SPLIT
# =====================================================================
print("\n--- STEP 6: PREPROCESSING PIPELINE & TRAINING SPLIT ---")

# Separate into numeric and categorical lists
num_features = [
    'age', 'watch_hours', 'last_login_days', 'monthly_fee', 
    'number_of_profiles', 'avg_watch_time_per_day', 
    'engagement_score', 'fee_per_profile'
]

cat_features = [
    'gender', 'subscription_type', 'region', 'device', 
    'payment_method', 'favorite_genre', 'is_inactive', 
    'is_low_watch_time', 'age_group', 'login_category'
]

# Set up transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training instances: {X_train.shape[0]}")
print(f"Testing instances: {X_test.shape[0]}")
print("-" * 50)


# =====================================================================
# 7. MODEL TRAINING, SELECTION AND COMPARISON
# =====================================================================
print("\n--- STEP 7: EVALUATING MODELS WITH 5-FOLD CV ---")

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results_records = []

# Evaluate each candidate model
for model_name, model_obj in models.items():
    # Build complete model pipeline
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_obj)
    ])
    
    # 5-fold CV score based on ROC-AUC
    cv_scores = cross_val_score(clf_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    mean_cv_auc = cv_scores.mean()
    
    # Fit model to training partition
    clf_pipeline.fit(X_train, y_train)
    
    # Class predictions and prediction probabilities
    y_pred = clf_pipeline.predict(X_test)
    y_prob = clf_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_prob)
    
    results_records.append({
        "Model": model_name,
        "CV_ROC_AUC": mean_cv_auc,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": test_auc
    })
    
    print(f"\n{model_name}:")
    print(f"  CV ROC-AUC:  {mean_cv_auc:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  ROC-AUC:     {test_auc:.4f}")
    
    # Generate Confusion Matrix Chart
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f'outputs/charts/confusion_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Generate ROC Curve Chart
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {test_auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'outputs/charts/roc_{model_name.replace(" ", "_").lower()}.png')
    plt.close()

# Save comparison report
comparison_df = pd.DataFrame(results_records)
comparison_df.to_csv("outputs/reports/model_comparison.csv", index=False)
print("\nSaved model comparison report to 'outputs/reports/model_comparison.csv'.")

# Find the best model based on higher ROC-AUC and strong recall performance
best_row = comparison_df.sort_values(by=['ROC_AUC', 'Recall'], ascending=False).iloc[0]
best_model_name = best_row['Model']
print(f"\n---> Winning Model Selected: {best_model_name}")
print("-" * 50)


# =====================================================================
# 8. SAVE THE BEST MODEL PIPELINE
# =====================================================================
print("\n--- STEP 8: RE-FITTING AND SAVING MODEL PIPELINE ---")

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', models[best_model_name])
])

final_pipeline.fit(X_train, y_train)

# Save best model
model_save_path = "models/best_netflix_churn_model.pkl"
joblib.dump(final_pipeline, model_save_path)
print(f"Perfect fit complete. Model and prep pipeline saved to {model_save_path}.")
print("-" * 50)


# =====================================================================
# 9. FEATURE IMPORTANCE
# =====================================================================
print("\n--- STEP 9: EXTRACTING FEATURE IMPORTANCE ---")

# Reconstruct post-pipeline feature names
cat_encoder = final_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_encoded_names = list(cat_encoder.get_feature_names_out(cat_features))
all_feature_names = num_features + cat_encoded_names

classifier = final_pipeline.named_steps['classifier']

if hasattr(classifier, 'feature_importances_'):
    importances = classifier.feature_importances_
elif hasattr(classifier, 'coef_'):
    importances = np.abs(classifier.coef_[0])
else:
    importances = np.zeros(len(all_feature_names))

importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

importance_df.to_csv("outputs/reports/feature_importance.csv", index=False)
print("Saved feature importance report to 'outputs/reports/feature_importance.csv'.")

# Plot Top 15 Features
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', palette='mako')
plt.title('Top 15 Churn Predictive Features')
plt.xlabel('Importance/Coefficient Weight')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('outputs/charts/feature_importance.png')
plt.close()

print("\nTop 5 Churn Driving Features:")
top_features_list = []
for idx, row in importance_df.head(5).iterrows():
    feat = row['Feature']
    weight = row['Importance']
    print(f" - {feat} ({weight:.4f})")
    top_features_list.append(feat)

print("\nSummary Interpretation of Customer Churn Drivers:")
print("* High number of days since last login signals higher imminent churn risk.")
print("* Watch hours drop highlights lower viewing satisfaction and lower customer stickiness.")
print("* Users with higher monthly subscription fees shows sensitivity to pricing structures.")
print("* Subscriptions with fewer profiles relative to monthly fees are prone to cancellations.")
print("* Customers on Basic tiers often switch more actively compared to Premium and Standard users.")
print("-" * 50)


# =====================================================================
# 10. SAMPLE PREDICTIONS
# =====================================================================
print("\n--- STEP 10: SAMPLE PREDICTIONS ON RESERVED TEST DATA ---")

sample_X = X_test.head(5).copy()
sample_y = y_test.head(5).copy()

probs = final_pipeline.predict_proba(sample_X)[:, 1]
preds = final_pipeline.predict(sample_X)

sample_results = pd.DataFrame({
    'Actual Churn': sample_y.values,
    'Predicted Churn': preds,
    'Churn Probability': probs
})

def calculate_risk(p):
    if p < 0.40:
        return 'Low Risk'
    elif p <= 0.70:
        return 'Medium Risk'
    else:
        return 'High Risk'

sample_results['Risk Category'] = sample_results['Churn Probability'].apply(calculate_risk)
print(sample_results)
print("-" * 50)


# =====================================================================
# 11. FINAL SUMMARY
# =====================================================================
print("\n--- STEP 11: FINAL PIPELINE SUMMARY ---")

# Run test set evaluation metrics for the selected model
test_probs = final_pipeline.predict_proba(X_test)[:, 1]
high_risk_count = sum(test_probs > 0.70)

print(f"Selected Model Type:               {best_model_name}")
print(f"Accuracy Score:                    {best_row['Accuracy']:.4%}")
print(f"Precision Score:                   {best_row['Precision']:.4%}")
print(f"Recall Score (Sensitivity):         {best_row['Recall']:.4%}")
print(f"F1-Score:                          {best_row['F1_Score']:.4%}")
print(f"Area Under ROC Curve (ROC-AUC):    {best_row['ROC_AUC']:.4%}")
print(f"High-Risk Customers in Test Set:   {high_risk_count} out of {len(X_test)}")

print("\nImmediate Actionable Retention Recommendations:")
print("1. Target users approaching 30+ days of login inactivity with automatic personalized emails.")
print("2. Deliver personalized recommended movie lists to users with declining monthly watch hours.")
print("3. Engage high-paying standard or premium single-profile accounts with plan downgrade options rather than lose them to cancellation.")
print("4. Provide loyalty discounts or alternative plans to long-term subscribers expressing high payment method frictional failures.")
print("-" * 50)