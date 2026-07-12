import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Netflix Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #141414; color: #FFFFFF; }
    h1, h2, h3 { color: #E50914 !important; }
    .stButton>button { background-color: #E50914 !important; color: white !important; font-weight: bold; border-radius: 4px; border: none; }
    .stSidebar { background-color: #1F1F1F !important; }
    div[data-testid="stMetricValue"] { color: #E50914 !important; }
</style>
""", unsafe_allow_html=True)

# Cache model loader
@st.cache_resource
def load_churn_model():
    return joblib.load("models/best_netflix_churn_model.pkl")

try:
    model_pipeline = load_churn_model()
except Exception as e:
    st.error("Error loading model. Please run the `main.py` script first to fit and export the pipeline model binary.")
    st.stop()

st.title("Netflix Customer Churn Prediction Dashboard")
st.write("Determine the likelihood of subscription churn for any Netflix user based on demographic data, subscription fees, and historical account engagement.")

st.sidebar.header("Customer Information Profile")

# Create user input sliders/drop-downs
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=35)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
subscription_type = st.sidebar.selectbox("Subscription Type", options=["Basic", "Standard", "Premium"])
watch_hours = st.sidebar.slider("Total Monthly Watch Hours", min_value=1.0, max_value=250.0, value=65.0)
last_login_days = st.sidebar.slider("Days Since Last Login", min_value=0, max_value=60, value=12)
region = st.sidebar.selectbox("Region", options=["North America", "Europe", "Asia-Pacific", "South America"])
device = st.sidebar.selectbox("Preferred Streaming Device", options=["Smart TV", "Mobile", "Laptop", "Tablet"])
monthly_fee = st.sidebar.number_input("Monthly Subscription Fee ($)", min_value=5.00, max_value=30.00, value=15.49, step=0.50)
payment_method = st.sidebar.selectbox("Payment Method", options=["Credit Card", "PayPal", "Direct Debit", "Gift Card"])
favorite_genre = st.sidebar.selectbox("Favorite Genre Category", options=["Sci-Fi", "Action", "Comedy", "Drama", "Documentary", "Thriller"])
number_of_profiles = st.sidebar.slider("Number of Profiles on Account", min_value=1, max_value=5, value=3)
avg_watch_time_per_day = st.sidebar.slider("Average Watch Time Per Day (Hours)", min_value=0.1, max_value=12.0, value=2.2)

# Auto-compute engineered features matching main.py logic
engagement_score = watch_hours * avg_watch_time_per_day
is_inactive = 1 if last_login_days > 30 else 0

# Safe dynamic setting of low watch hours based on typical 25th percentile (approx 35 hours)
is_low_watch_time = 1 if watch_hours < 35.0 else 0

safe_profiles = 1 if number_of_profiles == 0 else number_of_profiles
fee_per_profile = monthly_fee / safe_profiles

# Bin age group
if age <= 25:
    age_group = '18-25'
elif age <= 35:
    age_group = '26-35'
elif age <= 50:
    age_group = '36-50'
else:
    age_group = '51+'

# Bin login categories
if last_login_days <= 7:
    login_category = 'Active'
elif last_login_days <= 30:
    login_category = 'Less Active'
else:
    login_category = 'Inactive'

# Assemble inputs into a pandas row matching original schema
input_dict = {
    'age': age,
    'gender': gender,
    'subscription_type': subscription_type,
    'watch_hours': watch_hours,
    'last_login_days': last_login_days,
    'region': region,
    'device': device,
    'monthly_fee': monthly_fee,
    'payment_method': payment_method,
    'number_of_profiles': number_of_profiles,
    'avg_watch_time_per_day': avg_watch_time_per_day,
    'favorite_genre': favorite_genre,
    'engagement_score': engagement_score,
    'is_inactive': is_inactive,
    'is_low_watch_time': is_low_watch_time,
    'fee_per_profile': fee_per_profile,
    'age_group': age_group,
    'login_category': login_category
}

input_df = pd.DataFrame([input_dict])

st.subheader("Simulated Feature Set Passed to the Model Pipeline")
st.dataframe(input_df)

# Prediction Column Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Predictions")
    
    # Run prediction through pre-trained pipeline
    churn_probability = model_pipeline.predict_proba(input_df)[0, 1]
    churn_prediction = model_pipeline.predict(input_df)[0]
    
    # Categorize Risk
    if churn_probability < 0.40:
        risk_category = "Low Risk"
        risk_color = "green"
    elif churn_probability <= 0.70:
        risk_category = "Medium Risk"
        risk_color = "orange"
    else:
        risk_category = "High Risk"
        risk_color = "red"
        
    st.metric(label="Churn Probability Score", value=f"{churn_probability:.2%}")
    st.markdown(
    f"**Calculated Risk Category:** <span style='font-size:20px; color:{risk_color}; font-weight:bold;'>{risk_category}</span>",
    unsafe_allow_html=True)
    
    if churn_prediction == 1:
        st.error("Prediction Flag: Customer likely to cancel/churn.")
    else:
        st.success("Prediction Flag: Customer likely to stay.")

with col2:
    st.subheader("Custom Retention Action Items")
    
    # Conditional prescriptive recommendations based on inputs
    recs = []
    if is_inactive == 1:
        recs.append("📧 **Send Inactivity Recovery Email:** Trigger a push notification containing dynamic recommendations for trending titles.")
    if is_low_watch_time == 1:
        recs.append("🎬 **Personalized Content Promotion:** Promote new releases related to the customer's favorite genre.")
    if monthly_fee > 15.0 and number_of_profiles <= 2:
        recs.append("💳 **Recommend Plan Downgrade:** Offer standard/basic plans to prevent a full subscription cancellation.")
    if last_login_days > 14 and last_login_days <= 30:
        recs.append("⏳ **Mid-tier Re-engagement Push:** Prompt the user with high-rated content matching their favorite categories.")
    
    if len(recs) == 0:
        recs.append("✅ **Loyalty Reinforcement:** Keep accounts engaged with standard content updates and highlight features like mobile offline downloads.")
        
    for item in recs:
        st.write(item)

st.markdown("---")
st.subheader("Model & Feature Explanation")
st.info(
    "This system processes raw categorical and numerical variables through a calibrated machine learning pipeline. "
    "Features like 'engagement_score', 'fee_per_profile', and categorical binnings interact to construct predictions based on customer inactivity, pricing pain-points, and usage patterns."
)