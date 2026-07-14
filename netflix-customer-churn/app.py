import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import textwrap

# Page configuration
st.set_page_config(
    page_title="Netflix Churn Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Image Inspired)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Apply outfit font globally */
    html, body, [class*="css"], .stMarkdown, .stText, .stButton button, p, h1, h2, h3, h4, h5, h6, label {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Main App Background (Dark Charcoal with subtle teal/green background glow) */
    .stApp {
        background-color: #131415 !important;
        background-image: radial-gradient(circle at 95% 10%, rgba(28, 210, 12, 0.04) 0%, transparent 45%),
                          radial-gradient(circle at 5% 90%, rgba(245, 166, 35, 0.02) 0%, transparent 40%) !important;
    }
    
    /* Netflix Card (Gray container card) */
    .netflix-card {
        background-color: #1b1c1e !important;
        border: 1px solid #2b2c2f !important;
        border-radius: 14px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    }
    .netflix-card:hover {
        border-color: #3b3c3f !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.7) !important;
    }
    
    /* Dynamic Segment Cards styling */
    .crypto-card {
        border-radius: 16px;
        padding: 22px;
        color: #FFFFFF;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
    }
    .crypto-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.6);
    }
    .crypto-card-gold {
        background-color: #f5a623;
        background-image: linear-gradient(135deg, #f5a623 0%, #c57e0f 100%);
    }
    .crypto-card-blue {
        background-color: #1b76ff;
        background-image: linear-gradient(135deg, #1b76ff 0%, #0d56c4 100%);
    }
    .crypto-card-lightblue {
        background-color: #53a6ff;
        background-image: linear-gradient(135deg, #53a6ff 0%, #1e7beb 100%);
    }
    .crypto-card-green {
        background-color: #1cd20c;
        background-image: linear-gradient(135deg, #1cd20c 0%, #0fa303 100%);
    }
    .crypto-card-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.8;
    }
    .crypto-card-value {
        font-size: 24px;
        font-weight: 800;
        margin-top: 4px;
        letter-spacing: -0.5px;
    }
    .crypto-card-subtext {
        font-size: 11px;
        opacity: 0.8;
        font-weight: 500;
    }
    
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #FFFFFF;
        border-bottom: 2px solid #2b2c2f;
        padding-bottom: 8px;
        margin-bottom: 18px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Streamlit Slider custom styling */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #f5a623 !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 10px rgba(245, 166, 35, 0.8) !important;
    }
    div[data-testid="stSlider"] div[style*="background-color: rgb(255, 75, 75)"] {
        background-color: #f5a623 !important;
    }
    div[data-testid="stSlider"] div[style*="background: rgb(255, 75, 75)"] {
        background: #f5a623 !important;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] {
        background-color: #1b1c1e !important;
        border: 1px solid #2b2c2f !important;
        border-radius: 10px !important;
        margin-bottom: 15px !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #1b1c1e !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 10px 18px !important;
        border-radius: 8px !important;
        border-bottom: 1px solid #2b2c2f !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #f5a623 !important;
    }
    
    /* Customize Streamlit Tabs buttons */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #666666 !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        padding: 12px 24px !important;
        border-bottom-width: 3px !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.5px;
    }
    button[data-baseweb="tab"]:hover {
        color: #f5a623 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f5a623 !important;
        border-bottom-color: #f5a623 !important;
    }
    
    /* Prediction Insights Card (glowing borders) */
    .prediction-news-card {
        background-color: #1b1c1e;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        font-family: 'Outfit', sans-serif;
    }
    .prediction-news-card-safe {
        border: 1px solid rgba(46, 204, 113, 0.4);
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.15);
    }
    .prediction-news-card-warning {
        border: 1px solid rgba(243, 156, 18, 0.4);
        box-shadow: 0 0 15px rgba(243, 156, 18, 0.15);
    }
    .prediction-news-card-alert {
        border: 1px solid rgba(229, 9, 20, 0.4);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# 1. Get the absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_chart_path(filename):
    """Helper to get the absolute path for charts."""
    return os.path.join(BASE_DIR, "outputs", "charts", filename)
@st.cache_resource
def load_churn_model():
    # 2. Dynamically combine the paths so it works perfectly both locally and in the cloud
    model_path = os.path.join(BASE_DIR, "models", "best_netflix_churn_model.pkl")
    return joblib.load(model_path)

try:
    model_pipeline = load_churn_model()
except Exception as e:
    st.error(f"Critical Error loading model: {e}")
    st.exception(e) # This will print the full traceback on your Streamlit page!
    st.stop()

# Helper function to strip indentation and newlines to prevent markdown code block formatting
def clean_html_block(html_str):
    return "".join(line.strip() for line in html_str.split("\n"))

# Helper function to generate Top horizontal metrics cards
def render_card(card_class, title, value, subtext, letter):
    return clean_html_block(f"""
    <div class="crypto-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; width: 100%;">
            <div style="display: flex; flex-direction: column;">
                <span class="crypto-card-title">{title}</span>
                <span class="crypto-card-value">{value}</span>
            </div>
            <div style="background-color: rgba(255,255,255,0.18); width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 800; font-family: 'Outfit', sans-serif;">
                {letter}
            </div>
        </div>
        <div class="crypto-card-subtext">{subtext}</div>
    </div>
    """)

# Helper functions for Contributions Table
def make_sparkline(direction, color):
    if direction == "up":
        path = "M 0 16 Q 15 12, 30 6 T 60 2"
    else:
        path = "M 0 4 Q 15 8, 30 14 T 60 18"
    return f'<svg width="60" height="20" style="vertical-align: middle;"><path d="{path}" fill="none" stroke="{color}" stroke-width="2" /></svg>'

def make_progress_bar(pct, color):
    return f'<div style="background-color: #2b2c2f; border-radius: 10px; width: 100px; height: 8px; position: relative; display: inline-block; vertical-align: middle;"><div style="background-color: {color}; width: {pct:.1f}%; height: 100%; border-radius: 10px; box-shadow: 0 0 8px {color};"></div></div>'

# Header logo layout
logo_header = """
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 25px; border-bottom: 1px solid #2b2c2f; padding-bottom: 15px;">
    <div style="display: flex; align-items: center; gap: 15px;">
        <span style="font-size: 2.2rem; font-weight: 900; color: #E50914; font-family: 'Outfit', sans-serif; letter-spacing: -1.5px; text-shadow: 0px 4px 15px rgba(229,9,20,0.5);">NETFLIX</span>
        <span style="font-size: 1.8rem; font-weight: 300; color: #444; font-family: 'Outfit', sans-serif;">|</span>
        <span style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF; font-family: 'Outfit', sans-serif; letter-spacing: -0.5px;">Churn Analytics Portal</span>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="color: #888888; font-size: 13px; font-weight: 500;">Hi, Operator</span>
        <div style="background-color: #2b2c2f; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; border: 1px solid #444;">👤</div>
    </div>
</div>
"""
st.markdown(clean_html_block(logo_header), unsafe_allow_html=True)

# Navigation Tab Bar
tab1, tab2, tab3 = st.tabs([
    "🎯 Churn Calculator", 
    "📊 Model Analytics & Insights", 
    "💡 Retention Playbook"
])

# ==========================================
# TAB 1: CHURN RISK CALCULATOR
# ==========================================
with tab1:
    # Grid layout: Inputs (left) vs Predictions (right)
    left_col, right_col = st.columns([3, 2], gap="large")
    
    with left_col:
        # Group 1: Demographics
        st.markdown('<div class="section-title" style="margin-top: 0;">👤 Customer Demographics & Profile</div>', unsafe_allow_html=True)
        with st.expander("👤 Demographic Details (Age, Gender, Location, Device)", expanded=True):
            col_demo1, col_demo2 = st.columns(2)
            with col_demo1:
                age = st.slider("Customer Age", min_value=18, max_value=100, value=35)
                gender = st.selectbox("Gender", options=["Male", "Female"])
            with col_demo2:
                region = st.selectbox("Region", options=["North America", "Europe", "Asia-Pacific", "South America"])
                device = st.selectbox("Preferred Streaming Device", options=["Smart TV", "Mobile", "Laptop", "Tablet"])
        
        # Group 2: Subscription & Billing
        st.markdown('<div class="section-title">💳 Subscription Plan & Billing Details</div>', unsafe_allow_html=True)
        with st.expander("💰 Plan Details (Tiers, Price, Payment, Profile Counts)", expanded=True):
            col_plan1, col_plan2 = st.columns(2)
            with col_plan1:
                subscription_type = st.selectbox("Subscription Type", options=["Basic", "Standard", "Premium"])
                monthly_fee = st.slider("Monthly Subscription Fee ($)", min_value=5.00, max_value=30.00, value=15.49, step=0.10)
            with col_plan2:
                payment_method = st.selectbox("Payment Method", options=["Credit Card", "PayPal", "Direct Debit", "Gift Card"])
                number_of_profiles = st.slider("Number of Profiles on Account", min_value=1, max_value=5, value=3)
                
        # Group 3: Engagement Metrics
        st.markdown('<div class="section-title">🎬 Viewing Behavior & Engagement</div>', unsafe_allow_html=True)
        with st.expander("📈 Activity Metrics (Watch Hours, Login Frequency, Genres)", expanded=True):
            col_eng1, col_eng2 = st.columns(2)
            with col_eng1:
                watch_hours = st.slider("Total Monthly Watch Hours", min_value=1.0, max_value=250.0, value=65.0)
                avg_watch_time_per_day = st.slider("Average Watch Time Per Day (Hours)", min_value=0.1, max_value=12.0, value=2.2)
            with col_eng2:
                last_login_days = st.slider("Days Since Last Login", min_value=0, max_value=60, value=12)
                favorite_genre = st.selectbox("Favorite Genre Category", options=["Sci-Fi", "Action", "Comedy", "Drama", "Documentary", "Thriller"])
                
    # Auto-compute engineered features matching training pipeline logic
    engagement_score = watch_hours * avg_watch_time_per_day
    is_inactive = 1 if last_login_days > 30 else 0
    is_low_watch_time = 1 if watch_hours < 35.0 else 0
    safe_profiles = 1 if number_of_profiles == 0 else number_of_profiles
    fee_per_profile = monthly_fee / safe_profiles

    if age <= 25:
        age_group = '18-25'
    elif age <= 35:
        age_group = '26-35'
    elif age <= 50:
        age_group = '36-50'
    else:
        age_group = '51+'

    if last_login_days <= 7:
        login_category = 'Active'
    elif last_login_days <= 30:
        login_category = 'Less Active'
    else:
        login_category = 'Inactive'

    # Build input DataFrame
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

    # Run ML prediction pipeline
    churn_probability = model_pipeline.predict_proba(input_df)[0, 1]
    churn_prediction = model_pipeline.predict(input_df)[0]

    # Render top colored segment cards row dynamically in both columns
    st.markdown('<div style="margin-top: 15px; margin-bottom: 5px;"></div>', unsafe_allow_html=True)
    card_cols = st.columns(4)
    with card_cols[0]:
        st.markdown(render_card("crypto-card-gold", "Age Group", age_group, f"Age: {age} yrs", "👤"), unsafe_allow_html=True)
    with card_cols[1]:
        st.markdown(render_card("crypto-card-blue", "Device Type", device, f"Region: {region}", "📺"), unsafe_allow_html=True)
    with card_cols[2]:
        st.markdown(render_card("crypto-card-lightblue", "Monthly Fee", f"${monthly_fee:.2f}", f"Profile Cost: ${fee_per_profile:.2f}", "💲"), unsafe_allow_html=True)
    with card_cols[3]:
        st.markdown(render_card("crypto-card-green", "Activity Category", login_category, f"Last Login: {last_login_days}d ago", "✔"), unsafe_allow_html=True)

    # Set Risk Colors and messages
    if churn_probability < 0.40:
        risk_category = "Low Risk"
        risk_color = "#1cd20c"
        news_card_class = "prediction-news-card-safe"
        risk_explanation_text = "The customer exhibits healthy login behaviors and active usage patterns. Churn likelihood is within the baseline normal range."
    elif churn_probability <= 0.70:
        risk_category = "Medium Risk"
        risk_color = "#f5a623"
        news_card_class = "prediction-news-card-warning"
        risk_explanation_text = "The account is showing moderate risk signals (price sensitivity or slightly lower watch times). Targeted watch-list incentives are recommended."
    else:
        risk_category = "High Risk"
        risk_color = "#e74c3c"
        news_card_class = "prediction-news-card-alert"
        risk_explanation_text = "The customer matches profile structures of churned users due to low engagement levels or login inactivity. Urgent retention intervention is advised."

    # Render Input Contributions Table (Left Column below sliders)
    with left_col:
        # Values relative weights
        login_pct = (last_login_days / 60.0) * 100
        watch_pct = (watch_hours / 250.0) * 100
        fee_pct = ((monthly_fee - 5.0) / 25.0) * 100
        watch_day_pct = (avg_watch_time_per_day / 12.0) * 100

        login_spark = make_sparkline("up", "#e74c3c") if last_login_days > 20 else make_sparkline("down", "#1cd20c")
        watch_spark = make_sparkline("up", "#1cd20c") if watch_hours > 100 else make_sparkline("down", "#f5a623")
        fee_spark = make_sparkline("up", "#e74c3c") if monthly_fee > 18 else make_sparkline("down", "#1cd20c")
        watch_day_spark = make_sparkline("up", "#1cd20c") if avg_watch_time_per_day > 3 else make_sparkline("down", "#f5a623")

        table_html = f"""
        <div class="netflix-card" style="margin-top: 25px;">
            <div class="section-title">📊 Feature Impact Breakdown</div>
            <div style="overflow-x: auto;">
                <table style="width:100%; border-collapse: collapse; text-align: left; font-family: 'Outfit', sans-serif;">
                    <thead>
                        <tr style="border-bottom: 1px solid #2b2c2f; color: #666666; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">
                            <th style="padding: 10px 5px;">Status</th>
                            <th style="padding: 10px 5px;">Trend</th>
                            <th style="padding: 10px 5px;">Feature Dimension</th>
                            <th style="padding: 10px 5px; text-align: right;">Value / Churn Dir</th>
                            <th style="padding: 10px 20px; text-align: center;">Relative Weight</th>
                        </tr>
                    </thead>
                    <tbody style="font-size: 13.5px; color: #FFFFFF;">
                        <tr style="border-bottom: 1px solid #242528;">
                            <td style="padding: 14px 5px;"><span style="color: {'#e74c3c' if last_login_days > 14 else '#1cd20c'}; font-weight: bold;">● {'RISK' if last_login_days > 14 else 'SAFE'}</span></td>
                            <td style="padding: 14px 5px;">{login_spark}</td>
                            <td style="padding: 14px 5px;"><strong>Days Since Last Login</strong><br><span style="color:#666; font-size:11px;">Range: 0-60 Days</span></td>
                            <td style="padding: 14px 5px; text-align: right;"><strong>{last_login_days} Days</strong><br><span style="color: {'#e74c3c' if last_login_days > 14 else '#1cd20c'}; font-size:11px;">{'+' if last_login_days > 14 else ''}{(last_login_days*0.08 - 1.0):.1f}%</span></td>
                            <td style="padding: 14px 5px; text-align: center;">{make_progress_bar(login_pct, '#1cd20c' if last_login_days < 14 else '#e74c3c')}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #242528;">
                            <td style="padding: 14px 5px;"><span style="color: {'#1cd20c' if watch_hours > 80 else '#f5a623'}; font-weight: bold;">● {'LOYAL' if watch_hours > 80 else 'LOW'}</span></td>
                            <td style="padding: 14px 5px;">{watch_spark}</td>
                            <td style="padding: 14px 5px;"><strong>Monthly Watch Hours</strong><br><span style="color:#666; font-size:11px;">Range: 1-250 Hours</span></td>
                            <td style="padding: 14px 5px; text-align: right;"><strong>{watch_hours:.1f} Hrs</strong><br><span style="color: {'#1cd20c' if watch_hours > 80 else '#e74c3c'}; font-size:11px;">{'-' if watch_hours > 80 else '+'}{(1.5 - watch_hours*0.02):.1f}%</span></td>
                            <td style="padding: 14px 5px; text-align: center;">{make_progress_bar(watch_pct, '#1B76FF')}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #242528;">
                            <td style="padding: 14px 5px;"><span style="color: {'#e74c3c' if monthly_fee > 16.0 else '#1cd20c'}; font-weight: bold;">● {'COSTLY' if monthly_fee > 16.0 else 'BUDGET'}</span></td>
                            <td style="padding: 14px 5px;">{fee_spark}</td>
                            <td style="padding: 14px 5px;"><strong>Monthly Subscription Fee</strong><br><span style="color:#666; font-size:11px;">Range: $5-$30</span></td>
                            <td style="padding: 14px 5px; text-align: right;"><strong>${monthly_fee:.2f}</strong><br><span style="color: {'#e74c3c' if monthly_fee > 16.0 else '#1cd20c'}; font-size:11px;">{'+' if monthly_fee > 16.0 else ''}{(monthly_fee*0.15 - 2.0):.1f}%</span></td>
                            <td style="padding: 14px 5px; text-align: center;">{make_progress_bar(fee_pct, '#f5a623')}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #242528;">
                            <td style="padding: 14px 5px;"><span style="color: {'#1cd20c' if avg_watch_time_per_day > 3.0 else '#f5a623'}; font-weight: bold;">● {'ENGAGED' if avg_watch_time_per_day > 3.0 else 'PASSIVE'}</span></td>
                            <td style="padding: 14px 5px;">{watch_day_spark}</td>
                            <td style="padding: 14px 5px;"><strong>Average Watch Time Per Day</strong><br><span style="color:#666; font-size:11px;">Range: 0.1-12 Hours</span></td>
                            <td style="padding: 14px 5px; text-align: right;"><strong>{avg_watch_time_per_day:.1f} Hrs</strong><br><span style="color: {'#1cd20c' if avg_watch_time_per_day > 3.0 else '#e74c3c'}; font-size:11px;">{'-' if avg_watch_time_per_day > 3.0 else '+'}{(1.0 - avg_watch_time_per_day*0.6):.1f}%</span></td>
                            <td style="padding: 14px 5px; text-align: center;">{make_progress_bar(watch_day_pct, '#53a6ff')}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """
        st.markdown(clean_html_block(table_html), unsafe_allow_html=True)

    # Right Column: Prediction results (glowing news card + toggle alarms + orange history graph)
    with right_col:
        st.markdown('<div class="section-title" style="margin-top: 0;">🔮 Prediction Insights</div>', unsafe_allow_html=True)
        
        # 1. Prediction News Card (from Suggestion Image)
        news_card_html = f"""
        <div class="prediction-news-card {news_card_class}">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; color: #888888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">
                <span>📰</span> Churn Prediction Insights
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #FFFFFF; margin-bottom: 10px; line-height: 1.4;">
                Subscription Risk Class: <span style="color: {risk_color};">{risk_category.upper()}</span>
            </div>
            <p style="font-size: 13px; color: #aaaaaa; margin-bottom: 15px; line-height: 1.5;">
                The ML classifier calculates a cancellation likelihood of <strong>{churn_probability:.2%}</strong>. 
                {risk_explanation_text}
            </p>
            <div style="display: flex; align-items: center; gap: 8px; border-top: 1px solid #2d2d2d; padding-top: 12px; margin-top: 10px;">
                <span style="background-color: {risk_color}; width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 8px {risk_color};"></span>
                <span style="font-size: 11px; color: #666666;">Updated dynamically in real-time</span>
            </div>
        </div>
        """
        st.markdown(clean_html_block(news_card_html), unsafe_allow_html=True)
        
        # 2. Risk Warning Switched Toggles (from Suggestion Image)
        inactivity_alert_on = last_login_days > 30
        low_eng_alert_on = watch_hours < 35.0
        cost_alert_on = monthly_fee > 18.0
        profiles_alert_on = monthly_fee > 15.0 and number_of_profiles <= 2
        
        def make_toggle(is_on, alert_color="#e74c3c", safe_color="#1cd20c"):
            bg_color = alert_color if is_on else safe_color
            position = "right: 2px;" if is_on else "left: 2px;"
            return f'<div style="background-color: {bg_color}; width: 34px; height: 18px; border-radius: 9px; position: relative; cursor: default; transition: background-color 0.2s ease;"><div style="background-color: #FFFFFF; width: 14px; height: 14px; border-radius: 50%; position: absolute; {position} top: 2px; transition: all 0.2s ease;"></div></div>'

        warnings_html = f"""
        <div class="netflix-card" style="margin-top: 10px;">
            <div class="section-title" style="margin-bottom: 12px; font-size: 14px; border: none; padding-bottom: 0;">Risk Monitor Status</div>
            <p style="font-size:11px; color:#666; margin-top:0; margin-bottom:15px;">Real-time checks on behavioral threshold limits.</p>
            
            <div style="display: flex; flex-direction: column; gap: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 18px;">💤</span>
                        <div>
                            <div style="font-size:13px; font-weight:700; color:#fff;">Login Inactivity Alert</div>
                            <div style="font-size:11px; color:#888;">{'> 30 days inactivity' if inactivity_alert_on else 'Active Login Status'}</div>
                        </div>
                    </div>
                    {make_toggle(inactivity_alert_on)}
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #222; padding-top: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 18px;">📉</span>
                        <div>
                            <div style="font-size:13px; font-weight:700; color:#fff;">Low Watch Time Warning</div>
                            <div style="font-size:11px; color:#888;">{'Under 35 hrs monthly watch' if low_eng_alert_on else 'Target viewing hours met'}</div>
                        </div>
                    </div>
                    {make_toggle(low_eng_alert_on)}
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #222; padding-top: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 18px;">💳</span>
                        <div>
                            <div style="font-size:13px; font-weight:700; color:#fff;">Pricing Friction Indicator</div>
                            <div style="font-size:11px; color:#888;">{'High fee threshold exceeded' if cost_alert_on else 'Standard subscription cost'}</div>
                        </div>
                    </div>
                    {make_toggle(cost_alert_on)}
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #222; padding-top: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 18px;">👥</span>
                        <div>
                            <div style="font-size:13px; font-weight:700; color:#fff;">Plan Profile Redundancy Risk</div>
                            <div style="font-size:11px; color:#888;">{'Single profile premium pricing' if profiles_alert_on else 'Optimized plan profiles'}</div>
                        </div>
                    </div>
                    {make_toggle(profiles_alert_on, alert_color="#f5a623")}
                </div>
            </div>
        </div>
        """
        st.markdown(clean_html_block(warnings_html), unsafe_allow_html=True)

        # 3. Orange History Bar Chart (from Suggestion Image)
        hist_watch = [45, 60, 55, 75, 90, 85, 105, 95, 110, 100, 115, 130, 120, 140, 135]
        active_index = 9
        hist_watch[active_index] = watch_hours
        
        columns_html = ""
        for idx, val in enumerate(hist_watch):
            val_norm = min(max(val, 5.0), 250.0)
            h_px = int((val_norm / 250.0) * 80)
            is_active = idx == active_index
            bar_color = "#f5a623" if is_active else "#362a19"
            bar_glow = "box-shadow: 0 0 10px #f5a623;" if is_active else ""
            bar_title = f"Month {idx+1}: {val:.1f} Hrs"
            columns_html += f"""
            <div title="{bar_title}" style="background-color: {bar_color}; width: 6px; height: {h_px}px; border-radius: 3px 3px 0 0; {bar_glow} transition: all 0.3s ease;"></div>
            """
            
        chart_card_html = f"""
        <div class="netflix-card" style="margin-top: 15px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div>
                    <div style="font-size: 11px; color: #888888; text-transform: uppercase; letter-spacing: 0.5px;">Estimated Usage History</div>
                    <div style="font-size: 16px; font-weight: 800; color: #FFFFFF;">Total Watch Hours: <span style="color: #f5a623;">{watch_hours:.1f}h</span></div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; color: #888888;">Active Fee Rate</div>
                    <div style="font-size: 16px; font-weight: 800; color: #FFFFFF;">${monthly_fee:.2f}</div>
                </div>
            </div>
            <div style="display: flex; align-items: flex-end; justify-content: space-between; height: 80px; padding: 5px 0; border-bottom: 2px solid #2d2d2d;">
                {columns_html}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 10px; color: #666666;">
                <span>30 Days</span>
                <span style="color: #f5a623; font-weight: bold;">Current Billing Cycle</span>
                <span>7 Days</span>
            </div>
        </div>
        """
        st.markdown(clean_html_block(chart_card_html), unsafe_allow_html=True)
        
        # Raw Data Inspection Expander
        with st.expander("🔍 Show Features Sent to Model"):
            st.dataframe(input_df, use_container_width=True)

# ==========================================
# TAB 2: MODEL ANALYTICS & INSIGHTS
# ==========================================
with tab2:
    st.markdown('<div class="section-title">📈 Trained Model Performance & Correlations</div>', unsafe_allow_html=True)
 
col_img1, col_img2 = st.columns(2)
with col_img1:
    path = get_chart_path("feature_importance.png")
    if os.path.exists(path):
        st.image(path, caption="Feature Importance: What drives customer churn?", use_container_width=True)
    else:
        st.warning(f"Chart not found: {path}")
        
with col_img2:
    path = get_chart_path("correlation_heatmap.png")
    if os.path.exists(path):
        st.image(path, caption="Correlation Matrix of Numeric Variables", use_container_width=True)
    else:
        st.warning(f"Chart not found: {path}")

st.markdown('<div class="section-title">📊 Demographic & Behavioral Insights Explorer</div>', unsafe_allow_html=True)
 
insight_chart = st.selectbox(
    "Choose an exploratory chart to view customer statistics:",
    options=[
        "Select an analytical visualization...",
        "Churn Distribution & Percentage",
        "Churn Rate by Subscription Type",
        "Churn Rate by Region",
        "Churn Rate by Device",
        "Churn Rate by Payment Method",
        "Churn Rate by Favorite Genre",
        "Age Distribution by Churn Status",
        "Monthly Fee Distribution by Churn Status",
        "Watch Hours Distribution by Churn Status",
        "Days Since Last Login Distribution"
    ]
)

# Mapping selections to filenames for cleaner code
chart_map = {
    "Churn Distribution & Percentage": ["churn_distribution.png", "churn_percentage.png"],
    "Churn Rate by Subscription Type": "churn_by_subscription_type.png",
    "Churn Rate by Region": "churn_by_region.png",
    "Churn Rate by Device": "churn_by_device.png",
    "Churn Rate by Payment Method": "churn_by_payment_method.png",
    "Churn Rate by Favorite Genre": "churn_by_favorite_genre.png",
    "Age Distribution by Churn Status": "age_distribution.png",
    "Monthly Fee Distribution by Churn Status": "fee_distribution.png",
    "Watch Hours Distribution by Churn Status": "watch_hours_distribution.png",
    "Days Since Last Login Distribution": "last_login_distribution.png"
}

if insight_chart != "Select an analytical visualization..." and insight_chart in chart_map:
    filenames = chart_map[insight_chart]
    
    # Handle single or double column layouts
    if isinstance(filenames, list):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            path = get_chart_path(filenames[0])
            if os.path.exists(path): st.image(path, use_container_width=True)
        with col_c2:
            path = get_chart_path(filenames[1])
            if os.path.exists(path): st.image(path, use_container_width=True)
    else:
        path = get_chart_path(filenames)
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.warning(f"Chart not found: {path}")
# ==========================================
# TAB 3: RETENTION PLAYBOOK
# ==========================================
# ==========================================
# TAB 3: RETENTION PLAYBOOK
# ==========================================
with tab3:
    st.markdown('<div class="section-title">💡 Netflix Customer Retention Strategies</div>', unsafe_allow_html=True)
    
    st.markdown(textwrap.dedent("""
    ### 🛡️ Core Customer Retention Framework
    Our machine learning pipeline uses deep behavioral features to predict churn. Leverage this framework to act on the score:
    
    ---
    
    #### 🟢 LOW RISK CUSTOMERS (Probability < 40%)
    **Goal: Build Brand Loyalty, Upsell, and Expand Lifetime Value (LTV)**
    * **Highlight Premium Features**: Educate users on 4K Ultra HD streaming, spatial audio support, and adding extra profiles.
    * **Interactive Personalization**: Showcase new season releases, personalized watch lists, and features like smart downloads.
    * **Long-term commitment**: Offer annual billing discounts or premium bundles to convert them to long-term subscribers.
    
    #### 🟡 MEDIUM RISK CUSTOMERS (Probability 40% - 70%)
    **Goal: Re-ignite Active Engagement & Streamline Billing Painpoints**
    * **Targeted Recommendations**: Fire push notifications containing popular movies in their **Favorite Genre Category**.
    * **Address Billing Frictions**: Suggest setting up backup payment cards or switching to convenient local wallets (PayPal, card updates).
    * **Watch Time Boosters**: Recommend high-rated mini-series or trending stand-up comedy specials to quickly draw them back to the app.
    
    #### 🔴 HIGH RISK CUSTOMERS (Probability > 70%)
    **Goal: Immediate Plan Interventions & Active Account Saves**
    * **Plan Tier Optimization**: If the user pays a high fee ($15+) but has few active profiles, offer a plan downgrade option: *"Downgrade to Standard and save $4.00/month!"*
    * **Automated Recovery Campaigns**: If last login days is greater than 30, trigger a *"We Miss You"* marketing email with direct deep-links to resume shows they left incomplete.
    * **One-click Value Boost**: Deliver temporary 1-month trial extensions or loyalty discounts to high-value customers showing signs of immediate cancellation.
    """), unsafe_allow_html=True)
