import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image

# Load model and preprocessing objects
@st.cache_resource
def load_models():
    return {
        'model': joblib.load("model/random_forest_model.pkl"),
        'le_occupation': joblib.load("model/label_encoder_occupation.pkl"),
        'le_city': joblib.load("model/label_encoder_city_tier.pkl"),
        'le_class': joblib.load("model/label_encoder_class.pkl"),
        'scaler': joblib.load("model/standard_scaler.pkl")
    }

models = load_models()

# Configuration
st.set_page_config(
    page_title="Finance Personality Classifier",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #F5F7FA;}
    .sidebar .sidebar-content {background-color: #2E4053;}
    h1 {color: #2874A6;}
    .stButton>button {background-color: #28B463;color:white;border-radius:8px;padding:8px 16px;}
    .st-bb {background-color: white;}
    .st-at {background-color: #D5F5E3;}
    div[data-baseweb="select"] > div {border: 2px solid #2874A6;}
    .st-b7 {background-color: transparent;}
    .st-cq {background-color: #EAEDED;}
    </style>
    """, unsafe_allow_html=True)

# App Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/477/477103.png", width=100)
with col2:
    st.title("Personal Finance Personality Classifier")
    st.markdown("""
    <p style='font-size:18px; color:#5D6D7E;'>
    Discover whether someone is a <strong>Saver</strong>, <strong>Spender</strong>, or <strong>Neutral</strong> 
    based on their financial behavior patterns
    </p>
    """, unsafe_allow_html=True)

# Sidebar Input Section
with st.sidebar:
    st.markdown("<h2 style='color:white;'>📊 Input Parameters</h2>", unsafe_allow_html=True)
    
    with st.expander("💰 Financial Details", expanded=True):
        income = st.number_input("Monthly Income / Pocket Money (₹)", min_value=0, value=50000, step=1000)
        fixed = st.number_input("Fixed Expenses (₹)", min_value=0, value=20000, step=1000)
        investment = st.number_input("Investments (₹)", min_value=0, value=10000, step=1000)
        savings = st.number_input("Savings (₹)", min_value=0, value=10000, step=1000)
        loan = st.number_input("Loan Repayment (₹)", min_value=0, value=5000, step=500)
    
    with st.expander("👤 Personal Details"):
        age = st.slider("Age", 18, 80, 30)
        occupation = st.selectbox("Occupation", models['le_occupation'].classes_)
        city = st.selectbox("City Tier", models['le_city'].classes_)
        shop_freq = st.slider("Outings (monthly) [Restaurants,Malls,etc.]", 0, 30, 5)
    
    with st.expander("🎯 Lifestyle"):
        all_hobbies = ['Baking', 'Cooking', 'Electronics', 'Fitness', 'Gaming',
                      'Gardening', 'Photography', 'Reading', 'Social Media',
                      'Sports', 'Travel', 'Video Games', 'Writing']
        selected_hobbies = st.multiselect("Select Hobbies", all_hobbies)

# Main Content Area
tab1, tab2 = st.tabs(["🧠 Prediction", "📊 Insights"])

with tab1:
    if st.button("🔍 Analyze Financial Personality", use_container_width=True):
        # Prepare input data
        input_dict = {
            'Income': income,
            'Age': age,
            'Occupation': models['le_occupation'].transform([occupation])[0],
            'City_Tier': models['le_city'].transform([city])[0],
            'Loan_Repayment': loan,
            'Fixed Expenses': fixed,
            'Investment': investment,
            'Savings': savings,
            'Shopping_Frequency': shop_freq
        }
        
        # One-hot encode hobbies
        for hobby in all_hobbies:
            input_dict[hobby] = 1 if hobby in selected_hobbies else 0
        
        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_dict])
        numerical_cols = ['Income', 'Age', 'Loan_Repayment', 'Fixed Expenses', 
                         'Investment', 'Savings', 'Shopping_Frequency']
        input_df[numerical_cols] = models['scaler'].transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = models['model'].predict(input_df)
        predicted_class = models['le_class'].inverse_transform(prediction)[0]
        probabilities = models['model'].predict_proba(input_df)[0]
        
        # Display results
        st.markdown("---")
        
        # Personality Card
        class_colors = {
            "Saver": "#2ECC71",
            "Spender": "#E74C3C",
            "Neutral": "#F39C12"
        }
        
        st.markdown(f"""
        <div style='background-color:{class_colors[predicted_class]};padding:20px;border-radius:10px;color:white;'>
            <h2 style='color:white;text-align:center;'>Financial Personality: {predicted_class}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability Visualization
        prob_df = pd.DataFrame({
            "Class": models['le_class'].classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        
        fig = px.bar(prob_df, 
                     x="Probability", 
                     y="Class", 
                     color="Class",
                     color_discrete_map=class_colors,
                     text="Probability",
                     title="Prediction Confidence",
                     height=300)
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        
        # Financial Health Indicators
        st.subheader("💰 Financial Health Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            savings_ratio = savings / income if income > 0 else 0
            st.metric("Savings Ratio", f"{savings_ratio:.1%}", 
                     help="Percentage of income saved monthly")
        
        with col2:
            invest_ratio = investment / income if income > 0 else 0
            st.metric("Investment Ratio", f"{invest_ratio:.1%}", 
                     help="Percentage of income invested monthly")
        
        with col3:
            expense_ratio = (fixed + loan) / income if income > 0 else 0
            st.metric("Fixed Expense Ratio", f"{expense_ratio:.1%}", 
                     help="Percentage of income spent on fixed obligations")

with tab2:
    st.subheader("📈 Financial Personality Insights")
    
    # Personality Distribution Info
    st.markdown("""
    ### What Each Personality Means:
    - **Saver**: Typically saves >20% of income, invests regularly, low discretionary spending
    - **Spender**: High discretionary spending, saves <10% of income, minimal investments
    - **Neutral**: Balanced approach, saves 10-20% of income, moderate spending
    """)
    
    # Example Profiles
    st.markdown("### 🏆 Ideal Financial Ratios")
    st.table(pd.DataFrame({
        "Metric": ["Savings Ratio", "Investment Ratio", "Fixed Expense Ratio", "Discretionary Spending"],
        "Saver": [">20%", ">15%", "<50%", "<10%"],
        "Neutral": ["10-20%", "5-15%", "50-65%", "10-20%"],
        "Spender": ["<10%", "<5%", ">65%", ">20%"]
    }).set_index("Metric"))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#5D6D7E; font-size:14px;'>
    <i>Created with ❤️ by Ikram Shaik</i>
</div>
""", unsafe_allow_html=True)
