import streamlit as st
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import MalnutritionPredictor

# Page Configuration
st.set_page_config(
    page_title="Child Malnutrition Risk Predictor",
    page_icon="👶",
    layout="wide"
)

# Title and Header
st.title("👶 Child Malnutrition Risk Assessment Tool")
st.markdown("""
This application uses a **Decentralized Federated Learning Model** trained across multiple healthcare nodes to predict child malnutrition risk while protecting patient privacy.
""")

st.sidebar.header("⚙️ Settings & Thresholds")
# Threshold slider to control clinical sensitivity/recall
threshold = st.sidebar.slider(
    "Decision Threshold (Risk Sensitivity)", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.35, 
    step=0.05,
    help="Lower thresholds increase recall (sensitivity) to avoid missing high-risk malnutrition cases."
)

import os

@st.cache_resource
def load_model_and_scaler():
    """Loads dataset features to fit the scaler and loads global PyTorch model."""
    # Check both potential dataset locations
    data_path = os.path.join("dataset", "malnutrition_child_dataset_cleaned_v2.csv")
    if not os.path.exists(data_path):
        if os.path.exists("malnutrition_child_dataset_cleaned_v2.csv"):
            data_path = "malnutrition_child_dataset_cleaned_v2.csv"
        else:
            raise FileNotFoundError(
                f"Dataset file not found at '{data_path}' or root folder."
            )
    
    df = pd.read_csv(data_path)
    
    # Feature extraction matching training setup
    X = df.drop(columns=['is_malnourished', 'stunting_zscore', 'wasting_zscore', 'underweight_zscore', 'child_bmi_raw'], errors='ignore')
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    model = MalnutritionPredictor(input_dim=X.shape[1])
    model.load_state_dict(torch.load("global_malnutrition_model.pth"))
    model.eval()
    
    return model, scaler, X.columns.tolist()

try:
    model, scaler, feature_cols = load_model_and_scaler()
    st.sidebar.success("✅ Global Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Patient Clinical Data Form
st.subheader("📋 Enter Patient & Demographic Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👶 Child Details")
    child_age_months = st.number_input("Child Age (Months)", min_value=0, max_value=60, value=12)
    gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    birth_weight = st.number_input("Birth Weight (kg)", min_value=0.5, max_value=6.0, value=3.0, step=0.1)
    weight_kg = st.number_input("Current Weight (kg)", min_value=1.0, max_value=30.0, value=8.5, step=0.1)
    height_cm = st.number_input("Current Height (cm)", min_value=30.0, max_value=120.0, value=72.0, step=0.5)
    breastfeeding = st.selectbox("Currently Breastfeeding?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.markdown("### 🩺 Recent Symptoms")
    diarrhea = st.selectbox("Recent Diarrhea Episode?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    fever = st.selectbox("Recent Fever Episode?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    cough = st.selectbox("Recent Cough Episode?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.markdown("### 👩 Mother Details")
    mother_age = st.number_input("Mother Age (Years)", min_value=12, max_value=50, value=25)
    mother_education = st.selectbox("Mother Education Level", options=[0, 1, 2, 3], 
                                     format_func=lambda x: ["No Education", "Primary", "Secondary", "Higher"][x])
    mother_bmi = st.number_input("Mother BMI", min_value=12.0, max_value=45.0, value=21.5, step=0.1)
    children_ever_born = st.number_input("Total Children Ever Born", min_value=1, max_value=15, value=2)
    birth_order = st.number_input("Birth Order of Child", min_value=1, max_value=15, value=1)
    antenatal_visits = st.number_input("Antenatal Visits", min_value=0, max_value=20, value=4)

with col3:
    st.markdown("### 🏠 Household Details")
    urban_rural = st.selectbox("Location Type", options=[1, 2], format_func=lambda x: "Urban" if x == 1 else "Rural")
    wealth_index = st.selectbox("Household Wealth Index", options=[1, 2, 3, 4, 5], 
                                 format_func=lambda x: ["Poorest", "Poorer", "Middle", "Richer", "Richest"][x-1])
    water_source = st.number_input("Water Source Category Code", min_value=1, max_value=50, value=12)
    time_to_water = st.number_input("Time to Water Source (Minutes)", min_value=0, max_value=300, value=15)
    sanitation_facility = st.number_input("Sanitation/Toilet Facility Code", min_value=1, max_value=100, value=44)
    handwashing_facility = st.number_input("Handwashing Facility Code", min_value=1, max_value=100, value=34)
    household_members = st.number_input("Total Household Members", min_value=1, max_value=30, value=5)

child_age_years = float(child_age_months) / 12.0

st.markdown("---")

# Prediction Execution
if st.button("🔍 Assess Malnutrition Risk", type="primary", use_container_width=True):
    # Construct input feature dictionary in exact feature order
    input_data = {
        'child_age_months': child_age_months,
        'gender': gender,
        'birth_weight': birth_weight,
        'weight_kg': weight_kg,
        'height_cm': height_cm,
        'breastfeeding': breastfeeding,
        'diarrhea': diarrhea,
        'fever': fever,
        'cough': cough,
        'mother_age': mother_age,
        'mother_education': mother_education,
        'mother_bmi': mother_bmi,
        'children_ever_born': children_ever_born,
        'birth_order': birth_order,
        'urban_rural': urban_rural,
        'wealth_index': wealth_index,
        'water_source': water_source,
        'time_to_water': time_to_water,
        'sanitation_toilet_facility': sanitation_facility,
        'handwashing_facility': handwashing_facility,
        'child_age_years': child_age_years,
        'household_members': household_members,
        'antenatal_visits': antenatal_visits
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Scale input features
    scaled_input = scaler.transform(input_df)
    
    # Model Inference
    with torch.no_grad():
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        risk_probability = model(input_tensor).item()
    
    # Output Display
    st.subheader("📊 Assessment Result")
    
    risk_percentage = risk_probability * 100
    st.progress(risk_probability)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(label="Calculated Malnutrition Risk", value=f"{risk_percentage:.2f}%")
        
    with col_b:
        if risk_probability >= threshold:
            st.error("⚠️ HIGH RISK: Child is flagged for potential malnutrition.")
            st.markdown("""
            **Recommended Actions:**
            * Refer child for detailed Anthropometric Assessment (MUAC / Z-Score check).
            * Provide immediate dietary supplementation advice.
            * Schedule follow-up within 14 days.
            """)
        else:
            st.success("✅ LOW RISK: Child shows healthy growth indicators.")
            st.markdown("""
            **Recommended Actions:**
            * Continue standard routine checkups and immunizations.
            * Encourage optimal infant feeding and nutrition practices.
            """)