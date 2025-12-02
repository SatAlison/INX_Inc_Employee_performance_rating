import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="INX Future Inc. - Performance Predictor",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL AND RESOURCES
# ----------------------------
@st.cache_resource
def load_resources():
    resources = {
        'model': None,
        'feature_order': None,
        'binary_mappings': None,
        'mode': 'demo',
        'error': None
    }

    model_dir = 'models'
    if os.path.exists(model_dir):
        try:
            resources['model'] = joblib.load(os.path.join(model_dir, 'xgb_model.pkl'))
            resources['feature_order'] = joblib.load(os.path.join(model_dir, 'feature_order.pkl'))
            resources['binary_mappings'] = joblib.load(os.path.join(model_dir, 'binary_mappings.pkl'))
            resources['mode'] = 'production'
        except Exception as e:
            resources['error'] = str(e)
    return resources

resources = load_resources()

# ----------------------------
# ONE-HOT ENCODING FUNCTION
# ----------------------------
def preprocess_inputs(inputs):
    df = pd.DataFrame([inputs])
    
    # Map binary features
    for col, mapping in (resources['binary_mappings'] or {}).items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # One-hot encode categorical string columns
    cat_cols = ['EducationBackground_grouped', 'EmpDepartment_grouped', 'EmpJobRole_grouped']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    
    # Ensure all columns are in training order
    for col in resources['feature_order']:
        if col not in df.columns:
            df[col] = 0  # Add missing dummy columns as 0
    df = df[resources['feature_order']]
    
    return df

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(inputs):
    try:
        df = preprocess_inputs(inputs)
        proba = resources['model'].predict_proba(df)
        return proba
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def get_rating_label(proba):
    classes = ['Rating 2', 'Rating 3', 'Rating 4']
    return classes[np.argmax(proba)]

# ----------------------------
# UI
# ----------------------------
st.title("🔍 INX Future Inc. – Employee Performance Predictor")
st.markdown("Predict employee performance for talent strategy. All features included.")

if resources['mode'] == 'production':
    st.sidebar.success("✅ Production Mode")
else:
    st.sidebar.warning("🧪 Demo Mode")

if resources['error']:
    st.sidebar.error(f"Error loading resources: {resources['error']}")

st.markdown("---")

# INPUTS
inputs = {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographics & Basic Info")
    inputs['Age'] = st.slider("🎂 Age", 18, 65, 35)
    inputs['Gender'] = st.radio("🧑 Gender", ["Male", "Female"])
    inputs['DistanceFromHome'] = st.slider("📍 Distance From Home (km)", 1, 50, 10)
    inputs['EmpEducationLevel'] = st.select_slider("🎓 Education Level", [1,2,3,4,5], value=3)
    inputs['MaritalStatus'] = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
    inputs['Attrition'] = st.radio("🚪 Attrition Risk", ["No", "Yes"])

    st.subheader("💼 Work Experience")
    inputs['TotalWorkExperienceInYears'] = st.slider("📅 Total Work Experience", 0, 40, 10)
    inputs['ExperienceYearsAtThisCompany'] = st.slider("🏢 Years at INX", 0, 40, 5)
    inputs['ExperienceYearsInCurrentRole'] = st.slider("🎯 Years in Current Role", 0, 20, 3)
    inputs['YearsSinceLastPromotion'] = st.slider("⏰ Years Since Last Promotion", 0, 15, 2)
    inputs['YearsWithCurrManager'] = st.slider("👤 Years With Current Manager", 0, 15, 3)
    inputs['NumCompaniesWorked'] = st.slider("🏢 Past Companies Worked", 1, 10, 2)
    inputs['CompanyExperienceShare'] = st.slider("📈 % Career at INX", 0.0, 1.0, 0.7)
    inputs['RoleStabilityScore'] = st.slider("🛡️ Role Stability Score", 0.0, 1.0, 0.6)

with col2:
    st.subheader("🎯 Job & Satisfaction")
    inputs['EmpJobLevel'] = st.slider("📊 Job Level", 1, 5, 2)
    inputs['EmpHourlyRate'] = st.slider("💰 Hourly Rate", 30, 100, 65)
    inputs['EmpLastSalaryHikePercent'] = st.slider("📈 Last Salary Hike (%)", 0, 30, 12)
    inputs['OverTime'] = st.radio("🕒 Overtime", ["No", "Yes"])
    inputs['TrainingTimesLastYear'] = st.slider("🎓 Training Times Last Year", 0, 6, 3)
    inputs['EmpEnvironmentSatisfaction'] = st.select_slider("🌱 Environment Satisfaction", [1,2,3,4], value=3)
    inputs['EmpJobSatisfaction'] = st.select_slider("😊 Job Satisfaction", [1,2,3,4], value=3)
    inputs['EmpJobInvolvement'] = st.select_slider("🎯 Job Involvement", [1,2,3,4], value=3)
    inputs['EmpRelationshipSatisfaction'] = st.select_slider("👥 Relationship Satisfaction", [1,2,3,4], value=3)
    inputs['EmpWorkLifeBalance'] = st.select_slider("⚖️ Work-Life Balance", [1,2,3,4], value=3)

    st.subheader("🏢 Department & Role")
    inputs['EducationBackground_grouped'] = st.selectbox("🧪 Education Background", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
    inputs['EmpDepartment_grouped'] = st.selectbox("🏢 Department", ["Sales", "Development", "Research & Development", "Human Resources", "Finance", "Data Science"])
    inputs['EmpJobRole_grouped'] = st.selectbox("🎯 Job Role", ["Sales Executive", "Developer", "Manager R&D", "Manufacturing Director", "Healthcare Representative",
        "Lab Technician", "Sales Representative", "Research Scientist", "Data Scientist", 
        "Human Resources", "Senior Developer", "Manager", "Technical Lead", "Other"])
    inputs['CommuteCategory_Moderate'] = st.checkbox("Commute: Moderate")
    inputs['CommuteCategory_Far'] = st.checkbox("Commute: Far")
    inputs['MaritalStatus_Married'] = st.checkbox("Married")
    inputs['MaritalStatus_Single'] = st.checkbox("Single")
    inputs['PromotionWaitTime_Moderate'] = st.checkbox("Promotion Wait: Moderate")
    inputs['PromotionWaitTime_Long'] = st.checkbox("Promotion Wait: Long")
    inputs['BusinessTravelFrequency_Travel_Frequently'] = st.checkbox("Travel Frequently")
    inputs['BusinessTravelFrequency_Travel_Rarely'] = st.checkbox("Travel Rarely")

# ----------------------------
# PREDICT BUTTON
# ----------------------------
if st.button("🔮 Predict Performance"):
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    try:
        proba = predict(inputs)
        rating = get_rating_label(proba[0])
        confidence = np.max(proba)*100
        
        if rating == 'Rating 4':
            st.success(f"**{rating}** - Exceeds expectations 🌟")
        elif rating == 'Rating 3':
            st.info(f"**{rating}** - Meets expectations ✅")
        else:
            st.warning(f"**{rating}** - Needs improvement ⚠️")
        
        st.metric("Confidence", f"{confidence:.0f}%")
    except Exception as e:
        st.error(f"❌ {e}")

# ----------------------------
# FEATURE IMPORTANCE (optional)
# ----------------------------
st.markdown("---")
st.subheader("📊 Key Performance Drivers")
try:
    importance_scores = resources['model'].feature_importances_
    feature_names = resources['feature_order']
    top_idx = np.argsort(importance_scores)[-15:][::-1]
    features = [feature_names[i] for i in top_idx]
    importances = [importance_scores[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(features, importances, color="#4A90E2")
    ax.set_xlabel("Feature Importance")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
except:
    st.info("Feature importance not available in demo mode.")

st.markdown("---")
st.caption("© 2025 INX Future Inc. | For strategic HR use only")
