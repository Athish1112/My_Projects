
import streamlit as st
import pandas as pd
import pickle
import sklearn


# Load model and scaler (replace with actual file paths)
try:
    model = pickle.load(open("classifier.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"Error: Model and/or scaler files not found. ({e})")
    exit(1)

# Streamlit interface
st.title("Diabetes Prediction System")
st.header("Enter your Medical Info:")

# User input fields (using consistent feature names)
pregnancies = st.number_input("Number of Pregnancies:", min_value=0, step=1)
glucose = st.number_input("Glucose (mg/dL):", min_value=0, step=1)
blood_pressure = st.number_input("BloodPressure (mmHg):", min_value=0, step=1)
skin_thickness = st.number_input("SkinThickness (mm):", min_value=0, step=1)
insulin = st.number_input("Insulin (uU/mL):", min_value=0, step=1)
bmi = st.number_input("BMI (kg/m²):", min_value=0.0, step=0.2)
diabetes_pedigree_function = st.number_input("DiabetesPedigreeFunction:", min_value=0.0, step=0.3)
age = st.number_input("Age:", min_value=0, step=1)

if st.button("Predict"):
    data = pd.DataFrame({
        # Ensure feature names match training data
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree_function],
        "Age": [age],
    })
    std_data = scaler.transform(data)
    prediction = model.predict(std_data)

    if prediction == 0:
        st.write("Predicted outcome: You are unlikely to have diabetes.")
    else:
        st.write("Predicted outcome: You are at risk of diabetes.")

    st.write("Disclaimer: This website is for informational purposes only. Consult a healthcare professional for medical advice.")
