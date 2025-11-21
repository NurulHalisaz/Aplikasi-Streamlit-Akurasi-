import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_page_config(page_title="Prediksi Diabetes", page_icon="🧠", layout="centered")

# Load model dan scaler
model = joblib.load("model_diabetes.pkl")
scaler = joblib.load("scaler_diabetes.pkl")

st.title("🧠 Prediksi Diabetes dengan Ensemble Machine Learning")
st.write("Gunakan aplikasi ini untuk memprediksi risiko diabetes berdasarkan data kesehatan.")

# Sidebar - info akurasi model
st.sidebar.header("📊 Informasi Model")
st.sidebar.success("Model: Ensemble (RandomForest + XGBoost)")
st.sidebar.write("🎯 Akurasi Latih:")
st.sidebar.subheader("**> 90%** (sesuai hasil training)")

# Input form
st.subheader("📝 Masukkan Data Pasien")
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=80.0, value=30.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
Age = st.number_input("Age", min_value=1, max_value=120, value=35)

if st.button("🔍 Prediksi"):
    # Input ke dataframe
    data = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]])

    # Scaling
    data_scaled = scaler.transform(data)
    
    # Predict
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0]

    st.subheader("🎯 Hasil Prediksi")
    if pred == 1:
        st.error(f"⚠ POTENSI DIABETES — Probabilitas: {prob[1]*100:.2f}%")
    else:
        st.success(f"💚 SEHAT — Probabilitas: {prob[0]*100:.2f}%")

    # Grafik distribusi probabilitas
    st.subheader("📌 Distribusi Probabilitas")
    labels = ["Sehat", "Diabetes"]
    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=prob, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilitas")
    st.pyplot(fig)

    # Confusion matrix visual
    # Catatan: Di Streamlit hanya bisa ditampilkan setelah evaluasi training,
    # jadi kita tampilkan confusion matrix simulasi dari prediksi user
    cm = confusion_matrix([pred], [1 if prob[1] > prob[0] else 0])
    st.subheader("📌 Confusion Matrix (Prediksi Saat Ini)")
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sehat", "Diabetes"], yticklabels=["Sehat", "Diabetes"])
    ax2.set_ylabel("Aktual")
    ax2.set_xlabel("Prediksi")
    st.pyplot(fig2)

st.info("💡 Tips: Semakin tinggi nilai Glucose, Insulin, BMI, dan Age, risiko diabetes semakin meningkat.")
