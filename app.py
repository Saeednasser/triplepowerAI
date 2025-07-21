import os
import streamlit as st
import joblib

model_path = "rf_model.pkl"

if not os.path.exists(model_path):
    st.error("ملف النموذج rf_model.pkl غير موجود. يرجى تشغيل تدريب النموذج أولاً.")
    st.stop()

model = joblib.load(model_path)

# باقي الكود...
