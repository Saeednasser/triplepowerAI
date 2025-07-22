import os
import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="توقعات الأسهم بالذكاء الاصطناعي", layout="centered")
st.title("توقع اتجاهات الأسهم باستخدام نموذج القوة الثلاثية والذكاء الاصطناعي")

model_path = "rf_model.pkl"

if not os.path.exists(model_path):
    st.error("⚠️ ملف النموذج rf_model.pkl غير موجود. يرجى تشغيل تدريب النموذج أولاً.")
    st.stop()

model = joblib.load(model_path)

def prepare_features(df):
    df = df.copy()
    df['HL_range'] = df['High'] - df['Low']
    df['OC_change'] = df['Close'] - df['Open']
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df = df.dropna()
    features = df[['HL_range', 'OC_change', 'MA_5', 'MA_10']]
    return features

# إدخال المستخدم
symbol = st.text_input("أدخل رمز السهم (مثلاً AAPL):", value="AAPL").upper()
period = st.selectbox("اختر فترة التحليل", ["1mo", "3mo", "6mo", "1y"])

if st.button("توقع الاتجاه"):
    with st.spinner("جاري تحميل البيانات وتحليلها..."):
        try:
            data = yf.download(symbol, period=period, interval="1d", progress=False)
            if data.empty:
                st.warning("⚠️ لا توجد بيانات للسهم المدخل أو الفترة المحددة.")
            else:
                X = prepare_features(data)
                if X.empty:
                    st.warning("⚠️ البيانات غير كافية لتحليل السهم.")
                else:
                    pred = model.predict(X.tail(1))[0]
                    directions = {0: "هبوط", 1: "استقرار", 2: "صعود"}
                    st.success(f"الاتجاه المتوقع للسهم **{symbol}** هو: **{directions.get(pred, 'غير معروف')}**")
        except Exception as e:
            st.error(f"حدث خطأ أثناء تحميل البيانات أو التنبؤ: {e}")
