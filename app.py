import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta
from model import prepare_features

# تحميل النموذج المدرب
model = joblib.load("rf_model.pkl")

st.title("توقع اتجاهات الأسهم باستخدام نموذج القوة الثلاثية والذكاء الاصطناعي")

symbol = st.text_input("أدخل رمز السهم (مثلاً AAPL):", value="AAPL").upper()
period = st.selectbox("اختر فترة التحليل", ["1mo", "3mo", "6mo", "1y"])

if st.button("توقع الاتجاه"):
    data = yf.download(symbol, period=period, interval="1d", progress=False)
    if data.empty:
        st.error("لا توجد بيانات للسهم المدخل")
    else:
        # تجهيز الميزات
        X = prepare_features(data)
        # توقع الاتجاه لآخر يوم
        pred = model.predict(X.tail(1))[0]
        directions = {0: "هبوط", 1: "استقرار", 2: "صعود"}
        st.write(f"الاتجاه المتوقع للسهم {symbol} هو: **{directions.get(pred, 'غير معروف')}**")