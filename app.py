import streamlit as st
import joblib
import pandas as pd
import yfinance as yf

st.title("توقعات الأسهم باستخدام نموذج XGBoost")

MODEL_PATH = "xgb_model.pkl"

def prepare_features(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    required_cols = ['high', 'low', 'close', 'open']
    cols_lower = [c.lower() for c in df.columns]
    for col in required_cols:
        if col not in cols_lower:
            raise ValueError(f"العمود المطلوب غير موجود: {col}")

    df['hl_range'] = df['high'] - df['low']
    df['oc_change'] = df['close'] - df['open']
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df = df.dropna()
    features = df[['hl_range', 'oc_change', 'ma_5', 'ma_10']]
    return features

def train_model():
    st.info("النموذج غير موجود، جاري التدريب...")
    st.warning("يجب تدريب النموذج على بيانات محددة مسبقًا، لأن هذا الإصدار لا يحتوي على بيانات تدريب.")
    return None  # يمكنك تعديل هذه الدالة لاحقًا لتدريب النموذج على بيانات معينة

if st.button("تحميل النموذج"):
    if st.file_uploader("ارفع ملف النموذج (xgb_model.pkl)", type=["pkl"]) is not None:
        uploaded_file = st.file_uploader("ارفع ملف النموذج (xgb_model.pkl)", type=["pkl"])
        if uploaded_file:
            with open(MODEL_PATH, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("تم تحميل النموذج بنجاح.")
    else:
        st.warning("يرجى رفع ملف النموذج أولاً.")

if st.button("استخدام النموذج"):
    if not st.session_state.get("model_loaded", False):
        try:
            model = joblib.load(MODEL_PATH)
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("تم تحميل النموذج بنجاح.")
        except Exception as e:
            st.error(f"خطأ في تحميل النموذج: {e}")

symbol = st.text_input("أدخل رمز السهم (مثلاً AAPL):", value="AAPL").upper()
period = st.selectbox("اختر فترة التحليل", ["1mo", "3mo", "6mo", "1y"])

if st.button("توقع الاتجاه"):
    if not st.session_state.get("model_loaded", False):
        st.error("يرجى تحميل النموذج أولاً بالضغط على زر 'استخدام النموذج'")
    else:
        with st.spinner("جاري تحميل البيانات وتحليلها..."):
            try:
                data = yf.download(symbol, period=period, interval="1d", progress=False)
                if data.empty:
                    st.warning("⚠️ لا توجد بيانات للسهم أو الفترة المحددة.")
                    st.stop()

                X_pred = prepare_features(data)
                if X_pred.empty:
                    st.warning("⚠️ البيانات غير كافية لتحليل السهم.")
                else:
                    prediction = st.session_state.model.predict(X_pred.tail(1))[0]
                    directions = {0: "هبوط", 1: "استقرار", 2: "صعود"}
                    st.success(f"الاتجاه المتوقع للسهم **{symbol}** هو: **{directions.get(prediction, 'غير معروف')}**")
            except Exception as e:
                st.error(f"حدث خطأ أثناء التحميل أو التنبؤ: {e}")
