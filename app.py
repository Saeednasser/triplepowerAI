import streamlit as st
import joblib
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

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

def train_model(symbol="AAPL", period="2y"):
    st.info("جاري تدريب النموذج على بيانات السهم...")
    data = yf.download(symbol, period=period, interval="1d", progress=False)
    if data.empty:
        st.error("⚠️ لم يتم تحميل بيانات كافية للتدريب.")
        return None

    X = prepare_features(data)
    y = (data['Close'].shift(-1) - data['Close']).fillna(0)
    y = y.apply(lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1))
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    st.success("تم تدريب النموذج وحفظه بنجاح.")
    return model

def load_model():
    if st.session_state.get("model") is not None:
        return st.session_state.model

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            st.session_state.model = model
            st.success("تم تحميل النموذج بنجاح.")
            return model
        except Exception as e:
            st.error(f"خطأ في تحميل النموذج: {e}")
            return None
    else:
        return None

symbol = st.text_input("أدخل رمز السهم (مثلاً AAPL):", value="AAPL").upper()
period = st.selectbox("اختر فترة التحليل", ["1mo", "3mo", "6mo", "1y", "2y"])

model = load_model()
if model is None:
    model = train_model(symbol, "2y")  # تدريب النموذج على سهم AAPL افتراضياً بفترة سنتين

if st.button("توقع الاتجاه"):
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
                prediction = model.predict(X_pred.tail(1))[0]
                directions = {0: "هبوط", 1: "استقرار", 2: "صعود"}
                st.success(f"الاتجاه المتوقع للسهم **{symbol}** هو: **{directions.get(prediction, 'غير معروف')}**")
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحميل أو التنبؤ: {e}")
