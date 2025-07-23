import os
import streamlit as st
import joblib
import pandas as pd
import yfinance as yf

st.title("توقعات الأسهم باستخدام نموذج XGBoost")

MODEL_PATH = "xgb_model.pkl"
DATA_CSV = "AAPL_data.csv"

def load_clean_data(path):
    df = pd.read_csv(path, header=[0,1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(1)
    df.columns = df.columns.str.strip()  # إزالة المسافات الزائدة من أسماء الأعمدة
    return df

def prepare_features(df):
    df = df.copy()
    required_cols = ['High', 'Low', 'Close', 'Open']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"العمود المطلوب غير موجود: {col}")

    df['HL_range'] = df['High'] - df['Low']
    df['OC_change'] = df['Close'] - df['Open']
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df = df.dropna()
    features = df[['HL_range', 'OC_change', 'MA_5', 'MA_10']]
    return features

def train_model():
    st.info("النموذج غير موجود، جاري التدريب...")
    data = load_clean_data(DATA_CSV)
    X = prepare_features(data)
    y = (data['Close'].shift(-1) - data['Close']).fillna(0)
    y = y.apply(lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1))
    y = y.loc[X.index]

    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    st.success("تم تدريب النموذج وحفظه بنجاح.")
    return model

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.success("تم تحميل النموذج بنجاح.")
    except Exception as e:
        st.error(f"حدث خطأ عند تحميل النموذج: {e}")
        model = train_model()
else:
    model = train_model()

symbol = st.text_input("أدخل رمز السهم (مثلاً AAPL):", value="AAPL").upper()
period = st.selectbox("اختر فترة التحليل", ["1mo", "3mo", "6mo", "1y"])

if st.button("توقع الاتجاه"):
    with st.spinner("جاري تحميل البيانات وتحليلها..."):
        try:
            if symbol == "AAPL" and os.path.exists(DATA_CSV):
                data = load_clean_data(DATA_CSV)
            else:
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
