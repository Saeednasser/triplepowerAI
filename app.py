import os
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

    # تحويل MultiIndex إلى أسماء أعمدة مسطحة إذا لزم الأمر
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
    else:
        df.columns = df.columns.str.strip()

    df.columns = [col.lower() for col in df.columns]

    # دالة للعثور على العمود الذي يبدأ بالمفتاح المطلوب
    def find_col(cols, key):
        for c in cols:
            if c.startswith(key):
                return c
        return None

    required_keys = ['high', 'low', 'close', 'open']
    found_cols = {}
    for key in required_keys:
        col_name = find_col(df.columns, key)
        if col_name is None:
            raise ValueError(f"العمود المطلوب غير موجود: {key}")
        found_cols[key] = col_name

    df['hl_range'] = df[found_cols['high']] - df[found_cols['low']]
    df['oc_change'] = df[found_cols['close']] - df[found_cols['open']]
    df['ma_5'] = df[found_cols['close']].rolling(window=5).mean()
    df['ma_10'] = df[found_cols['close']].rolling(window=10).mean()
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

symbols_input = st.text_input("أدخل رمز أو أكثر من الأسهم (مفصول بمسافة أو فاصلة):", value="AAPL")
symbols = [s.strip().upper() for s in symbols_input.replace(",", " ").split() if s.strip()]

period = st.selectbox("اختر فترة التحليل", ["1mo", "3mo", "6mo", "1y", "2y"])

model = load_model()
if model is None:
    model = train_model("AAPL", "2y")  # تدريب افتراضي للنموذج

if st.button("توقع الاتجاه"):
    results = []
    with st.spinner("جاري تحميل البيانات وتحليلها..."):
        for symbol in symbols:
            try:
                data = yf.download(symbol, period=period, interval="1d", progress=False)
                if data.empty:
                    results.append((symbol, "⚠️ لا توجد بيانات"))
                    continue

                X_pred = prepare_features(data)
                if X_pred.empty:
                    results.append((symbol, "⚠️ بيانات غير كافية"))
                    continue

                prediction = model.predict(X_pred.tail(1))[0]
                directions = {0: "هبوط", 1: "استقرار", 2: "صعود"}
                results.append((symbol, directions.get(prediction, "غير معروف")))
            except Exception as e:
                results.append((symbol, f"خطأ: {e}"))

    df_results = pd.DataFrame(results, columns=["الرمز", "الاتجاه المتوقع"])
    st.table(df_results)
