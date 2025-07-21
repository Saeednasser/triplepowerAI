import os
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def prepare_features(df):
    df = df.copy()
    df['HL_range'] = df['High'] - df['Low']
    df['OC_change'] = df['Close'] - df['Open']
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df = df.dropna()
    features = df[['HL_range', 'OC_change', 'MA_5', 'MA_10']]
    return features

def main():
    symbol = "AAPL"
    data_file = f"{symbol}_data.csv"

    # تحميل البيانات من الإنترنت وحفظها إذا لم تكن موجودة
    if not os.path.exists(data_file):
        print("تحميل البيانات من yfinance...")
        data = yf.download(symbol, period="2y", interval="1d", progress=False)
        data.to_csv(data_file)
        print(f"تم حفظ بيانات السهم في ملف {data_file}")
    else:
        print(f"تحميل البيانات من الملف {data_file}")
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # تجهيز الميزات
    X = prepare_features(data)

    # تحضير الهدف
    y = (data['Close'].shift(-1) - data['Close']).fillna(0).squeeze()
    y = y.apply(lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1))
    y = y.loc[X.index]

    if len(X) == 0 or len(y) == 0:
        print("لا توجد بيانات كافية للتدريب.")
        return

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # تدريب النموذج
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # حفظ النموذج
    joblib.dump(model, "rf_model.pkl")
    print("تم حفظ النموذج rf_model.pkl")

if __name__ == "__main__":
    main()
