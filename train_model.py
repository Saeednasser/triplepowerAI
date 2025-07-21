import os
import yfinance as yf
import pandas as pd
from model import prepare_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def main():
    symbol = "AAPL"
    data_file = f"{symbol}_data.csv"

    # تحميل البيانات من الملف إذا موجود
    if os.path.exists(data_file):
        print(f"تحميل البيانات من الملف {data_file}")
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        print("تحميل البيانات من yfinance...")
        try:
            data = yf.download(symbol, period="2y", interval="1d", progress=False)
            data.to_csv(data_file)
        except Exception as e:
            print(f"خطأ أثناء تحميل البيانات: {e}")
            return

    X = prepare_features(data)

    if X.empty:
        print("لا توجد بيانات كافية بعد تجهيز الميزات.")
        return

    y = (data['Close'].shift(-1) - data['Close']).fillna(0).squeeze()
    y = y.apply(lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1))
    y = y.loc[X.index]

    if len(X) == 0 or len(y) == 0:
        print("لا توجد بيانات كافية للتدريب.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "rf_model.pkl")
    print("تم حفظ النموذج rf_model.pkl")

if __name__ == "__main__":
    main()
