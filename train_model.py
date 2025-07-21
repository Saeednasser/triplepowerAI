import yfinance as yf
import pandas as pd
from model import prepare_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def main():
    symbol = "AAPL"
    data = yf.download(symbol, period="2y", interval="1d", progress=False)
    X = prepare_features(data)

    y = (data['Close'].shift(-1) - data['Close']).fillna(0)
    y = y.apply(lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1))
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "rf_model.pkl")
    print("تم حفظ النموذج rf_model.pkl")

if __name__ == "__main__":
    main()