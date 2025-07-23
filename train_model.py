import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def load_clean_data(path):
    df = pd.read_csv(path, header=[0,1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(1)
    df.columns = df.columns.str.strip().str.lower()  # تحويل لأحرف صغيرة وتنظيف
    print("الأعمدة بعد التنظيف:", list(df.columns))
    return df

def prepare_features(df):
    df = df.copy()
    required_cols = ['high', 'low', 'close', 'open']
    cols_lower = [c.lower() for c in df.columns]
    for col in required_cols:
        if col not in cols_lower:
            raise ValueError(f"العمود المطلوب غير موجود: {col}")

    df.columns = df.columns.str.lower()
    df['hl_range'] = df['high'] - df['low']
    df['oc_change'] = df['close'] - df['open']
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df = df.dropna()
    features = df[['hl_range', 'oc_change', 'ma_5', 'ma_10']]
    return features

def main():
    data = load_clean_data("AAPL_data.csv")
    X = prepare_features(data)
    y = (data['close'].shift(-1) - data['close']).fillna(0)
    y = y.apply(lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1))
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "xgb_model.pkl")
    print("تم حفظ النموذج xgb_model.pkl")

if __name__ == "__main__":
    main()
