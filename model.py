import pandas as pd

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    تجهيز الميزات من بيانات الشموع لتدريب النموذج أو التنبؤ.

    المدخلات:
    df : DataFrame يحتوي على أعمدة ['Open', 'High', 'Low', 'Close']

    المخرجات:
    DataFrame يحتوي على الميزات التالية:
    - HL_range : الفرق بين أعلى وأدنى سعر في الشمعة
    - OC_change : الفرق بين سعر الإغلاق وسعر الفتح
    - MA_5 : المتوسط المتحرك لآخر 5 أيام لسعر الإغلاق
    - MA_10 : المتوسط المتحرك لآخر 10 أيام لسعر الإغلاق
    """

    df = df.copy()
    df['HL_range'] = df['High'] - df['Low']
    df['OC_change'] = df['Close'] - df['Open']
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()

    # حذف الصفوف التي تحتوي قيم مفقودة (ناجمة عن المتوسطات المتحركة)
    df = df.dropna()

    # اختيار الأعمدة كميزات
    features = df[['HL_range', 'OC_change', 'MA_5', 'MA_10']]

    return features
