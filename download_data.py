import yfinance as yf

def download_and_save(symbol="AAPL", period="2y", interval="1d"):
    """
    تحميل بيانات سهم من yfinance وحفظها في ملف CSV.

    المعطيات:
    - symbol: رمز السهم (مثلاً "AAPL")
    - period: فترة التحميل (مثلاً "2y" للسنتين الأخيرتين)
    - interval: فترة الشمعة (مثلاً "1d" ليومي)

    الناتج:
    - حفظ ملف CSV باسم {symbol}_data.csv في المجلد الحالي.
    """

    print(f"تحميل بيانات السهم {symbol} للفترة {period} بفاصل {interval}...")
    data = yf.download(symbol, period=period, interval=interval)
    if data.empty:
        print("⚠️ لم يتم تحميل بيانات، يرجى التحقق من الرمز أو الاتصال بالإنترنت.")
        return
    filename = f"{symbol}_data.csv"
    data.to_csv(filename)
    print(f"تم حفظ بيانات السهم في الملف: {filename}")

if __name__ == "__main__":
    # يمكنك تعديل رمز السهم أو الفترة هنا حسب الحاجة
    download_and_save(symbol="AAPL", period="2y", interval="1d")
