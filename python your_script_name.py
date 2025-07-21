import yfinance as yf

symbol = "AAPL"
data = yf.download(symbol, period="2y", interval="1d")
data.to_csv(f"{symbol}_data.csv")
print("تم حفظ بيانات السهم في ملف CSV.")
