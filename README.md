# توقع اتجاهات الأسهم باستخدام نموذج القوة الثلاثية والذكاء الاصطناعي

مشروع بسيط يستخدم بيانات yfinance مع نموذج تعلم آلي للتنبؤ باتجاهات الأسهم.

## الاستخدام

- قم بتثبيت المتطلبات:
```
pip install -r requirements.txt
```

- درب النموذج أولاً:
```
python train_model.py
```

- ثم شغل التطبيق:
```
streamlit run app.py
```

- أدخل رمز السهم واختر فترة التحليل ثم اضغط "توقع الاتجاه".