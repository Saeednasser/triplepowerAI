import pandas as pd
import numpy as np

def prepare_features(df):
    df = df.copy()
    df['HL_range'] = df['High'] - df['Low']
    df['OC_change'] = df['Close'] - df['Open']
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df = df.dropna()
    features = df[['HL_range', 'OC_change', 'MA_5', 'MA_10']]
    return features