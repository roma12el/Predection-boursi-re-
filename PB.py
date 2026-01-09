import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.mixture import GaussianMixture

import xgboost as xgb
import catboost as cb

from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import shap
from lime import lime_tabular

st.set_page_config(page_title="MASI AI Prediction", layout="wide")
st.title("📈 MASI – Prédiction Boursière par IA")

# =========================
# UPLOAD DATA
# =========================
st.sidebar.header("📂 Upload du dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is None:
    st.warning("Veuillez uploader le fichier CSV du MASI.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df['exchange_date'] = pd.to_datetime(df['exchange_date'])
    df = df.sort_values('exchange_date').set_index('exchange_date')

    for col in ['close','open','low','high','h-l','trade_price']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    df['%chg'] = df['%chg'].astype(str).str.replace('%','').astype(float)
    return df

df = load_data(file)

# =========================
# TABS
# =========================
tabs = st.tabs([
    "EDA", "Features", "ML", "Prophet", "LSTM/GRU",
    "Hybrid", "Explainability", "Backtesting", "Advanced"
])

# =========================
# EDA
# =========================
with tabs[0]:
    st.subheader("Exploratory Data Analysis")
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(df.index, df['close'])
    ax.set_title("Indice MASI")
    st.pyplot(fig)

    df['log_return'] = np.log(df['close']/df['close'].shift(1))
    st.line_chart(df['log_return'])

# =========================
# FEATURES
# =========================
with tabs[1]:
    st.subheader("Feature Engineering")

    for lag in range(1,21):
        df[f'lag_{lag}'] = df['log_return'].shift(lag)

    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['volatility_30'] = df['log_return'].rolling(30).std()
    df['target'] = df['log_return'].shift(-1)

    df_ml = df.dropna()
    st.write(df_ml.head())

# =========================
# ML MODELS
# =========================
with tabs[2]:
    st.subheader("Machine Learning Models")

    X = df_ml.drop(columns=['target'])
    y = df_ml['target']

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBRegressor(n_estimators=300)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    st.metric("RMSE XGBoost", rmse)

# =========================
# PROPHET
# =========================
with tabs[3]:
    st.subheader("Prophet Forecast")

    df_p = df.reset_index()[['exchange_date','close']]
    df_p.columns = ['ds','y']

    model_p = Prophet()
    model_p.fit(df_p)

    future = model_p.make_future_dataframe(periods=30)
    forecast = model_p.predict(future)

    fig = model_p.plot(forecast)
    st.pyplot(fig)

# =========================
# LSTM / GRU
# =========================
with tabs[4]:
    st.subheader("Deep Learning – LSTM")

    scaler_dl = MinMaxScaler()
    data = scaler_dl.fit_transform(df[['close']])

    def seq(data, w=30):
        X,y=[],[]
        for i in range(w,len(data)):
            X.append(data[i-w:i])
            y.append(data[i])
        return np.array(X),np.array(y)

    Xs,ys = seq(data)
    split = int(len(Xs)*0.8)
    Xtr,Xte = Xs[:split],Xs[split:]
    ytr,yte = ys[:split],ys[split:]

    model_lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(30,1)),
        LSTM(32),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(Xtr,ytr,epochs=5,verbose=0)

    preds = scaler_dl.inverse_transform(model_lstm.predict(Xte))
    real = scaler_dl.inverse_transform(yte)

    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(real,label="Real")
    ax.plot(preds,label="Pred")
    ax.legend()
    st.pyplot(fig)

# =========================
# BACKTESTING
# =========================
with tabs[7]:
    st.subheader("Backtesting Strategy")

    signal = np.sign(preds.flatten())
    strat = signal * yte.flatten()
    cum = (1+strat).cumprod()

    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(cum)
    ax.set_title("Cumulative Strategy Return")
    st.pyplot(fig)

st.success("Application prête à l'emploi 🚀")
