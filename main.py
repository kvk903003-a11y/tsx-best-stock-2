import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(page_title="TSX Best Stock + Entry Exit", layout="wide")

st.title("📌 Best Stock + Entry + Exit (EMA + RSI)")

# Load tickers
@st.cache_data
def load_tickers():
    return pd.read_csv("data/tsx_tickers.csv", header=None)[0].tolist()

# Indicators
def EMA(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

tickers = load_tickers()
results = []

for t in tickers:
    try:
        df = yf.download(t, period="30d", interval="1h")
        if df.empty:
            continue

        df["EMA20"] = EMA(df["Close"], 20)
        df["RSI14"] = RSI(df["Close"], 14)

        last = df.iloc[-1]
        price = last["Close"]
        ema = last["EMA20"]
        rsi = last["RSI14"]

        # Entry condition
        entry = None
        exit = None

        if price > ema and rsi < 70:
            entry = price
            exit = price * 1.03  # 3% target
        elif price < ema or rsi > 70:
            exit = price

        results.append({
            "Ticker": t,
            "Price": round(price, 2),
            "EMA20": round(ema, 2),
            "RSI14": round(rsi, 2),
            "Entry": round(entry, 2) if entry else None,
            "Exit": round(exit, 2) if exit else None
        })
    except:
        continue

df = pd.DataFrame(results)

if df.empty:
    st.warning("No data available.")
    st.stop()

# Sort by best trend
df = df.sort_values("RSI14")

best = df[df["Entry"].notna()].sort_values("RSI14").head(1)

if best.empty:
    st.warning("No trade signals right now.")
else:
    best = best.iloc[0]
    st.subheader("✅ Best Stock to Trade Now")
    st.metric("Ticker", best["Ticker"])
    st.metric("Entry Price", f"${best['Entry']}")
    st.metric("Exit Price", f"${best['Exit']}")

st.subheader("📊 All Stocks (EMA + RSI)")
st.dataframe(df, use_container_width=True)
