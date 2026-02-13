import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.title("🇨🇦 TSX Best Stock Finder (Live)")

stocks = ["SHOP.TO", "SU.TO", "RY.TO", "TD.TO", "BNS.TO"]

results = []

for ticker in stocks:
    df = yf.download(ticker, period="3mo", interval="1d")

    if df.empty:
        continue

    df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    last = df.iloc[-1]

    score = 0
    
    if last["Close"] > last["EMA20"]:
        score += 1
        
    if last["RSI"] > 50:
        score += 1

    results.append({
        "Stock": ticker,
        "Price": round(last["Close"], 2),
        "RSI": round(last["RSI"], 2),
        "Score": score
    })

if len(results) == 0:
    st.error("No data available.")
else:
    results_df = pd.DataFrame(results)
    best = results_df.sort_values(by="Score", ascending=False).iloc[0]

    st.subheader("🔥 Best Stock Right Now")
    st.write(best)
