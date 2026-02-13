import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import time

st.set_page_config(page_title="TSX Best Stock Live", layout="wide")

st.title("🇨🇦 TSX Best Stock + Entry & Exit (Live)")

# 🔥 Auto refresh every 60 seconds
st.caption("Auto-refreshing every 60 seconds...")
time.sleep(1)

stocks = [
    "SHOP.TO", "SU.TO", "RY.TO", "TD.TO", "BNS.TO",
    "ENB.TO", "CNQ.TO", "CP.TO", "CNR.TO", "BAM.TO"
]

results = []

for ticker in stocks:
    df = yf.download(ticker, period="3mo", interval="1d")

    if df.empty:
        continue

    # Fix MultiIndex issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    df["EMA20"] = ta.trend.ema_indicator(close=close, window=20)
    df["RSI"] = ta.momentum.rsi(close=close, window=14)
    df["ATR"] = ta.volatility.average_true_range(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    last = df.iloc[-1]

    score = 0

    if last["Close"] > last["EMA20"]:
        score += 1

    if last["RSI"] > 50:
        score += 1

    # Entry, Stop, Target
    entry = float(last["Close"])
    stop_loss = entry - float(last["ATR"])
    take_profit = entry + (2 * float(last["ATR"]))  # 2:1 risk reward

    confidence = round((score / 2) * 100, 0)

    results.append({
        "Stock": ticker,
        "Price": round(entry, 2),
        "Entry": round(entry, 2),
        "Stop Loss": round(stop_loss, 2),
        "Take Profit": round(take_profit, 2),
        "RSI": round(float(last["RSI"]), 2),
        "Confidence %": confidence,
        "Score": score
    })

if len(results) == 0:
    st.error("No data available.")
else:
    results_df = pd.DataFrame(results)
    best = results_df.sort_values(by="Score", ascending=False).iloc[0]

    st.subheader("🔥 BEST STOCK TO BUY RIGHT NOW")

    col1, col2, col3 = st.columns(3)

    col1.metric("Stock", best["Stock"])
    col2.metric("Price", best["Price"])
    col3.metric("Confidence", f"{best['Confidence %']}%")

    st.write("### Trade Setup")
    st.write(f"**Entry:** {best['Entry']}")
    st.write(f"**Stop Loss:** {best['Stop Loss']}")
    st.write(f"**Take Profit:** {best['Take Profit']}")

    st.write("### RSI")
    st.write(best["RSI"])

    st.divider()
    st.write("### 📊 All Stocks Ranked")
    st.dataframe(results_df.sort_values(by="Score", ascending=False))

# 🔥 Auto rerun
time.sleep(60)
st.rerun()
