import streamlit as st
import yfinance as yf
import pandas as pd

st.title("📈 Live Stock Test")

ticker = "SHOP.TO"

st.write(f"Fetching data for {ticker}...")

df = yf.download(ticker, period="1mo", interval="1d")

if df.empty:
    st.error("No data available from Yahoo Finance.")
else:
    st.success("Data loaded successfully!")
    st.dataframe(df.tail())
