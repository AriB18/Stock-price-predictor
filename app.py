import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import ta
import matplotlib.pyplot as plt

analyzer = SentimentIntensityAnalyzer()

def fetch_stock_data(ticker, days):
    stock = yf.Ticker(ticker)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    df = df.reset_index()
    return df

def fetch_news(ticker):
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)
    headlines = []
    for entry in feed.entries:
        title = entry.title
        link = entry.link
        score = analyzer.polarity_scores(title)['compound']
        headlines.append({
            'title': title,
            'snippet': '',
            'link': link,
            'score': score
        })
    return headlines

def run_pipeline(ticker, days):
    stock_data = fetch_stock_data(ticker, days)

    sentiment_results = fetch_news(ticker)
    scores = [item['score'] for item in sentiment_results]
    avg_sentiment = np.mean(scores) if scores else 0

    stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
    stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=14)

    stock_data['Prev_Close'] = stock_data['Close'].shift(1)
    stock_data['Sentiment'] = avg_sentiment
    stock_data['Day'] = pd.to_datetime(stock_data['Date']).dt.day
    stock_data['Month'] = pd.to_datetime(stock_data['Date']).dt.month

    stock_data = stock_data.dropna().copy()
    features = ['Prev_Close', 'SMA_10', 'EMA_10', 'RSI', 'Sentiment', 'Day', 'Month']
    X = stock_data[features]
    y = stock_data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

    rf_params = {'n_estimators': [50], 'max_depth': [5]}
    rf = RandomForestRegressor()
    rf_grid = GridSearchCV(rf, rf_params, cv=2, scoring='neg_root_mean_squared_error')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    xgb_params = {'n_estimators': [50], 'max_depth': [3]}
    xgb = XGBRegressor(objective='reg:squarederror')
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=2, scoring='neg_root_mean_squared_error')
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_

    rf_pred = best_rf.predict(X_test)
    xgb_pred = best_xgb.predict(X_test)
    blended_pred = (rf_pred + xgb_pred) / 2

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    blended_rmse = np.sqrt(mean_squared_error(y_test, blended_pred))

    rf_cv = cross_val_score(best_rf, X_train, y_train, cv=2, scoring='neg_root_mean_squared_error')
    xgb_cv = cross_val_score(best_xgb, X_train, y_train, cv=2, scoring='neg_root_mean_squared_error')

    return {
        'Data': stock_data,
        'X': X,
        'RF_Model': best_rf,
        'XGB_Model': best_xgb,
        'RMSEs': {
            'Random Forest RMSE': rf_rmse,
            'XGBoost RMSE': xgb_rmse,
            'Blended RMSE': blended_rmse,
            'Random Forest CV RMSE': -rf_cv.mean(),
            'XGBoost CV RMSE': -xgb_cv.mean()
        },
        'Sentiment Headlines': sentiment_results
    }

st.title("📈 Stock Price Predictor")

ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()

if st.button("Run Prediction"):
    result = run_pipeline(ticker, 365 * 1)

    st.subheader("Company Info")
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    st.write(f"**Full Name:** {info.get('longName', 'Unknown')}")
    st.write(f"**About:** {info.get('longBusinessSummary', 'No summary available.')[:200]}...")
    st.write(f"**Current Price:** ${info.get('currentPrice', 'NA')}")

    st.subheader("Model Performance (1 Year Window)")
    st.write(result['RMSEs'])

    latest_features = result['X'].iloc[[-1]]
    rf_next = result['RF_Model'].predict(latest_features)[0]
    xgb_next = result['XGB_Model'].predict(latest_features)[0]
    blended_next = (rf_next + xgb_next) / 2

    st.subheader("Predicted Next Closing Price")
    st.write(f"**Random Forest:** ${rf_next:.2f}")
    st.write(f"**XGBoost:** ${xgb_next:.2f}")
    st.write(f"**Blended:** ${blended_next:.2f}")

    st.subheader("📰 Why is this stock moving?")
    if result['Sentiment Headlines']:
        top_news = result['Sentiment Headlines'][0]
        st.write(f"**Headline:** {top_news['title']}")
        st.write(f"[Read more]({top_news['link']})")
    else:
        st.write("No recent headlines found.")

    st.subheader("Price Chart")
    fig, ax = plt.subplots()
    ax.plot(result['Data']['Date'], result['Data']['Close'], label='Close Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
