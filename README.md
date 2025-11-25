# Stock-price-predictor

This project is a Streamlit based web application that predicts the next closing price of a stock using machine learning models, technical indicators, and news sentiment analysis. It combines financial data from Yahoo Finance with real time sentiment from Yahoo Finance RSS feeds to generate insights and predictions.

**Features**
1. Stock Price Data

The app fetches historical price data using the yfinance library and builds technical indicators including:

Simple Moving Average (SMA)

Exponential Moving Average (EMA)

Relative Strength Index (RSI)

2. News Sentiment Analysis

The app retrieves the latest news headlines for the selected stock using RSS feeds and applies VADER sentiment analysis to quantify how positive or negative the market news is.

3. Machine Learning Models

Two regression models are trained:

Random Forest Regressor

XGBoost Regressor

Both models are tuned with GridSearchCV and evaluated with RMSE. A blended forecast average of both models is also generated for increased stability.

4. Prediction

The app predicts the next closing price of the stock using:

Random Forest prediction

XGBoost prediction

Blended prediction

5. Visuals and Insights

The app provides:

Company summary from Yahoo Finance

Recent sentiment driven headline explaining possible price movements

RMSE scores for model performance

A closing price chart for the last year

Project Structure
├── app.py                      # Streamlit application
├── README.md                   # Project documentation

How It Works
Data Pipeline

Fetch one year of stock price data.

Calculate technical indicators.

Pull recent news headlines and extract sentiment scores.

Prepare the dataset with engineered features.

Train and tune Random Forest and XGBoost models.

Generate predictions and evaluate using RMSE.

**User Workflow**

Enter a stock ticker (example: AAPL).

Click Run Prediction.

View company details, model results, sentiment insights, and price chart.

See the estimated next closing price.

Installation
1. Clone the repository
git clone <your_repo_url>
cd stock-price-predictor

2. Install dependencies
pip install -r requirements.txt


Required libraries include:

streamlit

yfinance

pandas

numpy

feedparser

vaderSentiment

scikit-learn

xgboost

ta

matplotlib

3. Run the app
streamlit run app.py

Requirements

Python 3.8 or above is recommended.

Notes

Predictions are based on historical patterns and sentiment. They are not financial advice.

Yahoo Finance does not guarantee the availability of all stock summaries or data fields.

Future Improvements

Add LSTM based deep learning models

Include more sentiment sources such as Twitter and financial blogs

Add option to forecast multiple future days

Deploy the app to Streamlit Cloud
