from flask import Flask, render_template, request
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_ticker = request.form['stock']
    days = int(request.form['days'])

    # Download past 60 days of stock data
    data = yf.download(stock_ticker, period="60d")
    if data.empty:
        return "Invalid stock ticker or no data available."

    data = data[['Close']].reset_index()
    data['Day'] = range(1, len(data) + 1)

    X = data[['Day']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.array(range(len(data) + 1, len(data) + days + 1)).reshape(-1, 1)
    predicted_prices = model.predict(future_days)

    predictions = []
    last_date = data['Date'].max()

    for i in range(days):
        date = (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
        predictions.append(f"{date}: {predicted_prices[i][0]:.2f}")

    return render_template('result.html', stock=stock_ticker.upper(), predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
