
# app.py
from flask import Flask, render_template, request
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_symbol = request.form['stock_symbol']
        days = int(request.form['days'])

        # Download historical stock data
        stock_data = yf.download(stock_symbol, start="2022-01-01", end=datetime.datetime.now())

        # Feature engineering
        stock_data['Date'] = stock_data.index
        stock_data['Date'] = stock_data['Date'].apply(lambda x: x.toordinal())

        X = np.array(stock_data['Date']).reshape(-1, 1)
        y = stock_data['Close']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        future_dates = np.array([max(stock_data['Date']) + i for i in range(1, days + 1)]).reshape(-1, 1)
        future_predictions = model.predict(future_dates)

        return render_template('result.html', predictions=future_predictions)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
