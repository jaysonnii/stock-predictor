# Stock Predictor

A machine-learning stock price forecasting application built with Python, FastAPI, scikit-learn, yfinance, and a browser-based frontend.

The application downloads historical market data, engineers technical indicators, trains a Random Forest regression model, and generates estimated future stock prices and returns.

> This project is educational and should not be considered financial advice.

## Features

- Search stocks by ticker symbol
- View current stock information
- Display historical price data
- Generate 1-month, 6-month, and 1-year forecasts
- Train separate machine-learning models for supported tickers
- Calculate expected returns
- Track stocks using a locally saved watchlist
- Access interactive FastAPI documentation
- Automatically save trained models for later use

## Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning and Data

- scikit-learn
- Random Forest Regression
- pandas
- NumPy
- yfinance
- joblib

### Frontend

- HTML
- CSS
- JavaScript

## How It Works

1. The application downloads historical stock data through yfinance.
2. It creates technical indicators and lag-based features.
3. It creates a target based on the future percentage return.
4. The data is divided chronologically into training and testing sets.
5. A Random Forest regression model is trained.
6. The trained model and scaler are saved locally.
7. The latest market data is processed and passed into the model.
8. The predicted return is converted into an estimated future price.

## Model Features

The model uses information including:

- Closing price
- Trading volume
- 20-day moving average
- 50-day moving average
- 200-day moving average
- Price relative to moving averages
- Daily returns
- 20-day volatility
- Relative Strength Index
- MACD
- MACD signal
- Historical lag prices
- Volume change

## Project Structure

```text
stockPredictor/
├── data.py
├── main.py
├── model.py
├── requirements.txt
├── README.md
├── static/
│   └── index.html
└── models/
    └── Generated model and scaler files
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jaysonnii/stockPredictor.git
cd stockPredictor
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

```powershell
python -m uvicorn main:app --reload
```

Open the application:

```text
http://127.0.0.1:8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Opens the web application |
| GET | `/predict/all?ticker=AAPL` | Returns multiple forecast horizons |
| GET | `/predict?ticker=AAPL&days=252` | Returns a prediction for a selected horizon |
| POST | `/train?ticker=AAPL` | Trains or retrains a model |
| GET | `/history?ticker=AAPL&period=1y` | Returns historical closing prices |
| GET | `/info?ticker=AAPL` | Returns general company and stock information |
| GET | `/docs` | Opens the FastAPI documentation |

## Example

After starting the application, search for a ticker such as:

```text
AAPL
MSFT
GOOGL
AMZN
NVDA
TSLA
META
JPM
SPY
BRK-B
```

The first prediction for a ticker may take longer because the application must download historical data and train a new model.

## Model Evaluation

During training, the application calculates:

- Mean Absolute Error
- Root Mean Squared Error
- Directional accuracy
- Number of training samples
- Number of testing samples

The project uses a chronological train-test split to avoid randomly mixing future and past market observations.

## Limitations

- Stock prices are influenced by events that historical price data cannot fully predict.
- Predictions are estimates and may be inaccurate.
- Shorter forecast horizons are derived from the one-year trend.
- The model does not currently include news, financial statements, economic indicators, or sentiment data.
- Trained models are saved locally and are not shared between devices.
- The watchlist is saved in the browser's local storage.

## Future Improvements

- Add automated model testing
- Add additional machine-learning models
- Compare model performance
- Add news and sentiment analysis
- Add financial statement features
- Add user accounts and cloud-based watchlists
- Add Docker support
- Add continuous integration with GitHub Actions
- Deploy the application online

## Disclaimer

This application was created for educational and portfolio purposes. Its predictions should not be treated as financial advice or used as the sole basis for investment decisions.

## Author

Jay Soni

- GitHub: [jaysonnii](https://github.com/jaysonnii)
- LinkedIn: [Jay Soni](https://www.linkedin.com/in/jayy-soni/)
