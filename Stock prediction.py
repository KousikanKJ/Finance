import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Download stock data
def get_stock_data(ticker, period="5y"):
    '''Download stock data from Yahoo Finance'''
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Example: Get S&P 500 data
ticker = "SPY"  # S&P 500 ETF
df = get_stock_data(ticker, "5y")
print(f"Downloaded {len(df)} days of data for {ticker}")
print(df.head())

# ================================
# 2. FEATURE ENGINEERING
# ================================

def create_features(df):
    '''Create technical indicators and features'''
    data = df.copy()

    # Basic price features
    data['Returns'] = data['Close'].pct_change()
    data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
    data['Price_Change'] = data['Close'] - data['Open']

    # Moving averages
    data['SMA_5'] = ta.trend.sma_indicator(data['Close'], window=5)
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)

    # Technical indicators
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd_diff(data['Close'])

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['BB_Width'] = data['BB_High'] - data['BB_Low']

    # Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()

    # Lagged features
    for i in [1, 2, 3, 5]:
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
        data[f'Volume_Lag_{i}'] = data['Volume'].shift(i)

    return data

# ================================
# 3. TARGET VARIABLE CREATION
# ================================

def create_target(df):
    '''Create binary target: 1 if next day price goes up, 0 if down'''
    data = df.copy()
    data['Tomorrow_Close'] = data['Close'].shift(-1)
    data['Target'] = (data['Tomorrow_Close'] > data['Close']).astype(int)
    return data

# ================================
# 4. MODEL TRAINING & EVALUATION
# ================================

def prepare_data(df):
    '''Prepare data for modeling'''
    # Remove rows with NaN values
    df_clean = df.dropna()

    # Select features (exclude price columns to avoid data leakage)
    feature_cols = [col for col in df_clean.columns if col not in 
                   ['Open', 'High', 'Low', 'Close', 'Tomorrow_Close', 'Target', 'Adj Close']]

    X = df_clean[feature_cols]
    y = df_clean['Target']

    return X, y, df_clean

def backtest_model(df, model, train_size=0.8):
    '''Perform time series backtesting'''
    # Sort by date to maintain temporal order
    df_sorted = df.sort_index()

    # Split data chronologically
    split_idx = int(len(df_sorted) * train_size)
    train_data = df_sorted.iloc[:split_idx]
    test_data = df_sorted.iloc[split_idx:]

    # Prepare features and targets
    X_train, y_train, _ = prepare_data(train_data)
    X_test, y_test, test_df = prepare_data(test_data)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'actual': y_test
    }

    return results, test_df

# ================================
# 5. IMPLEMENTATION EXAMPLE
# ================================

# Create features and target
df_features = create_features(df)
df_with_target = create_target(df_features)

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    result, test_df = backtest_model(df_with_target, model)
    results[name] = result
    print(f"\n{name} Results:")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Precision: {result['precision']:.3f}")
    print(f"Recall: {result['recall']:.3f}")

# ================================
# 6. VISUALIZATION
# ================================

def plot_predictions(test_df, predictions, actual):
    '''Plot prediction results'''
    plt.figure(figsize=(12, 8))

    # Plot stock price and predictions
    plt.subplot(2, 1, 1)
    plt.plot(test_df.index, test_df['Close'], label='Stock Price', alpha=0.7)
    plt.title('Stock Price Over Time')
    plt.legend()

    # Plot prediction accuracy over time
    plt.subplot(2, 1, 2)
    correct_predictions = (predictions == actual).astype(int)
    rolling_accuracy = pd.Series(correct_predictions, index=test_df.index).rolling(20).mean()
    plt.plot(test_df.index, rolling_accuracy, label='Rolling 20-day Accuracy')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
    plt.title('Model Accuracy Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()
