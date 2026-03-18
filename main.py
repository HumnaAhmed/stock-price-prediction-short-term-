import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -------------------------------
# 1. Fetch Stock Data
# -------------------------------
def fetch_data():
    """
    Fetch stock data using yfinance API
    """
    print("Downloading stock data...")
    stock = yf.download("AAPL", period="1y")

    print("\nDataset Info:")
    print("Start Date:", stock.index.min())
    print("End Date:", stock.index.max())
    print("Total Rows:", len(stock))

    return stock

# -------------------------------
# 2. Preprocess Data
# -------------------------------
def preprocess_data(stock):
    """
    Create target variable (Next Day Close)
    and select features
    """
    stock["Next_Close"] = stock["Close"].shift(-1)
    stock = stock.dropna()

    X = stock[["Open", "High", "Low", "Volume"]]
    y = stock["Next_Close"]

    return X, y, stock

# -------------------------------
# 3. Time Series Split
# -------------------------------
def split_data(X, y):
    """
    Split data chronologically (not randomly)
    to avoid future data leakage
    """
    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

# -------------------------------
# 4. Plot Stock Trend
# -------------------------------
def plot_stock(stock):
    """
    Plot closing price trend over time
    """
    plt.figure(figsize=(10, 5))
    plt.plot(stock["Close"])
    plt.title("Stock Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()

# -------------------------------
# 5. Train Model
# -------------------------------
def train_model(X_train, y_train):
    """
    Train Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModel training completed.")
    return model

# -------------------------------
# 6. Evaluate Model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Make predictions and calculate error
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    print("\nModel Evaluation:")
    print("Mean Absolute Error:", round(mae, 3))

    return predictions

# -------------------------------
# 7. Create Results Table
# -------------------------------
def create_results(y_test, predictions):
    """
    Create professional results table
    """
    results = pd.DataFrame({
        "Date": y_test.index,
        "Actual Price": y_test.values,
        "Predicted Price": predictions
    })

    results["Error"] = abs(results["Actual Price"] - results["Predicted Price"])
    results[["Actual Price","Predicted Price","Error"]] = results[["Actual Price","Predicted Price","Error"]].round(2)
    results = results.reset_index(drop=True)

    print("\n" + "="*60)
    print("\t\tPREDICTION RESULTS TABLE")
    print("="*60)
    print(results.head(10).to_string(index=False))
    print("="*60)

    return results

# -------------------------------
# 8. Plot Predictions
# -------------------------------
def plot_predictions(results):
    """
    Plot actual vs predicted prices
    """
    plt.figure(figsize=(12, 6))

    plt.plot(results["Date"], results["Actual Price"], label="Actual Price")
    plt.plot(results["Date"], results["Predicted Price"], label="Predicted Price")

    plt.title("Actual vs Predicted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")

    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

# -------------------------------
# 9. Predict Next Day Price
# -------------------------------
def predict_next_day(model, X):
    """
    Predict next day closing price
    """
    latest_data = X.tail(1)
    prediction = model.predict(latest_data)

    print("\nNext Day Prediction:")
    print("Predicted Closing Price:", round(prediction[0], 2))

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    stock = fetch_data()
    X, y, stock = preprocess_data(stock)
    X_train, X_test, y_train, y_test = split_data(X, y)

    plot_stock(stock)

    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)

    results = create_results(y_test, predictions)
    plot_predictions(results)

    predict_next_day(model, X)

# Run program
if __name__ == "__main__":
    main()