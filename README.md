# 📈 Stock Price Prediction (Short-Term)

## Task Objective
Predict the **next day’s closing stock price** using historical stock data and machine learning. The goal is to understand how past stock features can help estimate future prices.

---

## Dataset
Stock market data retrieved from Yahoo Finance using the `yfinance` Python library.

Features used:
- `Open` – Opening price  
- `High` – Highest price  
- `Low` – Lowest price  
- `Volume` – Number of shares traded  

Target variable:
- `Next_Close` – Next day closing price (created using shift)

Stock used:
- Apple Inc. (AAPL)

---

## Tools Used
- Python  
- Pandas  
- Matplotlib  
- Scikit-learn  
- yfinance  

---

## Steps Performed
1. Fetched historical stock data using API.  
2. Explored dataset (date range and size).  
3. Created target variable (`Next_Close`) using time shift.  
4. Selected relevant features (Open, High, Low, Volume).  
5. Applied time-series split for training and testing.  
6. Visualized stock closing price trend.  
7. Trained regression model.  
8. Evaluated model using Mean Absolute Error (MAE).  
9. Compared actual vs predicted prices using plots.  
10. Displayed results in a structured table.  

---

## Model Applied
- Linear Regression  

---

## Key Results and Findings
- Model predicts next day closing price with reasonable accuracy  
- Mean Absolute Error (MAE) shows prediction performance  
- Predicted values follow actual trends closely  
- Time-series split prevents data leakage  
- Stock prices show trends but also fluctuations  

---

## Insights
- Linear Regression works for basic trend prediction  
- Stock market is highly unpredictable  
- More advanced models can improve results  

---

## Files
- [main.py](main.py) – Python code for stock price prediction  
- [README.md](README.md) – Project documentation  

---

## Outputs & Results

### 1. Stock Closing Price Trend
*Shows the historical closing prices for Apple stock.*

<img width="490" height="245" alt="image" src="https://github.com/user-attachments/assets/06a97463-7a5e-4695-b0fb-3f1fa045a6c3" />

### 2. Actual vs Predicted Prices
*Comparison of actual and predicted next-day closing prices.*

<img width="488" height="244" alt="image" src="https://github.com/user-attachments/assets/b7752e3b-b841-45cf-aa08-1a721c22e4c5" />

### 3. Prediction Results Table
*Structured table showing actual price, predicted price, and error.*

<img width="365" height="242" alt="image" src="https://github.com/user-attachments/assets/47855132-39da-4f44-aaf8-b37749235f62" />

### 4. Next Day Prediction
*Predicted next day closing price from the trained model.*

<img width="201" height="36" alt="image" src="https://github.com/user-attachments/assets/d2e3ebda-1bc7-4a84-87e3-6f450e38ca1a" />

