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
[main.py](main.py)
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
- `main.py` – Python code for stock price prediction  
- `README.md` – Project documentation  
