import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Get historical data for Amazon (AMZN)
ticker = "AMZN"
amzn = yf.Ticker(ticker)
data = amzn.history(period="1y")

# Prepare data for training
data = data[["Close"]]
X = np.array(range(len(data))).reshape(-1, 1)
y = data.values.ravel()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make a prediction for the next day
next_day = len(data)
next_day_price = model.predict([[next_day]])

# Evaluate the model on the test set
test_predictions = model.predict(X_test)
mse = np.mean((y_test - test_predictions) ** 2)
print(f"Mean Squared Error: {mse:.2f}")

print(f"Predicted closing price for {ticker} on the next day: ${next_day_price[0]:.2f}")
