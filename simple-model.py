import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# Directory where the CSV files are stored
directory_path = 'stock_market_data'

# List to hold dataframes
dataframes = []

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        # Read each file into a DataFrame
        df = pd.read_csv(file_path)
        # Optionally add a column to identify the stock ticker
        df['Ticker'] = filename.replace('.csv', '')
        dataframes.append(df)

# Combine all DataFrames into a single DataFrame
combined_data = pd.concat(dataframes, ignore_index=True)


# Handle missing values
# Instead of combined_data.fillna(method='ffill', inplace=True)
combined_data.ffill(inplace=True)

# Convert date column to datetime type if exists
if 'Date' in combined_data.columns:
    # combined_data['Date'] = pd.to_datetime(combined_data['Date'])

    # If your date format is day-month-year, use:
    combined_data['Date'] = pd.to_datetime(combined_data['Date'], dayfirst=True)

    # Or specify the exact format
    # combined_data['Date'] = pd.to_datetime(combined_data['Date'], format='%d-%m-%Y')

# Feature Engineering
# Calculate moving averages
combined_data['MA20'] = combined_data['Close'].rolling(window=20).mean()
combined_data['MA50'] = combined_data['Close'].rolling(window=50).mean()

# Calculate daily returns
combined_data['Returns'] = combined_data['Close'].pct_change()

# Add more features that might be useful
combined_data['Volatility'] = combined_data['Returns'].rolling(window=20).std()

# Normalize volume data
combined_data['Volume_norm'] = combined_data['Volume'] / combined_data['Volume'].max()

# Display the first few rows to check everything
print(combined_data.head())


# Selecting features and target
features = ['MA20', 'MA50', 'Volatility', 'Volume_norm']
target = 'Returns'

# Preparing data
X = combined_data[features].ffill()
y = combined_data[target].ffill()

# Ensure no NaN values are in your features or target
if X.isnull().any().any() or combined_data[target].isnull().any():
    print("NaN values detected in X or y, filling missing values")
    X.ffill(inplace=True)
    X.bfill(inplace=True)
    combined_data[target].ffill(inplace=True)
    combined_data[target].bfill(inplace=True)

# Now you can safely split your data
y = combined_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Check for infinite values and handle them
if np.isinf(X_train).any().any() or np.isinf(y_train).any():
    print("Infinite values detected, replacing with NaN and filling")
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.ffill(inplace=True)
    X_train.bfill(inplace=True)
    y_train.ffill(inplace=True)
    y_train.bfill(inplace=True)

combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
assert not combined_data.isnull().any().any(), "NaN values exist in the DataFrame after handling."


# Initialize and train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Final check before training
if X_train.isnull().any().any() or y_train.isnull().any():
    raise ValueError("NaN or Inf values present in training data after all checks.")
else:
    print("Data is clean. Proceeding with model training.")

# Prepare feature matrix X and target vector y
X = combined_data[features]
y = combined_data[target]

# Replace and fill infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
y.replace([np.inf, -np.inf], np.nan, inplace=True)
X.ffill(inplace=True)
X.bfill(inplace=True)
y.ffill(inplace=True)
y.bfill(inplace=True)

# Check again for any remaining missing or infinite values
if np.isinf(X.values).any() or np.isinf(y.values).any():
    raise ValueError("Data contains infinite values after all handling.")

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and calculate MSE
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


# Initialize and train RandomForest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
print(f'Random Forest MSE: {rf_mse}')


def simulate_trading(data, model, threshold=0.001):
    buy_signals, sell_signals = [], []
    for index, row in data.iterrows():
        prediction = model.predict([row[features].values])[0]
        if prediction > threshold:  # Threshold to decide buy
            buy_signals.append((index, row['Close']))
        elif prediction < -threshold:  # Threshold to decide sell
            sell_signals.append((index, row['Close']))

    # Example output
    return buy_signals, sell_signals


# Simulate trading on the test set
buy_signals, sell_signals = simulate_trading(combined_data.loc[X_test.index], model)


# Ensure all NaNs are filled after all calculations
# Instead of modifying the slice in-place, modify the whole DataFrame directly for the target column
combined_data[target] = combined_data[target].ffill()
combined_data[target] = combined_data[target].bfill()



print("Buy signals:", buy_signals[:5])  # print first 5 buy signals
print("Sell signals:", sell_signals[:5])  # print first 5 sell signals

# Example: Filtering data for a specific stock and plotting

apple_data = combined_data[combined_data['Ticker'] == 'AAPL']
plt.figure(figsize=(10, 6))
plt.plot(apple_data['Date'], apple_data['MA20'], label='20-day MA')
plt.title('20-Day Moving Average for Apple')
plt.xlabel('Date')
plt.ylabel('Moving Average Price')
plt.legend()
plt.show()