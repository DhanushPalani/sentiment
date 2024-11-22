# Ex.No.6 Sentiment Analysis

## Part A: Sentiment Analysis
## Aim
To analyze textual data, derive sentiment scores, and classify text as positive, negative, or neutral using Python's NLTK and TextBlob libraries.
## Step 1: Setup and Installation
Install necessary libraries:
```
pip install nltk textblob
```
Download additional resources for NLTK:
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
## Step 2: Sentiment Analysis Using TextBlob
## code:
```
from textblob import TextBlob

# Sample text for sentiment analysis
text = "I love programming in Python. It is such a powerful and enjoyable language!"

# Create a TextBlob object
blob = TextBlob(text)

# Perform sentiment analysis
sentiment = blob.sentiment

# Output the sentiment polarity and subjectivity
print("Sentiment Analysis using TextBlob:")
print("Polarity:", sentiment.polarity)  # -1 (negative) to 1 (positive)
print("Subjectivity:", sentiment.subjectivity)  # 0 (objective) to 1 (subjective)
```
## Output:
![image](https://github.com/user-attachments/assets/adb0ac28-aef9-4b34-a3a0-b60ed7e652cb)
## Step 3: Sentiment Analysis Using NLTK (VADER)
## code:
```
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Sample text
text = "I love programming in Python. It is such a powerful and enjoyable language!"

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis
sentiment_score = sia.polarity_scores(text)

# Output the sentiment scores
print("Sentiment Analysis using NLTK (VADER):")
print("Positive:", sentiment_score['pos'])
print("Neutral:", sentiment_score['neu'])
print("Negative:", sentiment_score['neg'])
print("Compound:", sentiment_score['compound'])  # Overall sentiment (-1 to 1)
```
## output:
![image](https://github.com/user-attachments/assets/ecbdce9a-552c-4957-b46e-eef7a1509d5b)
## Part B: Time Series Forecasting with ARIMA
## Step 1: Setup and Installation
Install necessary libraries:
```
pip install statsmodels pandas matplotlib
```
## Step 2: Generate Sample Time Series Data
## code:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(0)
date_rng = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
data = np.random.normal(100, 10, size=(len(date_rng)))

# Create a pandas DataFrame
df = pd.DataFrame(data, index=date_rng, columns=["Value"])

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df, label="Sales Data")
plt.title("Synthetic Sales Data (Time Series)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
```
## Step 3: ARIMA Model for Forecasting
## Code:
```
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Fit ARIMA model (p=5, d=1, q=0)
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=len(test))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df.index[:train_size], train, label="Training Data")
plt.plot(df.index[train_size:], test, label="Test Data", color='orange')
plt.plot(df.index[train_size:], forecast, label="Forecast", color='red')
plt.title("ARIMA Forecasting")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
```
## Step 4: Evaluate Forecast Accuracy
## code:
```
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate evaluation metrics
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)

# Output evaluation metrics
print("Forecast Accuracy:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
```
## output:
![image](https://github.com/user-attachments/assets/437afb17-e2f4-4f75-b719-1055926fb9f4)

## Result Interpretation
1.Sentiment Analysis: TextBlob and NLTK effectively classify text sentiment and compute scores for emotional tone.
2.ARIMA Forecasting: The model successfully forecasts time series data with acceptable accuracy, demonstrating its utility for trend prediction.





