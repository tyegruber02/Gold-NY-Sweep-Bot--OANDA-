import pandas as pd
import yfinance as yf

print("Pandas version:", pd.__version__)
print("Fetching AAPL price data as a test...")

data = yf.download("AAPL", start="2024-01-01", end="2024-01-10", progress=False)

if data.empty:
    print("no data returned, check internet connection")
else:
    print("Success! Here is a sample:")
    print(data.head())
