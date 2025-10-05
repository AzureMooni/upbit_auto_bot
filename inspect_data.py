import pandas as pd
import os

file_path = os.path.join(os.getcwd(), "cache", "BTC_KRW_1h.feather")
df = pd.read_feather(file_path)
print("DataFrame Head:")
print(df.head())
print("\nDataFrame Tail:")
print(df.tail())
print("\nDataFrame Info:")
df.info()
print("\nMissing values:")
print(df.isnull().sum())
