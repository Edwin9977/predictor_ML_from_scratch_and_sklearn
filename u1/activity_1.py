import pandas as pd
separator = ';'
df = pd.read_csv("bank+marketing/bank/bank-full.csv", sep= separator)
print(df.head(1))
print(df.describe())