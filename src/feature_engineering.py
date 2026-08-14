import pandas as pd

df = pd.read_csv("cars_clean.csv")

df['car_age'] = 2026 - df['year']

df.to_csv("cars_engineering.csv", index=False)