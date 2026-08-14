import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preprocessing import (
    split_features_and_target,
    build_preprocessor
)
from sklearn.model_selection import train_test_split

df = pd.read_csv("cars_engineering.csv")

X, y = split_features_and_target(df)

print("Creating the same train/test split...")
 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Loading trained model...")
 
model = joblib.load("car_price_model.joblib")

print("Making predictions...")
 
y_pred = model.predict(X_test)

print("Calculating regression metrics...")
 
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],
})
 
print("\nRegression metrics:")
print(metrics)