import joblib
import pandas as pd
 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
 
from data_preprocessing import (
    split_features_and_target,
    build_preprocessor
)

df = pd.read_csv("cars_engineering.csv")

print("Splitting features and target...")
 
X, y = split_features_and_target(df)

print("Splitting data into training and test sets...")
 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Creating model pipeline...")
 
model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor()),
    ]
)

print("Training model...")
 
model.fit(X_train, y_train)

print("Saving model...")
 
joblib.dump(model, "car_price_model.joblib")
 
print("Model saved to: ")

df = pd.read_csv("cars_engineering.csv")

X, y = split_features_and_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

loaded_model = joblib.load("car_price_model.joblib")

sample_X = X_test.sample(10, random_state=42)
sample_y = y_test.loc[sample_X.index]

sample_predictions = loaded_model.predict(sample_X)

prediction_preview = pd.DataFrame({
    "actual_price_USD": sample_y.values,
    "predicted_price_USD": sample_predictions,
})
 
print(prediction_preview)

print("Based on the mae metric, the best model for this given data is Random Forest model.")