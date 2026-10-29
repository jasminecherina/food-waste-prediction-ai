
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the cleaned training data
data_path = Path("dataset/train_cleaned.csv")
df = pd.read_csv(data_path)

# 2. Standardize category labels
df["staff_experience"] = (
    df["staff_experience"].astype("string").str.strip().str.title()
)
df["waste_category"] = (
    df["waste_category"].astype("string").str.strip().str.title()
)

# 3. Select input features and prediction target
numeric_features = [
    "meals_served",
    "kitchen_staff",
    "temperature_C",
    "humidity_percent",
    "day_of_week",
    "special_event",
    "past_waste_kg",
    "year",
    "month",
    "day",
]

categorical_features = [
    "staff_experience",
    "waste_category",
]

features = numeric_features + categorical_features
target = "food_waste_kg"

# 4. Sort records chronologically
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Remove records with an unavailable target, if any
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# 5. Use the earlier 80% of records for training
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_valid = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_valid = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Validation records:", len(X_valid))

# 6. Prepare numerical and categorical data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 7. Baseline model
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_valid)

# 8. Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_valid)

# 9. Evaluate both models
def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print(f"MAE:  {mae:.3f} kg")
    print(f"RMSE: {rmse:.3f} kg")
    print(f"R2:   {r2:.3f}")

evaluate("Baseline Model", y_valid, baseline_predictions)
evaluate("Random Forest", y_valid, predictions)

# 10. Save the trained model
Path("models").mkdir(exist_ok=True)
dump(model, "models/food_waste_model.joblib")

print("\nModel saved to models/food_waste_model.joblib")
print("Practice model training completed.")
