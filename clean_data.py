
import pandas as pd
from pathlib import Path

# Project dataset location
data_folder = Path("dataset")

# Load the original files
train = pd.read_csv(data_folder / "train.csv")
test = pd.read_csv(data_folder / "test.csv")

# Standardize text columns
for df in [train, test]:
    df["waste_category"] = (
        df["waste_category"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "meat": "Meat",
            "dairy": "Dairy",
            "vegetables": "Vegetables",
            "grains": "Grains"
        })
    )

    df["staff_experience"] = (
        df["staff_experience"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "beginner": "Beginner",
            "intermediate": "Intermediate",
            "advanced": "Advanced"
        })
        .fillna("Unknown")
    )

    # Convert dates into a consistent format
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Create useful calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

# Save cleaned copies without changing the originals
train.to_csv(data_folder / "train_cleaned.csv", index=False)
test.to_csv(data_folder / "test_cleaned.csv", index=False)

# Display cleaning results
print("Data cleaning completed.")
print("\nTraining data shape:", train.shape)
print("Test data shape:", test.shape)

print("\nMissing values in training data:")
print(train.isnull().sum())

print("\nStandardized waste categories:")
print(train["waste_category"].value_counts(dropna=False))

print("\nStaff experience:")
print(train["staff_experience"].value_counts(dropna=False))

print("\nDate range:")
print(train["date"].min(), "to", train["date"].max())

print("\nCleaned files saved successfully.")
