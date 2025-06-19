import pandas as pd
import numpy as np
from pathlib import Path

# Step 1: Load the data (change path if needed)
data_file = Path(r"C:\Users\kdine\OneDrive\Documents\Desktop\Cognifyz Internship\Dataset .csv")
raw_df = pd.read_csv(data_file)
# Target column we want to predict
target_col = "Aggregate rating"
# These two look like they directly tell you the rating, so we'll drop them
leaky_cols = ["Rating color", "Rating text"]
X_all = raw_df.drop(columns=[target_col] + leaky_cols)
y_all = raw_df[target_col]


# Step 2: Split into train and test sets (80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)


# Step 3: Set up preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
# Separate columns into numeric and categorical (some bools might sneak in)
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()
# Probably safe to fill missing numbers with median
numeric_transform = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
# Fill missing categories with mode (most common value), then one-hot encode
categorical_transform = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])
# Combine preprocessing for all columns
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transform, numeric_features),
    ("cat", categorical_transform, categorical_features)
])


# Step 4: Train Gradient Boosting model
from sklearn.ensemble import GradientBoostingRegressor
# Using basic GradientBoosting — haven’t tuned params yet
gbr_model = GradientBoostingRegressor(random_state=42)
# Full pipeline: preprocess → model
full_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", gbr_model)
])
# Fit everything on training data
full_pipeline.fit(X_train, y_train)


# Step 5: Evaluate model on the test set
from sklearn.metrics import mean_squared_error, r2_score
predictions = full_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print("\nModel Performance on Test Set:")
print(f" - RMSE: {rmse:.3f}")
print(f" - R² Score: {r2:.3f}")


# Step 6: Interpret using permutation importance
from sklearn.inspection import permutation_importance
# Note: This will take a bit of time
print("\nRunning permutation importance... (could be slow)")
perm_results = permutation_importance(
    full_pipeline, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_root_mean_squared_error"
)
# Match importance to the raw column names
feature_importances = pd.Series(perm_results.importances_mean, index=X_test.columns)
top_features = feature_importances.sort_values(ascending=False).head(15)
print("\nTop 15 Most Influential Features:")
print(top_features)