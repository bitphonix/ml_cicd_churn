import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os

# Load data
df = pd.read_csv("data/customer_churn.csv")

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Define X and y
X = df.drop("Churn", axis=1)
y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Categorical and numerical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

# Final pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/churn_model.pkl")