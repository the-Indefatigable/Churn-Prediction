import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# ------------------------------
# Load data
# ------------------------------
data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Total Cols -> ", data.shape[1])
print("Total Rows -> ", data.shape[0])
print("Total Missing Data -> ", data.isna().sum().sum())

# ------------------------------
# Clean data
# ------------------------------
# Convert TotalCharges to numeric, coerce errors to NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])

# ------------------------------
# Define features and target
# ------------------------------
target = 'Churn'
X = data.drop(columns=[target, 'customerID'])
y = data[target].apply(lambda x: 1 if x=='Yes' else 0)

# ------------------------------
# Column types
# ------------------------------
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [col for col in X.columns if col not in num_cols]

# ------------------------------
# Preprocessing pipelines
# ------------------------------
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# ------------------------------
# XGBClassifier Pipeline with scale_pos_weight
# ------------------------------
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    ))
])

# ------------------------------
# Train/Test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Fit XGBClassifier with GridSearch
# ------------------------------
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 4],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

print("Training model with GridSearch...")
grid_search = GridSearchCV(
    xgb_pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# ------------------------------
# Evaluate
# ------------------------------
y_pred_xgb = grid_search.predict(X_test)

print("\n=== MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# ------------------------------
# Save model
# ------------------------------
print("\nSaving model...")
joblib.dump(grid_search.best_estimator_, 'churn_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("✓ Model saved as 'churn_model.pkl'")
print("✓ Feature columns saved as 'feature_columns.pkl'")
