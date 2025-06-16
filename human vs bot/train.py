import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Fill missing values
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# Encode binary categorical columns
binary_cols = ['profile pic', 'private', 'name==username']
for col in binary_cols:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

# Drop columns not useful for modeling
train_df.drop('external URL', axis=1, inplace=True)
test_df.drop('external URL', axis=1, inplace=True)

# Define features and target
X = train_df.drop('fake', axis=1)
y = train_df['fake']

# Split training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# Models
# ===========================

# 1. XGBoost
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)

# 3. K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_val)

# ===========================
# Evaluation Function
# ===========================
def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# ===========================
# Model Results
# ===========================
evaluate_model("XGBoost", y_val, xgb_pred)
evaluate_model("Random Forest", y_val, rf_pred)
evaluate_model("K-Nearest Neighbors", y_val, knn_pred)
