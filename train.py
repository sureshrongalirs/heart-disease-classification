"""
Heart Disease Classification - Model Training Script
This script trains all 6 classification models and saves them along with their metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report)
import pickle
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load dataset
df = pd.read_csv('heart.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nTarget distribution:")
print(df['target'].value_counts())

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save test data for Streamlit app
test_data = pd.concat([pd.DataFrame(X_test_scaled, columns=X.columns), pd.DataFrame(y_test.values, columns=['target'])], axis=1)
test_data.to_csv('model/test_data.csv', index=False)

print("\n" + "="*80)
print("Training Models and Calculating Metrics")
print("="*80)

# Dictionary to store all models and their metrics
models = {}
metrics_results = {}

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr

# 2. Decision Tree
print("2. Training Decision Tree Classifier...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
models['Decision Tree'] = dt

# 3. K-Nearest Neighbors
print("3. Training K-Nearest Neighbors...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
models['KNN'] = knn

# 4. Naive Bayes (Gaussian)
print("4. Training Naive Bayes (Gaussian)...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
models['Naive Bayes'] = nb

# 5. Random Forest
print("5. Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf

# 6. Gradient Boosting (Alternative to XGBoost)
print("6. Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb

# Calculate metrics for all models
print("\n" + "="*80)
print("Calculating Metrics")
print("="*80)

for model_name, model in models.items():
    print(f"\n{model_name}:")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Store metrics
    metrics_results[model_name] = {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc
    }
    
    # Print metrics
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    
    # Save model
    model_filename = f"model/{model_name.replace(' ', '_').lower()}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved: {model_filename}")

# Save all metrics to CSV for easy access
metrics_df = pd.DataFrame(metrics_results).T
metrics_df.to_csv('model/metrics_comparison.csv')
print("\n" + "="*80)
print("Metrics Comparison Table:")
print("="*80)
print(metrics_df.to_string())

print("\n" + "="*80)
print("All models trained and saved successfully!")
print("="*80)