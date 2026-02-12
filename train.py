"""
Heart Disease Classification - Streamlit Web Application
Interactive app for demonstrating classification models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef,confusion_matrix, classification_report)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Disease Classification System")
st.markdown("""
This interactive application demonstrates multiple machine learning models for predicting heart disease.
Upload test data, select a model, and view detailed performance metrics and predictions.
""")

# Sidebar for model selection
st.sidebar.header("Configuration")

# Model selection
model_options = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'K-Nearest Neighbors (KNN)': 'model/knn.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Random Forest': 'model/random_forest.pkl',
    'Gradient Boosting': 'model/gradient_boosting.pkl'
}

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_options.keys())
)

# Load scaler
@st.cache_resource
def load_scaler():
    try:
        with open('model/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        st.error("Scaler file not found. Please run train_models.py first.")
        return None

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except:
        st.error(f"Model file not found: {model_path}")
        return None

# Load metrics comparison
@st.cache_data
def load_metrics():
    try:
        return pd.read_csv('model/metrics_comparison.csv', index_col=0)
    except:
        return None

scaler = load_scaler()
model = load_model(model_options[selected_model_name])

# File upload section
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data"
)

# Use default test data if no file uploaded
if uploaded_file is None:
    if os.path.exists('model/test_data.csv'):
        st.sidebar.info("Using default test dataset")
        test_data = pd.read_csv('model/test_data.csv')
        use_default = True
    else:
        st.warning("Please upload a test dataset or run train_models.py to generate default test data.")
        st.stop()
else:
    test_data = pd.read_csv(uploaded_file)
    use_default = False

# Display dataset info
st.header("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples", len(test_data))
with col2:
    st.metric("Number of Features", len(test_data.columns) - 1)
with col3:
    if 'target' in test_data.columns:
        st.metric("Positive Cases", int(test_data['target'].sum()))

# Show data sample
with st.expander("View Data Sample"):
    st.dataframe(test_data.head(10))

# Feature information
with st.expander("Feature Descriptions"):
    feature_info = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'restecg': 'Resting electrocardiographic results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes; 0 = no)',
        'oldpeak': 'ST depression induced by exercise',
        'slope': 'Slope of peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)',
        'target': 'Heart disease (1 = disease; 0 = no disease)'
    }
    st.table(pd.DataFrame(feature_info.items(), columns=['Feature', 'Description']))

# Make predictions if model and scaler are loaded
if model is not None and scaler is not None and 'target' in test_data.columns:
    
    # Separate features and target
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Display model performance
    st.header(f"🎯 {selected_model_name} Performance")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
    with col2:
        st.metric("AUC Score", f"{auc:.4f}")
        st.metric("Recall", f"{recall:.4f}")
    with col3:
        st.metric("F1 Score", f"{f1:.4f}")
        st.metric("MCC Score", f"{mcc:.4f}")
    
    # Confusion Matrix
    st.subheader("📈 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Disease', 'Predicted Disease'],
        y=['Actual No Disease', 'Actual Disease'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"))
    
    # Prediction distribution
    st.subheader("🔍 Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction distribution
        pred_counts = pd.Series(y_pred).value_counts()
        fig = px.pie(
            values=pred_counts.values,
            names=['No Disease', 'Disease'],
            title='Prediction Distribution',
            color_discrete_sequence=['#00cc96', '#ef553b']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Probability distribution
        fig = px.histogram(
            y_pred_proba,
            nbins=30,
            title='Prediction Probability Distribution',
            labels={'value': 'Probability of Disease'},
            color_discrete_sequence=['#636efa']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample predictions
    st.subheader("🔬 Sample Predictions")
    results_df = test_data.copy()
    results_df['Predicted'] = y_pred
    results_df['Probability'] = y_pred_proba
    results_df['Correct'] = (results_df['target'] == results_df['Predicted'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 10 Predictions:**")
        st.dataframe(results_df[['target', 'Predicted', 'Probability', 'Correct']].head(10))
    with col2:
        correct_preds = results_df['Correct'].sum()
        st.metric("Correct Predictions", f"{correct_preds} / {len(results_df)}")
        st.metric("Error Count", len(results_df) - correct_preds)

# Model comparison section
st.header("📊 All Models Comparison")
metrics_df = load_metrics()

if metrics_df is not None:
    # Rename index for display
    display_names = {
        'Logistic Regression': 'Logistic Regression',
        'Decision Tree': 'Decision Tree',
        'KNN': 'K-Nearest Neighbors',
        'Naive Bayes': 'Naive Bayes',
        'Random Forest': 'Random Forest',
        'Gradient Boosting': 'Gradient Boosting'
    }
    
    metrics_df.index = metrics_df.index.map(lambda x: display_names.get(x, x))
    
    # Display metrics table
    st.dataframe(metrics_df.style.format("{:.4f}").highlight_max(axis=0, color='lightgreen'))
    
    # Visualize comparison
    st.subheader("Visual Comparison of Models")
    
    # Create bar chart for all metrics
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(metrics_df.columns)
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
    for idx, (col, pos) in enumerate(zip(metrics_df.columns, positions)):
        fig.add_trace(
            go.Bar(
                x=metrics_df.index,
                y=metrics_df[col],
                name=col,
                showlegend=False
            ),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(height=600, title_text="Model Performance Metrics Comparison")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model observations
    st.subheader("Model Performance Observations")
    
    observations = {
        'Logistic Regression': 'Good baseline model with decent performance. Shows moderate accuracy (81%) with strong AUC (93%), indicating good class separation. High recall (91%) means it catches most positive cases.',
        'Decision Tree': 'Excellent performance with 98.5% accuracy. Perfect precision (100%) with very high recall (97%). Might be slightly overfitting on this dataset.',
        'K-Nearest Neighbors': 'Strong performance with 86% accuracy. Balanced precision (87%) and recall (86%) with high AUC (96%), making it a reliable choice for this problem.',
        'Naive Bayes': 'Solid performance with 83% accuracy. Good recall (88%) but slightly lower precision (81%). Fast and efficient for real-time predictions.',
        'Random Forest': 'Perfect performance (100%) across all metrics. Best performing model but shows signs of potential overfitting. Excellent for this dataset but may need validation on new data.',
        'Gradient Boosting': 'Outstanding performance with 97.6% accuracy. Near-perfect scores across all metrics. Excellent balance and strong generalization capability.'
    }
    
    for model_name, observation in observations.items():
        with st.expander(f"**{model_name}**"):
            st.write(observation)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Heart Disease Classification System | Built with Streamlit</p>
    <p>Dataset: UCI Heart Disease Dataset from Kaggle</p>
</div>
""", unsafe_allow_html=True)