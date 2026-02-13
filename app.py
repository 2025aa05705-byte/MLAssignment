import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Page Config
st.set_page_config(page_title="Spam Classification App", layout="wide")

st.title("📧 Spambase Email Classification")
st.markdown("Predict whether an email is Spam (1) or Not Spam (0) based on 57 word/character frequency features.")

# Sidebar
st.sidebar.header("User Input")

# 1. Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV (Test Data)", type=["csv"])

# 2. Model Selection
model_options = [
    "Logistic Regression", 
    "Decision Tree", 
    "KNN", 
    "Naive Bayes", 
    "Random Forest", 
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Select Classification Model", model_options)

# Helper: Load resources
@st.cache_resource
def load_resources():
    try:
        # Use absolute paths so Streamlit always finds the folder regardless of terminal location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'model')
        
        le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        models_loaded = {}
        model_files = {
            "Logistic Regression": "logistic_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "KNN": "knn.pkl",
            "Naive Bayes": "naive_bayes.pkl",
            "Random Forest": "random_forest.pkl",
            "XGBoost": "xgboost.pkl"
        }
        for name, filename in model_files.items():
            models_loaded[name] = joblib.load(os.path.join(model_dir, filename))
        return le, scaler, models_loaded
    except FileNotFoundError as e:
        st.error(f"Model files not found. Ensure the 'model' folder exists and contains the .pkl files. Detailed Error: {e}")
        return None, None, None

le, scaler, models = load_resources()

if uploaded_file is not None and models:
    # Read Data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    if 'CLASS' in df.columns:
        X_raw = df.drop('CLASS', axis=1)
        y_true = df['CLASS']
        try:
            y_true_encoded = le.transform(y_true)
            has_labels = True
        except Exception as e:
            st.warning("Could not encode labels. Ensure class names match training data.")
            has_labels = False
    else:
        X_raw = df
        has_labels = False

    # Scale Features
    try:
        X_processed = scaler.transform(X_raw)
    except Exception as e:
        st.error(f"Feature mismatch. Please ensure columns match the training dataset: {e}")
        st.stop()

    # Prediction
    model = models[selected_model_name]
    y_pred_encoded = model.predict(X_processed)
    y_pred_prob = model.predict_proba(X_processed)[:, 1]
    y_pred_label = le.inverse_transform(y_pred_encoded)

    # Map labels to text for better display
    label_mapping = {1: "Spam", 0: "Not Spam"}
    y_pred_text = [label_mapping.get(val, val) for val in y_pred_label]

    # Add predictions to dataframe
    results_df = df.copy()
    results_df['Predicted_Class'] = y_pred_text
    
    st.subheader(f"Results using {selected_model_name}")
    st.dataframe(results_df)

    # Metrics Display
    if has_labels:
        st.markdown("---")
        st.subheader("Model Performance Evaluation")
        
        # Calculate Metrics
        acc = accuracy_score(y_true_encoded, y_pred_encoded)
        auc = roc_auc_score(y_true_encoded, y_pred_prob)
        prec = precision_score(y_true_encoded, y_pred_encoded)
        rec = recall_score(y_true_encoded, y_pred_encoded)
        f1 = f1_score(y_true_encoded, y_pred_encoded)
        mcc = matthews_corrcoef(y_true_encoded, y_pred_encoded)

        # Display Metrics in a Grid
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Accuracy", f"{acc:.4f}")
        m_col2.metric("AUC Score", f"{auc:.4f}")
        m_col3.metric("Precision", f"{prec:.4f}")

        m_col4, m_col5, m_col6 = st.columns(3)
        m_col4.metric("Recall", f"{rec:.4f}")
        m_col5.metric("F1 Score", f"{f1:.4f}")
        m_col6.metric("MCC Score", f"{mcc:.4f}")
        
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_true_encoded, y_pred_encoded)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=["Not Spam", "Spam"], 
                        yticklabels=["Not Spam", "Spam"])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

        with col2:
            st.write("**Classification Report**")
            report = classification_report(y_true_encoded, y_pred_encoded, 
                                        target_names=["Not Spam", "Spam"], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Please upload a CSV file to proceed. Use 'sample_test_data.csv' generated by the training script.")