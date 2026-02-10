import streamlit as st
import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Bank Marketing Classification App",
    layout="wide"
)

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

preprocessor = load_pickle("model/preprocessor.pkl")

models = {
    "Logistic Regression": load_pickle("model/logistic_regression.pkl"),
    "Decision Tree": load_pickle("model/decision_tree.pkl"),
    "KNN": load_pickle("model/k-nearest_neighbors.pkl"),
    "Naive Bayes": load_pickle("model/naive_bayes.pkl"),
    "Random Forest": load_pickle("model/random_forest.pkl"),
    "XGBoost": load_pickle("model/xgboost.pkl"),
}

with st.sidebar:
    st.title("Bank Marketing Classification")
    st.write("Predict whether a client will subscribe to a term deposit.")

    use_default = st.checkbox("Use default test data (test_data.csv)")
    st.markdown("[Download Test Dataset](https://github.com/AplicanU/ml_assignment_2/blob/e022eb9dc6e8a66558d09baa5cb0c165f79d8923/test_data.csv)")

    uploaded_file = st.file_uploader(
        "Upload test dataset (CSV)",
        type=["csv"]
    )

    selected_model_name = st.selectbox(
        "Select a Model",
        list(models.keys())
    )

    selected_model = models[selected_model_name]

    st.markdown("---")
    st.write("**</> Developer:** Ashish Upadhyay")
    

    

if use_default or uploaded_file is not None:
    if use_default:
        df = pd.read_csv("test_data.csv")
        data_source = "default test data"
    else:
        df = pd.read_csv(uploaded_file)
        data_source = "uploaded file"

    st.subheader(f"Data Preview ({data_source})")
    st.dataframe(df.head())

    if "y" in df.columns:
        X = df.drop(columns=["y"])
        y_true = df["y"]

        # Preprocess
        X_processed = preprocessor.transform(X)

        # Predictions
        y_pred = selected_model.predict(X_processed)

        if hasattr(selected_model, "predict_proba"):
            y_prob = selected_model.predict_proba(X_processed)[:, 1]
        else:
            y_prob = None


        st.subheader(f"Evaluation Metrics for {selected_model_name} model")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
        col3.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")

        col1.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
        col2.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

        if y_prob is not None:
            col3.metric("AUC", f"{roc_auc_score(y_true, y_prob):.4f}")


        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {selected_model_name}")

        st.pyplot(fig)
    else:
        st.error("Target column 'y' not found in the data.")
        st.stop()
else:
    st.info("Please select to use default test data or upload a CSV file.")
