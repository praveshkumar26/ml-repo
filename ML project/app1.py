import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Bank Customer Model Lab", layout="wide")
st.title("Bank Customer Model Lab")
st.write("Train three models on bank customer data, compare them and try a live prediction.")

# session_state keeps our results when streamlit reruns the page
if "trained" not in st.session_state:
    st.session_state.trained = None


# ---------- 1. Load the data ----------
st.sidebar.header("1. Data")

csv_path = "miniproject/data_set/Customer-Churn-Records.csv"
if not os.path.exists(csv_path):
    csv_path = "Customer-Churn-Records.csv"

use_bundled = st.sidebar.checkbox("Use bundled dataset", value=True) if os.path.exists(csv_path) else False
uploaded = st.sidebar.file_uploader("Or upload a CSV", type=["csv"])
st.sidebar.caption("Custom upload is not optimised for every CSV yet.")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_bundled:
    df = pd.read_csv(csv_path)
else:
    st.info("Pick the bundled dataset or upload a CSV from the sidebar.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(), use_container_width=True)
st.caption(f"{df.shape[0]} rows and {df.shape[1]} columns")


# ---------- 2. Quick plots ----------
numeric_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in ["RowNumber", "CustomerId"]]

with st.expander("Dataset plots", expanded=True):
    tab1, tab2, tab3 = st.tabs(["Correlation", "Scatter", "Distribution"])

    with tab1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        img = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1)
        fig.colorbar(img, ax=ax)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=40, ha="right")
        ax.set_yticklabels(numeric_cols)
        # write the number inside every box
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

    with tab2:
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("X axis", numeric_cols)
        y_col = col2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
        sample = df.sample(min(1000, len(df)), random_state=1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(sample[x_col], sample[y_col], alpha=0.5, s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

    with tab3:
        hist_col = st.selectbox("Feature", numeric_cols)
        bins = st.slider("Bins", 10, 60, 25)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[hist_col].dropna(), bins=bins, color="#3b82f6", edgecolor="white")
        ax.set_xlabel(hist_col)
        ax.set_ylabel("Count")
        st.pyplot(fig)


# ---------- 3. Sidebar settings ----------
columns = list(df.columns)
default_target = columns.index("NumOfProducts") if "NumOfProducts" in columns else len(columns) - 1
target = st.sidebar.selectbox("Target column (y)", columns, index=default_target)

st.sidebar.header("2. Models")
use_lr = st.sidebar.checkbox("Logistic Regression", value=True)
use_dt = st.sidebar.checkbox("Decision Tree", value=True)
use_rf = st.sidebar.checkbox("Random Forest", value=True)

seed = st.sidebar.number_input("Random seed", min_value=1, max_value=9999, value=42)
train_clicked = st.sidebar.button("Train and compare", type="primary")


# ---------- 4. Train the models ----------
if train_clicked:
    models = {}
    if use_lr:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=seed)
    if use_dt:
        models["Decision Tree"] = DecisionTreeClassifier(max_depth=5, random_state=seed)
    if use_rf:
        models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=seed)

    if not models:
        st.warning("Please pick at least one model.")
    else:
        # remove id columns and turn text columns into numbers
        data = df.drop(columns=[c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns])
        X = pd.get_dummies(data.drop(columns=[target]), drop_first=True)
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)

        # scale only the big numeric columns
        scale_cols = [c for c in ["CreditScore", "Age", "Tenure", "Balance"] if c in X.columns]
        scaler = StandardScaler()
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])

        scores = []
        predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            scores.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            })

        # save everything in one place
        st.session_state.trained = {
            "models": models,
            "scaler": scaler,
            "columns": list(X.columns),
            "scale_cols": scale_cols,
            "y_test": y_test,
            "predictions": predictions,
            "scores": pd.DataFrame(scores).sort_values("F1", ascending=False).reset_index(drop=True),
            "target": target,
        }


# ---------- 5. Show results ----------
saved = st.session_state.trained

if saved is not None:
    st.markdown("---")
    st.subheader("Model comparison")
    st.dataframe(saved["scores"].style.format(precision=4), use_container_width=True)

    # bar chart of all metrics
    fig, ax = plt.subplots(figsize=(9, 4))
    bar_width = 0.2
    positions = np.arange(len(saved["scores"]))
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(positions + i * bar_width, saved["scores"][metric], bar_width, label=metric)
    ax.set_xticks(positions + bar_width * 1.5)
    ax.set_xticklabels(saved["scores"]["Model"])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # ----- confusion matrix -----
    st.markdown("---")
    st.subheader("Confusion matrix")
    cm_name = st.selectbox("Model", list(saved["predictions"].keys()), key="cm_model")
    as_percent = st.checkbox("Show row percentages")

    y_test = saved["y_test"]
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, saved["predictions"][cm_name], labels=labels,
                          normalize="true" if as_percent else None)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format=".2f" if as_percent else "d")
    ax.set_title(cm_name)
    st.pyplot(fig)

    # ----- live prediction -----
    st.markdown("---")
    st.subheader("Predict for one customer")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        credit_score = col1.number_input("Credit score", 300, 850, 650)
        age = col1.slider("Age", 18, 100, 35)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        geography = col2.selectbox("Geography", ["France", "Germany", "Spain"])
        tenure = col2.slider("Tenure (years)", 0, 10, 5)
        balance = col3.number_input("Balance", 0.0, 300000.0, 50000.0, step=1000.0)
        model_name = col3.selectbox("Model", list(saved["models"].keys()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        # start with a row of zeros that has the same columns as the training data
        row = pd.DataFrame(0, index=[0], columns=saved["columns"])
        values = {"CreditScore": credit_score, "Age": age, "Tenure": tenure, "Balance": balance,
                  f"Gender_{gender}": 1, f"Geography_{geography}": 1}
        for name, value in values.items():
            if name in row.columns:
                row.at[0, name] = value

        row[saved["scale_cols"]] = saved["scaler"].transform(row[saved["scale_cols"]])

        model = saved["models"][model_name]
        st.success(f"Predicted {saved['target']}: {model.predict(row)[0]}  ({model_name})")

        probs = model.predict_proba(row)[0]
        st.table(pd.DataFrame({
            "Class": model.classes_,
            "Probability": [f"{p * 100:.1f}%" for p in probs],
        }))
