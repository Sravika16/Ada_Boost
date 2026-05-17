import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="AdaBoost Regression",
    layout="wide"
)

st.title("AdaBoost Regression on Asteroid Dataset")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Asteroid.csv")
    return data

try:
    # -------------------------------
    # Read Dataset
    # -------------------------------
    data = load_data()

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # -------------------------------
    # Missing Values
    # -------------------------------
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # -------------------------------
    # Encode Categorical Columns
    # -------------------------------
    encoder = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = encoder.fit_transform(
                data[col].astype(str)
            )

    # -------------------------------
    # Target Column
    # -------------------------------
    target_column = "H"

    st.subheader("Target Column")
    st.write(target_column)

    # -------------------------------
    # Features and Target
    # -------------------------------
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Convert to numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    y = pd.to_numeric(y, errors="coerce")
    y = y.fillna(0)

    # -------------------------------
    # Train Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    st.subheader("Train Test Split")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Training Shape")
        st.write(X_train.shape)

    with col2:
        st.write("Testing Shape")
        st.write(X_test.shape)

    # -------------------------------
    # Base Model
    # -------------------------------
    base_model = DecisionTreeRegressor(
        max_depth=4
    )

    # -------------------------------
    # AdaBoost Regressor
    # -------------------------------
    model = AdaBoostRegressor(
        estimator=base_model,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    # -------------------------------
    # Train Model
    # -------------------------------
    model.fit(X_train, y_train)

    # -------------------------------
    # Prediction
    # -------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------
    # Evaluation Metrics
    # -------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("MAE", f"{mae:.4f}")
        st.metric("MSE", f"{mse:.4f}")

    with col4:
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("R2 Score", f"{r2:.4f}")

    # -------------------------------
    # Actual vs Predicted Plot
    # -------------------------------
    st.subheader("Actual vs Predicted Values")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        y_train,
        model.predict(X_train),
        color="blue",
        label="Train Data"
    )

    ax.scatter(
        y_test,
        y_pred,
        color="red",
        label="Test Data"
    )

    ax.plot(
        [y.min(), y.max()],
        [y.min(), y.max()],
        "k--",
        lw=2
    )

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted Values")
    ax.legend()

    st.pyplot(fig)

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    st.subheader("Correlation Heatmap")

    # Convert all columns to numeric
    heatmap_data = data.copy()

    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(
            heatmap_data[col],
            errors="coerce"
        )

    # Fill NaN values
    heatmap_data = heatmap_data.fillna(0)

    # Heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        heatmap_data.corr(),
        cmap="coolwarm",
        ax=ax2
    )

    ax2.set_title("Correlation Heatmap")

    st.pyplot(fig2)

except FileNotFoundError:
    st.error(
        "Asteroid.csv file not found. "
        "Please place the dataset in the same folder as app.py"
    )

except Exception as e:
    st.error(f"Error: {e}")