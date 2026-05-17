import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("📰 Online News Popularity Prediction using AdaBoost")

# Load dataset
df = pd.read_csv("OnlineNewsPopularity.csv")

# Fix column spaces issue (VERY IMPORTANT)
df.columns = df.columns.str.strip()

df = df.drop(columns=["url"])

# Create target column
df["popularity"] = (df["shares"] > 1400).astype(int)

# Features and target
X = df.drop(columns=["shares", "popularity"])
y = df["popularity"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Sidebar hyperparameters
st.sidebar.header("Hyperparameters")

n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100)
learning_rate = st.sidebar.slider("learning_rate", 0.1, 2.0, 1.0)

# Model
model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=learning_rate,
    algorithm="SAMME",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.success(f"Accuracy Score: {acc:.2f}")

# Feature Importance
st.subheader("Top Important Features")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df.head(15))
st.bar_chart(importance_df.set_index("Feature").head(10))

# Prediction section
st.subheader("Predict News Popularity")

input_data = []

# Use limited features for UI (avoid overload)
selected_features = X.columns[:10]

for feature in selected_features:
    value = st.slider(
        feature,
        float(X[feature].min()),
        float(X[feature].max()),
        float(X[feature].mean())
    )
    input_data.append(value)

# Predict button
if st.button("Predict Popularity"):

    input_df = pd.DataFrame([input_data], columns=selected_features)

    # Fill remaining columns with mean values
    for col in X.columns[10:]:
        input_df[col] = X[col].mean()

    # Ensure correct order
    input_df = input_df[X.columns]

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("🔥 This News Article is POPULAR")
    else:
        st.warning("😐 This News Article is NOT Popular")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())
