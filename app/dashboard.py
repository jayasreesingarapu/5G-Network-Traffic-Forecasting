import streamlit as st

# Page configuration
st.set_page_config(
    page_title="5G Traffic Prediction",
    page_icon="📡",
    layout="wide"
)

# Dashboard title
st.title("📡 5G Network Traffic Forecasting & Congestion Prediction")

st.write(
    "Machine Learning-Based Network Traffic "
    "Forecasting and Potential Congestion Analysis"
)

st.divider()

# Project overview
st.subheader("Project Overview")

st.info(
    "This dashboard forecasts internet activity using "
    "machine learning and identifies potential high-traffic "
    "conditions across geographical regions."
)

st.success("Dashboard initialized successfully!")
from pathlib import Path
import pandas as pd

# Locate the saved dataset
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "traffic_5grids.csv"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_PATH,
        parse_dates=["datetime"]
    )

traffic = load_data()

st.divider()
st.subheader("📊 Network Traffic Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{len(traffic):,}")
col2.metric("Geographical Grids", traffic["GridID"].nunique())
col3.metric("Time Intervals", traffic["datetime"].nunique())

st.write("### Sample Traffic Data")
st.dataframe(traffic.head(10), use_container_width=True)
# Interactive Internet Traffic Visualization

st.divider()
st.subheader("📈 Internet Traffic Analysis")

selected_grid = st.selectbox(
    "Select Geographical Grid",
    sorted(traffic["GridID"].unique())
)

grid_data = traffic[
    traffic["GridID"] == selected_grid
].sort_values("datetime")

st.line_chart(
    grid_data.set_index("datetime")["internet"]
)
import joblib
from pathlib import Path

# Path to trained Random Forest model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "random_forest.pkl"
)

# Load trained model
@st.cache_resource
def load_forecasting_model():
    return joblib.load(MODEL_PATH)

rf_model = load_forecasting_model()

st.divider()
st.subheader("🤖 AI-Based Traffic Forecasting")
# Step 57: Predict next 10-minute internet activity

recent = grid_data.sort_values("datetime").tail(144)

next_time = recent["datetime"].iloc[-1] + pd.Timedelta(minutes=10)
local_time = next_time.tz_convert("Europe/Rome")

input_data = pd.DataFrame([{
    "GridID": int(selected_grid),
    "hour": local_time.hour,
    "day_of_week": local_time.dayofweek,
    "is_weekend": int(local_time.dayofweek >= 5),
    "minute_slot": local_time.minute // 10,
    "lag_1": recent["internet"].iloc[-1],
    "lag_2": recent["internet"].iloc[-2],
    "lag_3": recent["internet"].iloc[-3],
    "lag_6": recent["internet"].iloc[-6],
    "lag_144": recent["internet"].iloc[-144],
    "rolling_mean_6": recent["internet"].tail(6).mean(),
    "rolling_std_6": recent["internet"].tail(6).std(),
    "rolling_mean_144": recent["internet"].mean(),
    "rolling_std_144": recent["internet"].std()
}])

predicted_traffic = rf_model.predict(input_data)[0]

st.metric(
    "Predicted Internet Activity (Next 10 Minutes)",
    f"{predicted_traffic:.2f}"
)

st.success("Random Forest forecasting model loaded successfully!")
# Step 58: Predict Traffic-Load Status

import joblib

# Load thresholds calculated during training
THRESHOLD_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "traffic_thresholds.pkl"
)

thresholds = joblib.load(THRESHOLD_PATH)

# Classify predicted traffic activity
if predicted_traffic <= thresholds["Normal"]:
    traffic_status = "Normal"

elif predicted_traffic <= thresholds["Moderate"]:
    traffic_status = "Moderate"

elif predicted_traffic <= thresholds["High"]:
    traffic_status = "High"

else:
    traffic_status = "Potential Congestion"

# Display prediction
st.subheader("🚦 Predicted Traffic Status")

if traffic_status == "Normal":
    st.success(f"Traffic Status: {traffic_status}")

elif traffic_status == "Moderate":
    st.info(f"Traffic Status: {traffic_status}")

elif traffic_status == "High":
    st.warning(f"Traffic Status: {traffic_status}")

else:
    st.error(f"Traffic Status: {traffic_status}")

st.caption(
    "Traffic status is based on historical activity thresholds, "
    "not measured network congestion."
)
# Step 59: Actual vs Predicted Traffic

st.divider()
st.subheader("📊 Actual vs Predicted Internet Activity")

# Load previously engineered test features
FEATURES_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "traffic_features.csv"
)

@st.cache_data
def load_features():
    return pd.read_csv(
        FEATURES_PATH,
        parse_dates=["datetime"]
    )

feature_data = load_features()

# Use the same final test period as model evaluation
timestamps = sorted(feature_data["datetime"].unique())
test_start = timestamps[int(len(timestamps) * 0.85)]

# Filter selected grid and test period
test_grid = feature_data[
    (feature_data["GridID"] == selected_grid) &
    (feature_data["datetime"] >= test_start)
].sort_values("datetime").copy()

features = [
    "GridID", "hour", "day_of_week", "is_weekend",
    "minute_slot", "lag_1", "lag_2", "lag_3",
    "lag_6", "lag_144", "rolling_mean_6",
    "rolling_std_6", "rolling_mean_144",
    "rolling_std_144"
]

test_grid["Predicted"] = rf_model.predict(
    test_grid[features]
)

comparison = test_grid.set_index("datetime")[
    ["internet", "Predicted"]
].rename(columns={"internet": "Actual"})

st.line_chart(comparison)

st.caption(
    "Historical test-period predictions for the selected grid. "
    "These observations were not used to train the model."
)
# Step 60: Model Performance Comparison

st.divider()
st.subheader("📊 Forecasting Model Performance Comparison")

# Final test results
model_results = pd.DataFrame({
    "Model": [
        "Persistence",
        "Historical Average",
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "LSTM",
        "GRU"
    ],
    "MAE": [
        2.7589, 2.8919, 2.4840, 2.4501,
        2.4744, 2.5196, 2.5704
    ],
    "RMSE": [
        4.9404, 4.9535, 4.3500, 4.3093,
        4.3368, 4.3840, 4.4622
    ],
    "R2": [
        0.7978, 0.7967, 0.8432, 0.8461,
        0.8442, 0.8408, 0.8350
    ]
})

# Display results table
st.dataframe(model_results, hide_index=True)

# Compare RMSE
st.write("### RMSE Comparison (Lower is Better)")

st.bar_chart(
    model_results.set_index("Model")["RMSE"]
)

# Highlight best model
best_model = model_results.loc[
    model_results["RMSE"].idxmin()
]

st.success(
    f"Best Forecasting Model: {best_model['Model']} "
    f"(RMSE: {best_model['RMSE']:.4f}, "
    f"R²: {best_model['R2']:.4f})"
)

st.caption(
    "Results are from the held-out test period "
    "using five selected geographical grids."
)
# Step 61: Traffic-Load Classification Performance

st.divider()
st.subheader("🚦 Traffic-Load Classification Performance")

# Load saved classification results
CLASSIFICATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "classification_test_results.csv"
)

classification_results = pd.read_csv(
    CLASSIFICATION_PATH,
    index_col=0
)

# Overall classification accuracy
accuracy = classification_results.loc["accuracy", "f1-score"]

st.metric(
    "Classification Test Accuracy",
    f"{accuracy * 100:.2f}%"
)

# Display performance for each traffic category
categories = [
    "Normal",
    "Moderate",
    "High",
    "Potential Congestion"
]

category_results = classification_results.loc[
    categories,
    ["precision", "recall", "f1-score"]
]

st.write("### Precision, Recall and F1-Score")

st.dataframe(category_results.round(4))

# Compare classification performance
st.bar_chart(category_results)

st.caption(
    "The categories represent relative internet activity levels. "
    "Potential Congestion is a high-load proxy, not confirmed "
    "5G network congestion."
)
# Step 64: Geographical Grid Heatmap

import numpy as np
import matplotlib.pyplot as plt

st.divider()
st.subheader("🗺️ Geographical Internet Activity Heatmap")

# Average activity for each available grid
grid_averages = traffic.groupby("GridID")["internet"].mean()

# Create 100 x 100 grid
heatmap = np.full((100, 100), np.nan)

for grid_id, activity in grid_averages.items():
    # Grid IDs range from 1 to 10000
    index = int(grid_id) - 1
    row = index // 100
    col = index % 100

    heatmap[row, col] = activity

# Display heatmap
fig, ax = plt.subplots(figsize=(8, 7))

image = ax.imshow(
    heatmap,
    cmap="YlOrRd",
    origin="lower",
    interpolation="nearest"
)

fig.colorbar(image, ax=ax, label="Average Internet Activity")

ax.set_title("Internet Activity Across Selected Milan Grid Cells")
ax.set_xlabel("Grid Column")
ax.set_ylabel("Grid Row")

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Only five sampled geographical cells contain data. "
    "Blank cells were not analyzed. Grid positioning assumes "
    "sequential row-major grid numbering."
)