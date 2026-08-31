import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

from datetime import timedelta


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide"
)


# ==========================================================
# CONSTANTS
# ==========================================================

FORECAST_HOURS = 72
MODEL_VERSION = 2

MODEL_FEATURES = [
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3",
    "hour",
    "day",
    "month",
    "day_of_week",
    "aqi",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_72",
    "pm2_5_lag_1",
    "pm10_lag_1",
    "aqi_rolling_6",
    "aqi_rolling_24",
    "aqi_rolling_72",
    "aqi_change"
]


# ==========================================================
# EPA AQI CALCULATION
# ==========================================================

def calculate_pm25_aqi(pm25):

    if pd.isna(pm25):
        return None

    pm25 = float(pm25)

    if pm25 < 0:
        return None

    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ]

    if pm25 > 325.4:
        pm25 = 325.4

    for c_low, c_high, i_low, i_high in breakpoints:

        if c_low <= pm25 <= c_high:

            aqi = (
                (i_high - i_low)
                / (c_high - c_low)
            ) * (pm25 - c_low) + i_low

            return round(aqi)

    return None


# ==========================================================
# AQI CATEGORY
# ==========================================================

def aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


# ==========================================================
# TITLE
# ==========================================================

st.title("🌍 Pearls AQI Predictor")

st.subheader("3-Day Air Quality Index Forecast")

st.write(
    "AI-powered AQI forecasting using Machine Learning "
    "with automated feature engineering."
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    return joblib.load("aqi_model.pkl")


try:

    model = load_model()

    st.success(
        "✅ Random Forest model loaded successfully"
    )

except Exception as e:

    st.error(
        f"Unable to load model: {e}"
    )

    st.stop()


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("features.csv")

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    df = (
        df
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df


try:

    df = load_data()

except Exception as e:

    st.error(
        f"Unable to load features.csv: {e}"
    )

    st.stop()


# ==========================================================
# TIME FEATURES
# ==========================================================

df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["day_of_week"] = df["datetime"].dt.dayofweek


# ==========================================================
# CURRENT DATA
# ==========================================================

latest = df.iloc[-1]

latest_time = latest["datetime"]

current_pm25 = latest["pm2_5"]

# IMPORTANT:
# Do NOT use OpenWeather 1-5 AQI as the displayed AQI.
# Calculate actual EPA-style AQI from PM2.5.

current_aqi = calculate_pm25_aqi(
    current_pm25
)


# ==========================================================
# CURRENT AIR QUALITY
# ==========================================================

st.divider()

st.header("📍 Current Air Quality")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Current EPA AQI",
        f"{current_aqi:.0f}"
    )

with col2:

    st.metric(
        "AQI Category",
        aqi_category(current_aqi)
    )

with col3:

    st.metric(
        "PM2.5",
        f"{current_pm25:.2f}"
    )

with col4:

    st.metric(
        "Latest Data",
        latest_time.strftime(
            "%Y-%m-%d %H:%M"
        )
    )


# ==========================================================
# 72-HOUR FORECAST
# ==========================================================

st.divider()

st.header("🔮 Next 3 Days AQI Forecast")

st.write(
    "The Random Forest model generates hourly AQI predictions "
    "for the next 72 hours."
)


predictions = []

latest_row = latest.copy()


for i in range(1, FORECAST_HOURS + 1):

    future_time = (
        latest_time
        + timedelta(hours=i)
    )

    future_row = latest_row.copy()

    future_row["datetime"] = future_time

    future_row["hour"] = future_time.hour
    future_row["day"] = future_time.day
    future_row["month"] = future_time.month
    future_row["day_of_week"] = (
        future_time.dayofweek
    )

    # Ensure all required features exist
    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in future_row.index
    ]

    if missing_features:

        st.error(
            "Missing model features: "
            + ", ".join(missing_features)
        )

        st.stop()

    X_future = pd.DataFrame(
        [future_row]
    )[MODEL_FEATURES]

    predicted_aqi = model.predict(
        X_future
    )[0]

    predicted_aqi = max(
        0,
        min(
            500,
            predicted_aqi
        )
    )

    predictions.append({

        "datetime": future_time,

        "predicted_aqi": round(
            predicted_aqi,
            1
        )

    })


forecast_df = pd.DataFrame(
    predictions
)


# ==========================================================
# FORECAST CHART
# ==========================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    forecast_df["datetime"],
    forecast_df["predicted_aqi"],
    linewidth=2
)

ax.set_title(
    "72-Hour Predicted EPA AQI"
)

ax.set_xlabel(
    "Date and Time"
)

ax.set_ylabel(
    "Predicted AQI"
)

ax.grid(
    True,
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig)
# ==========================================================
# KEY FORECAST OUTPUTS — 24 / 48 / 72 HOURS
# ==========================================================

st.subheader("📌 Key Forecast Predictions")

# Get the predictions corresponding to +24h, +48h and +72h
day1_aqi = forecast_df.iloc[23]["predicted_aqi"]
day2_aqi = forecast_df.iloc[47]["predicted_aqi"]
day3_aqi = forecast_df.iloc[71]["predicted_aqi"]

day1_time = forecast_df.iloc[23]["datetime"]
day2_time = forecast_df.iloc[47]["datetime"]
day3_time = forecast_df.iloc[71]["datetime"]

forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

with forecast_col1:

    st.metric(
        "Day +1 (24 Hours)",
        f"{day1_aqi:.1f} AQI"
    )

    st.caption(
        day1_time.strftime("%Y-%m-%d %H:%M")
    )

with forecast_col2:

    st.metric(
        "Day +2 (48 Hours)",
        f"{day2_aqi:.1f} AQI"
    )

    st.caption(
        day2_time.strftime("%Y-%m-%d %H:%M")
    )

with forecast_col3:

    st.metric(
        "Day +3 (72 Hours)",
        f"{day3_aqi:.1f} AQI"
    )

    st.caption(
        day3_time.strftime("%Y-%m-%d %H:%M")
    )

# ==========================================================
# DAILY FORECAST
# ==========================================================

st.subheader(
    "📅 Daily Forecast Summary"
)

forecast_df["date"] = (
    forecast_df["datetime"]
    .dt.date
)

daily_forecast = (
    forecast_df
    .groupby("date")
    ["predicted_aqi"]
    .agg(
        [
            ("Minimum AQI", "min"),
            ("Average AQI", "mean"),
            ("Maximum AQI", "max")
        ]
    )
    .reset_index()
)

daily_forecast["Average AQI"] = (
    daily_forecast["Average AQI"].round(1)
)

daily_forecast["Minimum AQI"] = (
    daily_forecast["Minimum AQI"].round(1)
)

daily_forecast["Maximum AQI"] = (
    daily_forecast["Maximum AQI"].round(1)
)

st.dataframe(
    daily_forecast,
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# FORECAST ALERT
# ==========================================================

forecast_max = forecast_df[
    "predicted_aqi"
].max()


if forecast_max > 300:

    st.error(
        f"🚨 Hazardous AQI levels may occur. "
        f"Maximum predicted AQI: {forecast_max:.1f}"
    )

elif forecast_max > 200:

    st.warning(
        f"⚠️ Very unhealthy AQI levels may occur. "
        f"Maximum predicted AQI: {forecast_max:.1f}"
    )

elif forecast_max > 150:

    st.warning(
        f"⚠️ Unhealthy AQI levels may occur. "
        f"Maximum predicted AQI: {forecast_max:.1f}"
    )

elif forecast_max > 100:

    st.warning(
        f"⚠️ Unhealthy for sensitive groups AQI levels "
        f"may occur. Maximum predicted AQI: {forecast_max:.1f}"
    )

else:

    st.success(
        f"✅ Air quality is expected to remain relatively good. "
        f"Maximum predicted AQI: {forecast_max:.1f}"
    )


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

st.divider()

st.header("📊 Model Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "MAE",
        "3.72 AQI points"
    )

with metric2:

    st.metric(
        "RMSE",
        "6.58 AQI points"
    )

with metric3:

    st.metric(
        "R²",
        "0.9483"
    )

st.caption(
    "Evaluation performed using a chronological 80/20 "
    "train-test split on actual EPA-style AQI values."
)

# ==========================================================
# MODEL EXPLAINABILITY — SHAP
# ==========================================================

st.divider()

st.header("🤖 Model Explainability — SHAP")

st.write(
    "SHAP (SHapley Additive exPlanations) shows how each input "
    "feature contributes to the model's AQI prediction. "
    "Larger absolute SHAP values indicate a stronger influence "
    "on the prediction."
)


# ----------------------------------------------------------
# PREPARE SHAP DATA
# ----------------------------------------------------------

try:

    # Use only the model input features
    X_shap = df[MODEL_FEATURES].dropna().copy()

    # Keep the computation lightweight while preserving
    # a representative sample of the available data.
    shap_sample_size = min(300, len(X_shap))

    X_shap_sample = (
        X_shap
        .sample(
            n=shap_sample_size,
            random_state=42
        )
        .reset_index(drop=True)
    )


    # ------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # ------------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_shap_sample
    )


    # ------------------------------------------------------
    # HANDLE SHAP OUTPUT
    # ------------------------------------------------------

    if isinstance(shap_values, list):

        shap_values_plot = shap_values[0]

    else:

        shap_values_plot = shap_values


    # ------------------------------------------------------
    # GLOBAL SHAP IMPORTANCE
    # ------------------------------------------------------

    mean_abs_shap = (
        abs(shap_values_plot)
        .mean(axis=0)
    )


    shap_importance = pd.DataFrame({

        "Feature": MODEL_FEATURES,

        "Mean |SHAP Value|": mean_abs_shap

    })


    shap_importance = (
        shap_importance
        .sort_values(
            "Mean |SHAP Value|",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # ------------------------------------------------------
    # DISPLAY TOP FEATURES
    # ------------------------------------------------------

    st.subheader(
        "📊 Global Feature Impact"
    )

    st.caption(
        "Average absolute SHAP value across a representative "
        f"sample of {shap_sample_size} observations."
    )


    st.dataframe(
        shap_importance,
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------------------------------
    # SHAP BAR CHART
    # ------------------------------------------------------

    fig_shap, ax_shap = plt.subplots(
        figsize=(10, 8)
    )


    top_shap = (
        shap_importance
        .head(15)
        .sort_values(
            "Mean |SHAP Value|",
            ascending=True
        )
    )


    ax_shap.barh(
        top_shap["Feature"],
        top_shap["Mean |SHAP Value|"]
    )


    ax_shap.set_title(
        "Top 15 Features by Mean Absolute SHAP Value"
    )

    ax_shap.set_xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.tight_layout()

    st.pyplot(
        fig_shap
    )


    # ------------------------------------------------------
    # LOCAL EXPLANATION
    # ------------------------------------------------------

    st.subheader(
        "🔍 Individual Prediction Explanation"
    )

    st.write(
        "The chart below explains one representative AQI "
        "prediction and shows which features pushed the "
        "prediction higher or lower."
    )


    explanation_index = 0

    explanation_features = (
        X_shap_sample
        .iloc[
            explanation_index
        ]
    )


    explanation_values = (
        shap_values_plot[
            explanation_index
        ]
    )


    local_explanation = pd.DataFrame({

        "Feature": MODEL_FEATURES,

        "SHAP Value": explanation_values,

        "Feature Value": (
            explanation_features.values
        )

    })


    local_explanation[
        "Absolute SHAP"
    ] = abs(
        local_explanation[
            "SHAP Value"
        ]
    )


    local_explanation = (
        local_explanation
        .sort_values(
            "Absolute SHAP",
            ascending=False
        )
        .head(10)
        .sort_values(
            "SHAP Value"
        )
        .reset_index(
            drop=True
        )
    )


    fig_local, ax_local = plt.subplots(
        figsize=(10, 6)
    )


    ax_local.barh(
        local_explanation["Feature"],
        local_explanation["SHAP Value"]
    )


    ax_local.axvline(
        0,
        linewidth=1
    )


    ax_local.set_title(
        "SHAP Explanation for One AQI Prediction"
    )

    ax_local.set_xlabel(
        "SHAP Value"
    )

    ax_local.set_ylabel(
        "Feature"
    )

    plt.tight_layout()

    st.pyplot(
        fig_local
    )


    # ------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------

    strongest_feature = (
        shap_importance.iloc[0]["Feature"]
    )

    strongest_value = (
        shap_importance.iloc[0]["Mean |SHAP Value|"]
    )


    st.info(
        f"💡 The feature with the strongest overall "
        f"SHAP influence in the sampled data is "
        f"**{strongest_feature}**, with a mean absolute "
        f"SHAP value of **{strongest_value:.4f}**."
    )


except Exception as e:

    st.warning(
        "SHAP explainability could not be generated."
    )

    st.code(
        str(e)
    )

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.divider()

st.header("🧠 Model Information")

info1, info2, info3, info4 = st.columns(4)

with info1:

    st.metric(
        "Model",
        "Random Forest"
    )

with info2:

    st.metric(
        "Input Features",
        len(MODEL_FEATURES)
    )

with info3:

    st.metric(
        "Forecast Horizon",
        "72 Hours"
    )

with info4:

    st.metric(
        "Model Version",
        MODEL_VERSION
    )


# ==========================================================
# DATA INFORMATION
# ==========================================================

st.divider()

st.header("📊 Data Information")

data1, data2, data3 = st.columns(3)

with data1:

    st.metric(
        "Available Records",
        len(df)
    )

with data2:

    st.metric(
        "Feature Columns",
        len(df.columns)
    )

with data3:

    st.metric(
        "Latest Data Time",
        latest_time.strftime(
            "%Y-%m-%d %H:%M"
        )
    )


# ==========================================================
# ABOUT
# ==========================================================

st.divider()

st.header("ℹ️ About the Project")

st.write(
    """
**Pearls AQI Predictor** is an end-to-end machine learning
system designed to forecast Air Quality Index values for
the next three days.

### Technologies

- Python
- Pandas
- Scikit-learn
- Random Forest
- Hopsworks Feature Store
- GitHub Actions
- Streamlit

### Key Features

- Automated feature engineering
- Chronological model evaluation
- EPA-style PM2.5 AQI target
- 72-hour AQI forecasting
- Recursive future prediction
- Model comparison
- Feature importance analysis
- AQI health alerts
- Interactive Streamlit dashboard
"""
)

st.success(
    "🎉 Pearls AQI Predictor Dashboard Loaded Successfully!"
)