
import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import os
import joblib
import matplotlib.pyplot as plt

from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🌍 Pearls AQI Predictor")

st.subheader("3-Day Air Quality Index Forecast")

st.write(
    "AI-powered AQI forecasting using Machine Learning "
    "and Hopsworks Feature Store."
)


# =========================================================
# CONSTANTS
# =========================================================

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

MODEL_NAME = "aqi_random_forest"
MODEL_VERSION = 2

FORECAST_HOURS = 72


# =========================================================
# MODEL FEATURES
# EXACTLY MATCHES THE TRAINING PIPELINE
# =========================================================

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


# =========================================================
# CONNECT TO HOPSWORKS
# =========================================================

@st.cache_resource
def connect_hopsworks():

    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY was not found. "
            "Please check your .env file."
        )

    project = hopsworks.login(
        api_key_value=api_key
    )

    feature_store = project.get_feature_store()

    return project, feature_store


try:

    project, fs = connect_hopsworks()

    st.success(
        "✅ Connected to Hopsworks Feature Store"
    )

except Exception as e:

    st.error(
        "❌ Could not connect to Hopsworks"
    )

    st.code(str(e))

    st.stop()


# =========================================================
# LOAD FEATURE DATA
# =========================================================

@st.cache_data(ttl=300)
def load_data(_fs):

    fg = _fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    data = fg.select_all().read()

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    data = data.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return data


try:

    df = load_data(fs)

except Exception as e:

    st.error(
        "❌ Could not load AQI data from Hopsworks."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# VALIDATE FEATURE DATA
# =========================================================

missing_features = [
    feature
    for feature in MODEL_FEATURES
    if feature not in df.columns
]

if missing_features:

    st.error(
        "❌ Required model features are missing "
        "from the Hopsworks Feature Group."
    )

    st.write("Missing features:")

    st.code(
        "\n".join(missing_features)
    )

    st.stop()


# =========================================================
# REMOVE INVALID ROWS
# =========================================================

df = df.dropna(
    subset=MODEL_FEATURES + ["datetime"]
).reset_index(
    drop=True
)


if df.empty:

    st.error(
        "❌ No valid AQI feature records are available."
    )

    st.stop()


# =========================================================
# CURRENT AQI
# =========================================================

latest = df.iloc[-1]

current_aqi = float(
    latest["aqi"]
)

current_pm25 = float(
    latest["pm2_5"]
)


def get_aqi_status(aqi):

    if aqi <= 3:

        return "Good 🟢"

    elif aqi <= 4:

        return "Moderate 🟡"

    else:

        return "Unhealthy 🔴"


current_status = get_aqi_status(
    current_aqi
)


# =========================================================
# DASHBOARD METRICS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Latest AQI",
        f"{current_aqi:.2f}"
    )


with col2:

    st.metric(
        "Latest PM2.5",
        f"{current_pm25:.2f}"
    )


with col3:

    st.metric(
        "Air Quality Status",
        current_status
    )


st.divider()


# =========================================================
# LOAD MODEL FROM HOPSWORKS MODEL REGISTRY
# =========================================================

st.header(
    "📈 72-Hour AQI Forecast"
)

st.caption(
    f"Using {MODEL_NAME} — Model Registry Version {MODEL_VERSION}"
)


@st.cache_resource
def load_model(_project):

    model_registry = (
        _project.get_model_registry()
    )

    model = model_registry.get_model(
        name=MODEL_NAME,
        version=MODEL_VERSION
    )

    model_dir = model.download()

    pkl_files = []

    for root, dirs, files in os.walk(
        model_dir
    ):

        for file in files:

            if file.endswith(".pkl"):

                pkl_files.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    if not pkl_files:

        raise FileNotFoundError(
            "No .pkl model file was found "
            "in the Hopsworks Model Registry."
        )

    model_file = pkl_files[0]

    trained_model = joblib.load(
        model_file
    )

    return trained_model


try:

    trained_model = load_model(project)

    st.success(
        "✅ Random Forest model loaded successfully "
        "from Hopsworks Model Registry"
    )

except Exception as e:

    st.error(
        "❌ Could not load Random Forest model "
        "from Model Registry"
    )

    st.code(str(e))

    st.stop()


# =========================================================
# VERIFY MODEL FEATURE SCHEMA
# =========================================================

if not hasattr(
    trained_model,
    "feature_names_in_"
):

    st.error(
        "❌ The registered model does not contain "
        "feature name information."
    )

    st.stop()


model_features = list(
    trained_model.feature_names_in_
)


if set(model_features) != set(
    MODEL_FEATURES
):

    st.error(
        "❌ Model feature schema does not match "
        "the Hopsworks feature data."
    )

    st.write(
        "Model expects:"
    )

    st.code(
        "\n".join(model_features)
    )

    st.write(
        "Application expects:"
    )

    st.code(
        "\n".join(MODEL_FEATURES)
    )

    st.stop()


# =========================================================
# HELPER FUNCTION
# =========================================================

def get_lag(values, lag):

    if len(values) >= lag:

        return float(
            values[-lag]
        )

    return float(
        values[0]
    )


# =========================================================
# RECURSIVE 72-HOUR FORECAST
# =========================================================

try:

    history = df.copy()

    predictions = []

    future_dates = []

    last_time = history[
        "datetime"
    ].max()


    for step in range(
        FORECAST_HOURS
    ):

        # -------------------------------------------------
        # NEXT TIMESTAMP
        # -------------------------------------------------

        next_time = (
            last_time
            + pd.Timedelta(
                hours=1
            )
        )


        # -------------------------------------------------
        # LATEST VALUES
        # -------------------------------------------------

        latest_row = history.iloc[-1]


        # -------------------------------------------------
        # CREATE FUTURE ROW
        # -------------------------------------------------

        future_row = {}


        # -------------------------------------------------
        # ENVIRONMENTAL FEATURES
        #
        # These are carried forward because this model
        # forecasts AQI, not future pollutant concentrations.
        # -------------------------------------------------

        pollutant_features = [

            "co",
            "no",
            "no2",
            "o3",
            "so2",
            "pm2_5",
            "pm10",
            "nh3"

        ]


        for feature in pollutant_features:

            future_row[feature] = float(
                latest_row[feature]
            )


        # -------------------------------------------------
        # TIME FEATURES
        # -------------------------------------------------

        future_row["hour"] = (
            next_time.hour
        )

        future_row["day"] = (
            next_time.day
        )

        future_row["month"] = (
            next_time.month
        )

        future_row["day_of_week"] = (
            next_time.dayofweek
        )


        # -------------------------------------------------
        # AQI HISTORY
        # -------------------------------------------------

        aqi_values = (
            history["aqi"]
            .astype(float)
            .tolist()
        )


        # -------------------------------------------------
        # CURRENT AQI
        # -------------------------------------------------

        future_row["aqi"] = (
            float(
                latest_row["aqi"]
            )
        )


        # -------------------------------------------------
        # AQI LAG FEATURES
        # -------------------------------------------------

        future_row["aqi_lag_1"] = get_lag(
            aqi_values,
            1
        )

        future_row["aqi_lag_3"] = get_lag(
            aqi_values,
            3
        )

        future_row["aqi_lag_6"] = get_lag(
            aqi_values,
            6
        )

        future_row["aqi_lag_12"] = get_lag(
            aqi_values,
            12
        )

        future_row["aqi_lag_24"] = get_lag(
            aqi_values,
            24
        )

        future_row["aqi_lag_48"] = get_lag(
            aqi_values,
            48
        )

        future_row["aqi_lag_72"] = get_lag(
            aqi_values,
            72
        )


        # -------------------------------------------------
        # PM2.5 / PM10 LAG FEATURES
        # -------------------------------------------------

        pm25_values = (
            history["pm2_5"]
            .astype(float)
            .tolist()
        )

        pm10_values = (
            history["pm10"]
            .astype(float)
            .tolist()
        )


        future_row["pm2_5_lag_1"] = get_lag(
            pm25_values,
            1
        )

        future_row["pm10_lag_1"] = get_lag(
            pm10_values,
            1
        )


        # -------------------------------------------------
        # ROLLING AQI FEATURES
        # -------------------------------------------------

        aqi_series = pd.Series(
            aqi_values
        )


        future_row["aqi_rolling_6"] = float(
            aqi_series
            .tail(6)
            .mean()
        )

        future_row["aqi_rolling_24"] = float(
            aqi_series
            .tail(24)
            .mean()
        )

        future_row["aqi_rolling_72"] = float(
            aqi_series
            .tail(72)
            .mean()
        )


        # -------------------------------------------------
        # AQI CHANGE
        # -------------------------------------------------

        if len(aqi_values) >= 2:

            future_row["aqi_change"] = (

                float(
                    aqi_values[-1]
                )

                -

                float(
                    aqi_values[-2]
                )

            )

        else:

            future_row[
                "aqi_change"
            ] = 0.0


        # -------------------------------------------------
        # CREATE MODEL INPUT
        #
        # IMPORTANT:
        # Exact same feature order as training.
        # -------------------------------------------------

        X_future = pd.DataFrame(
            [[
                future_row[
                    feature
                ]

                for feature
                in model_features

            ]],

            columns=model_features
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = float(
            trained_model.predict(
                X_future
            )[0]
        )


        # -------------------------------------------------
        # AQI RANGE SAFETY
        # -------------------------------------------------

        prediction = max(
            0.0,
            min(
                5.0,
                prediction
            )
        )


        predictions.append(
            prediction
        )

        future_dates.append(
            next_time
        )


        # -------------------------------------------------
        # ADD PREDICTION TO HISTORY
        #
        # This makes the forecast recursive.
        # -------------------------------------------------

        new_row = future_row.copy()

        new_row["datetime"] = (
            next_time
        )

        new_row["aqi"] = (
            prediction
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [new_row]
                )
            ],
            ignore_index=True
        )


        last_time = (
            next_time
        )


    # =====================================================
    # FORECAST DATAFRAME
    # =====================================================

    forecast_df = pd.DataFrame({

        "datetime": future_dates,

        "predicted_aqi": predictions

    })


    st.success(
        "✅ Real Random Forest 72-hour "
        "recursive forecast generated successfully"
    )


except Exception as e:

    st.error(
        "❌ 72-hour forecast could not be generated."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# FORECAST SUMMARY
# =========================================================

forecast_max = float(
    forecast_df[
        "predicted_aqi"
    ].max()
)

forecast_min = float(
    forecast_df[
        "predicted_aqi"
    ].min()
)

forecast_average = float(
    forecast_df[
        "predicted_aqi"
    ].mean()
)


summary_col1, summary_col2, summary_col3 = st.columns(3)


with summary_col1:

    st.metric(
        "Forecast Maximum",
        f"{forecast_max:.2f}"
    )


with summary_col2:

    st.metric(
        "Forecast Minimum",
        f"{forecast_min:.2f}"
    )


with summary_col3:

    st.metric(
        "Forecast Average",
        f"{forecast_average:.2f}"
    )


# =========================================================
# FORECAST TABLE
# =========================================================

st.subheader(
    "Next 72 Hours"
)


display_df = forecast_df.copy()


display_df["datetime"] = (
    display_df[
        "datetime"
    ].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
)


display_df["predicted_aqi"] = (
    display_df[
        "predicted_aqi"
    ].round(2)
)


st.dataframe(
    display_df,
    use_container_width=True,
    height=350
)


# =========================================================
# FORECAST CHART
# =========================================================

st.subheader(
    "📊 AQI Forecast Trend"
)


fig, ax = plt.subplots(
    figsize=(12, 5)
)


ax.plot(
    forecast_df["datetime"],
    forecast_df[
        "predicted_aqi"
    ],
    linewidth=2
)


ax.set_title(
    "Next 72 Hours AQI Forecast"
)

ax.set_xlabel(
    "Time"
)

ax.set_ylabel(
    "Predicted AQI"
)

ax.set_ylim(
    0,
    5
)

ax.grid(
    alpha=0.3
)


plt.xticks(
    rotation=45
)

plt.tight_layout()


st.pyplot(
    fig
)


# =========================================================
# AQI ALERT
# =========================================================

st.divider()

st.header(
    "🚨 Air Quality Alert"
)


if forecast_max >= 5:

    st.error(
        f"⚠️ Hazardous AQI levels may occur. "
        f"Maximum predicted AQI: "
        f"{forecast_max:.2f}"
    )

elif forecast_max >= 4:

    st.warning(
        f"⚠️ Moderate to unhealthy AQI levels "
        f"may occur. Maximum predicted AQI: "
        f"{forecast_max:.2f}"
    )

else:

    st.success(
        f"✅ Air quality is expected to remain "
        f"relatively good over the forecast period. "
        f"Maximum predicted AQI: "
        f"{forecast_max:.2f}"
    )


# =========================================================
# MODEL EXPLAINABILITY
# =========================================================

st.divider()

st.header(
    "🔍 Model Explainability — Feature Importance"
)


importance_values = (
    trained_model.feature_importances_
)


feature_importance = pd.DataFrame({

    "Feature": model_features,

    "Importance": importance_values

})


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ---------------------------------------------------------
# Feature importance table
# ---------------------------------------------------------

st.dataframe(
    feature_importance,
    use_container_width=True
)


# ---------------------------------------------------------
# Feature importance chart
# ---------------------------------------------------------

fig2, ax2 = plt.subplots(
    figsize=(10, 8)
)


ax2.barh(
    feature_importance[
        "Feature"
    ],
    feature_importance[
        "Importance"
    ]
)


ax2.set_title(
    "Random Forest Feature Importance"
)

ax2.set_xlabel(
    "Importance"
)

ax2.invert_yaxis()


plt.tight_layout()


st.pyplot(
    fig2
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.header(
    "🤖 Model Information"
)


info_col1, info_col2, info_col3, info_col4 = (
    st.columns(4)
)


with info_col1:

    st.metric(
        "Model",
        "Random Forest"
    )


with info_col2:

    st.metric(
        "Input Features",
        len(model_features)
    )


with info_col3:

    st.metric(
        "Forecast Horizon",
        "72 Hours"
    )


with info_col4:

    st.metric(
        "Model Version",
        MODEL_VERSION
    )


# =========================================================
# DATA INFORMATION
# =========================================================

st.divider()

st.header(
    "📊 Data Information"
)


data_col1, data_col2, data_col3 = (
    st.columns(3)
)


with data_col1:

    st.metric(
        "Available Records",
        len(df)
    )


with data_col2:

    st.metric(
        "Feature Columns",
        len(df.columns)
    )


with data_col3:

    st.metric(
        "Latest Data Time",
        latest["datetime"].strftime(
            "%Y-%m-%d %H:%M"
        )
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.header(
    "ℹ️ About the Project"
)


st.write(
    """
**Pearls AQI Predictor** is an end-to-end
machine learning system designed to forecast
Air Quality Index values for the next 3 days.

The system uses:

- Hopsworks Feature Store
- Python
- Pandas
- Scikit-learn
- Random Forest
- Hopsworks Model Registry
- Recursive 72-hour AQI forecasting
- Streamlit dashboard
- Automated feature engineering
- Model feature validation
- Model explainability
- AQI hazard alerts
"""
)


st.success(
    "🎉 Pearls AQI Predictor Dashboard Loaded Successfully!"
)
