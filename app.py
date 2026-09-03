import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import requests

from datetime import timedelta
from dotenv import load_dotenv


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Pearls AQI Predictor - Lahore",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CONSTANTS
# ==========================================================

CITY_NAME = "Lahore"
COUNTRY_NAME = "Pakistan"

FORECAST_HOURS = 72

LATITUDE = 31.5204
LONGITUDE = 74.3587

OPENWEATHER_API_URL = (
    "https://api.openweathermap.org/data/2.5/air_pollution"
)

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
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🌫️ Pearls AQI Predictor")

    st.caption(
        "10Pearls SHINE Internship · Data Sciences Track"
    )

    st.divider()

    st.subheader("📍 Location")

    st.write(
        f"**{CITY_NAME}, {COUNTRY_NAME}**"
    )

    st.divider()

    st.subheader("🌿 EPA AQI Scale")

    st.write("0–50  🙂 Good")
    st.write("51–100  😐 Moderate")
    st.write("101–150  😷 Unhealthy for Sensitive Groups")
    st.write("151–200  🚫 Unhealthy")
    st.write("201–300  ☣️ Very Unhealthy")
    st.write("301–500+  ☠️ Hazardous")

    st.divider()

    st.subheader("🤖 Model")

    st.write("Forecast Horizon: **72 Hours**")
    st.write("Input Features: **26**")

    st.divider()

    st.subheader("📚 Documentation")

    st.markdown(
        "[GitHub Repository]"
        "(https://github.com/anusha-awan/pearls-aqi-predictor)"
    )

    st.markdown(
        "[Final Report]"
        "(https://github.com/anusha-awan/pearls-aqi-predictor/blob/main/Final_Report.md)"
    )

    st.markdown(
        "[Build Journey & EDA]"
        "(https://github.com/anusha-awan/pearls-aqi-predictor/blob/main/EDA_Writeup.md)"
    )


# ==========================================================
# EPA PM2.5 AQI CALCULATION
# ==========================================================

def calculate_pm25_aqi(pm25):

    if pd.isna(pm25):
        return None

    try:
        pm25 = float(pm25)

    except (ValueError, TypeError):
        return None

    if pm25 < 0:
        return None

    pm25 = int(pm25 * 10) / 10

    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500)
    ]

    if pm25 > 325.4:
        return 500

    for (
        concentration_low,
        concentration_high,
        aqi_low,
        aqi_high
    ) in breakpoints:

        if (
            concentration_low
            <= pm25
            <= concentration_high
        ):

            aqi = (
                (
                    aqi_high - aqi_low
                )
                /
                (
                    concentration_high
                    - concentration_low
                )
            ) * (
                pm25
                - concentration_low
            ) + aqi_low

            return int(round(aqi))

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
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "aqi_model.pkl"
    )


# ==========================================================
# LOAD METADATA
# ==========================================================

@st.cache_resource
def load_metadata():

    try:

        return joblib.load(
            "model_metadata.pkl"
        )

    except Exception:

        return {}


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "features.csv"
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce"
    )

    df = (
        df
        .dropna(
            subset=["datetime"]
        )
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=["datetime"],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ==========================================================
# FETCH LIVE AIR QUALITY FROM OPENWEATHER
# ==========================================================

@st.cache_data(ttl=300)
def fetch_live_air_quality():

    # Load local .env if available.
    # This is used during local development.

    load_dotenv()

    api_key = None

    # ------------------------------------------------------
    # Streamlit Cloud Secrets
    # ------------------------------------------------------

    try:

        api_key = st.secrets[
            "OPENWEATHER_API_KEY"
        ]

    except Exception:

        api_key = None

    # ------------------------------------------------------
    # Local .env fallback
    # ------------------------------------------------------

    if not api_key:

        api_key = os.getenv(
            "OPENWEATHER_API_KEY"
        )

    if not api_key:

        raise ValueError(
            "OPENWEATHER_API_KEY is not configured."
        )

    # ------------------------------------------------------
    # API request
    # ------------------------------------------------------

    params = {

        "lat":
            LATITUDE,

        "lon":
            LONGITUDE,

        "appid":
            api_key

    }

    response = requests.get(
        OPENWEATHER_API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    pollution = data["list"][0]

    # ------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------

    timestamp = pd.to_datetime(
        pollution["dt"],
        unit="s",
        utc=True
    )

    components = pollution[
        "components"
    ]

    # ------------------------------------------------------
    # Return live pollutant values
    # ------------------------------------------------------

    return {

        "datetime":
            timestamp,

        "co":
            components["co"],

        "no":
            components["no"],

        "no2":
            components["no2"],

        "o3":
            components["o3"],

        "so2":
            components["so2"],

        "pm2_5":
            components["pm2_5"],

        "pm10":
            components["pm10"],

        "nh3":
            components["nh3"]

    }


# ==========================================================
# PAGE HEADER
# ==========================================================

st.title(
    "🌍 Pearls AQI Predictor"
)

st.subheader(
    "3-Day Air Quality Index Forecast"
)

st.write(
    f"AI-powered AQI forecasting for "
    f"**{CITY_NAME}, {COUNTRY_NAME}** using "
    "machine learning and automated feature engineering."
)


# ==========================================================
# LOAD FILES
# ==========================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        f"Unable to load aqi_model.pkl: {e}"
    )

    st.stop()


try:

    metadata = load_metadata()

except Exception:

    metadata = {}


try:

    df = load_data()

except Exception as e:

    st.error(
        f"Unable to load features.csv: {e}"
    )

    st.stop()


# ==========================================================
# MODEL METADATA
# ==========================================================

model_name = metadata.get(
    "model_name",
    metadata.get(
        "model",
        "Gradient Boosting"
    )
)

model_version = metadata.get(
    "model_version",
    1
)

mae = metadata.get(
    "mae",
    None
)

rmse = metadata.get(
    "rmse",
    None
)

r2 = metadata.get(
    "r2",
    None
)


# ==========================================================
# VALIDATE MODEL FEATURES
# ==========================================================

actual_features = getattr(
    model,
    "n_features_in_",
    None
)

expected_features = len(
    MODEL_FEATURES
)

if actual_features is not None:

    if actual_features != expected_features:

        st.error(
            f"Model expects {actual_features} features, "
            f"but the dashboard provides "
            f"{expected_features} features."
        )

        st.stop()


missing_features = [
    feature
    for feature in MODEL_FEATURES
    if feature not in df.columns
]

if missing_features:

    st.error(
        "Missing model features:\n"
        + "\n".join(missing_features)
    )

    st.stop()


# ==========================================================
# TIME FEATURES
# ==========================================================

df["hour"] = (
    df["datetime"].dt.hour
)

df["day"] = (
    df["datetime"].dt.day
)

df["month"] = (
    df["datetime"].dt.month
)

df["day_of_week"] = (
    df["datetime"].dt.dayofweek
)


# ==========================================================
# CURRENT OBSERVATION — LIVE API
# ==========================================================

try:

    live_data = fetch_live_air_quality()

    st.caption(
        "🟢 Live air-quality data from OpenWeather API · "
        "Data refreshes every 5 minutes."
    )

except Exception:

    st.warning(
        "⚠️ Live OpenWeather data is temporarily unavailable. "
        "Using the latest stored observation instead."
    )

    latest_stored = df.iloc[-1].copy()

    live_data = {

        "datetime":
            latest_stored["datetime"],

        "co":
            latest_stored["co"],

        "no":
            latest_stored["no"],

        "no2":
            latest_stored["no2"],

        "o3":
            latest_stored["o3"],

        "so2":
            latest_stored["so2"],

        "pm2_5":
            latest_stored["pm2_5"],

        "pm10":
            latest_stored["pm10"],

        "nh3":
            latest_stored["nh3"]

    }


latest_time = live_data["datetime"]

current_pm25 = float(
    live_data["pm2_5"]
)

current_aqi = calculate_pm25_aqi(
    current_pm25
)

if current_aqi is None:

    st.error(
        "Unable to calculate current EPA AQI."
    )

    st.stop()


# ==========================================================
# LOCATION
# ==========================================================

st.divider()

location_col1, location_col2 = st.columns(2)

with location_col1:

    st.metric(
        "📍 City",
        CITY_NAME
    )

with location_col2:

    st.metric(
        "🌎 Country",
        COUNTRY_NAME
    )


# ==========================================================
# CURRENT AIR QUALITY
# ==========================================================

st.header(
    "📍 Current Air Quality"
)

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
        "Latest Observation",
        latest_time.strftime(
            "%Y-%m-%d %H:%M"
        )
    )


# ==========================================================
# 72-HOUR FORECAST
# ==========================================================

st.divider()

st.header(
    "🔮 Next 3 Days AQI Forecast"
)

st.write(
    "The trained machine learning model generates "
    "recursive hourly EPA-style AQI predictions "
    "for the next 72 hours."
)


# ==========================================================
# HISTORICAL AQI
# ==========================================================

historical_aqi = (

    df["aqi"]

    .dropna()

    .astype(float)

    .tolist()

)

if len(historical_aqi) == 0:

    st.error(
        "No historical AQI values are available."
    )

    st.stop()


# Synchronize latest historical AQI
# with current live EPA AQI.

historical_aqi[-1] = float(
    current_aqi
)


# ==========================================================
# FORECAST
# ==========================================================

predictions = []


for step in range(
    1,
    FORECAST_HOURS + 1
):

    future_time = (

        latest_time

        + timedelta(
            hours=step
        )

    )


    # ------------------------------------------------------
    # AQI LAG FEATURES
    # ------------------------------------------------------

    aqi_lag_1 = (
        historical_aqi[-1]
    )

    aqi_lag_3 = (

        historical_aqi[-3]

        if len(historical_aqi) >= 3

        else historical_aqi[0]

    )

    aqi_lag_6 = (

        historical_aqi[-6]

        if len(historical_aqi) >= 6

        else historical_aqi[0]

    )

    aqi_lag_12 = (

        historical_aqi[-12]

        if len(historical_aqi) >= 12

        else historical_aqi[0]

    )

    aqi_lag_24 = (

        historical_aqi[-24]

        if len(historical_aqi) >= 24

        else historical_aqi[0]

    )

    aqi_lag_48 = (

        historical_aqi[-48]

        if len(historical_aqi) >= 48

        else historical_aqi[0]

    )

    aqi_lag_72 = (

        historical_aqi[-72]

        if len(historical_aqi) >= 72

        else historical_aqi[0]

    )


    # ------------------------------------------------------
    # ROLLING FEATURES
    # ------------------------------------------------------

    rolling_6_values = (
        historical_aqi[-6:]
    )

    rolling_24_values = (
        historical_aqi[-24:]
    )

    rolling_72_values = (
        historical_aqi[-72:]
    )


    aqi_rolling_6 = (

        sum(rolling_6_values)
        /
        len(rolling_6_values)

    )

    aqi_rolling_24 = (

        sum(rolling_24_values)
        /
        len(rolling_24_values)

    )

    aqi_rolling_72 = (

        sum(rolling_72_values)
        /
        len(rolling_72_values)

    )


    # ------------------------------------------------------
    # AQI CHANGE
    # ------------------------------------------------------

    previous_aqi = (

        historical_aqi[-2]

        if len(historical_aqi) >= 2

        else aqi_lag_1

    )

    aqi_change = (

        aqi_lag_1
        -
        previous_aqi

    )


    # ------------------------------------------------------
    # FUTURE POLLUTANT BASELINE
    # ------------------------------------------------------
    # Use LIVE OpenWeather values.

    future_co = float(
        live_data["co"]
    )

    future_no = float(
        live_data["no"]
    )

    future_no2 = float(
        live_data["no2"]
    )

    future_o3 = float(
        live_data["o3"]
    )

    future_so2 = float(
        live_data["so2"]
    )

    future_pm25 = float(
        live_data["pm2_5"]
    )

    future_pm10 = float(
        live_data["pm10"]
    )

    future_nh3 = float(
        live_data["nh3"]
    )


    # ------------------------------------------------------
    # FUTURE FEATURE ROW
    # ------------------------------------------------------

    future_features = {

        "co":
            future_co,

        "no":
            future_no,

        "no2":
            future_no2,

        "o3":
            future_o3,

        "so2":
            future_so2,

        "pm2_5":
            future_pm25,

        "pm10":
            future_pm10,

        "nh3":
            future_nh3,

        "hour":
            future_time.hour,

        "day":
            future_time.day,

        "month":
            future_time.month,

        "day_of_week":
            future_time.dayofweek,

        "aqi":
            aqi_lag_1,

        "aqi_lag_1":
            aqi_lag_1,

        "aqi_lag_3":
            aqi_lag_3,

        "aqi_lag_6":
            aqi_lag_6,

        "aqi_lag_12":
            aqi_lag_12,

        "aqi_lag_24":
            aqi_lag_24,

        "aqi_lag_48":
            aqi_lag_48,

        "aqi_lag_72":
            aqi_lag_72,

        "pm2_5_lag_1":
            future_pm25,

        "pm10_lag_1":
            future_pm10,

        "aqi_rolling_6":
            aqi_rolling_6,

        "aqi_rolling_24":
            aqi_rolling_24,

        "aqi_rolling_72":
            aqi_rolling_72,

        "aqi_change":
            aqi_change

    }


    # ------------------------------------------------------
    # MODEL INPUT
    # ------------------------------------------------------

    X_future = pd.DataFrame(
        [future_features]
    )[MODEL_FEATURES]


    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    predicted_aqi = model.predict(
        X_future
    )[0]


    predicted_aqi = max(
        0,
        min(
            500,
            float(predicted_aqi)
        )
    )


    predicted_aqi = round(
        predicted_aqi,
        1
    )


    # ------------------------------------------------------
    # RECURSIVE UPDATE
    # ------------------------------------------------------

    historical_aqi.append(
        predicted_aqi
    )


    predictions.append({

        "datetime":
            future_time,

        "predicted_aqi":
            predicted_aqi

    })


# ==========================================================
# FORECAST DATAFRAME
# ==========================================================

forecast_df = pd.DataFrame(
    predictions
)


# ==========================================================
# FORECAST CHART
# ==========================================================

st.subheader(
    "📈 72-Hour Predicted AQI"
)

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
    "Forecast Hour"
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

st.pyplot(
    fig
)

plt.close(
    fig
)


# ==========================================================
# KEY FORECAST OUTPUTS
# ==========================================================

st.subheader(
    "📌 Key Forecast Predictions"
)

day1 = forecast_df.iloc[23]

day2 = forecast_df.iloc[47]

day3 = forecast_df.iloc[71]


forecast_col1, forecast_col2, forecast_col3 = (
    st.columns(3)
)


with forecast_col1:

    st.metric(
        "Day 1",
        f"{day1['predicted_aqi']:.1f} AQI"
    )

    st.caption(
        "24-hour forecast"
    )


with forecast_col2:

    st.metric(
        "Day 2",
        f"{day2['predicted_aqi']:.1f} AQI"
    )

    st.caption(
        "48-hour forecast"
    )


with forecast_col3:

    st.metric(
        "Day 3",
        f"{day3['predicted_aqi']:.1f} AQI"
    )

    st.caption(
        "72-hour forecast"
    )


# ==========================================================
# DAILY FORECAST
# ==========================================================

forecast_df["forecast_day"] = [

    "Day 1"

    if i < 24

    else "Day 2"

    if i < 48

    else "Day 3"

    for i in range(
        len(forecast_df)
    )

]


daily_forecast = (

    forecast_df

    .groupby(
        "forecast_day",
        sort=False
    )["predicted_aqi"]

    .agg(

        [
            ("Minimum AQI", "min"),

            ("Average AQI", "mean"),

            ("Maximum AQI", "max")

        ]

    )

    .reset_index()

)


daily_forecast[
    "Minimum AQI"
] = (

    daily_forecast[
        "Minimum AQI"
    ].round(1)

)


daily_forecast[
    "Average AQI"
] = (

    daily_forecast[
        "Average AQI"
    ].round(1)

)


daily_forecast[
    "Maximum AQI"
] = (

    daily_forecast[
        "Maximum AQI"
    ].round(1)

)


# ==========================================================
# DAILY FORECAST EXPANDER
# ==========================================================

with st.expander(
    "📅 Daily Forecast Summary",
    expanded=False
):

    st.write(
        "A summary of the predicted AQI range for each "
        "of the next three days."
    )

    st.dataframe(
        daily_forecast,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# FORECAST ALERT
# ==========================================================

forecast_max = (

    forecast_df[
        "predicted_aqi"
    ].max()

)


if forecast_max > 300:

    st.error(

        f"🚨 Hazardous AQI levels may occur. "
        f"Maximum predicted AQI: "
        f"{forecast_max:.1f}"

    )

elif forecast_max > 200:

    st.warning(

        f"⚠️ Very unhealthy AQI levels may occur. "
        f"Maximum predicted AQI: "
        f"{forecast_max:.1f}"

    )

elif forecast_max > 150:

    st.warning(

        f"⚠️ Unhealthy AQI levels may occur. "
        f"Maximum predicted AQI: "
        f"{forecast_max:.1f}"

    )

elif forecast_max > 100:

    st.warning(

        f"⚠️ Unhealthy for sensitive groups "
        f"levels may occur. "
        f"Maximum predicted AQI: "
        f"{forecast_max:.1f}"

    )

else:

    st.success(

        f"✅ Air quality is expected to remain "
        f"relatively good. "
        f"Maximum predicted AQI: "
        f"{forecast_max:.1f}"

    )


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

st.divider()


with st.expander(
    "📊 Model Performance",
    expanded=False
):

    st.write(
        "Performance metrics calculated during model evaluation."
    )

    metric1, metric2, metric3 = (
        st.columns(3)
    )


    with metric1:

        st.metric(
            "MAE",
            f"{float(mae):.2f} AQI points"
            if mae is not None
            else "N/A"
        )


    with metric2:

        st.metric(
            "RMSE",
            f"{float(rmse):.2f} AQI points"
            if rmse is not None
            else "N/A"
        )


    with metric3:

        st.metric(
            "R²",
            f"{float(r2):.4f}"
            if r2 is not None
            else "N/A"
        )


    st.caption(
        "Evaluation uses a chronological 80/20 "
        "train-test split on EPA-style AQI values."
    )


# ==========================================================
# SHAP EXPLAINABILITY
# ==========================================================

st.divider()


with st.expander(
    "🤖 Model Explainability — SHAP",
    expanded=False
):

    st.write(
        "SHAP (SHapley Additive exPlanations) "
        "shows how model input features influence "
        "AQI predictions."
    )


    try:

        import shap

        X_shap = (

            df[
                MODEL_FEATURES
            ]

            .dropna()

            .copy()

        )


        if len(X_shap) > 0:

            shap_sample_size = min(
                300,
                len(X_shap)
            )


            X_shap_sample = (

                X_shap

                .sample(
                    n=shap_sample_size,
                    random_state=42
                )

                .reset_index(
                    drop=True
                )

            )


            explainer = shap.TreeExplainer(
                model
            )


            shap_values = (

                explainer.shap_values(
                    X_shap_sample
                )

            )


            if isinstance(
                shap_values,
                list
            ):

                shap_values_plot = (
                    shap_values[0]
                )

            else:

                shap_values_plot = (
                    shap_values
                )


            shap_values_plot = (

                pd.DataFrame(

                    shap_values_plot,

                    columns=MODEL_FEATURES

                )

                .astype(float)

                .values

            )


            mean_abs_shap = (

                abs(
                    shap_values_plot
                )

                .mean(
                    axis=0
                )

            )


            shap_importance = pd.DataFrame({

                "Feature":
                    MODEL_FEATURES,

                "Mean |SHAP Value|":
                    mean_abs_shap

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


            st.subheader(
                "📊 Global Feature Impact"
            )


            st.caption(

                "Average absolute SHAP influence "
                f"across {shap_sample_size} observations."

            )


            st.dataframe(

                shap_importance,

                use_container_width=True,

                hide_index=True

            )


            top_shap = (

                shap_importance

                .head(15)

                .sort_values(

                    "Mean |SHAP Value|",

                    ascending=True

                )

            )


            fig_shap, ax_shap = plt.subplots(

                figsize=(10, 8)

            )


            ax_shap.barh(

                top_shap["Feature"],

                top_shap[
                    "Mean |SHAP Value|"
                ]

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


            plt.close(
                fig_shap
            )


            st.subheader(
                "🔍 Individual Prediction Explanation"
            )


            st.write(
                "The following explanation shows the "
                "strongest feature contributions for "
                "one representative observation."
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

                "Feature":
                    MODEL_FEATURES,

                "SHAP Value":
                    explanation_values,

                "Feature Value":
                    explanation_features.values

            })


            local_explanation[
                "Absolute SHAP"
            ] = (

                abs(

                    local_explanation[
                        "SHAP Value"
                    ]

                )

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

                local_explanation[
                    "Feature"
                ],

                local_explanation[
                    "SHAP Value"
                ]

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


            plt.close(
                fig_local
            )


            strongest_feature = (

                shap_importance.iloc[0][
                    "Feature"
                ]

            )


            strongest_value = (

                shap_importance.iloc[0][
                    "Mean |SHAP Value|"
                ]

            )


            st.info(

                f"💡 The feature with the strongest "
                f"overall SHAP influence is "
                f"**{strongest_feature}**, with a mean "
                f"absolute SHAP value of "
                f"**{strongest_value:.4f}**."

            )


        else:

            st.warning(
                "No valid observations are available for SHAP analysis."
            )


    except Exception:

        st.warning(
            "SHAP explainability could not be generated. "
            "The forecasting model can still operate normally."
        )


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.divider()


with st.expander(
    "🧠 Model Information",
    expanded=False
):

    st.write(
        "Technical information about the trained forecasting model."
    )


    info1, info2, info3, info4 = (
        st.columns(4)
    )


    with info1:

        st.metric(
            "Model",
            str(model_name)
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
            str(model_version)
        )


# ==========================================================
# DATA INFORMATION
# ==========================================================

st.divider()


with st.expander(
    "📊 Data Information",
    expanded=False
):

    st.write(
        "Information about the data currently used by the dashboard."
    )


    data1, data2, data3 = (
        st.columns(3)
    )


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
# ABOUT THE PROJECT
# ==========================================================

st.divider()


with st.expander(
    "ℹ️ About the Project",
    expanded=False
):

    st.write(

        f"""
**Pearls AQI Predictor** is an end-to-end machine
learning system designed to forecast Air Quality Index
values for the next three days in **{CITY_NAME},
{COUNTRY_NAME}**.

### Technologies

- Python
- Pandas
- Scikit-learn
- Gradient Boosting
- SHAP
- OpenWeather Air Pollution API
- Hopsworks Feature Store
- GitHub Actions
- Streamlit

### Key Features

- Live OpenWeather air-quality data
- Automated feature engineering
- EPA-style PM2.5 AQI calculation
- Chronological model evaluation
- Gradient Boosting model
- 72-hour AQI forecasting
- Recursive future prediction
- Model comparison
- SHAP feature importance
- Individual prediction explanation
- AQI health alerts
- Interactive Streamlit dashboard
"""
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Pearls AQI Predictor · 10Pearls SHINE Internship · Data Sciences Track"
)