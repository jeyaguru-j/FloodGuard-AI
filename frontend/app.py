import requests
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        }

        .risk-box {
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            background-color: #f5f5f5;
        }

        .risk-value {
            font-size: 38px;
            font-weight: 700;
        }

        .section-title {
            font-size: 24px;
            font-weight: 600;
            margin-top: 25px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">🌊 FloodGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-Powered Urban Flood Prediction & Early Warning System"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Prediction Settings")

selected_date = st.sidebar.date_input(
    "Select Date",
)

predict_button = st.sidebar.button(
    "🔍 Predict Flood Risk",
    use_container_width=True,
)


# ============================================================
# API call
# ============================================================

if predict_button:

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={
                "date": selected_date.strftime("%Y-%m-%d")
            },
            timeout=10,
        )

        if response.status_code != 200:
            st.error(response.json().get("detail", "Prediction failed."))

        else:

            result = response.json()

            probability = result["flood_probability_percent"]
            risk = result["risk_level"]

            # ------------------------------------------------
            # Main metrics
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Flood Probability",
                    f"{probability:.2f}%",
                )

            with col2:
                st.metric(
                    "Risk Level",
                    risk,
                )

            with col3:
                st.metric(
                    "Forecast Horizon",
                    "Next 3 Days",
                )

            st.markdown("---")

            # ------------------------------------------------
            # Risk indicator
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">Flood Risk Assessment</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="risk-box">
                    <div>Predicted Risk</div>
                    <div class="risk-value">{risk}</div>
                    <div>
                        Estimated probability:
                        <strong>{probability:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # Weather indicators
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">Weather Indicators</div>',
                unsafe_allow_html=True,
            )

            weather = result["weather"]

            w1, w2, w3, w4 = st.columns(4)

            with w1:
                st.metric(
                    "Rainfall",
                    f'{weather["rainfall_mm"]} mm',
                )

            with w2:
                st.metric(
                    "Temperature",
                    f'{weather["temperature_mean_c"]} °C',
                )

            with w3:
                st.metric(
                    "Humidity",
                    f'{weather["humidity_percent"]}%',
                )

            with w4:
                st.metric(
                    "Wind Speed",
                    f'{weather["wind_speed_kmh"]} km/h',
                )

            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">Why this prediction?</div>',
                unsafe_allow_html=True,
            )

            for indicator in result["key_indicators"]:
                st.write(f"• {indicator}")

            # ------------------------------------------------
            # System information
            # ------------------------------------------------

            st.markdown("---")

            st.caption(
                "FloodGuard AI uses a Random Forest model trained on "
                "historical Chennai weather and flood-event data. "
                "The model forecasts whether flood conditions may "
                "occur within the following three days."
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "⚠️ Cannot connect to FloodGuard API. "
            "Make sure FastAPI is running on port 8000."
        )

else:

    # --------------------------------------------------------
    # Landing screen
    # --------------------------------------------------------

    st.info(
        "Select a date from the sidebar and click "
        "**Predict Flood Risk** to generate an early-warning assessment."
    )

    st.markdown("### System Capabilities")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🌧️ Flood Prediction")
        st.write(
            "Forecast flood risk within the next three days "
            "using machine learning."
        )

    with c2:
        st.markdown("### 📊 Risk Assessment")
        st.write(
            "Convert model probabilities into interpretable "
            "risk levels and indicators."
        )

    with c3:
        st.markdown("### ⚙️ MLOps Ready")
        st.write(
            "Designed for model monitoring, drift detection, "
            "versioning and automated retraining."
        )