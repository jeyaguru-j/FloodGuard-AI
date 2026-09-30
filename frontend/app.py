import streamlit as st
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_dataset"
    / "chennai_flood_3day_ml_dataset.csv"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================================================
       FLOODGUARD AI — PREMIUM DASHBOARD THEME
       ========================================================= */

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 5% 5%, rgba(14,165,233,0.10), transparent 25%),
            radial-gradient(circle at 95% 15%, rgba(37,99,235,0.08), transparent 25%),
            linear-gradient(180deg, #f7faff 0%, #f3f7fc 100%);
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 20% 10%, rgba(14,165,233,0.12), transparent 28%),
            linear-gradient(180deg, #06162c 0%, #092746 55%, #071d36 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #eaf4ff !important;
    }

    .sidebar-brand {
        padding: 10px 4px 20px 4px;
    }

    .sidebar-logo {
        font-size: 34px;
        margin-bottom: 5px;
        filter: drop-shadow(0 4px 8px rgba(56,189,248,0.25));
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 850;
        letter-spacing: -0.7px;
    }

    .sidebar-subtitle {
        color: #8fa8c5 !important;
        font-size: 13px;
        margin-top: 6px;
    }

    .sidebar-divider {
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.15),
            transparent
        );
        margin: 22px 0;
    }

    .sidebar-section {
        font-size: 12px;
        font-weight: 800;
        color: #72c7ff !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 20px 0 11px 0;
    }

    .api-online {
        background: linear-gradient(
            135deg,
            rgba(34,197,94,0.12),
            rgba(16,185,129,0.08)
        );
        border: 1px solid rgba(34,197,94,0.25);
        border-radius: 14px;
        padding: 13px;
        color: #75e6a5 !important;
        font-size: 13px;
        font-weight: 700;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    }

    .api-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow:
            0 0 0 4px rgba(34,197,94,0.10),
            0 0 14px rgba(34,197,94,0.8);
        animation: apiPulse 2s infinite;
    }

    @keyframes apiPulse {
        0%, 100% {
            box-shadow:
                0 0 0 4px rgba(34,197,94,0.10),
                0 0 14px rgba(34,197,94,0.8);
        }
        50% {
            box-shadow:
                0 0 0 7px rgba(34,197,94,0.03),
                0 0 20px rgba(34,197,94,0.95);
        }
    }

    .sidebar-stat {
        margin-top: 14px;
        padding: 14px;
        border-radius: 13px;
        background: linear-gradient(
            135deg,
            rgba(255,255,255,0.07),
            rgba(255,255,255,0.035)
        );
        border: 1px solid rgba(255,255,255,0.06);
        transition: transform 0.2s ease, background 0.2s ease;
    }

    .sidebar-stat:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.09);
    }

    .sidebar-stat-label {
        font-size: 10px;
        color: #8fa8c5 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .sidebar-stat-value {
        font-size: 16px;
        font-weight: 750;
        margin-top: 5px;
    }

    /* ---------- HERO ---------- */

    .hero {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(
                circle at 100% 0%,
                rgba(56,189,248,0.20),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #06172f 0%,
                #0a2d50 52%,
                #075985 100%
            );
        border-radius: 26px;
        padding: 34px 40px;
        color: white;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow:
            0 24px 55px rgba(7,42,78,0.20),
            inset 0 1px 0 rgba(255,255,255,0.08);
    }

    .hero:before {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        right: -140px;
        top: -170px;
        border-radius: 50%;
        background: rgba(56,189,248,0.09);
        border: 1px solid rgba(125,211,252,0.12);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        left: -100px;
        bottom: -120px;
        border-radius: 50%;
        background: rgba(37,99,235,0.10);
    }

    .hero-top {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .hero-icon {
        font-size: 44px;
        filter: drop-shadow(0 5px 12px rgba(56,189,248,0.35));
    }

    .hero-title {
        font-size: 39px;
        font-weight: 850;
        letter-spacing: -1.8px;
        line-height: 1;
    }

    .hero-subtitle {
        color: #b8d8f2;
        font-size: 15px;
        margin-top: 9px;
        letter-spacing: 0.1px;
    }

    .hero-badges {
        position: relative;
        z-index: 2;
        display: flex;
        gap: 9px;
        margin-top: 21px;
        flex-wrap: wrap;
    }

    .hero-badge {
        background: rgba(255,255,255,0.075);
        border: 1px solid rgba(255,255,255,0.13);
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        color: #e3f2ff;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }

    .hero-badge:hover {
        background: rgba(255,255,255,0.13);
        transform: translateY(-1px);
    }

    /* ---------- KPI CARDS ---------- */

    .kpi-card {
        position: relative;
        overflow: hidden;
        background: rgba(255,255,255,0.92);
        border: 1px solid #e3eaf3;
        border-radius: 19px;
        padding: 20px;
        box-shadow:
            0 9px 28px rgba(25,55,95,0.065),
            inset 0 1px 0 rgba(255,255,255,0.8);
        min-height: 120px;
        transition: all 0.22s ease;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow:
            0 16px 34px rgba(25,55,95,0.10),
            inset 0 1px 0 rgba(255,255,255,0.9);
        border-color: #cbdced;
    }

    .kpi-card:after {
        content: "";
        position: absolute;
        width: 90px;
        height: 90px;
        right: -35px;
        bottom: -45px;
        border-radius: 50%;
        background: rgba(14,165,233,0.06);
    }

    .kpi-label {
        color: #74869d;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.9px;
    }

    .kpi-value {
        color: #10243e;
        font-size: 28px;
        font-weight: 850;
        margin-top: 9px;
        letter-spacing: -0.8px;
    }

    .kpi-icon {
        float: right;
        font-size: 26px;
    }

    /* ---------- SECTION HEADINGS ---------- */

    .section-title {
        font-size: 23px;
        font-weight: 850;
        color: #10243e;
        margin-top: 30px;
        margin-bottom: 14px;
        letter-spacing: -0.5px;
    }

    .section-caption {
        color: #718198;
        font-size: 13px;
        margin-top: -7px;
        margin-bottom: 15px;
    }

    /* ---------- RISK CARD ---------- */

    .risk-card {
        border-radius: 21px;
        padding: 25px 28px;
        border: 1px solid;
        margin-bottom: 13px;
        box-shadow: 0 12px 32px rgba(20,40,70,0.07);
        position: relative;
        overflow: hidden;
    }

    .risk-card:after {
        content: "";
        position: absolute;
        width: 170px;
        height: 170px;
        right: -80px;
        top: -90px;
        border-radius: 50%;
        background: rgba(255,255,255,0.45);
    }

    .risk-high {
        background: linear-gradient(135deg, #fff0f2, #fff8f8);
        border-color: #fecdd3;
    }

    .risk-moderate {
        background: linear-gradient(135deg, #fffbea, #fffef5);
        border-color: #fde68a;
    }

    .risk-low {
        background: linear-gradient(135deg, #ecfdf5, #f7fffb);
        border-color: #bbf7d0;
    }

    .risk-header {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 17px;
        font-weight: 850;
    }

    .risk-pulse {
        width: 13px;
        height: 13px;
        border-radius: 50%;
        display: inline-block;
    }

    .pulse-high {
        background: #ef4444;
        animation: riskPulse 1.7s infinite;
    }

    .pulse-moderate {
        background: #f59e0b;
        animation: riskPulseOrange 1.9s infinite;
    }

    .pulse-low {
        background: #22c55e;
        box-shadow: 0 0 10px rgba(34,197,94,0.45);
    }

    @keyframes riskPulse {
        0% {
            box-shadow: 0 0 0 0 rgba(239,68,68,0.45);
        }
        70% {
            box-shadow: 0 0 0 13px rgba(239,68,68,0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(239,68,68,0);
        }
    }

    @keyframes riskPulseOrange {
        0% {
            box-shadow: 0 0 0 0 rgba(245,158,11,0.45);
        }
        70% {
            box-shadow: 0 0 0 13px rgba(245,158,11,0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(245,158,11,0);
        }
    }

    .risk-probability {
        position: relative;
        z-index: 2;
        margin-top: 12px;
        font-size: 14px;
        color: #53657b;
    }

    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #e2eaf3;
        border-radius: 18px;
        padding: 19px;
        min-height: 112px;
        box-shadow: 0 8px 23px rgba(25,55,95,0.045);
        transition: all 0.2s ease;
    }

    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 13px 28px rgba(25,55,95,0.075);
    }

    .info-icon {
        font-size: 22px;
    }

    .info-label {
        color: #708198;
        font-size: 12px;
        margin-top: 7px;
        font-weight: 650;
    }

    .info-value {
        color: #142b48;
        font-size: 25px;
        font-weight: 850;
        margin-top: 4px;
    }

    /* ---------- INDICATORS ---------- */

    .indicator-card {
        background: linear-gradient(135deg, #edf6ff, #f8fbff);
        border: 1px solid #d7e8fa;
        border-radius: 16px;
        padding: 17px;
        min-height: 115px;
        transition: all 0.2s ease;
    }

    .indicator-card:hover {
        transform: translateY(-3px);
        border-color: #bddaf5;
        box-shadow: 0 10px 25px rgba(30,100,170,0.08);
    }

    .indicator-number {
        color: #1473c9;
        font-size: 11px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .indicator-text {
        color: #1f4264;
        font-size: 14px;
        font-weight: 650;
        margin-top: 11px;
        line-height: 1.45;
    }

    /* ---------- SUMMARY ---------- */

    .summary-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid #e1e9f2;
        border-radius: 19px;
        padding: 23px;
        box-shadow: 0 8px 25px rgba(25,55,95,0.05);
    }

    .summary-label {
        color: #718198;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .summary-value {
        color: #172f4e;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 17px;
    }

    /* ---------- STREAMLIT METRICS ---------- */

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid #e2eaf3;
        padding: 14px;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(25,55,95,0.045);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 700;
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        border-radius: 13px;
        border: 0;
        padding: 0.78rem 1rem;
        font-weight: 750;
        background: linear-gradient(135deg, #1473c9, #0ea5e9);
        color: white;
        box-shadow:
            0 8px 20px rgba(14,116,200,0.22),
            inset 0 1px 0 rgba(255,255,255,0.18);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 12px 27px rgba(14,116,200,0.32),
            inset 0 1px 0 rgba(255,255,255,0.22);
    }

    /* ---------- DATE INPUT ---------- */

    div[data-baseweb="input"] {
        border-radius: 12px !important;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        padding: 35px 0 12px 0;
        color: #8090a5;
        font-size: 12px;
    }

    .footer strong {
        color: #47617d;
    }

    /* ---------- SCROLLBAR ---------- */

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #eef3f8;
    }

    ::-webkit-scrollbar-thumb {
        background: #b7c8da;
        border-radius: 20px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #8ea8c0;
    }

    /* ---------- HIDE STREAMLIT UI ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- RESPONSIVE ---------- */

    @media (max-width: 900px) {
        .hero {
            padding: 27px;
        }

        .hero-title {
            font-size: 31px;
        }

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200, response.json()
    except Exception:
        return False, {}


def get_prediction(selected_date):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"date": selected_date.strftime("%Y-%m-%d")},
            timeout=10,
        )

        if response.status_code == 200:
            return response.json(), None

        try:
            return None, response.json().get("detail", "Prediction failed.")
        except Exception:
            return None, "Prediction API returned an error."

    except requests.exceptions.ConnectionError:
        return None, "Prediction API is offline. Start FastAPI on port 8000."

    except Exception as e:
        return None, str(e)


def risk_details(level):
    if level == "HIGH":
        return {
            "class": "risk-high",
            "pulse": "pulse-high",
            "icon": "🔴",
            "title": "HIGH FLOOD RISK",
        }

    if level == "MODERATE":
        return {
            "class": "risk-moderate",
            "pulse": "pulse-moderate",
            "icon": "🟠",
            "title": "MODERATE FLOOD RISK",
        }

    return {
        "class": "risk-low",
        "pulse": "pulse-low",
        "icon": "🟢",
        "title": "LOW FLOOD RISK",
    }


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


historical_data = load_data()

# ============================================================
# API STATUS
# ============================================================

api_online, health_data = check_api()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🌊</div>
            <div class="sidebar-title">FloodGuard AI</div>
            <div class="sidebar-subtitle">
                Urban Flood Intelligence Platform
            </div>
        </div>

        <div class="sidebar-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">🎯 Prediction Control</div>',
        unsafe_allow_html=True,
    )

    if not historical_data.empty:
        min_date = historical_data["date"].min().date()
        max_date = historical_data["date"].max().date()
        default_date = date(2021, 11, 5)

        selected_date = st.date_input(
            "Analysis Date",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
        )
    else:
        selected_date = st.date_input(
            "Analysis Date",
            value=date(2021, 11, 5),
        )

    analyze = st.button("🔎  Analyze Flood Risk")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-section">⚙️ System Status</div>',
        unsafe_allow_html=True,
    )

    if api_online:
        st.markdown(
            """
            <div class="api-online">
                <span class="api-dot"></span>
                Prediction API Online
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("Prediction API Offline")

    model_name = health_data.get("model_type", "RandomForestClassifier")
    feature_count = health_data.get("features", 52)

    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Machine Learning Model</div>
            <div class="sidebar-stat-value">🌲 {model_name}</div>
        </div>

        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Feature Count</div>
            <div class="sidebar-stat-value">🧠 {feature_count}</div>
        </div>

        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Forecast Horizon</div>
            <div class="sidebar-stat-value">📅 Next 3 Days</div>
        </div>

        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Deployment</div>
            <div class="sidebar-stat-value">⚡ FastAPI + Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HERO
# ============================================================

# ---------- HERO ----------

st.html("""
<div class="hero">
    <div class="hero-top">
        <div class="hero-icon">🌊</div>

        <div>
            <div class="hero-title">FloodGuard AI</div>
            <div class="hero-subtitle">
                Intelligent Urban Flood Prediction &amp; Early Warning System
            </div>
        </div>
    </div>

    <div class="hero-badges">
        <div class="hero-badge">📍 Chennai</div>
        <div class="hero-badge">🌲 Random Forest</div>
        <div class="hero-badge">🧠 52 Features</div>
        <div class="hero-badge">📅 3-Day Forecast</div>
        <div class="hero-badge">⚙️ MLOps Ready</div>
    </div>
</div>
""")

# ============================================================
# TOP KPIs
# ============================================================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        """
        <div class="kpi-card">
            <span class="kpi-icon">📍</span>
            <div class="kpi-label">Location</div>
            <div class="kpi-value">Chennai</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        """
        <div class="kpi-card">
            <span class="kpi-icon">🌲</span>
            <div class="kpi-label">Model</div>
            <div class="kpi-value">Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        """
        <div class="kpi-card">
            <span class="kpi-icon">🧠</span>
            <div class="kpi-label">Features</div>
            <div class="kpi-value">52</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        """
        <div class="kpi-card">
            <span class="kpi-icon">📅</span>
            <div class="kpi-label">Forecast Horizon</div>
            <div class="kpi-value">3 Days</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Flood Risk Assessment</div>',
    unsafe_allow_html=True,
)

prediction = None

if analyze or "prediction" not in st.session_state:

    prediction, error = get_prediction(selected_date)

    if prediction:
        st.session_state["prediction"] = prediction

    elif error:
        st.error(error)

else:
    prediction = st.session_state["prediction"]

# ============================================================
# PREDICTION DISPLAY
# ============================================================

if prediction:

    level = prediction.get("risk_level", "LOW")
    probability = prediction.get("flood_probability_percent", 0)

    details = risk_details(level)

    st.html(
        f"""
        <div class="risk-card {details['class']}">

            <div class="risk-header">
                <span class="risk-pulse {details['pulse']}"></span>
                <span>{details['title']}</span>
            </div>

            <div class="risk-probability">
                Flood probability within the next 3 days:
                <strong>{probability:.2f}%</strong>
            </div>

        </div>
        """
    )

    st.progress(min(probability / 100, 1.0))

    p1, p2 = st.columns(2)

    with p1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Predicted Flood Probability</div>
                <div class="kpi-value">{probability:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Current Risk Level</div>
                <div class="kpi-value">{level}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # ENVIRONMENTAL CONDITIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">🌦️ Environmental Conditions</div>',
        unsafe_allow_html=True,
    )

    weather = prediction.get("weather", {})

    e1, e2, e3, e4 = st.columns(4)

    metrics = [
        ("🌧️", "Rainfall", f"{weather.get('rainfall_mm', 0):.1f} mm"),
        ("🌡️", "Temperature", f"{weather.get('temperature_mean_c', 0):.1f} °C"),
        ("💧", "Humidity", f"{weather.get('humidity_percent', 0):.1f}%"),
        ("💨", "Wind Speed", f"{weather.get('wind_speed_kmh', 0):.1f} km/h"),
    ]

    for col, (icon, label, value) in zip(
        [e1, e2, e3, e4],
        metrics
    ):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-icon">{icon}</div>
                    <div class="info-label">{label}</div>
                    <div class="info-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
# RISK INDICATORS
# ========================================================

st.markdown(
    '<div class="section-title">⚠️ Key Risk Indicators</div>',
    unsafe_allow_html=True,
)

indicators = prediction.get("key_indicators", [])

if indicators:

    cols = st.columns(min(len(indicators), 4))

    for i, indicator in enumerate(indicators):
        with cols[i % len(cols)]:
            st.html(
                f"""
                <div class="indicator-card">
                    <div class="indicator-number">
                        Indicator {i + 1}
                    </div>

                    <div class="indicator-text">
                        {indicator}
                    </div>
                </div>
                """
            )

    # ========================================================
    # RAINFALL TREND
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Rainfall Trend</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">14-day rainfall pattern preceding the selected analysis date.</div>',
        unsafe_allow_html=True,
    )

    if not historical_data.empty:

        start_date = pd.Timestamp(selected_date) - timedelta(days=13)
        end_date = pd.Timestamp(selected_date)

        trend = historical_data[
            (historical_data["date"] >= start_date)
            & (historical_data["date"] <= end_date)
        ][["date", "rainfall_mm"]].copy()

        if not trend.empty:

            trend = trend.set_index("date")

            st.line_chart(
                trend,
                height=320,
                use_container_width=True,
            )

    # ========================================================
# PREDICTION SUMMARY
# ========================================================

st.markdown(
    '<div class="section-title">📋 Prediction Summary</div>',
    unsafe_allow_html=True,
)

predicted = prediction.get("prediction", 0)

prediction_text = (
    "Potential flood risk detected"
    if predicted == 1
    else "No immediate flood event detected"
)

st.markdown(
    f"""<div class="summary-card">
<div class="summary-label">Analysis Date</div>
<div class="summary-value">{prediction.get("date", "-")}</div>

<div class="summary-label">Location</div>
<div class="summary-value">📍 {prediction.get("location", "Chennai")}</div>

<div class="summary-label">Forecast Horizon</div>
<div class="summary-value">🌧️ Flood risk within next 3 days</div>

<div class="summary-label">Model</div>
<div class="summary-value">🌲 Random Forest Classifier</div>

<div class="summary-label">Prediction</div>
<div class="summary-value">{prediction_text}</div>

<div class="summary-label">Risk Level</div>
<div class="summary-value">{level}</div>

<div class="summary-label">Flood Probability</div>
<div class="summary-value">{prediction.get("flood_probability_percent", 0):.2f}%</div>
</div>""",
    unsafe_allow_html=True,
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align: center;">

    ### 🌊 FloodGuard AI

    **Intelligent Urban Flood Prediction & Early Warning System**

    `Random Forest` · `FastAPI` · `Streamlit` · `MLOps`

    *3-Day Flood Risk Forecasting · Chennai Urban Region*

    <br>

    <small>Built for academic research and intelligent flood-risk analysis.</small>

    </div>
    """,
    unsafe_allow_html=True,
)