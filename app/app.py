import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="Delivery Time Estimator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Clean CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header */
    .system-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .system-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 1.75rem;
    }
    
    /* Primary ETA card */
    .prediction-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 28px 32px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .card-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 8px;
    }
    .eta-display {
        font-size: 3.6rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .eta-unit {
        font-size: 1.35rem;
        font-weight: 500;
        color: #64748b;
        margin-left: 6px;
    }
    
    /* Confidence Window Tag */
    .window-tag {
        display: inline-block;
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        color: #334155;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 16px;
    }
    .priority-tag {
        display: inline-block;
        background-color: #e0f2fe;
        border: 1px solid #7dd3fc;
        color: #0369a1;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
        margin-left: 8px;
    }
    
    /* Visible Toggle Styling */
    div[data-testid="stToggle"] {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 12px 0 16px 0;
    }
    div[data-testid="stToggle"] label {
        font-weight: 600 !important;
        color: #1e293b !important;
        font-size: 0.92rem !important;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 14px;
        letter-spacing: -0.01em;
    }
    
    /* Summary Card */
    .summary-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.92rem;
    }
    .summary-row:last-child {
        border-bottom: none;
    }
    .summary-label {
        color: #64748b;
    }
    .summary-value {
        font-weight: 600;
        color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model Pipeline Artifact
@st.cache_resource
def load_trained_artifact():
    candidate_paths = [
        'models/best_delivery_model.pkl',
        '../models/best_delivery_model.pkl',
        '/Users/rizwansalmani/Desktop/ML_Main_Project/models/best_delivery_model.pkl'
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return joblib.load(p)
    return None

artifact = load_trained_artifact()

# Top Title
st.markdown('<div class="system-title">Delivery Time Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="system-subtitle">Calculate arrival estimates and expected delivery windows based on current conditions.</div>', unsafe_allow_html=True)

if artifact is None:
    st.error("Prediction engine not initialized. Please ensure model artifact is loaded.")
    st.stop()

model = artifact.get('model', artifact.get('pipeline'))
scaler = artifact.get('scaler', None)
features = artifact.get('features', None)
model_mae = float(artifact.get('mae', 5.0))

# Sidebar Controls
with st.sidebar:
    st.subheader("Order Parameters")
    
    preset = st.selectbox(
        "Order Profile",
        ["Custom Settings", "Short Distance Route", "Peak Hour Congestion", "Late Night Route"]
    )
    
    if preset == "Short Distance Route":
        default_dist, default_prep, default_traffic, default_weather, default_tod, default_rating = (
            3.5, 18, "Medium", "Sunny", "Afternoon", 4.7
        )
    elif preset == "Peak Hour Congestion":
        default_dist, default_prep, default_traffic, default_weather, default_tod, default_rating = (
            14.0, 32, "Jam", "Stormy", "Evening", 4.1
        )
    elif preset == "Late Night Route":
        default_dist, default_prep, default_traffic, default_weather, default_tod, default_rating = (
            5.0, 15, "Low", "Clear", "Night", 4.8
        )
    else:
        default_dist, default_prep, default_traffic, default_weather, default_tod, default_rating = (
            7.5, 20, "Medium", "Clear", "Evening", 4.5
        )

    st.markdown("---")
    st.caption("Route and Preparation")
    distance_km = st.slider("Delivery Distance (km)", min_value=0.5, max_value=25.0, value=float(default_dist), step=0.1)
    prep_time_min = st.slider("Kitchen Preparation Time (minutes)", min_value=5, max_value=50, value=int(default_prep), step=1)

    st.caption("Current Conditions")
    traffic_density = st.selectbox(
        "Traffic Level",
        options=["Low", "Medium", "High", "Jam"],
        index=["Low", "Medium", "High", "Jam"].index(default_traffic)
    )
    weather_condition = st.selectbox(
        "Weather Condition",
        options=["Clear", "Sunny", "Cloudy", "Foggy", "Rainy", "Stormy"],
        index=["Clear", "Sunny", "Cloudy", "Foggy", "Rainy", "Stormy"].index(default_weather)
    )

    st.caption("Delivery Details")
    time_of_day = st.selectbox(
        "Time of Day",
        options=["Morning", "Afternoon", "Evening", "Night"],
        index=["Morning", "Afternoon", "Evening", "Night"].index(default_tod)
    )
    partner_rating = st.slider("Partner Rating", min_value=3.0, max_value=5.0, value=float(default_rating), step=0.05)

    st.markdown("---")
    # Visible Sidebar Toggle: Priority Dispatch
    priority_mode = st.toggle("Priority Order Dispatch", value=False)

# Main Layout
col1, col2 = st.columns([1.15, 0.85], gap="large")

# Build input DataFrame
input_df = pd.DataFrame([{
    'distance_km': distance_km,
    'prep_time_min': prep_time_min,
    'traffic_density': traffic_density,
    'weather_condition': weather_condition,
    'time_of_day': time_of_day,
    'delivery_partner_rating': partner_rating
}])

# Compute Baseline Prediction
if scaler is not None and features is not None:
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    predicted_eta = float(model.predict(input_scaled)[0])
else:
    predicted_eta = float(model.predict(input_df)[0])

# Adjust for Priority Dispatch if toggled
base_transit = max(2.0, predicted_eta - prep_time_min)
if priority_mode:
    # Priority dispatch reduces transit buffer by 15% through dedicated courier routing
    adjusted_transit = max(2.0, base_transit * 0.85)
    final_eta = prep_time_min + adjusted_transit
else:
    adjusted_transit = base_transit
    final_eta = predicted_eta

lower_bound = max(10.0, round(final_eta - model_mae, 1))
upper_bound = round(final_eta + model_mae, 1)

with col1:
    # Primary Prediction Display Card
    priority_badge_html = '<span class="priority-tag">Priority Courier Active</span>' if priority_mode else ''
    
    st.markdown(f"""
        <div class="prediction-card">
            <div class="card-label">Estimated Delivery Time</div>
            <div class="eta-display">{final_eta:.0f}<span class="eta-unit">minutes</span></div>
            <div>
                <span class="window-tag">Expected Arrival Window: {lower_bound:.0f} to {upper_bound:.0f} minutes</span>
                {priority_badge_html}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Visible Page Toggle: Show/Hide Route Breakdown
    show_details = st.toggle("Show Route Allocation Details", value=True)
    
    if show_details:
        st.markdown('<div class="section-title">Time Allocation Breakdown</div>', unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Kitchen Preparation", f"{prep_time_min} min")
        with m_col2:
            st.metric("Estimated Transit", f"{adjusted_transit:.0f} min")
        with m_col3:
            st.metric("Expected Variation", f"&plusmn;{model_mae:.0f} min")

with col2:
    # Order Parameters Summary
    st.markdown('<div class="section-title">Order Summary</div>', unsafe_allow_html=True)
    dispatch_type_label = "Priority (Expedited)" if priority_mode else "Standard"
    
    st.markdown(f"""
        <div class="summary-card">
            <div class="summary-row">
                <span class="summary-label">Dispatch Mode</span>
                <span class="summary-value">{dispatch_type_label}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Distance</span>
                <span class="summary-value">{distance_km:.1f} km</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Preparation Time</span>
                <span class="summary-value">{prep_time_min} min</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Traffic Density</span>
                <span class="summary-value">{traffic_density}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Weather Condition</span>
                <span class="summary-value">{weather_condition}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Time of Day</span>
                <span class="summary-value">{time_of_day}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Partner Rating</span>
                <span class="summary-value">{partner_rating:.2f} / 5.0</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 8px 0;">
    Delivery Time Estimator
</div>
""", unsafe_allow_html=True)
