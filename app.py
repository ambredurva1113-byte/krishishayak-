"""
KrishiSahayak — Agricultural Risk Advisory Dashboard
Run with: streamlit run app.py

Data: real Maharashtra district-crop-year records (1999-2017), sourced from
Runax15/crop-yield-prediction-maharashtra on GitHub. See README.md and
01_generate_data.py for full source documentation and methodology.
"""
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from reference_data import DISTRICT_COORDS

st.set_page_config(page_title="KrishiSahayak | Agricultural Risk Advisory",
                    page_icon=None, layout="wide")

# ---------------------------------------------------------------------------
# Professional dark theme styling (no emoji, muted palette)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .risk-banner {
        padding: 20px 28px; border-radius: 6px; margin: 12px 0 20px 0;
        border-left: 5px solid; font-family: 'Segoe UI', sans-serif;
    }
    .risk-banner-low { background-color: #10241a; border-color: #2E7D32; }
    .risk-banner-medium { background-color: #241f10; border-color: #C99A2E; }
    .risk-banner-high { background-color: #2a1414; border-color: #B23A3A; }
    .risk-title { font-size: 15px; letter-spacing: 1px; text-transform: uppercase;
                  color: #9AA0A6; margin-bottom: 6px; }
    .risk-value { font-size: 28px; font-weight: 600; }
    .risk-value-low { color: #4CAF50; }
    .risk-value-medium { color: #D9A93B; }
    .risk-value-high { color: #D9534F; }
    .risk-note { color: #B0B4B9; font-size: 14px; margin-top: 8px; }
    .section-header { font-size: 13px; letter-spacing: 1.5px; text-transform: uppercase;
                       color: #8A8F98; border-bottom: 1px solid #2A2E37; padding-bottom: 6px;
                       margin: 24px 0 12px 0; }
    .alert-row { padding: 10px 14px; border-radius: 4px; margin-bottom: 6px;
                 font-size: 14px; border-left: 3px solid; }
    .alert-high { background-color: #221515; border-color: #B23A3A; color: #E0B4B4; }
    .alert-medium { background-color: #221E12; border-color: #C99A2E; color: #E0D2A9; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    le = joblib.load("label_encoder.pkl")
    features = joblib.load("feature_list.pkl")
    return model, le, features


@st.cache_data
def load_data():
    return pd.read_csv("krishisahayak_dataset.csv")


model, le, FEATURES = load_artifacts()
df = load_data()

# ---------------------------------------------------------------------------
# Language toggle
# ---------------------------------------------------------------------------
LANG = st.sidebar.radio("Language / भाषा", ["English", "मराठी"])

TXT = {
    "English": dict(
        title="KrishiSahayak", subtitle="Agricultural Risk Advisory System — Maharashtra",
        details="Plot Details", select_district="District",
        select_crop="Crop", select_year="Assessment Year",
        prediction="Distress Risk Assessment", low="LOW RISK", medium="MEDIUM RISK", high="HIGH RISK",
        low_note="Yield and rainfall are close to this crop's normal range here.",
        medium_note="Some stress signals present. Monitor conditions closely.",
        high_note="Strong stress signals present relative to this crop's recent history.",
        recommendations="Contributing Factors", map_title="Regional Risk Overview — Maharashtra",
        trend_title="Yield History — Last 10 Years", drivers="Model Confidence by Category",
        alerts_title="District-Crop Advisory Alerts", confidence="Model Confidence",
        no_data="No record for this district-crop-year combination in the dataset.",
        source_note="Data source: real Maharashtra district agricultural statistics (1999–2017). "
                     "See README for full citation and methodology.",
    ),
    "मराठी": dict(
        title="कृषीसहायक", subtitle="कृषी जोखीम सल्ला प्रणाली — महाराष्ट्र",
        details="शेत तपशील", select_district="जिल्हा",
        select_crop="पीक", select_year="मूल्यांकन वर्ष",
        prediction="संकट जोखीम मूल्यांकन", low="कमी जोखीम", medium="मध्यम जोखीम", high="उच्च जोखीम",
        low_note="उत्पादन आणि पाऊस या पिकाच्या सामान्य श्रेणीजवळ आहेत.",
        medium_note="काही ताण निर्देशक आढळले आहेत. परिस्थितीचे बारकाईने निरीक्षण करा.",
        high_note="या पिकाच्या अलीकडील इतिहासाच्या तुलनेत तीव्र ताण निर्देशक आढळले आहेत.",
        recommendations="कारणीभूत घटक", map_title="प्रादेशिक जोखीम आढावा — महाराष्ट्र",
        trend_title="उत्पादन इतिहास — गेली 10 वर्षे", drivers="श्रेणीनुसार मॉडेल विश्वासार्हता",
        alerts_title="जिल्हा-पीक सल्ला सूचना", confidence="मॉडेल विश्वासार्हता",
        no_data="या जिल्हा-पीक-वर्ष संयोजनासाठी डेटासेटमध्ये नोंद नाही.",
        source_note="डेटा स्रोत: वास्तविक महाराष्ट्र जिल्हा कृषी आकडेवारी (1999–2017).",
    ),
}
t = TXT[LANG]

RISK_CLASS = {"Low": "low", "Medium": "medium", "High": "high"}
RISK_HEX = {"Low": "#4CAF50", "Medium": "#D9A93B", "High": "#D9534F"}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(t["title"])
st.caption(t["subtitle"])

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.markdown(f'<div class="section-header">{t["details"]}</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    district = st.selectbox(t["select_district"], sorted(df["district"].unique()))
available_crops = sorted(df.loc[df["district"] == district, "crop"].unique())
with c2:
    crop = st.selectbox(t["select_crop"], available_crops)
available_years = sorted(
    df.loc[(df["district"] == district) & (df["crop"] == crop), "year"].unique(), reverse=True)
with c3:
    year = st.selectbox(t["select_year"], available_years)

row = df[(df["district"] == district) & (df["crop"] == crop) & (df["year"] == year)]

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.markdown(f'<div class="section-header">{t["prediction"]}</div>', unsafe_allow_html=True)

if row.empty:
    st.info(t["no_data"])
else:
    X_row = row[FEATURES]
    pred = model.predict(X_row)[0]
    proba = model.predict_proba(X_row)[0]
    risk_label = le.inverse_transform([pred])[0]
    confidence = proba[pred]

    note_map = {"Low": t["low_note"], "Medium": t["medium_note"], "High": t["high_note"]}
    st.markdown(f"""
    <div class="risk-banner risk-banner-{RISK_CLASS[risk_label]}">
        <div class="risk-title">{t['prediction']} — {district} / {crop.title()} ({year})</div>
        <div class="risk-value risk-value-{RISK_CLASS[risk_label]}">{t[risk_label.lower()]}</div>
        <div class="risk-note">{note_map[risk_label]}</div>
        <div class="risk-note">{t['confidence']}: {confidence:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f'<div class="section-header">{t["recommendations"]}</div>', unsafe_allow_html=True)
        factor_labels = {
            "area_1000ha": "Area sown (1000 ha)",
            "rainfall_mm": "Rainfall this year (mm)",
            "avg_temp_c": "Average temperature (°C)",
            "rainfall_deviation_pct": "Rainfall deviation from district normal (%)",
            "prev_year_yield": "Previous year's yield (kg/ha)",
            "prior_3yr_avg_yield": "Prior 3-year average yield (kg/ha)",
            "prior_3yr_yield_volatility": "Prior 3-year yield volatility",
        }
        factor_df = row[FEATURES].T.reset_index()
        factor_df.columns = ["Factor", "Value"]
        factor_df["Factor"] = factor_df["Factor"].map(factor_labels)
        factor_df["Value"] = factor_df["Value"].round(2)
        st.dataframe(factor_df, use_container_width=True, hide_index=True, height=280)

    with col_right:
        st.markdown(f'<div class="section-header">{t["drivers"]}</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Risk Category": le.classes_, "Probability": proba})
        fig_prob = go.Figure(go.Bar(
            x=prob_df["Probability"], y=prob_df["Risk Category"], orientation="h",
            marker_color=[RISK_HEX[r] for r in prob_df["Risk Category"]],
            text=[f"{p:.0%}" for p in prob_df["Probability"]], textposition="outside",
        ))
        fig_prob.update_layout(
            height=180, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(range=[0, 1], showgrid=False, visible=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B0B4B9"),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown(f'<div class="section-header">{t["trend_title"]}</div>', unsafe_allow_html=True)
        history = df[(df["district"] == district) & (df["crop"] == crop)].sort_values("year")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=history["year"], y=history["yield_kg_per_ha"],
            mode="lines+markers", line=dict(color="#5B8DEF", width=2), marker=dict(size=5),
        ))
        fig_trend.update_layout(
            height=200, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B0B4B9"),
            xaxis=dict(gridcolor="#262B36"), yaxis=dict(gridcolor="#262B36", title="Yield (kg/ha)"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------------------------------------------------------
# Regional risk map (for the selected crop, latest available year)
# ---------------------------------------------------------------------------
st.markdown(f'<div class="section-header">{t["map_title"]}</div>', unsafe_allow_html=True)

crop_subset = df[df["crop"] == crop]
latest_year_for_crop = crop_subset["year"].max()
overview = crop_subset[crop_subset["year"] == latest_year_for_crop].copy()
overview["predicted_risk"] = le.inverse_transform(model.predict(overview[FEATURES]))
overview["lat"] = overview["district"].map(lambda d: DISTRICT_COORDS.get(d, (None, None))[0])
overview["lon"] = overview["district"].map(lambda d: DISTRICT_COORDS.get(d, (None, None))[1])
overview = overview.dropna(subset=["lat", "lon"])

if not overview.empty:
    fig_map = px.scatter_mapbox(
        overview, lat="lat", lon="lon", color="predicted_risk",
        color_discrete_map=RISK_HEX, hover_name="district",
        hover_data={"lat": False, "lon": False, "predicted_risk": True},
        zoom=5.4, height=440, size=[14] * len(overview), size_max=14,
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#B0B4B9"), title=None),
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(f"Showing {crop.title()} risk across districts for {latest_year_for_crop} "
               f"(most recent year with data for this crop).")

# ---------------------------------------------------------------------------
# Advisory alerts panel
# ---------------------------------------------------------------------------
st.markdown(f'<div class="section-header">{t["alerts_title"]}</div>', unsafe_allow_html=True)

high_risk = overview[overview["predicted_risk"] == "High"].sort_values("district")
medium_risk = overview[overview["predicted_risk"] == "Medium"].sort_values("district")

for _, r in high_risk.head(5).iterrows():
    st.markdown(f"""<div class="alert-row alert-high">
        <strong>{r['district']}</strong> — {crop.title()}: high risk. Rainfall deviation
        {r['rainfall_deviation_pct']:.0f}% vs. district normal.
    </div>""", unsafe_allow_html=True)

for _, r in medium_risk.head(3).iterrows():
    st.markdown(f"""<div class="alert-row alert-medium">
        <strong>{r['district']}</strong> — {crop.title()}: medium risk. Monitor rainfall trend.
    </div>""", unsafe_allow_html=True)

if high_risk.empty and medium_risk.empty:
    st.caption("No elevated-risk districts for this crop in the latest available year.")

st.markdown("---")
st.caption(t["source_note"])
