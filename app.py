import json
import streamlit as st
from extractor_engine import DashboardExtractor

st.set_page_config(layout="wide")
st.title("Newton SkillUP KPI Extractor (DSPy Optimized)")

pipeline = DashboardExtractor()
try:
    pipeline.load('gemini_optimized_dashboard_state.json')
except Exception:
    st.warning(
        "Optimized pipeline not found. Run `python compile_and_test.py` first to "
        "generate `gemini_optimized_dashboard_state.json`, then refresh this page."
    )

transcript = st.text_area(
    "Meeting Transcript",
    value=(
        "Final quarter push. We hit $900k in revenue! "
        "Signups are booming at 12,000. Churn is incredibly low at 0.5%."
    ),
    height=150,
)

if st.button("Extract KPIs"):
    with st.spinner("Extracting parameters..."):
        result = pipeline(raw_transcript=transcript)

    st.subheader("Extracted KPIs")
    try:
        cleaned = result.json_kpis.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        st.json(parsed)
    except Exception:
        st.code(result.json_kpis)

    with st.expander("🔍 View DSPy Reasoning (Chain of Thought)"):
        st.write(result.reasoning)