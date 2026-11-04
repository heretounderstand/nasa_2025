"""Exoplanet Hunter – entry point.  Run with:  streamlit run app.py"""
import streamlit as st

from views.common import inject_css

st.set_page_config(page_title="Exoplanet Hunter", page_icon="🪐", layout="wide")
inject_css()

pages = st.navigation(
    {
        "Explore": [
            st.Page("views/overview.py", title="Overview", icon=":material/public:", default=True),
            st.Page("views/candidates.py", title="Candidate explorer", icon=":material/travel_explore:"),
            st.Page("views/classify.py", title="Vet a signal", icon=":material/insights:"),
        ],
        "Model": [
            st.Page("views/performance.py", title="Performance", icon=":material/monitoring:"),
            st.Page("views/lab.py", title="Training lab", icon=":material/science:"),
        ],
    }
)

with st.sidebar:
    st.markdown("### 🪐 Exoplanet Hunter")
    st.caption(
        "NASA Space Apps 2025  \n*A World Away: Hunting for Exoplanets with AI*  \n\n"
        "Data: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu) · TOI table"
    )

pages.run()
