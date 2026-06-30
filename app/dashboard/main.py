import sys
from pathlib import Path

# Add the project root to sys.path dynamically so that 'app' can be found
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import datetime
from app.dashboard.styles import inject_styles
from app.dashboard.data_loader import DashboardDataLoader

# Set page config FIRST before any other Streamlit calls
st.set_page_config(
    page_title="EvalForge | AI Evaluation Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom styles & keyframes
inject_styles()

# Initialize data loader in session state
if "loader" not in st.session_state:
    st.session_state.loader = DashboardDataLoader()

# Import pages (we will implement show() for each page)
import app.dashboard.pages.overview as overview
import app.dashboard.pages.benchmarks as benchmarks
import app.dashboard.pages.dataset_manager as dataset_manager
import app.dashboard.pages.eval_runs as eval_runs
import app.dashboard.pages.reports as reports
import app.dashboard.pages.settings as settings

# Sidebar Branding
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2px;">
        <span style="font-size: 1.8rem;">⚡</span>
        <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800; background: linear-gradient(90deg, #3b82f6, #0d9488); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            EvalForge
        </h2>
    </div>
    <div style="font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace; margin-bottom: 24px; padding-left: 2px;">
        v0.1.0 // DEVELOPER PLATFORM
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation setup using callables
pg = st.navigation({
    "Dashboard": [
        st.Page(overview.show, title="Overview", icon="📊", url_path="overview"),
        st.Page(benchmarks.show, title="Benchmark Comparison", icon="⚔️", url_path="benchmarks"),
    ],
    "Assets & Runs": [
        st.Page(dataset_manager.show, title="Dataset Manager", icon="📂", url_path="dataset_manager"),
        st.Page(eval_runs.show, title="Evaluation Runs", icon="🚀", url_path="eval_runs"),
    ],
    "Analytics & Config": [
        st.Page(reports.show, title="Reports & Export", icon="📥", url_path="reports"),
        st.Page(settings.show, title="Settings", icon="⚙️", url_path="settings"),
    ]
})

# Sidebar Global Filters
st.sidebar.markdown("<hr style='border-color: #1e293b; margin: 16px 0;' />", unsafe_allow_html=True)
st.sidebar.subheader("Global Filters")

datasets = st.session_state.loader.get_datasets()
selected_dataset = st.sidebar.selectbox("Active Dataset", datasets)

runs = st.session_state.loader.get_runs()
models = list(set([r["model_name"] for r in runs])) if runs else ["llama3.2:3b", "gpt-4o-mini", "gemini-1.5-flash", "mistral-7b-instruct"]
selected_model = st.sidebar.selectbox("Active Model Config", ["All Models"] + sorted(models))

available_metrics = [
    "Faithfulness", "Answer Relevancy", "Context Precision", 
    "Context Recall", "Answer Correctness", "Hallucination", "Toxicity", "Bias"
]
selected_metrics = st.sidebar.multiselect("Active Metrics", available_metrics, default=available_metrics[:5])

# Date range selection
today = datetime.date.today()
last_week = today - datetime.timedelta(days=7)
date_range = st.sidebar.date_input("Date Range", [last_week, today])

# Handle date input unpack safely (user might click only one date)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range
    end_date = start_date

# Save selected filters to session state for pages to access
st.session_state.global_filters = {
    "dataset": selected_dataset,
    "model": selected_model,
    "metrics": selected_metrics,
    "start_date": start_date,
    "end_date": end_date
}

# Sidebar footer
st.sidebar.markdown("<hr style='border-color: #1e293b; margin: 24px 0;' />", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style="font-size: 0.7rem; color: #475569; font-family: 'JetBrains Mono', monospace; text-align: center;">
        CONNECTED TO SQLITE & LOCAL LLM
    </div>
    """,
    unsafe_allow_html=True
)

# Run navigation
pg.run()
