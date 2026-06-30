import streamlit as st

def inject_styles():
    """Injects premium dark slate and neon CSS rules, custom card behaviors, and micro-animations."""
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    /* Global layout & theme overrides */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid #1e293b !important;
    }

    [data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid #1e293b !important;
    }

    /* Streamlit widget modifications */
    div[data-testid="stMarkdownContainer"] p {
        font-family: 'Outfit', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #f8fafc !important;
    }

    /* Metric modifications */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
        box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.15);
    }

    /* Custom Cards and UI components */
    .eval-card {
        background: linear-gradient(135deg, #0f172a 0%, #172033 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .eval-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #0d9488);
        opacity: 0.7;
    }

    .eval-card:hover {
        transform: translateY(-4px);
        border-color: #3b82f6;
        box-shadow: 0 12px 30px -10px rgba(59, 130, 246, 0.25);
    }

    /* Small/Grid Cards */
    .metric-grid-card {
        background: #0d1321;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    
    .metric-grid-card:hover {
        transform: translateY(-3px);
        border-color: #0d9488;
        box-shadow: 0 8px 25px -10px rgba(13, 148, 136, 0.3);
    }

    /* Glowing Banners */
    .best-config-banner {
        background: linear-gradient(135deg, #062b3d 0%, #081d2a 100%);
        border: 1px solid #0e7490;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 0 20px -3px rgba(14, 116, 144, 0.2);
        animation: pulseGlow 4s infinite alternate ease-in-out;
        position: relative;
    }
    
    .best-config-banner::after {
        content: '';
        position: absolute;
        top: -1px; left: -1px; right: -1px; bottom: -1px;
        border-radius: 12px;
        background: linear-gradient(90deg, #0d9488, #3b82f6, #d97706);
        z-index: -1;
        filter: blur(4px);
        opacity: 0.4;
    }

    /* Pulse Glow Keyframes */
    @keyframes pulseGlow {
        0% {
            box-shadow: 0 0 15px -3px rgba(14, 116, 144, 0.15);
            border-color: #0e7490;
        }
        50% {
            box-shadow: 0 0 25px 2px rgba(14, 116, 144, 0.35);
            border-color: #14b8a6;
        }
        100% {
            box-shadow: 0 0 15px -3px rgba(14, 116, 144, 0.15);
            border-color: #0e7490;
        }
    }

    /* Shimmer effect for skeleton loading */
    .shimmer-bg {
        background: linear-gradient(90deg, #161f30 25%, #24324d 50%, #161f30 75%);
        background-size: 200% 100%;
        animation: shimmerWave 1.6s infinite linear;
        border-radius: 8px;
    }

    @keyframes shimmerWave {
        0% {
            background-position: 200% 0;
        }
        100% {
            background-position: -200% 0;
        }
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-success {
        background-color: rgba(13, 148, 136, 0.15);
        color: #14b8a6;
        border: 1px solid rgba(13, 148, 136, 0.3);
    }
    .badge-warning {
        background-color: rgba(217, 119, 6, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(217, 119, 6, 0.3);
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Code & Technical Data Styles */
    .tech-code {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem;
        color: #38bdf8;
        background: #090d16;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #1e293b;
    }

    /* Interactive Inputs Styling */
    div[data-baseweb="select"] > div {
        background-color: #0d1321 !important;
        border-color: #1e293b !important;
        color: #f8fafc !important;
    }
    div[role="listbox"] {
        background-color: #0d1321 !important;
        color: #f8fafc !important;
    }

    /* Custom Tables */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 0.9rem;
    }
    .custom-table th {
        background-color: #090d16;
        color: #94a3b8;
        text-align: left;
        padding: 12px;
        font-weight: 500;
        border-bottom: 2px solid #1e293b;
        font-family: 'Outfit', sans-serif;
    }
    .custom-table td {
        padding: 12px;
        border-bottom: 1px solid #1e293b;
        color: #cbd5e1;
    }
    .custom-table tr:hover {
        background-color: rgba(59, 130, 246, 0.05);
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0f19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
