import streamlit as st

def best_config_banner(provider: str, model_name: str, chunk_size: int, top_k: int, faithfulness: float, relevancy: float):
    """Renders a pulsing, modern banner for the best RAG configuration found so far."""
    html_content = f"""
    <div class="best-config-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="badge badge-success" style="margin-bottom: 8px;">★ Best Configuration</span>
                <h3 style="margin: 4px 0 0 0; font-size: 1.4rem; color: #f8fafc;">
                    {provider.upper()} / <span class="tech-code">{model_name}</span>
                </h3>
                <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #94a3b8;">
                    Parameters: Chunk Size: <b>{chunk_size}</b> | Chunk Overlap: <b>16</b> | Top-k: <b>{top_k}</b>
                </p>
            </div>
            <div style="display: flex; gap: 24px; margin-top: 10px;">
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Faithfulness</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #14b8a6;">{faithfulness:.2f}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Answer Relevancy</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #14b8a6;">{relevancy:.2f}</div>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def metric_grid_card(title: str, value: str, trend: str = None, trend_type: str = "success", subtitle: str = None):
    """Renders a compact card for key metrics in the Overview Grid."""
    trend_html = ""
    if trend:
        badge_class = "badge-success" if trend_type == "success" else "badge-warning" if trend_type == "warning" else "badge-danger"
        trend_html = f'<span class="badge {badge_class}" style="margin-top: 4px;">{trend}</span>'
        
    subtitle_html = f'<p style="margin: 4px 0 0 0; font-size: 0.75rem; color: #64748b;">{subtitle}</p>' if subtitle else ""

    html_content = f"""
    <div class="metric-grid-card">
        <div>
            <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-top: 6px; font-family: 'Outfit', sans-serif;">{value}</div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px; min-height: 24px;">
            {trend_html}
            {subtitle_html}
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def status_badge(status: str) -> str:
    """Returns HTML for a success, warning, or danger badge based on status text."""
    status_lower = status.lower()
    if status_lower in ["success", "completed", "passed", "active", "ok"]:
        return f'<span class="badge badge-success">{status}</span>'
    elif status_lower in ["warning", "pending", "running", "moderate"]:
        return f'<span class="badge badge-warning">{status}</span>'
    else:
        return f'<span class="badge badge-danger">{status}</span>'

def empty_state(title: str, message: str, icon: str = "⚡"):
    """Renders a high-contrast empty state placeholder when no runs or data exist."""
    html_content = f"""
    <div class="eval-card" style="text-align: center; padding: 48px 24px; border-style: dashed; border-width: 2px;">
        <div style="font-size: 3rem; margin-bottom: 16px;">{icon}</div>
        <h4 style="margin: 0; font-size: 1.25rem; color: #f8fafc;">{title}</h4>
        <p style="margin: 8px auto 0 auto; max-width: 320px; font-size: 0.875rem; color: #94a3b8;">{message}</p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def shimmer_loader(height: str = "80px", margin: str = "10px 0"):
    """Renders a shimmer wave loading skeleton."""
    html_content = f"""
    <div class="shimmer-bg" style="height: {height}; margin: {margin}; width: 100%;"></div>
    """
    st.markdown(html_content, unsafe_allow_html=True)
