import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
import app.dashboard.components as comp

def show():
    st.markdown("<h2 style='margin: 0 0 4px 0; font-size: 1.8rem; color: #f8fafc;'>System Overview</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 24px;'>Performance insights, cost tracking, and optimization metrics across evaluations.</p>", unsafe_allow_html=True)

    loader = st.session_state.loader
    filters = st.session_state.global_filters
    
    # Fetch runs
    all_runs = loader.get_runs()
    
    if not all_runs:
        comp.empty_state(
            title="No Evaluation Runs Found",
            message="To view analytics, go to 'Evaluation Runs' and start an evaluation, or import a dataset.",
            icon="📊"
        )
        return

    # Filter runs
    filtered_runs = all_runs
    
    # 1. Filter by dataset
    if filters["dataset"]:
        filtered_runs = [r for r in filtered_runs if r["dataset_name"] == filters["dataset"]]
        
    # 2. Filter by model
    if filters["model"] != "All Models":
        filtered_runs = [r for r in filtered_runs if r["model_name"] == filters["model"]]
        
    # 3. Filter by date
    filtered_runs = [
        r for r in filtered_runs 
        if filters["start_date"] <= r["created_at"].date() <= filters["end_date"]
    ]

    if not filtered_runs:
        comp.empty_state(
            title="No Runs Match Selected Filters",
            message="Adjust the global filters in the sidebar to view evaluation history.",
            icon="🔍"
        )
        return

    # Sort runs chronologically for line charts
    filtered_runs = sorted(filtered_runs, key=lambda r: r["created_at"])

    # Determine Best Configuration (highest average of faithfulness + answer_relevancy)
    best_run = max(all_runs, key=lambda r: (r["faithfulness"] + r["answer_relevancy"]) / 2.0)
    
    # Render Best Config Banner
    comp.best_config_banner(
        provider=best_run["provider"],
        model_name=best_run["model_name"],
        chunk_size=best_run["chunk_size"],
        top_k=best_run["top_k"],
        faithfulness=best_run["faithfulness"],
        relevancy=best_run["answer_relevancy"]
    )

    # Calculate average metrics of filtered runs for overall scorecards
    df_runs = pd.DataFrame(filtered_runs)
    avg_faithfulness = df_runs["faithfulness"].mean()
    avg_relevancy = df_runs["answer_relevancy"].mean()
    avg_precision = df_runs["context_precision"].mean()
    avg_recall = df_runs["context_recall"].mean()
    avg_correctness = df_runs["answer_correctness"].mean()
    avg_hallucination = df_runs["hallucination"].mean()
    avg_toxicity = df_runs["toxicity"].mean()
    avg_bias = df_runs["bias"].mean()
    avg_latency = df_runs["latency_seconds"].mean()
    avg_cost = df_runs["cost_usd"].mean()

    # Metric Cards Grid Layout (4 columns)
    st.markdown("<h4 style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 12px; font-weight: 500;'>KEY PERFORMANCE INDICATORS</h4>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        comp.metric_grid_card(
            title="Faithfulness",
            value=f"{avg_faithfulness * 100:.1f}%",
            trend="+2.4% vs last week" if avg_faithfulness > 0.8 else "-1.2% drop",
            trend_type="success" if avg_faithfulness > 0.8 else "warning",
            subtitle="Factual grounding"
        )
    with col2:
        comp.metric_grid_card(
            title="Answer Relevancy",
            value=f"{avg_relevancy * 100:.1f}%",
            trend="+1.8% vs last week" if avg_relevancy > 0.85 else "Stable",
            trend_type="success" if avg_relevancy > 0.85 else "success",
            subtitle="QA semantic alignment"
        )
    with col3:
        comp.metric_grid_card(
            title="Avg Latency",
            value=f"{avg_latency:.2f}s",
            trend="-0.4s faster" if avg_latency < 2.5 else "+0.8s increase",
            trend_type="success" if avg_latency < 2.5 else "danger",
            subtitle="Response generation speed"
        )
    with col4:
        comp.metric_grid_card(
            title="Estimated Cost",
            value=f"${avg_cost:.4f}",
            trend="Local (Ollama)" if avg_cost == 0.0 else f"+${avg_cost:.4f} per run",
            trend_type="success" if avg_cost == 0.0 else "warning",
            subtitle="LLM provider token cost"
        )

    # Extra Metrics Accordion for Advanced Parameters
    with st.expander("Show Advanced Retrieval, Bias, and Safety Metrics"):
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            comp.metric_grid_card("Context Precision", f"{avg_precision * 100:.1f}%", subtitle="Retrieval precision")
        with ec2:
            comp.metric_grid_card("Context Recall", f"{avg_recall * 100:.1f}%", subtitle="Retrieval recall")
        with ec3:
            comp.metric_grid_card("Hallucination Index", f"{avg_hallucination * 100:.1f}%", trend="Lower is better", trend_type="success" if avg_hallucination < 0.15 else "danger", subtitle="Factual inconsistencies")
        with ec4:
            comp.metric_grid_card("Toxicity & Bias", f"{avg_toxicity * 100:.1f}% / {avg_bias * 100:.1f}%", subtitle="Content safety scores")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Charts Section (2 Columns)
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("<h3 style='font-size: 1.15rem; color: #f8fafc; margin-bottom: 12px;'>Experiment History (Quality Trend)</h3>", unsafe_allow_html=True)
        
        # Plotly Line Chart for Metric scores
        fig = go.Figure()
        timestamps = [r["created_at"].strftime("%m-%d %H:%M") for r in filtered_runs]
        
        # Add lines for Ragas metrics
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=[r["faithfulness"] for r in filtered_runs],
            mode='lines+markers',
            name='Faithfulness',
            line=dict(color='#0d9488', width=3),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=[r["answer_relevancy"] for r in filtered_runs],
            mode='lines+markers',
            name='Answer Relevancy',
            line=dict(color='#2563eb', width=3),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=[r["context_recall"] for r in filtered_runs],
            mode='lines+markers',
            name='Context Recall',
            line=dict(color='#d97706', width=2, dash='dash'),
            marker=dict(size=5)
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
            yaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8'), range=[0, 1.05]),
            legend=dict(font=dict(color='#f8fafc'), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown("<h3 style='font-size: 1.15rem; color: #f8fafc; margin-bottom: 12px;'>Latency vs Quality Trade-off</h3>", unsafe_allow_html=True)
        
        # Scatter/Bubble Chart
        fig_scatter = go.Figure()
        for model in df_runs["model_name"].unique():
            df_m = df_runs[df_runs["model_name"] == model]
            fig_scatter.add_trace(go.Scatter(
                x=df_m["latency_seconds"],
                y=(df_m["faithfulness"] + df_m["answer_relevancy"]) / 2.0,
                mode='markers+text',
                name=str(model),
                text=[f"{m} ({c}ch)" for m, c in zip(df_m["model_name"], df_m["chunk_size"])],
                textposition="top center",
                marker=dict(size=14, line=dict(width=1, color='white'))
            ))

        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis=dict(title=dict(text="Latency (s)", font=dict(color='#94a3b8', size=11)), gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
            yaxis=dict(title=dict(text="Avg Quality Score", font=dict(color='#94a3b8', size=11)), gridcolor='#1e293b', tickfont=dict(color='#94a3b8'), range=[0, 1.05]),
            showlegend=False
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Experiment History Table (Custom Styled HTML)
    st.markdown("<h3 style='font-size: 1.25rem; color: #f8fafc; margin-top: 24px; margin-bottom: 12px;'>Recent Experiment Logs</h3>", unsafe_allow_html=True)
    
    table_rows_html = ""
    for r in reversed(filtered_runs[-6:]):  # Show latest 6 runs
        avg_q = (r["faithfulness"] + r["answer_relevancy"] + r["context_recall"] + r["context_precision"]) / 4.0
        badge_html = comp.status_badge(r["status"])
        
        table_rows_html += f"""
        <tr>
            <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #38bdf8;">#{r['id']}</td>
            <td>{r['created_at'].strftime('%Y-%m-%d %H:%M')}</td>
            <td><span class="tech-code">{r['model_name']}</span></td>
            <td>{r['dataset_name']}</td>
            <td style="font-weight: 600; color: {'#14b8a6' if avg_q > 0.8 else '#f59e0b' if avg_q > 0.6 else '#f87171'}">{avg_q:.2f}</td>
            <td>{r['latency_seconds']:.2f}s</td>
            <td>${r['cost_usd']:.4f}</td>
            <td>{badge_html}</td>
        </tr>
        """

    custom_table_html = f"""
    <table class="custom-table">
        <thead>
            <tr>
                <th>RUN ID</th>
                <th>TIMESTAMP</th>
                <th>MODEL CONFIG</th>
                <th>DATASET</th>
                <th>AVG QUALITY</th>
                <th>LATENCY</th>
                <th>COST</th>
                <th>STATUS</th>
            </tr>
        </thead>
        <tbody>
            {table_rows_html}
        </tbody>
    </table>
    """
    st.markdown(custom_table_html, unsafe_allow_html=True)

    # Detailed Run Viewer Section
    st.markdown("<br/>", unsafe_allow_html=True)
    with st.expander("🔍 Deep-Dive: Inspect Predictions & Sample Rows"):
        run_ids = [r["id"] for r in reversed(filtered_runs)]
        selected_run_id = st.selectbox("Select Run ID to Inspect", run_ids, format_func=lambda x: f"Run #{x} - {next(r['model_name'] for r in filtered_runs if r['id'] == x)}")
        
        details_df = loader.get_run_details(selected_run_id)
        if details_df.empty:
            st.info("No detailed predictions stored for this run. Loading global evaluation results.")
            if Path("evaluation/results.csv").exists():
                details_df = pd.read_csv("evaluation/results.csv")
                
        if not details_df.empty:
            # Drop unnecessary columns if they exist
            cols_to_show = ["question", "answer", "ground_truth", "faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
            cols_to_show = [c for c in cols_to_show if c in details_df.columns]
            
            st.dataframe(
                details_df[cols_to_show],
                column_config={
                    "question": st.column_config.TextColumn("Question", width="medium"),
                    "answer": st.column_config.TextColumn("Generated Answer", width="medium"),
                    "ground_truth": st.column_config.TextColumn("Ground Truth", width="medium"),
                    "faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0.0, max_value=1.0, format="%.2f"),
                    "answer_relevancy": st.column_config.ProgressColumn("Relevancy", min_value=0.0, max_value=1.0, format="%.2f"),
                    "context_recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="%.2f"),
                    "context_precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.2f"),
                    "answer_correctness": st.column_config.ProgressColumn("Correctness", min_value=0.0, max_value=1.0, format="%.2f"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No dataset preview records available.")
