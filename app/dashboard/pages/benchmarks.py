import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import app.dashboard.components as comp

def show():
    st.markdown("<h2 style='margin: 0 0 4px 0; font-size: 1.8rem; color: #f8fafc;'>Benchmark Comparison</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 24px;'>Compare configurations, rank model capabilities, and analyze multi-dimensional retrieval trade-offs.</p>", unsafe_allow_html=True)

    loader = st.session_state.loader
    all_runs = loader.get_runs()

    if not all_runs:
        comp.empty_state(
            title="No Data for Benchmarking",
            message="Run multiple evaluations to generate comparison tables and charts.",
            icon="🏆"
        )
        return

    # Group runs by model and chunk parameters to aggregate performance
    df = pd.DataFrame(all_runs)
    
    # Create group column
    df["config_label"] = df["model_name"] + " (ch:" + df["chunk_size"].astype(str) + ")"
    
    # Calculate average scores per configuration
    metrics_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
    
    grouped = df.groupby(["config_label", "model_name", "provider"]).agg({
        "faithfulness": "mean",
        "answer_relevancy": "mean",
        "context_precision": "mean",
        "context_recall": "mean",
        "answer_correctness": "mean",
        "latency_seconds": "mean",
        "cost_usd": "mean",
        "id": "count"
    }).reset_index()

    # Calculate overall quality score as average of key metrics
    grouped["overall_score"] = grouped[metrics_cols].mean(axis=1)
    
    # Rank configurations by overall score
    grouped = grouped.sort_values(by="overall_score", ascending=False).reset_index(drop=True)
    grouped["rank"] = grouped.index + 1

    # Render Leaderboard Card
    st.markdown("<h3 style='font-size: 1.25rem; color: #f8fafc; margin-bottom: 12px;'>🏆 Model Configurations Leaderboard</h3>", unsafe_allow_html=True)
    
    leaderboard_rows = ""
    for _, row in grouped.iterrows():
        rank_badge = f'<span style="font-weight: 700; color: #14b8a6;"># {row["rank"]}</span>' if row["rank"] == 1 else f'# {row["rank"]}'
        provider_badge = comp.status_badge("SUCCESS") if row["provider"] == "ollama" else comp.status_badge("WARNING")
        
        leaderboard_rows += f"""
        <tr>
            <td style="text-align: center;">{rank_badge}</td>
            <td style="font-weight: 600;">{row["config_label"]}</td>
            <td><span class="tech-code">{row["provider"].upper()}</span></td>
            <td style="font-weight: 700; color: #14b8a6;">{row["overall_score"]:.2f}</td>
            <td>{row["faithfulness"]:.2f}</td>
            <td>{row["answer_relevancy"]:.2f}</td>
            <td>{row["context_precision"]:.2f}</td>
            <td>{row["context_recall"]:.2f}</td>
            <td>{row["latency_seconds"]:.2f}s</td>
            <td>${row["cost_usd"]:.4f}</td>
            <td style="text-align: center;"><span class="badge badge-success">{row["id"]} runs</span></td>
        </tr>
        """

    leaderboard_html = f"""
    <table class="custom-table">
        <thead>
            <tr>
                <th style="width: 60px; text-align: center;">RANK</th>
                <th>CONFIGURATION</th>
                <th>PROVIDER</th>
                <th>OVERALL SCORE</th>
                <th>FAITHFULNESS</th>
                <th>RELEVANCY</th>
                <th>PRECISION</th>
                <th>RECALL</th>
                <th>LATENCY</th>
                <th>COST</th>
                <th style="text-align: center;">SAMPLE SIZE</th>
            </tr>
        </thead>
        <tbody>
            {leaderboard_rows}
        </tbody>
    </table>
    """
    st.markdown(leaderboard_html, unsafe_allow_html=True)

    st.markdown("<br/><hr style='border-color: #1e293b;' /><br/>", unsafe_allow_html=True)

    # Multi-select models to compare
    st.markdown("<h3 style='font-size: 1.25rem; color: #f8fafc; margin-bottom: 12px;'>📊 Side-by-Side Analysis</h3>", unsafe_allow_html=True)
    
    configs_to_compare = st.multiselect(
        "Select configurations to plot:",
        options=grouped["config_label"].tolist(),
        default=grouped["config_label"].tolist()[:3]
    )

    if not configs_to_compare:
        st.warning("Please select at least one configuration to view comparison charts.")
        return

    compare_df = grouped[grouped["config_label"].isin(configs_to_compare)]

    col_radar, col_bar = st.columns(2, gap="large")

    with col_radar:
        st.markdown("<h4 style='font-size: 1.05rem; color: #94a3b8; margin-bottom: 8px;'>Radar Profile (Multi-Dimensional Quality)</h4>", unsafe_allow_html=True)
        
        # Plotly Radar Chart
        fig_radar = go.Figure()
        categories = ["Faithfulness", "Relevancy", "Precision", "Recall", "Correctness"]
        
        # Color palette for comparison
        colors = ["#2563eb", "#0d9488", "#d97706", "#ef4444", "#8b5cf6"]

        for idx, (_, row) in enumerate(compare_df.iterrows()):
            color = colors[idx % len(colors)]
            r_values = [
                row["faithfulness"],
                row["answer_relevancy"],
                row["context_precision"],
                row["context_recall"],
                row["answer_correctness"]
            ]
            # Radar loop back to start value
            r_values.append(r_values[0])
            categories_loop = categories + [categories[0]]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values,
                theta=categories_loop,
                fill='toself',
                name=row["config_label"],
                line=dict(color=color, width=2),
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)" if len(color) == 7 else "rgba(59, 130, 246, 0.15)"
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1e293b', linecolor='#1e293b', tickfont=dict(color='#94a3b8')),
                angularaxis=dict(gridcolor='#1e293b', linecolor='#1e293b', tickfont=dict(color='#f8fafc', size=11))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=20, b=20),
            height=340,
            legend=dict(font=dict(color='#f8fafc'), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bar:
        st.markdown("<h4 style='font-size: 1.05rem; color: #94a3b8; margin-bottom: 8px;'>Metric Scores Comparison</h4>", unsafe_allow_html=True)
        
        # Grouped Bar Chart
        fig_bar = go.Figure()
        
        categories_short = ["Faithfulness", "Relevancy", "Precision", "Recall", "Correctness"]
        
        for idx, (_, row) in enumerate(compare_df.iterrows()):
            color = colors[idx % len(colors)]
            fig_bar.add_trace(go.Bar(
                x=categories_short,
                y=[
                    row["faithfulness"],
                    row["answer_relevancy"],
                    row["context_precision"],
                    row["context_recall"],
                    row["answer_correctness"]
                ],
                name=row["config_label"],
                marker_color=color
            ))

        fig_bar.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            height=340,
            xaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#f8fafc')),
            yaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8'), range=[0, 1.05]),
            legend=dict(font=dict(color='#f8fafc'), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
