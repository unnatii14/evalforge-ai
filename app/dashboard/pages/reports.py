import streamlit as st
import pandas as pd
from pathlib import Path
from app.reports.report_service import ReportService
from app.models.schemas import (
    ReportRequest, 
    ReportFormat, 
    EvaluationResponse, 
    EvaluationMetricResult, 
    BenchmarkResponse, 
    BenchmarkRow
)
import app.dashboard.components as comp

REPORTS_DIR = Path("data") / "reports"

def show():
    st.markdown("<h2 style='margin: 0 0 4px 0; font-size: 1.8rem; color: #f8fafc;'>Reports & Export</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 24px;'>Generate structured PDF digests, export raw metric CSVs, or package JSON logs for external BI tools.</p>", unsafe_allow_html=True)

    loader = st.session_state.loader
    all_runs = loader.get_runs()

    if not all_runs:
        comp.empty_state(
            title="No Data to Package",
            message="Perform evaluations or import runs to enable document rendering services.",
            icon="📄"
        )
        return

    # Two columns: Left for configuration form, Right for list of downloaded historical reports
    left_col, right_col = st.columns([6, 6], gap="large")

    with left_col:
        st.markdown("<h3 style='font-size: 1.25rem; color: #f8fafc; margin-bottom: 12px;'>Document Builder</h3>", unsafe_allow_html=True)
        
        with st.form("document_generator_form"):
            report_type = st.radio("Report Source", ["Single Evaluation Run", "Aggregated Benchmark Leaderboard"])
            
            # Contextual inputs
            if report_type == "Single Evaluation Run":
                run_options = [r["id"] for r in reversed(all_runs)]
                selected_run_id = st.selectbox(
                    "Select Run ID", 
                    run_options, 
                    format_func=lambda x: f"Run #{x} - {next(r['model_name'] for r in all_runs if r['id'] == x)} ({next(r['dataset_name'] for r in all_runs if r['id'] == x)})"
                )
            else:
                selected_run_id = None
                
            report_name = st.text_input("Filename (without extension)", value="evalforge_digest")
            report_format_val = st.selectbox("Export Format", [f.value for f in ReportFormat], index=2) # Default PDF
            
            st.markdown("<br/>", unsafe_allow_html=True)
            generate_btn = st.form_submit_button("🔨 Build Report Document", use_container_width=True)

        if generate_btn:
            report_service = ReportService()
            fmt = ReportFormat(report_format_val)
            
            # Construct payload
            eval_resp = None
            bench_resp = None
            
            if report_type == "Single Evaluation Run" and selected_run_id:
                # Fetch database record
                run_rec = loader.repo.get_run(selected_run_id)
                if run_rec:
                    meta = run_rec.metadata or {}
                    scores = meta.get("scores", {})
                    
                    # Convert to EvaluationMetricResult schemas
                    metrics_schemas = [
                        EvaluationMetricResult(name=k, score=float(v))
                        for k, v in scores.items()
                    ]
                    
                    eval_resp = EvaluationResponse(
                        run=run_rec,
                        metrics=metrics_schemas,
                        latency_seconds=meta.get("latency_seconds", 0.0),
                        cost_usd=meta.get("cost_usd", 0.0)
                    )
            else:
                # Build BenchmarkResponse
                df_runs = pd.DataFrame(all_runs)
                df_runs["config_label"] = df_runs["model_name"]
                
                rows = []
                for _, r in df_runs.iterrows():
                    meta = r.get("metadata", {}) or {}
                    scores = meta.get("scores", {})
                    metrics_schemas = [
                        EvaluationMetricResult(name=k, score=float(v))
                        for k, v in scores.items()
                    ]
                    
                    # Estimate overall score
                    overall = sum(scores.values()) / len(scores) if scores else 0.0
                    
                    rows.append(BenchmarkRow(
                        scenario_name=r["dataset_name"],
                        model_name=r["model_name"],
                        provider=r["provider"],
                        overall_score=overall,
                        latency_seconds=r["latency_seconds"],
                        metric_scores=metrics_schemas
                    ))
                
                best_scenario_name = max(all_runs, key=lambda r: (r["faithfulness"] + r["answer_relevancy"]) / 2.0)["dataset_name"]
                
                bench_resp = BenchmarkResponse(
                    benchmark_name="evalforge_global_benchmark",
                    rows=rows,
                    best_scenario_name=best_scenario_name
                )

            # Build Request
            req = ReportRequest(
                report_name=report_name,
                format=fmt,
                evaluation=eval_resp,
                benchmark=bench_resp
            )

            try:
                res = report_service.generate(req)
                st.success(f"✓ Report '{res.report_name}' generated successfully!")
                
                # Load file bytes for Streamlit downloader
                filepath = Path(res.file_path)
                if filepath.exists():
                    file_bytes = filepath.read_bytes()
                    mime_type = "application/pdf" if fmt == ReportFormat.pdf else "text/csv" if fmt == ReportFormat.csv else "application/json"
                    
                    st.download_button(
                        label=f"📥 Download {fmt.value.upper()} Report",
                        data=file_bytes,
                        file_name=filepath.name,
                        mime=mime_type,
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Failed to generate report: {str(e)}")

    with right_col:
        st.markdown("<h3 style='font-size: 1.25rem; color: #f8fafc; margin-bottom: 12px;'>Export Repository</h3>", unsafe_allow_html=True)
        
        # List files in data/reports directory
        reports_dir = REPORTS_DIR
        reports_dir.mkdir(parents=True, exist_ok=True)
        files = list(reports_dir.glob("*.*"))
        
        if not files:
            comp.empty_state(
                title="No Documents Rendered",
                message="Newly compiled documents will appear in this repository list.",
                icon="🗂️"
            )
        else:
            # Table of generated reports
            rep_rows = ""
            for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                size_kb = f.stat().st_size / 1024.0
                mod_time = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                ext_badge = "badge-success" if f.suffix == ".pdf" else "badge-warning" if f.suffix == ".json" else "badge-danger"
                
                rep_rows += f"""
                <tr>
                    <td style="font-weight: 500; color: #f8fafc;">{f.stem}</td>
                    <td><span class="badge {ext_badge}">{f.suffix[1:].upper()}</span></td>
                    <td>{size_kb:.1f} KB</td>
                    <td>{mod_time}</td>
                </tr>
                """
                
            reports_table_html = f"""
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>REPORT NAME</th>
                        <th>FORMAT</th>
                        <th>SIZE</th>
                        <th>COMPILED AT</th>
                    </tr>
                </thead>
                <tbody>
                    {rep_rows}
                </tbody>
            </table>
            """
            st.markdown(reports_table_html, unsafe_allow_html=True)
