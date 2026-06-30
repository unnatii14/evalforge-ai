import streamlit as st
import pandas as pd
from pathlib import Path
from app.services.dataset_service import DatasetService, DatasetValidationError
import app.dashboard.components as comp

DATASETS_DIR = Path("data") / "datasets"

def show():
    st.markdown("<h2 style='margin: 0 0 4px 0; font-size: 1.8rem; color: #f8fafc;'>Dataset Manager</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 24px;'>Upload, inspect, and validate evaluation test sets for RAG and LLM benchmarking.</p>", unsafe_allow_html=True)

    dataset_service = DatasetService()
    loader = st.session_state.loader

    # Create two columns: Left for list & preview, Right for uploading new datasets
    left_col, right_col = st.columns([7, 5], gap="large")

    with left_col:
        st.markdown("<h3 style='font-size: 1.2rem; color: #f8fafc; margin-bottom: 12px;'>Saved Datasets</h3>", unsafe_allow_html=True)
        
        datasets = loader.get_datasets()
        
        if not datasets:
            st.info("No datasets saved yet. Use the upload panel on the right to add a dataset.")
        else:
            selected_dataset_name = st.selectbox("Select Dataset to Preview", datasets)
            
            # Load dataset file
            dataset_path = DATASETS_DIR / f"{selected_dataset_name}.csv"
            
            # Fallback if it's the seed dataset and file does not exist in folder yet
            if not dataset_path.exists() and selected_dataset_name == "halueval_rag_test":
                dataset_path = Path("evaluation/testset.csv")
                
            if dataset_path.exists():
                try:
                    df = pd.read_csv(dataset_path)
                    
                    # Schema Analysis Card
                    st.markdown("<div class='eval-card'>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <h4 style="margin: 0 0 12px 0; font-size: 1rem; color: #f8fafc;">Schema Verification</h4>
                        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px;">
                            <div>
                                <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Row Count</span>
                                <div style="font-size: 1.4rem; font-weight: 700; color: #38bdf8;">{len(df)}</div>
                            </div>
                            <div>
                                <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Column Count</span>
                                <div style="font-size: 1.4rem; font-weight: 700; color: #38bdf8;">{len(df.columns)}</div>
                            </div>
                            <div>
                                <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Status</span>
                                <div style="margin-top: 4px;">{comp.status_badge("SUCCESS")}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Column badges
                    col_status_html = ""
                    for col in df.columns:
                        is_req = col in dataset_service.required_columns
                        is_opt = col in dataset_service.optional_columns
                        if is_req:
                            badge = '<span class="badge badge-success" style="margin-right: 6px;">[REQ] ' + col + '</span>'
                        elif is_opt:
                            badge = '<span class="badge badge-warning" style="margin-right: 6px;">[OPT] ' + col + '</span>'
                        else:
                            badge = '<span class="badge" style="background-color: #1e293b; color: #94a3b8; margin-right: 6px;">[META] ' + col + '</span>'
                        col_status_html += badge
                        
                    st.markdown(f"**Detected Fields:**<br/>{col_status_html}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Sample Preview Table
                    st.markdown("<h4 style='font-size: 1rem; color: #f8fafc; margin-top: 16px; margin-bottom: 8px;'>Sample Rows Preview</h4>", unsafe_allow_html=True)
                    st.dataframe(
                        df.head(10),
                        hide_index=True,
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Failed to load dataset: {str(e)}")
            else:
                st.warning("Selected dataset file was not found in path.")

    with right_col:
        st.markdown("<h3 style='font-size: 1.2rem; color: #f8fafc; margin-bottom: 12px;'>Upload New Dataset</h3>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            # Create a temporary file to run verification on it
            temp_dir = Path("data") / "uploads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            
            try:
                temp_path.write_bytes(uploaded_file.getvalue())
                
                # Check validation using service
                df_temp = pd.read_csv(temp_path)
                dataset_service.validate_dataframe(df_temp)
                
                st.success("✓ Dataset schema is valid! (Required field 'question' found)")
                
                # Show upload validation details
                missing_opts = [col for col in dataset_service.optional_columns if col not in df_temp.columns]
                if missing_opts:
                    st.warning(f"Note: Optional fields {missing_opts} are missing. Evaluators will need to synthesize these runtime contexts/ground truths.")
                
                st.markdown("<hr style='border-color: #1e293b;' />", unsafe_allow_html=True)
                
                # Form to save and rename dataset
                with st.form("persist_dataset_form"):
                    cleaned_name = Path(uploaded_file.name).stem.lower().replace(" ", "_")
                    dataset_name = st.text_input("Save Dataset As", value=cleaned_name)
                    
                    submit_btn = st.form_submit_button("Persist & Register Dataset", use_container_width=True)
                    
                    if submit_btn:
                        res = dataset_service.persist_dataset(temp_path, DATASETS_DIR, dataset_name)
                        st.session_state.loader._ensure_dirs() # Refresh loader state
                        st.success(f"✓ Registered '{res.dataset_name}' with {res.row_count} rows!")
                        st.rerun()
            except DatasetValidationError as dve:
                st.error(f"Validation Error: {str(dve)}")
                st.markdown(
                    """
                    <div style="background-color: rgba(239, 68, 68, 0.1); border: 1px dashed #ef4444; border-radius: 8px; padding: 12px; font-size: 0.85rem; color: #f87171;">
                        <b>Requirements:</b><br/>
                        1. Must be a comma-separated values (.csv) file.<br/>
                        2. Must contain at least a <b>question</b> column.<br/>
                        3. Optional columns: <b>ground_truth</b>, <b>context</b>, <b>expected_answer</b>.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error validating file: {str(e)}")
            finally:
                # Cleanup temp file
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
        else:
            # Empty state upload help
            st.markdown(
                """
                <div class="eval-card" style="border-style: dashed; border-width: 1px; padding: 24px; text-align: center; margin-top: 12px;">
                    <div style="font-size: 2.2rem; margin-bottom: 8px;">📤</div>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">
                        Upload a CSV containing your evaluation prompts. You can download a standard template below to format your data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Download Template
            template_df = pd.DataFrame(columns=["question", "ground_truth", "context", "expected_answer"])
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Template CSV",
                data=csv_template,
                file_name="evalforge_template.csv",
                mime="text/csv",
                use_container_width=True
            )
