from __future__ import annotations

import requests
import pandas as pd
import streamlit as st


st.set_page_config(page_title="PURR", page_icon="RW", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .hero {
        background: linear-gradient(120deg, #001f3f 0%, #004a7c 45%, #1f8a70 100%);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .small-note {color: #dbe9f4; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
    <h2 style="margin:0;">PURR Requirement Analysis Workbench</h2>
      <div class="small-note">Table-first workflow: each analyzed requirement is appended as a new row.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "rows" not in st.session_state:
    st.session_state.rows = []

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API Base URL", value="http://127.0.0.1:8000")
    top_k_context = st.slider("RAG top-k context", min_value=0, max_value=8, value=3)
    include_rewrite = st.checkbox("Enable LLM rewrite", value=False, help="Can be slow on CPU. Leave off for fast analysis.")

    st.subheader("Knowledge Base (PDF)")
    pdf = st.file_uploader("Upload requirements PDF", type=["pdf"])
    if st.button("Ingest PDF", use_container_width=True):
        if pdf is None:
            st.warning("Choose a PDF first.")
        else:
            try:
                files = {"file": (pdf.name, pdf.read(), "application/pdf")}
                resp = requests.post(f"{api_base}/rag/upload-pdf", files=files, timeout=120)
                resp.raise_for_status()
                st.success("PDF ingested successfully.")
            except Exception as exc:
                st.error(f"Failed to ingest PDF: {exc}")

    if st.button("Clear Table", use_container_width=True):
        st.session_state.rows = []

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Single Requirement")
    requirement_text = st.text_area("Requirement", placeholder="Enter one requirement statement...")
    if st.button("Analyze and Add to Table", type="primary"):
        if len(requirement_text.strip()) < 3:
            st.warning("Enter a requirement with at least 3 characters.")
        else:
            try:
                payload = {
                    "requirement": requirement_text.strip(),
                    "top_k_context": top_k_context,
                    "include_rewrite": include_rewrite,
                }
                timeout_seconds = 420 if include_rewrite else 120
                resp = requests.post(f"{api_base}/analyze/text", json=payload, timeout=timeout_seconds)
                resp.raise_for_status()
                result = resp.json()
                st.session_state.rows.append(
                    {
                        "Requirement": result["requirement"],
                        "Class": result["predicted_class"],
                        "Subtype": result["predicted_subtype"],
                        "Ambiguity": result["predicted_ambiguity"],
                        "Rewrite": result.get("rewrite") or "",
                        "Class Confidence": round(float(result["confidence_class"]), 4),
                        "Ambiguity Confidence": round(float(result["confidence_ambiguity"]), 4),
                        "Explain Terms": ", ".join(result.get("explain_terms", [])),
                    }
                )
                st.success("Requirement added to table.")
            except Exception as exc:
                st.error(f"Analyze failed: {exc}")

with right:
    st.subheader("Batch CSV (max 50 per run)")
    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if st.button("Run Batch and Append", use_container_width=True):
        if batch_file is None:
            st.warning("Upload a CSV file first.")
        else:
            try:
                files = {"file": (batch_file.name, batch_file.read(), "text/csv")}
                data = {
                    "top_k_context": top_k_context,
                    "include_rewrite": str(include_rewrite).lower(),
                }
                timeout_seconds = 900 if include_rewrite else 300
                resp = requests.post(f"{api_base}/analyze/batch", files=files, data=data, timeout=timeout_seconds)
                resp.raise_for_status()
                payload = resp.json()
                for result in payload.get("results", []):
                    st.session_state.rows.append(
                        {
                            "Requirement": result["requirement"],
                            "Class": result["predicted_class"],
                            "Subtype": result["predicted_subtype"],
                            "Ambiguity": result["predicted_ambiguity"],
                            "Rewrite": result.get("rewrite") or "",
                            "Class Confidence": round(float(result["confidence_class"]), 4),
                            "Ambiguity Confidence": round(float(result["confidence_ambiguity"]), 4),
                            "Explain Terms": ", ".join(result.get("explain_terms", [])),
                        }
                    )
                st.success(f"Added {payload.get('count', 0)} rows to the table.")
            except Exception as exc:
                st.error(f"Batch failed: {exc}")

st.subheader("Analysis Table")
if st.session_state.rows:
    table_df = pd.DataFrame(st.session_state.rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True, height=480)
    csv_bytes = table_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Table as CSV", data=csv_bytes, file_name="reqwise_analysis_results.csv", mime="text/csv")
else:
    st.info("No rows yet. Analyze a requirement to start building the table.")
