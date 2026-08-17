import os
import sys
import time

# Ensure parent and current directory are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

try:
    from frontend.config import PAGE_TITLE, PAGE_ICON, BACKEND_URL
    from frontend.utils import (
        check_backend_health,
        upload_document,
        execute_query,
        fetch_metrics,
        fetch_documents,
        delete_document,
        reset_database
    )
except ImportError:
    from config import PAGE_TITLE, PAGE_ICON, BACKEND_URL
    from utils import (
        check_backend_health,
        upload_document,
        execute_query,
        fetch_metrics,
        fetch_documents,
        delete_document,
        reset_database
    )

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .citation-badge {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
    }
    .latency-pill-fast {
        background-color: #dcfce7;
        color: #15803d;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        display: inline-block;
    }
    .latency-pill-slow {
        background-color: #fee2e2;
        color: #b91c1c;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        display: inline-block;
    }
    .chunk-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Check backend health
is_backend_online = check_backend_health()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/database.png", width=60)
    st.title("System Controls")
    
    # Backend status
    if is_backend_online:
        st.success("🟢 Backend Connected (Port 8000)")
    else:
        st.error(f"🔴 Backend Offline at {BACKEND_URL}")
        st.info("Start the backend with: `uvicorn backend.app.main:app --port 8000`")

    st.markdown("---")
    st.subheader("1. Ingestion & Chunking")
    uploaded_files = st.file_uploader(
        "Upload Unstructured Context Files",
        type=["txt", "md", "pdf", "docx", "json", "csv"],
        accept_multiple_files=True
    )
    
    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        ["recursive", "sliding_window", "fixed"],
        index=0,
        help="Recursive preserves semantic paragraphs/sentences; Sliding Window preserves boundary tokens."
    )
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=50)
    with col_c2:
        chunk_overlap = st.slider("Overlap", min_value=0, max_value=500, value=100, step=25)

    if st.button("📥 Index Uploaded Files", use_container_width=True, disabled=not uploaded_files):
        with st.spinner("Processing and indexing documents..."):
            for f in uploaded_files:
                try:
                    res = upload_document(
                        file_bytes=f.getvalue(),
                        filename=f.name,
                        strategy=chunk_strategy,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    st.toast(f"✅ {f.name}: {res['message']}", icon="📄")
                except Exception as e:
                    st.error(f"Failed to ingest {f.name}: {e}")
            st.rerun()

    st.markdown("---")
    st.subheader("2. Hybrid Retrieval Parameters")
    
    hybrid_alpha = st.slider(
        "Sparse (BM25) vs Dense (Vector) Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0.0 = Pure BM25 Keyword Search | 0.5 = Balanced Hybrid | 1.0 = Pure Dense Vector Search"
    )
    st.caption(f"Weights: **BM25 {(1.0-hybrid_alpha):.2f}** | **Dense {hybrid_alpha:.2f}**")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        top_k_dense = st.number_input("Top-K Dense", min_value=1, max_value=50, value=10)
        top_k_sparse = st.number_input("Top-K BM25", min_value=1, max_value=50, value=10)
    with col_k2:
        top_k_rerank = st.number_input("Top-K Rerank", min_value=1, max_value=20, value=5)
        enable_reranker = st.checkbox("Cross-Encoder Reranker", value=True)

    use_cache = st.checkbox("⚡ KV Query Cache", value=True, help="Enable sub-second LRU query caching")

    st.markdown("---")
    st.subheader("3. Generation Engine")
    llm_provider = st.selectbox(
        "LLM Provider",
        ["local_extractive", "gemini", "openai", "huggingface"],
        index=0,
        help="Choose generation backend. 'local_extractive' runs locally with sub-second latency and zero API keys."
    )
    custom_api_key = None
    if llm_provider in ["gemini", "openai"]:
        custom_api_key = st.text_input(f"{llm_provider.capitalize()} API Key", type="password")


# Main Layout Header
st.markdown('<div class="main-header">⚡ Hybrid Retrieval RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">FastAPI • ChromaDB • PyTorch • BM25 • Cross-Encoder Reranking • p50/p95 Latency Tracking</div>', unsafe_allow_html=True)

# Tabs
tab_chat, tab_metrics, tab_docs = st.tabs(["💬 Hybrid RAG Chat", "📊 System Telemetry & p50/p95 Metrics", "📁 Document Management"])

# ----------------- TAB 1: HYBRID RAG CHAT -----------------
with tab_chat:
    # Display conversation messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "latencies" in msg:
                tot_lat = msg["latencies"].get("total_pipeline_ms", 0.0)
                pill_class = "latency-pill-fast" if tot_lat < 1000 else "latency-pill-slow"
                hit_badge = "⚡ [Cache Hit]" if msg.get("cache_hit") else ""
                st.markdown(f'<span class="{pill_class}">⏱ {tot_lat} ms {hit_badge}</span> Groundedness: **{int(msg.get("groundedness", 1.0)*100)}%**', unsafe_allow_html=True)

    # Chat Input
    query_input = st.chat_input("Ask a question grounded across the uploaded documents...")
    
    if query_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": query_input})
        with st.chat_message("user"):
            st.markdown(query_input)

        with st.chat_message("assistant"):
            with st.spinner("Executing hybrid retrieval and cross-encoder reranking..."):
                try:
                    result = execute_query(
                        query=query_input,
                        top_k_dense=top_k_dense,
                        top_k_sparse=top_k_sparse,
                        top_k_rerank=top_k_rerank,
                        hybrid_alpha=hybrid_alpha,
                        enable_reranker=enable_reranker,
                        use_cache=use_cache,
                        llm_provider=llm_provider,
                        custom_api_key=custom_api_key
                    )
                    
                    st.session_state.last_query_result = result
                    answer_text = result["answer"]
                    st.markdown(answer_text)

                    # Latency & Groundedness Badge
                    tot_lat = result["latencies"]["total_pipeline_ms"]
                    pill_class = "latency-pill-fast" if tot_lat < 1000 else "latency-pill-slow"
                    cache_label = "⚡ [Cache Hit]" if result["cache_hit"] else ""
                    st.markdown(f'<span class="{pill_class}">⏱ {tot_lat} ms {cache_label}</span> Groundedness: **{int(result["groundedness_score"]*100)}%**', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "latencies": result["latencies"],
                        "cache_hit": result["cache_hit"],
                        "groundedness": result["groundedness_score"]
                    })

                except Exception as e:
                    st.error(f"Query execution failed: {e}")

    # Deep-Dive Inspection Drawer for the last query
    if st.session_state.last_query_result:
        res = st.session_state.last_query_result
        st.markdown("---")
        with st.expander("🔍 Deep Dive: Retrieved Chunks, Cross-Encoder Scoring & Latency Breakdown", expanded=False):
            col_l1, col_l2 = st.columns([1, 1])
            
            with col_l1:
                st.subheader("⏱ Stage Latencies")
                lats = res["latencies"]
                df_lat = pd.DataFrame({
                    "Pipeline Stage": ["Embedding", "Dense Search (ChromaDB)", "Sparse Search (BM25)", "Hybrid Fusion", "Cross-Encoder Rerank", "Generation", "Total Pipeline"],
                    "Latency (ms)": [
                        lats.get("embedding_ms", 0.0),
                        lats.get("dense_search_ms", 0.0),
                        lats.get("sparse_search_ms", 0.0),
                        lats.get("fusion_ms", 0.0),
                        lats.get("rerank_ms", 0.0),
                        lats.get("generation_ms", 0.0),
                        lats.get("total_pipeline_ms", 0.0)
                    ]
                })
                st.dataframe(df_lat, hide_index=True, use_container_width=True)

            with col_l2:
                st.subheader("📚 Verified Citations")
                if res.get("citations"):
                    for cit in res["citations"]:
                        st.markdown(f"""
                        <div class="chunk-card">
                            <b>Citation {cit['label']}</b> — <i>{cit['filename']}</i> (Score: {cit['relevance_score']})<br/>
                            <small style="color: #475569;">"{cit['snippet']}"</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No citations generated.")

            st.subheader(f"📑 Top Reranked Context Chunks ({len(res.get('retrieved_chunks', []))})")
            for chunk in res.get("retrieved_chunks", []):
                st.markdown(f"""
                <div class="chunk-card">
                    <div style="display:flex; justify-content:space-between;">
                        <b>Rank #{chunk['final_rank']}: {chunk['filename']} [Chunk {chunk['chunk_index']}]</b>
                        <span>
                            <b>Dense:</b> {chunk.get('dense_score', 0.0)} | 
                            <b>BM25:</b> {chunk.get('sparse_score', 0.0)} | 
                            <b>Fused:</b> {chunk.get('fused_score', 0.0)} | 
                            <span style="color:#0284c7; font-weight:bold;">Rerank: {chunk.get('rerank_score', 0.0)}</span>
                        </span>
                    </div>
                    <p style="margin-top:8px; font-size:0.92rem; color:#1e293b;">{chunk['text']}</p>
                </div>
                """, unsafe_allow_html=True)


# ----------------- TAB 2: SYSTEM TELEMETRY & METRICS -----------------
with tab_metrics:
    st.subheader("📈 Real-Time Latency Percentiles & Pipeline Analytics")
    
    if st.button("🔄 Refresh Telemetry"):
        st.rerun()

    try:
        metrics_data = fetch_metrics()
        
        # Top KPI cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        pct = metrics_data["latency_percentiles"]
        cache_m = metrics_data["cache_metrics"]
        
        with col_m1:
            st.metric("Total Queries", metrics_data["total_queries"])
        with col_m2:
            st.metric("p50 Latency (Median)", f"{pct['p50_ms']} ms")
        with col_m3:
            st.metric("p95 Latency", f"{pct['p95_ms']} ms", delta="Sub-second" if pct['p95_ms'] < 1000 else "High")
        with col_m4:
            st.metric("p99 Latency", f"{pct['p99_ms']} ms")
        with col_m5:
            st.metric("Cache Hit Rate", f"{cache_m['hit_rate_pct']}%", f"{cache_m['hits']} hits / {cache_m['misses']} misses")

        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("⚡ Latency Percentile Distribution")
            df_pct = pd.DataFrame({
                "Percentile": ["p50 (Median)", "p90", "p95", "p99", "Average"],
                "Latency (ms)": [pct["p50_ms"], pct["p90_ms"], pct["p95_ms"], pct["p99_ms"], pct["avg_ms"]]
            })
            st.bar_chart(df_pct.set_index("Percentile"))

        with col_g2:
            st.subheader("⏱ Average Stage Breakdown")
            stage_avg = metrics_data["stage_breakdown_avg"]
            df_stages = pd.DataFrame({
                "Stage": list(stage_avg.keys()),
                "Avg Latency (ms)": list(stage_avg.values())
            })
            st.dataframe(df_stages, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("💾 Index & Cache Statistics")
        st.write(f"• **Total Indexed Documents:** {metrics_data['total_indexed_documents']}")
        st.write(f"• **Total Vector/BM25 Chunks:** {metrics_data['total_indexed_chunks']}")
        st.write(f"• **Active Cache Entries:** {cache_m['total_entries']} / {cache_m['max_size']}")

    except Exception as e:
        st.warning(f"Could not load telemetry metrics: {e}")


# ----------------- TAB 3: DOCUMENT MANAGEMENT -----------------
with tab_docs:
    st.subheader("📚 Indexed Document Registry")
    
    if st.button("🔄 Refresh Document List"):
        st.rerun()

    try:
        docs_res = fetch_documents()
        docs = docs_res.get("documents", [])
        
        if docs:
            df_docs = pd.DataFrame([
                {
                    "Document ID": d["doc_id"],
                    "Filename": d["filename"],
                    "Chunks Indexed": d["num_chunks"],
                    "Timestamp": d.get("created_at", "N/A"),
                    "File Size (Bytes)": d.get("metadata", {}).get("file_size_bytes", "N/A")
                }
                for d in docs
            ])
            st.dataframe(df_docs, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.subheader("🗑 Manage Documents")
            doc_to_delete = st.selectbox("Select document to delete", [f"{d['filename']} ({d['doc_id']})" for d in docs])
            if st.button("❌ Delete Selected Document"):
                sel_id = doc_to_delete.split("(")[-1].replace(")", "").strip()
                res = delete_document(sel_id)
                st.success(res["message"])
                st.rerun()

        else:
            st.info("No documents currently indexed. Upload files in the sidebar to begin.")

        st.markdown("---")
        if st.button("⚠️ Reset Entire Database & Clear All Indices"):
            res = reset_database()
            st.warning("All vector stores, BM25 indices, caches, and telemetry have been reset.")
            st.session_state.messages = []
            st.session_state.last_query_result = None
            st.rerun()

    except Exception as e:
        st.error(f"Failed to fetch documents: {e}")
