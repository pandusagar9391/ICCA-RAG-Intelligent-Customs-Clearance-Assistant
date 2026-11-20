"""
ICCA-RAG: Intelligent Customs Clearance Assistant Web Application.

A professional Streamlit interface for:
- Customs compliance Q&A chat with strict grounding citations and anti-hallucination controls
- Multi-modal document ingestion & indexing (PDF, DOCX, TXT, CSV, Excel)
- Knowledge Graph Explorer for HS codes, regulatory acts, and clearance requirements
- LLM provider management (Offline Deterministic Synthesizer, Google Gemini, OpenAI)

Project Team:
- K Dhanush (1602-23-733-142)
- B Gautham (1602-23-733-143)
- B Pandu Sagar (1602-23-733-161)
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:
    print(
        "Streamlit is not installed in the current environment.\n"
        "Please run: pip install streamlit\n"
        "Then launch the app with: streamlit run icca/ui/app.py"
    )
    sys.exit(1)

from icca.config import default_config, ICCAConfig
from icca.coordinator import CustomsClearanceCoordinator

# Streamlit version-safe full-width keyword argument (use width='stretch' for modern Streamlit)
import inspect
_button_sig = inspect.signature(st.button)
FULL_WIDTH = {"width": "stretch"} if "width" in _button_sig.parameters else {"use_container_width": True}


# ---------------------------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ICCA-RAG | Customs Clearance Assistant",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Styles */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 24px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0 0 8px 0;
        font-size: 2.1rem;
        font-weight: 700;
    }
    .main-header p {
        color: #cfd8dc;
        margin: 0;
        font-size: 1.05rem;
    }
    .team-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(4px);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-top: 12px;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .confidence-mid {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .provider-pill {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .source-box {
        background: #fdfdfe;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 18px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------
@st.cache_resource
def get_coordinator() -> CustomsClearanceCoordinator:
    """Initialize and return the central coordinator singleton."""
    coord = CustomsClearanceCoordinator()
    coord.initialize()
    return coord


coordinator = get_coordinator()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_docs_count" not in st.session_state:
    st.session_state.indexed_docs_count = 0

if "indexed_chunks_count" not in st.session_state:
    st.session_state.indexed_chunks_count = 0

if "corpus_ingested" not in st.session_state:
    # Auto-ingest sample corpus on first launch if available
    try:
        res = coordinator.ingest_documents()
        if res.get("status") == "success":
            st.session_state.indexed_chunks_count = res.get("chunks_count", 0)
            st.session_state.corpus_ingested = True
    except Exception as e:
        st.session_state.corpus_ingested = False


# ---------------------------------------------------------------------------
# Header Component
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🚢 ICCA-RAG: Intelligent Customs Clearance Assistant</h1>
        <p>Knowledge-Graph Augmented Hybrid Retrieval-Augmented Generation for Cross-Border Customs Compliance</p>
       <!-- <div class="team-badge">
            👥 <b>Project Investigators:</b> K Dhanush (1602-23-733-142) &bull; B Gautham (1602-23-733-146) &bull; B Pandu Sagar (1602-23-733-157)
        </div> -->
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Sidebar: System Status & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Active LLM Provider Status
    current_provider = coordinator.config.llM_provider.upper()
    if coordinator.compliance_agent and hasattr(coordinator.compliance_agent, "llm_client"):
        active_llm = coordinator.compliance_agent.llm_client.active_provider.upper()
    else:
        active_llm = current_provider

    st.markdown(f"**LLM Backend:** <span class='provider-pill'>{active_llm}</span>", unsafe_allow_html=True)
    st.markdown(f"**Chunks Indexed:** `{st.session_state.indexed_chunks_count}`")
    
    # KG statistics
    kg_entities_count = 0
    kg_relations_count = 0
    if coordinator.kg_agent and hasattr(coordinator.kg_agent, "graph_store"):
        stats = coordinator.kg_agent.graph_store.get_stats()
        kg_entities_count = stats.get("total_entities", 0)
        kg_relations_count = stats.get("total_relations", 0)
    
    st.markdown(f"**KG Entities:** `{kg_entities_count}`")
    st.markdown(f"**KG Relations:** `{kg_relations_count}`")

    st.divider()
    st.subheader("🛠️ Quick Ingestion")
    if st.button("🔄 Re-Index Sample Corpus", **FULL_WIDTH):
        with st.spinner("Indexing sample documents from data/sample_docs..."):
            res = coordinator.ingest_documents()
            if res.get("status") == "success":
                st.session_state.indexed_chunks_count = res.get("chunks_count", 0)
                st.success(f"Indexed {res.get('chunks_count', 0)} chunks successfully!")
            else:
                st.error(f"Ingestion failed: {res.get('message')}")

    if st.button("🗑️ Clear Chat History", **FULL_WIDTH):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("ICCA-RAG Architecture: Section IV-B (Hybrid BM25 + Dense Embeddings + Multi-Hop KG Traversal)")


# ---------------------------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------------------------
tab_chat, tab_ingestion, tab_kg, tab_settings = st.tabs([
    "💬 Assistant Chat & Regulatory Guidance",
    "📂 Document Ingestion & Corpus",
    "🕸️ Knowledge Graph Explorer",
    "⚙️ Settings & System Status"
])


# ===========================================================================
# TAB 1: Assistant Chat & Regulatory Guidance
# ===========================================================================
with tab_chat:
    st.subheader("Cross-Border Customs Question Answering")
    st.caption("Ask regulatory clearance questions, tariff classification duties, or mandatory documentation rules.")

    if "current_query" not in st.session_state:
        st.session_state.current_query = "What documents are required for customs clearance of machinery imports under HS 8429?"

    if "query_input_box" not in st.session_state:
        st.session_state.query_input_box = st.session_state.current_query

    def execute_query(q_text: str):
        """Execute query workflow and update history."""
        cleaned = q_text.strip()
        if not cleaned:
            return
        with st.spinner("Executing Section IV-B: Reformulating query -> Hybrid Retrieval -> KG Traversal -> LLM Grounding..."):
            start_time = time.time()
            result = coordinator.process_query(cleaned)
            elapsed = time.time() - start_time

            # Prepend to chat history
            st.session_state.chat_history.insert(0, {
                "query": cleaned,
                "result": result,
                "elapsed": elapsed
            })

    def handle_sample_click(sample_text: str):
        """Callback executed before widget instantiation to safely update query state."""
        st.session_state.query_input_box = sample_text
        st.session_state.current_query = sample_text
        execute_query(sample_text)

    # Sample Question Pills - clicking directly runs the query!
    st.markdown("**Sample Compliance Queries (Click to Run):**")
    sample_queries = [
        "What documents are required for customs clearance of machinery imports under HS 8429?",
        "What are the import clearance procedures and BIS certification rules for electronic integrated circuits (HS 8542)?",
        "Explain customs duty, IGST, and required hazmat documentation for importing industrial chemicals under Chapter 29.",
        "What are the dual-use export control guidelines and SCOMET licensing requirements?",
        "What is the standard Bill of Entry clearance timeline and customs examination workflow?"
    ]

    cols = st.columns(2)
    for idx, sample in enumerate(sample_queries):
        col = cols[idx % 2]
        col.button(
            f"📌 {sample}",
            key=f"sample_{idx}",
            on_click=handle_sample_click,
            args=(sample,),
            **FULL_WIDTH
        )

    st.divider()

    # Query Input & Search Controls wrapped in a form to support Enter key submission
    with st.form("customs_query_form", clear_on_submit=False):
        user_input = st.text_input(
            "Enter your customs clearance question (Press Enter to submit):",
            key="query_input_box",
            placeholder="e.g., What are the mandatory clearance documents and BIS rules for machinery under HS 8429?"
        )
        submit_clicked = st.form_submit_button("🔍 Submit Query & Verify Compliance", **FULL_WIDTH)

    if submit_clicked:
        target_query = user_input.strip() if user_input else ""
        if target_query:
            st.session_state.current_query = target_query
            execute_query(target_query)
            st.rerun()
        else:
            st.warning("⚠️ Please enter a question or click one of the sample compliance queries above.")

    # Display Query Results & History
    if not st.session_state.chat_history:
        st.info(
            "💡 **Ready for Compliance Q&A**: Select any sample compliance query above or type your question into the "
            "search box and press **Enter** or click **Submit Query & Verify Compliance**."
        )
    else:
        for entry_idx, entry in enumerate(st.session_state.chat_history):
            res = entry["result"]
            query_str = entry["query"]
            elapsed_sec = entry["elapsed"]

            with st.container():
                st.markdown(f"### ❓ Query: *\"{query_str}\"*")
                
                # Badges row
                conf = res.get("confidence", 0.85)
                conf_class = "confidence-high" if conf >= 0.75 else "confidence-mid"
                provider_tag = res.get("provider", "offline").upper()

                bcol1, bcol2, bcol3, bcol4 = st.columns([2, 2, 2, 2])
                with bcol1:
                    st.markdown(f"**Confidence:** <span class='{conf_class}'>{int(conf * 100)}% Grounded</span>", unsafe_allow_html=True)
                with bcol2:
                    st.markdown(f"**LLM Engine:** <span class='provider-pill'>{provider_tag}</span>", unsafe_allow_html=True)
                with bcol3:
                    st.markdown(f"**Execution Time:** `{elapsed_sec:.2f}s`")
                with bcol4:
                    st.markdown(f"**Retrieved Sources:** `{len(res.get('sources', []))}`")

                # Reformulated Query Callout
                if res.get("reformulated_query"):
                    st.info(
                        f"**Section IV-B1 Query Reformulation:** `{res.get('reformulated_query')}`\n\n"
                        f"*Technical Keywords Extracted (Q):* {', '.join(f'`{k}`' for k in res.get('keywords', []))}"
                    )

                # Grounded Response Box
                st.markdown(res.get("answer", "No answer generated."))

                # Expandable Inspector Panels
                with st.expander("🔍 Inspect Retrieved Context Chunks (BM25 + Semantic)", expanded=False):
                    sources = res.get("sources", [])
                    if sources:
                        for s_idx, s in enumerate(sources, 1):
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <b>[Source {s_idx}] {s.get('doc_id')}</b> (Chunk: <code>{s.get('chunk_id')}</code>, Relevance Score: <code>{s.get('relevance_score')}</code>, Modality: <code>{s.get('modality')}</code>)
                                    <p style="margin-top: 6px; font-size: 0.92rem; color: #374151;">{s.get('snippet')}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.write("No direct source chunks retrieved.")

                with st.expander("🕸️ Inspect Knowledge Graph Reasoning Paths", expanded=False):
                    kg_triples = res.get("kg_triples", [])
                    kg_paths = res.get("kg_paths", [])
                    if kg_triples or kg_paths:
                        if kg_triples:
                            st.markdown(f"**Entity Relationships Traversed ({len(kg_triples)} total, displaying top {min(len(kg_triples), 20)}):**")
                            for t in kg_triples[:20]:
                                st.write(f"- {t}")
                        if kg_paths:
                            st.markdown(f"**Multi-Hop Reasoning Chains ({len(kg_paths)} total, displaying top {min(len(kg_paths), 15)}):**")
                            for p in kg_paths[:15]:
                                st.write(f"  * {p}")
                    else:
                        st.info("No explicit KG triples traversed for this query.")

                with st.expander("📋 Inspect Grounded Prompt (Section IV-B3 Anti-Hallucination Template)", expanded=False):
                    st.code(res.get("structured_prompt", ""), language="markdown")

                st.divider()


# ===========================================================================
# TAB 2: Document Ingestion & Corpus Manager
# ===========================================================================
with tab_ingestion:
    st.subheader("Customs Document Parser & Indexer")
    st.caption("Upload customs statutory documents, tariffs, circulars, or SOPs (PDF, DOCX, TXT, CSV, Excel).")

    up_col1, up_col2 = st.columns([3, 2])

    with up_col1:
        uploaded_file = st.file_uploader(
            "Choose a customs regulatory file:",
            type=["pdf", "docx", "txt", "csv", "xlsx"]
        )

        if uploaded_file is not None:
            st.write(f"**Filename:** `{uploaded_file.name}` ({round(uploaded_file.size / 1024, 2)} KB)")
            if st.button("📥 Parse, Chunk & Index Document", **FULL_WIDTH):
                # Save file temporarily to sample_docs
                save_dir = default_config.sample_docs_dir
                save_dir.mkdir(parents=True, exist_ok=True)
                dest_path = save_dir / uploaded_file.name
                
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Executing Algorithm 1 (Parsing & Semantic Chunking) on {uploaded_file.name}..."):
                    ingest_res = coordinator.ingest_documents(str(dest_path))
                    if ingest_res.get("status") == "success":
                        st.success(f"Successfully processed and indexed {ingest_res.get('chunks_count', 0)} chunks!")
                        st.session_state.indexed_chunks_count = (
                            st.session_state.indexed_chunks_count + ingest_res.get("chunks_count", 0)
                        )
                    else:
                        st.error(f"Error during ingestion: {ingest_res.get('message')}")

    with up_col2:
        st.markdown(
            """
            <div class="metric-card">
                <h4>Algorithm 1: Unified Multimodal Parser</h4>
                <p>Supports multimodal ingestion of customs schedules, regulatory notifications, and trade manuals:</p>
                <ul>
                    <li>PDF / DOCX text and table structure recovery</li>
                    <li>CSV / Excel tariff rate schedule parsing</li>
                    <li>Sliding window semantic chunking (500 tokens, 50 overlap)</li>
                    <li>Dual-index generation (BM25 sparse + dense vector embeddings)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    st.subheader("📚 Pre-loaded Customs Regulatory Corpus")
    sample_files = list(default_config.sample_docs_dir.glob("*.*"))
    if sample_files:
        corpus_data = []
        for sf in sample_files:
            corpus_data.append({
                "Document Name": sf.name,
                "Format": sf.suffix.upper(),
                "Size (KB)": round(sf.stat().st_size / 1024, 2),
                "Path": str(sf)
            })
        st.dataframe(corpus_data, **FULL_WIDTH)
    else:
        st.info("No documents currently stored in data/sample_docs.")


# ===========================================================================
# TAB 3: Knowledge Graph Explorer
# ===========================================================================
with tab_kg:
    st.subheader("Customs Knowledge Graph & Regulatory Ontologies")
    st.caption("Explore entities (HS Codes, Regulations, Required Documents) and multi-hop compliance relationships.")

    kg_stats = {"total_entities": 0, "total_relations": 0, "engine": "in-memory"}
    if coordinator.kg_agent and hasattr(coordinator.kg_agent, "graph_store"):
        kg_stats = coordinator.kg_agent.graph_store.get_stats()

    kcol1, kcol2, kcol3 = st.columns(3)
    kcol1.metric("Total Customs Entities", kg_stats.get("total_entities", 0))
    kcol2.metric("Total Relationships / Edges", kg_stats.get("total_relations", 0))
    kcol3.metric("Storage Engine", kg_stats.get("engine", "InMemory / NetworkX"))

    st.divider()

    # Entity Search and Traversal
    kg_search_query = st.text_input(
        "Search Knowledge Graph by Entity or HS Code:",
        placeholder="e.g., 8429, Bill of Entry, BIS, SCOMET, Machinery"
    )

    if kg_search_query.strip() and coordinator.kg_agent:
        with st.spinner("Traversing Knowledge Graph..."):
            kg_res = coordinator.kg_agent.execute({
                "action": "get_context_triples",
                "query": kg_search_query.strip(),
                "max_hops": 2
            })
            if kg_res.get("status") == "success":
                triples = kg_res.get("data", {}).get("triples_text", [])
                paths = kg_res.get("data", {}).get("paths", [])

                st.markdown(f"**Found {len(triples)} Connected Relationships for `{kg_search_query}`:**")
                if triples:
                    for t in triples:
                        st.markdown(f"- 🔗 `{t}`")
                else:
                    st.info("No matching entity relationships found in the active graph.")

                if paths:
                    st.markdown("**Multi-Hop Reasoning Paths:**")
                    for p in paths:
                        st.markdown(f"  * ➡️ `{p}`")
    else:
        # Show default customs schema overview
        st.markdown(
            """
            ### 🏛️ Customs Regulatory Ontology Schema
            The ICCA-RAG Knowledge Graph models key customs compliance relationships:
            
            ```mermaid
            graph LR
                Product["Product / Commodity"] -->|CLASSIFIED_AS| HS["HS Code (e.g. 8429.52)"]
                HS -->|REQUIRES_DOC| Doc["Mandatory Document (Bill of Entry, BIS)"]
                HS -->|REGULATED_BY| Reg["Customs Act / DGFT Policy"]
                HS -->|HAS_TARIFF| Duty["Tariff (BCD, IGST, Cess)"]
                Doc -->|ISSUED_BY| Agency["PGA (BIS, FSSAI, DGFT)"]
            ```
            """
        )


# ===========================================================================
# TAB 4: Settings & System Status
# ===========================================================================
with tab_settings:
    st.subheader("System Configuration & Provider Settings")

    set_col1, set_col2 = st.columns(2)

    with set_col1:
        st.markdown("### 🤖 LLM Generation Provider")
        provider_options = ["offline", "gemini", "openai"]
        current_idx = 0
        if coordinator.compliance_agent and hasattr(coordinator.compliance_agent, "llm_client"):
            cur_p = coordinator.compliance_agent.llm_client.active_provider.lower()
            if cur_p in provider_options:
                current_idx = provider_options.index(cur_p)

        selected_provider = st.selectbox(
            "Select Active LLM Provider:",
            options=provider_options,
            index=current_idx,
            format_func=lambda x: {
                "offline": "Deterministic Offline Synthesizer (Zero API Keys Required)",
                "gemini": "Google Gemini API (models/gemini-1.5-flash)",
                "openai": "OpenAI API (gpt-4o-mini)"
            }.get(x, x)
        )

        gemini_key_input = st.text_input(
            "Google Gemini API Key (GEMINI_API_KEY):",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password"
        )

        openai_key_input = st.text_input(
            "OpenAI API Key (OPENAI_API_KEY):",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password"
        )

        if st.button("💾 Apply LLM Settings", **FULL_WIDTH):
            if selected_provider == "gemini" and gemini_key_input:
                os.environ["GEMINI_API_KEY"] = gemini_key_input
                coordinator.config.gemini_api_key = gemini_key_input
            elif selected_provider == "openai" and openai_key_input:
                os.environ["OPENAI_API_KEY"] = openai_key_input
                coordinator.config.openai_api_key = openai_key_input

            coordinator.config.llM_provider = selected_provider
            if coordinator.compliance_agent:
                coordinator.compliance_agent.execute({
                    "action": "set_provider",
                    "provider": selected_provider,
                    "api_key": gemini_key_input if selected_provider == "gemini" else openai_key_input
                })
            st.success(f"Active provider switched to '{selected_provider}'.")
            st.rerun()

    with set_col2:
        st.markdown("### ⚙️ Retrieval Hyperparameters (Paper Section IV-B)")
        st.markdown(f"- **BM25 $k_1$ (Term Frequency Saturation):** `{coordinator.config.bm25_k1}`")
        st.markdown(f"- **BM25 $b$ (Document Length Normalization):** `{coordinator.config.bm25_b}`")
        st.markdown(f"- **Sparse Top-K Candidates:** `{coordinator.config.sparse_top_k}`")
        st.markdown(f"- **Dense Top-K Candidates:** `{coordinator.config.dense_top_k}`")
        st.markdown(f"- **Embedding Dimensions:** `{coordinator.config.embedding_dimension}`")
        st.markdown(f"- **KG Max Graph Traversal Hops:** `{coordinator.config.kg_max_hops}`")

        st.divider()
        st.markdown("### 🧪 Self-Diagnostics")
        if st.button("Run System Diagnostics Check", **FULL_WIDTH):
            diag = []
            diag.append(f"Ingestion Agent: {'OK' if coordinator.ingestion_agent else 'Degraded'}")
            diag.append(f"Retrieval Agent: {'OK' if coordinator.retrieval_agent else 'Degraded'}")
            diag.append(f"Knowledge Graph Agent: {'OK' if coordinator.kg_agent else 'Degraded'}")
            diag.append(f"Compliance Agent: {'OK' if coordinator.compliance_agent else 'Degraded'}")
            for d in diag:
                st.write(f"- {d}")
            st.success("Diagnostics completed.")
