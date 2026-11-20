# ICCA-RAG: Intelligent Customs Clearance Assistant
### Hybrid RAG (Dense + Sparse) with Knowledge Graph Architecture for Multi-Hop Compliance Reasoning

**Project Team:**
- K Dhanush (1602-23-733-142)
- B Gautham (1602-23-733-143)
- B Pandu Sagar (1602-23-733-161)

**Based on:** IEEE Access Research Paper *"ICCA-RAG: Intelligent Customs Clearance Assistant Using Retrieval-Augmented Generation"* (R. Hu et al., 2025) and extended with an integrated **Knowledge Graph Layer** for multi-hop regulatory reasoning.

---

## 📌 Executive Summary & Problem Statement
Customs clearance in international trade involves navigating voluminous, fragmented, and rapidly changing regulations, tariff schedules (HS Codes), and inspection requirements across various government departments (Customs, DGFT, FSSAI, Pollution Control, BIS, CDSCO). Small and Medium Enterprises (SMEs) face high compliance costs, classification errors, penalties, and delayed shipments.

Traditional keyword search fails on complex regulatory semantics, while pure vector RAG often misses relationship-based and multi-hop compliance requirements (e.g., *"If I import machinery under HS 8429, which specific compliance declarations, BIS certificates, and port clearance procedures are mandatory?"*).

**ICCA-RAG** solves this through a tri-fold hybrid architecture:
1. **Multimodal Document Parsing (Algorithm 1)**: Ingests PDFs, Word DOCX, Excel/CSV tariff schedules, and text into normalized markdown tables and semantic paragraphs.
2. **Hybrid Storage & Retrieval**:
   - **Sparse Index**: Inverted Index with BM25 scoring ($k_1=1.5, b=0.75$) for exact match of regulatory clauses, act numbers, and HS codes.
   - **Dense Index**: High-dimensional embeddings with vector similarity (HNSW / FAISS with pure-Python/NumPy fallback) for semantic intent capture.
3. **Knowledge Graph Layer (Multi-Hop Compliance)**:
   - Entities: `HS_Code`, `Product_Category`, `Regulation`, `Required_Document`, `Tariff_Rate`, `Inspection_Rule`.
   - Relations: `REQUIRES_DOC`, `REGULATED_BY`, `HAS_TARIFF`, `CLASSIFIED_AS`, `SUBJECT_TO`.
   - NetworkX graph store (with direct Neo4j extension interface) providing multi-hop traversal and regulatory compliance paths.
4. **Context Construction & LLM Generation**:
   - Query Reformulation to eliminate ambiguity.
   - Structured context builder with strict anti-hallucination instructions.
   - Generation with source attribution, document citations, and compliance checklists.

---

## 🏗️ System Architecture

```
                    ┌───────────────────────────────┐
                    │          User Query           │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Central Coordinator      │
                    │ (CustomsClearanceCoordinator) │
                    └───────────────┬───────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌──────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│ Ingestion Agent  │    │   Retrieval Agent    │    │  Knowledge Graph Agent │
│ • Algorithm 1    │    │ • BM25 Sparse Search │    │ • Entity Linking       │
│ • PDF/DOCX/Excel │    │ • Dense Vector Index │    │ • Multi-Hop Traversal  │
│ • Markdown Tables│    │ • Reciprocal Fusion  │    │ • NetworkX / Neo4j     │
└──────────────────┘    └──────────────────────┘    └────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Compliance Agent        │
                    │ • Query Reformulation         │
                    │ • Structured Context Builder  │
                    │ • Grounded LLM Response       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Verified Cited Response     │
                    │  + Required Forms Checklist   │
                    │  + Source Attribution Chunks  │
                    └───────────────────────────────┘
```

---

## 🚀 Quick Start Guide (Running on Any System)

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Installation
Clone or copy the repository to your system:
```bash
# Navigate to project folder
cd "Major Project(Pandu)"

# (Optional but recommended) Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Configuration (Optional)
The system includes a smart **Offline/Mock LLM engine** that operates immediately without any external API keys or GPU.
To use Cloud LLMs (Google Gemini or OpenAI):
```bash
# Copy template
cp .env.example .env

# Edit .env and add your API key:
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_key_here
```

---

## 💻 Running the Application

### Option A: Interactive Web Application (Streamlit)
Launch the modern, responsive web dashboard:
```bash
python run_app.py
```
Or directly with Streamlit:
```bash
streamlit run icca/ui/app.py
```
The browser will open at `http://localhost:8501`.

**Web Dashboard Features:**
- **Assistant Chat**: Ask complex customs questions with real-time citations, confidence levels, and expandable retrieved contexts.
- **Pre-configured Prompts**: One-click questions for Machinery Imports (HS 8429/8471), Hazmat Chemicals (HS 2901), Dual-Use Goods (SCOMET), and Tariff Calculations.
- **Document Ingestion**: Upload custom PDFs, DOCX, CSV/Excel tariffs, or text files to dynamically index them into the knowledge base.
- **Knowledge Graph Explorer**: Interactive inspection of customs entities, relations, and multi-hop compliance pathways.
- **Settings & Status**: Toggle between Offline Mock, Gemini, and OpenAI LLM backends.

### Option B: Interactive Command Line Interface (CLI)
For rapid terminal querying and batch testing:
```bash
# Interactive mode:
python run_cli.py

# Direct single query:
python run_cli.py --query "What documents are required for customs clearance of machinery imports?"
```

---

## 🧪 Running Automated Tests
Run the test suite to verify all modules (parser, BM25 sparse index, vector index, knowledge graph, coordinator):
```bash
pytest tests/ -v
```

---

## 📂 Project Structure
```
├── README.md                              # Comprehensive Documentation
├── requirements.txt                       # Cross-platform dependencies
├── .env.example                           # Configuration environment template
├── run_app.py                             # Streamlit Web UI launcher
├── run_cli.py                             # Interactive CLI launcher
├── data/
│   ├── sample_docs/                       # Realistic customs regulatory corpus
│   │   ├── customs_machinery_imports_guide.txt
│   │   ├── customs_tariff_schedule_ch84_85.csv
│   │   ├── customs_chemical_hazmat_regulations.txt
│   │   ├── customs_dual_use_export_control.txt
│   │   └── customs_clearance_standard_sop.txt
│   ├── knowledge_graph/                   # Seed Knowledge Graph & Schema
│   │   └── customs_kg.json
│   └── storage/                           # Inverted index & vector index cache
├── icca/                                  # Core Package
│   ├── __init__.py
│   ├── config.py                          # Hyperparameters & path settings
│   ├── coordinator.py                     # Central Coordinator orchestrator
│   ├── document_parser/                   # Algorithm 1: Multimodal Ingestion
│   │   ├── parser.py                      # PDF, DOCX, Excel/CSV, Image OCR
│   │   └── chunker.py                     # Section/paragraph segmentation
│   ├── indexing/                          # Hybrid Indexing Engine
│   │   ├── bm25_index.py                  # Inverted index & BM25 algorithm
│   │   ├── embeddings.py                  # Multi-backend embedder with fallback
│   │   └── vector_index.py                # HNSW / FAISS / NumPy cosine index
│   ├── knowledge_graph/                   # Knowledge Graph Layer
│   │   ├── schema.py                      # Entities, Relations, & Data models
│   │   ├── graph_store.py                 # NetworkX store with Neo4j extension hooks
│   │   ├── entity_extractor.py            # HS Code & regulatory entity extraction
│   │   └── traverser.py                   # Multi-hop regulatory reasoning
│   ├── retrieval/                         # Retrieval & Context Building
│   │   ├── query_reformulator.py          # Section IV-B1 Query Optimization
│   │   ├── hybrid_retriever.py            # Dense + Sparse score fusion
│   │   └── context_builder.py             # Section IV-B3 Prompt Engineering
│   ├── generation/                        # LLM Generation Engine
│   │   ├── llm_client.py                  # Gemini / OpenAI / Offline synthesizer
│   │   └── generator.py                   # Context-grounded generation & citations
│   ├── agents/                            # Specialized Sub-Agents
│   │   ├── base_agent.py                  # Standard agent communication contract
│   │   ├── ingestion_agent.py             # Document parsing worker
│   │   ├── retrieval_agent.py             # Hybrid search worker
│   │   ├── kg_agent.py                    # Graph reasoning worker
│   │   └── compliance_agent.py            # Answer generation worker
│   └── ui/
│       └── app.py                         # Streamlit Web Application
└── tests/                                 # Unit & Integration tests
```

---

## 🔌 Knowledge Graph Extension (NetworkX to Neo4j)
The system is built with a pluggable graph abstraction layer in `icca/knowledge_graph/graph_store.py`:
- Out-of-the-box, it runs fully in memory with `NetworkX` and persists to `customs_kg.json`.
- To connect to an enterprise `Neo4j` instance:
  1. Set `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in `.env`.
  2. Call `graph_store.sync_to_neo4j()` to sync all nodes and relationships.
  3. The `CustomsGraphStore` automatically switches to Cypher query execution for distributed enterprise scale.

