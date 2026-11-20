"""
ICCA-RAG Interactive Command-Line Interface.

Usage:
    # One-shot query:
    python run_cli.py --query "What documents are required for machinery imports under HS 8429?"

    # Interactive session:
    python run_cli.py

    # Select LLM provider:
    python run_cli.py --provider offline  # or 'gemini', 'openai'

    # Ingest document or directory:
    python run_cli.py --ingest data/sample_docs
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from icca.config import default_config, ICCAConfig
from icca.coordinator import CustomsClearanceCoordinator

BANNER = """
========================================================================
  ICCA-RAG: Intelligent Customs Clearance Assistant
  Knowledge-Graph Augmented Hybrid RAG for Customs Compliance
  Team: K Dhanush (142) | B Gautham (146) | B Pandu Sagar (157)
========================================================================
"""

SAMPLE_QUERIES = [
    "What documents are required for customs clearance of machinery imports under HS 8429?",
    "What are the import clearance procedures and BIS certification rules for electronic integrated circuits (HS 8542)?",
    "Explain customs duty, IGST, and required hazmat documentation for importing industrial chemicals under Chapter 29.",
    "What are the dual-use export control guidelines and SCOMET licensing requirements?",
    "What is the standard Bill of Entry clearance timeline and customs examination workflow?"
]


def print_formatted_result(result: dict, elapsed: float):
    """Render grounded customs response in a clean terminal format."""
    print("\n" + "=" * 70)
    print(f"⏱️  Execution Time: {elapsed:.2f}s  |  Provider: {result.get('provider', 'offline').upper()}")
    print("=" * 70)

    # Reformulation info
    ref_query = result.get("reformulated_query")
    keywords = result.get("keywords", [])
    if ref_query:
        print(f"\n🔍 [Section IV-B1 Reformulated Query]:")
        print(f"   {ref_query}")
        if keywords:
            print(f"   Extracted Technical Terms (Q): {', '.join(keywords)}")

    # Grounded Answer
    print(f"\n📑 [Grounded Customs Response]:")
    print("-" * 70)
    print(result.get("answer", "No answer generated."))
    print("-" * 70)

    # Knowledge Graph reasoning
    kg_triples = result.get("kg_triples", [])
    kg_paths = result.get("kg_paths", [])
    if kg_triples or kg_paths:
        print(f"\n🕸️  [Knowledge Graph Reasoning Paths]:")
        for t in kg_triples[:4]:
            print(f"   • {t}")
        for p in kg_paths[:3]:
            print(f"   ➡️  {p}")

    # Source Citations
    sources = result.get("sources", [])
    if sources:
        print(f"\n📚 [Source Citations & Relevance Scores]:")
        for i, s in enumerate(sources[:4], 1):
            print(f"   [{i}] {s.get('doc_id')} (Chunk: {s.get('chunk_id')}, Score: {s.get('relevance_score')}, Modality: {s.get('modality')})")
            snippet = s.get('snippet', '').strip().replace('\n', ' ')
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            print(f"       \"{snippet}\"")
    print("=" * 70 + "\n")


def interactive_loop(coordinator: CustomsClearanceCoordinator):
    """Run an interactive CLI session."""
    print(BANNER)
    print("Commands:")
    print("  'samples'  : Display sample customs queries")
    print("  'ingest'   : Ingest sample documents from data/sample_docs")
    print("  'provider' : Switch LLM provider (offline, gemini, openai)")
    print("  'exit'     : Quit assistant\n")

    while True:
        try:
            user_input = input("ICCA-RAG > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting ICCA-RAG. Goodbye!")
                break

            elif user_input.lower() == "samples":
                print("\nSample Compliance Queries:")
                for i, q in enumerate(SAMPLE_QUERIES, 1):
                    print(f"  [{i}] {q}")
                print()
                continue

            elif user_input.lower().startswith("sample "):
                try:
                    num = int(user_input.split()[1]) - 1
                    if 0 <= num < len(SAMPLE_QUERIES):
                        user_input = SAMPLE_QUERIES[num]
                        print(f"\nRunning Sample Query: {user_input}\n")
                    else:
                        print("Invalid sample number.")
                        continue
                except ValueError:
                    print("Usage: sample <number>")
                    continue

            elif user_input.lower() == "ingest":
                print("[ICCA-RAG] Ingesting documents from data/sample_docs...")
                res = coordinator.ingest_documents()
                print(f"[ICCA-RAG] Ingested {res.get('chunks_count', 0)} chunks.\n")
                continue

            elif user_input.lower().startswith("provider"):
                parts = user_input.split()
                if len(parts) > 1:
                    new_p = parts[1].lower()
                    if coordinator.compliance_agent:
                        set_res = coordinator.compliance_agent.execute({
                            "action": "set_provider",
                            "provider": new_p
                        })
                        print(f"[ICCA-RAG] {set_res.get('message')}\n")
                else:
                    cur_p = "offline"
                    if coordinator.compliance_agent and hasattr(coordinator.compliance_agent, "llm_client"):
                        cur_p = coordinator.compliance_agent.llm_client.active_provider
                    print(f"[ICCA-RAG] Active provider: {cur_p} (options: offline, gemini, openai)\n")
                continue

            # Process query
            start_t = time.time()
            res = coordinator.process_query(user_input)
            elapsed = time.time() - start_t
            print_formatted_result(res, elapsed)

        except KeyboardInterrupt:
            print("\nExiting ICCA-RAG. Goodbye!")
            break
        except Exception as e:
            print(f"\n[!] Error processing request: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ICCA-RAG: Intelligent Customs Clearance Assistant CLI"
    )
    parser.add_argument("-q", "--query", type=str, help="Single query to execute in one-shot mode")
    parser.add_argument("-p", "--provider", type=str, default="offline", choices=["offline", "gemini", "openai"], help="LLM Provider")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of retrieved chunks (default: 5)")
    parser.add_argument("-i", "--ingest", type=str, help="Path to document or directory to ingest")
    parser.add_argument("--no-kg", action="store_true", help="Disable Knowledge Graph traversal")

    args = parser.parse_args()

    # Configure
    config = default_config
    config.llM_provider = args.provider
    coordinator = CustomsClearanceCoordinator(config=config)
    coordinator.initialize()

    # Set provider if compliance agent is loaded
    if coordinator.compliance_agent:
        coordinator.compliance_agent.execute({
            "action": "set_provider",
            "provider": args.provider
        })

    # Ingest if specified or auto-ingest default sample corpus
    if args.ingest:
        print(f"[ICCA-RAG] Ingesting from: {args.ingest}")
        ingest_res = coordinator.ingest_documents(args.ingest)
        print(f"[ICCA-RAG] Ingestion result: {ingest_res.get('message')}")
    else:
        # Check if default corpus can be indexed
        coordinator.ingest_documents()

    if args.query:
        # One-shot mode
        print(BANNER)
        print(f"Query: \"{args.query}\"")
        start_t = time.time()
        res = coordinator.process_query(
            user_query=args.query,
            top_k=args.top_k,
            use_kg=not args.no_kg
        )
        elapsed = time.time() - start_t
        print_formatted_result(res, elapsed)
    else:
        # Interactive mode
        interactive_loop(coordinator)


if __name__ == "__main__":
    main()
