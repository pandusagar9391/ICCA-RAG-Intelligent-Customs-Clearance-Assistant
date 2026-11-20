"""
Unit and Integration Tests for ICCA-RAG Generation, Reformulation, and Compliance Modules.
"""

import unittest
from icca.config import default_config
from icca.retrieval.query_reformulator import QueryReformulator
from icca.retrieval.context_builder import ContextBuilder, token_jaccard_similarity
from icca.generation.llm_client import UnifiedLLMClient, OfflineCustomsSynthesizer
from icca.generation.generator import ResponseGenerator
from icca.agents.compliance_agent import ComplianceAgent
from icca.coordinator import CustomsClearanceCoordinator


class TestQueryReformulator(unittest.TestCase):
    def setUp(self):
        self.reformulator = QueryReformulator()

    def test_clean_noise(self):
        raw = "Could you please tell me about the clearance process for machinery?"
        cleaned = self.reformulator.clean_noise(raw)
        self.assertNotIn("Could you please tell me about", cleaned)
        self.assertIn("machinery", cleaned)

    def test_extract_hs_codes(self):
        query = "What is the duty on HS 8429.52 and Chapter 85 items?"
        hs_codes = self.reformulator.extract_hs_codes(query)
        self.assertTrue(any("8429" in c for c in hs_codes))
        self.assertTrue(any("Chapter 85" in c for c in hs_codes))

    def test_intent_detection(self):
        intent_doc = self.reformulator.detect_query_intent("What documents are required for import?")
        self.assertEqual(intent_doc, "CLEARANCE_DOCUMENTS")

        intent_duty = self.reformulator.detect_query_intent("What is the basic customs duty and IGST rate?")
        self.assertEqual(intent_duty, "TARIFF_AND_DUTY")

        intent_scomet = self.reformulator.detect_query_intent("Are there dual-use SCOMET restrictions?")
        self.assertEqual(intent_scomet, "REGULATORY_RESTRICTION")

    def test_reformulate(self):
        query = "Can you please tell me what documents are needed to clear machinery under HS 8429?"
        res = self.reformulator.reformulate(query)
        self.assertIn("reformulated_query", res)
        self.assertIn("keywords", res)
        self.assertTrue(len(res["keywords"]) > 0)
        self.assertEqual(res["intent"], "CLEARANCE_DOCUMENTS")


class TestContextBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = ContextBuilder(redundancy_threshold=0.80)

    def test_jaccard_similarity(self):
        s1 = "Customs clearance requires bill of entry and invoice"
        s2 = "Customs clearance requires bill of entry and invoice"
        self.assertEqual(token_jaccard_similarity(s1, s2), 1.0)

        s3 = "Completely different topic with nothing common"
        self.assertEqual(token_jaccard_similarity(s1, s3), 0.0)

    def test_deduplicate_chunks(self):
        chunks = [
            {"chunk_id": "c1", "text": "Machinery clearance rules require BIS certificate and invoice."},
            {"chunk_id": "c1", "text": "Machinery clearance rules require BIS certificate and invoice."},  # duplicate ID
            {"chunk_id": "c2", "text": "Machinery clearance rules require BIS certificate and invoice."},  # duplicate text
            {"chunk_id": "c3", "text": "Chemical hazmat regulations mandate MSDS sheets."}
        ]
        deduped = self.builder.deduplicate_chunks(chunks)
        self.assertEqual(len(deduped), 2)

    def test_build_prompt_structure(self):
        query = "Machinery import clearance"
        sparse = [{"chunk_id": "s1", "doc_id": "doc1.pdf", "text": "Machinery import text", "score": 0.8}]
        dense = [{"chunk_id": "d1", "doc_id": "doc2.pdf", "text": "Dense import text", "score": 0.75}]
        kg_triples = ["[Machinery] -> REQUIRES_DOC -> [Bill of Entry]"]
        kg_paths = ["Machinery -> Standard -> Clearance"]

        prompt = self.builder.build_prompt(
            query=query,
            sparse_results=sparse,
            dense_results=dense,
            kg_triples=kg_triples,
            kg_paths=kg_paths
        )
        self.assertIn("### USER QUERY:\nMachinery import clearance", prompt)
        self.assertIn("### KNOWLEDGE GRAPH RELATIONSHIPS:", prompt)
        self.assertIn("### SPARSE RETRIEVAL RESULTS", prompt)
        self.assertIn("### DENSE RETRIEVAL RESULTS", prompt)
        self.assertIn("CRITICAL ANTI-HALLUCINATION CONSTRAINTS:", prompt)


class TestUnifiedLLMClientAndSynthesizer(unittest.TestCase):
    def setUp(self):
        self.client = UnifiedLLMClient()
        self.synthesizer = OfflineCustomsSynthesizer()

    def test_offline_synthesizer_grounding(self):
        prompt = """### USER QUERY:
What documents are required for machinery imports?

### KNOWLEDGE GRAPH RELATIONSHIPS:
- [Machinery] --REQUIRES_DOC--> [Bill of Entry]
- [Machinery] --REQUIRES_DOC--> [BIS Mandatory Certification]

### SPARSE RETRIEVAL RESULTS (Lexical / BM25):
[Sparse Retrieval Document 1] (Source: machinery_guide.pdf, Chunk ID: C1, Score: 0.95):
Importers of machinery must submit a Bill of Entry, Commercial Invoice, and obtain BIS certification.

### DENSE RETRIEVAL RESULTS (Semantic Embeddings):
[Dense Retrieval Document 1] (Source: tariff_schedule.pdf, Chunk ID: C2, Score: 0.90):
Earth moving machinery under HS 8429 incurs Basic Customs Duty of 7.5% and IGST of 18%.

### RESPONSE:
"""
        ans = self.synthesizer.synthesize(prompt)
        self.assertIn("Machinery & Capital Goods Import Clearance", ans)
        self.assertIn("Bill of Entry", ans)
        self.assertIn("8429", ans)
        self.assertIn("Checklist", ans)

    def test_offline_synthesizer_anti_hallucination(self):
        # When unrelated query is submitted with no relevant context
        prompt = """### USER QUERY:
How do I make chocolate chip cookies at home?

### KNOWLEDGE GRAPH RELATIONSHIPS:
No matching Knowledge Graph relations found for this query.

### SPARSE RETRIEVAL RESULTS (Lexical / BM25):
No sparse retrieval results available.

### DENSE RETRIEVAL RESULTS (Semantic Embeddings):
No dense retrieval results available.

### RESPONSE:
"""
        ans = self.synthesizer.synthesize(prompt)
        self.assertIn("does not contain sufficient information", ans)


class TestResponseGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ResponseGenerator()

    def test_extract_sources(self):
        chunks = [
            {"chunk_id": "c1", "doc_id": "customs_sop.pdf", "text": "Sample clearance text", "score": 0.88, "modality": "Hybrid"}
        ]
        sources = self.generator.extract_sources_from_chunks(chunks)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["doc_id"], "customs_sop.pdf")
        self.assertEqual(sources[0]["relevance_score"], 0.88)

    def test_compute_confidence(self):
        chunks = [{"chunk_id": "c1", "score": 0.9}]
        paths = ["Path 1 -> Path 2"]
        ans = "- [ ] Bill of Entry"
        conf = self.generator.compute_confidence(chunks, paths, ans)
        self.assertGreaterEqual(conf, 0.70)


class TestComplianceAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ComplianceAgent()

    def test_execute_reformulate(self):
        res = self.agent.execute({
            "action": "reformulate_query",
            "query": "What are the rules for importing chemical goods?"
        })
        self.assertEqual(res["status"], "success")
        self.assertIn("reformulated_query", res["data"])

    def test_execute_build_context(self):
        res = self.agent.execute({
            "action": "build_context",
            "query": "Chemical import clearance",
            "sparse_results": [],
            "dense_results": []
        })
        self.assertEqual(res["status"], "success")
        self.assertIn("prompt", res["data"])

    def test_execute_generate_answer(self):
        res = self.agent.execute({
            "action": "generate_answer",
            "query": "Machinery import clearance under HS 8429",
            "context_chunks": [
                {"chunk_id": "c1", "doc_id": "machinery.txt", "text": "Machinery clearance requires Bill of Entry and BIS.", "score": 0.8}
            ]
        })
        self.assertEqual(res["status"], "success")
        self.assertIn("answer", res["data"])
        self.assertIn("sources", res["data"])


class TestEndToEndCoordinator(unittest.TestCase):
    def test_end_to_end_query(self):
        coord = CustomsClearanceCoordinator()
        coord.initialize()
        coord.ingest_documents()

        res = coord.process_query("What documents are required for customs clearance of machinery imports under HS 8429?")
        self.assertEqual(res["status"], "success")
        self.assertIn("answer", res)
        self.assertTrue(len(res["answer"]) > 50)
        self.assertTrue(len(res["sources"]) > 0)


if __name__ == "__main__":
    unittest.main()
