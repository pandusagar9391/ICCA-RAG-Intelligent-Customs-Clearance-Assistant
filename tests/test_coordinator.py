"""
End-to-end Integration Test for Central CustomsClearanceCoordinator.
Validates the complete workflow:
1. Coordinator initialization
2. Multimodal document ingestion & dual indexing
3. Knowledge graph pre-loading and entity linking
4. Query reformulation
5. Hybrid retrieval (Sparse BM25 + Dense Vector Index)
6. Knowledge graph multi-hop traversal
7. Grounded answer generation with citations & compliance checklists
"""

import unittest
from pathlib import Path
from icca.config import ICCAConfig
from icca.coordinator import CustomsClearanceCoordinator


class TestCustomsClearanceCoordinator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = ICCAConfig()
        cls.config.llM_provider = "offline"  # Run deterministic offline engine
        cls.coordinator = CustomsClearanceCoordinator(config=cls.config)
        cls.coordinator.initialize()
        # Pre-ingest sample docs once for all tests
        cls.ingest_result = cls.coordinator.ingest_documents()

    def test_01_coordinator_initialization(self):
        """Verify that all 4 sub-agents are registered with the central coordinator."""
        self.assertTrue(self.coordinator.is_initialized)
        self.assertIsNotNone(self.coordinator.ingestion_agent)
        self.assertIsNotNone(self.coordinator.retrieval_agent)
        self.assertIsNotNone(self.coordinator.kg_agent)
        self.assertIsNotNone(self.coordinator.compliance_agent)

    def test_02_ingest_sample_customs_documents(self):
        """Verify ingestion and dual indexing of sample documents."""
        res = self.coordinator.ingest_documents()
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["chunks_count"], 0)
        print(f"\n[Test] Ingested and dual-indexed {res['chunks_count']} customs document chunks.")

    def test_03_query_machinery_imports(self):
        """Test query: 'What documents are required for customs clearance of machinery imports?' (matches Slide 8)."""
        query = "What documents are required for customs clearance of machinery imports?"
        res = self.coordinator.process_query(query, top_k=5, use_kg=True)

        self.assertEqual(res["status"], "success")
        self.assertIn("machinery", res["reformulated_query"].lower())
        self.assertIsNotNone(res["answer"])
        self.assertTrue(len(res["answer"]) > 50)

        # Check required documents identified
        self.assertTrue(len(res["required_documents"]) > 0 or len(res["sources"]) > 0)

        # Check sparse and dense retrieval
        self.assertGreater(len(res["sparse_results"]), 0)
        self.assertGreater(len(res["dense_results"]), 0)

        # Check Knowledge Graph paths
        self.assertGreater(len(res["kg_triples"]), 0)

        print("\n--- Test Query 1 Output ---")
        print("Query:", res["query"])
        print("Reformulated:", res["reformulated_query"])
        print("HS Codes:", res.get("hs_codes"))
        print("Required Documents:", res.get("required_documents"))
        print("KG Paths found:", len(res["kg_paths"]))
        print("Answer Summary:", res["answer"][:200] + "...")

    def test_04_query_hazardous_chemicals(self):
        """Test query for hazardous materials and MSDS compliance."""
        query = "What regulations and MSDS requirements apply to chemical hazardous goods imports?"
        res = self.coordinator.process_query(query, top_k=5, use_kg=True)

        self.assertEqual(res["status"], "success")
        self.assertTrue(len(res["answer"]) > 50)
        self.assertGreater(len(res["sparse_results"]), 0)
        print(f"\n[Test] Hazardous chemicals query answered successfully with {len(res['sources'])} cited sources.")

    def test_05_query_dual_use_goods(self):
        """Test query for SCOMET / dual use electronics and licensing."""
        query = "What license is needed for SCOMET dual-use items and End-User Certificate?"
        res = self.coordinator.process_query(query, top_k=5, use_kg=True)

        self.assertEqual(res["status"], "success")
        self.assertTrue(len(res["answer"]) > 50)
        print(f"\n[Test] Dual-use SCOMET query answered successfully.")


if __name__ == "__main__":
    unittest.main()
