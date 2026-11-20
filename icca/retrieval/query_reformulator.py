"""
Query Reformulation Module for ICCA-RAG.

Implements Section IV-B.1 and Equation (3) of the ICCA-RAG paper:
    q_query = Reformulate(q_input) = Concat(Q) U ExpansionTerms(Q)

Eliminates ambiguities, removes conversational fillers, identifies Harmonized System (HS)
codes, trade entities, regulatory terms, and produces an optimized search representation
for dual sparse (BM25) and dense (vector) retrieval.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Conversational filler prefixes and patterns to strip
CONVERSATIONAL_FILLERS: List[str] = [
    r"^can you please (tell me|explain|list|provide|show)\s+(about\s+)?",
    r"^could you please (tell me|explain|list|provide|show)\s+(about\s+)?",
    r"^please (tell me|explain|list|provide|show)\s+(about\s+)?",
    r"^i would like to know\s+(about\s+)?",
    r"^i want to know\s+(about\s+)?",
    r"^what are the (requirements|rules|documents|procedures|duties|tariffs)\s+(for|to|on)\s+",
    r"^what is the (process|procedure|duty rate|tariff rate|hs code)\s+(for|to|on|of)\s+",
    r"^how do i (clear|import|export|classify|pay duty for)\s+",
    r"^tell me (about|how to|what is)\s+",
    r"^kindly inform me (about|regarding)\s+",
    r"^what is required to\s+",
    r"^do i need (a|an|any)?\s+",
    r"^is it possible to\s+",
    r"\bplease\b",
    r"\bkindly\b",
    r"\bthank you\b",
    r"\bthanks\b",
]

# Trade domain acronyms and regulatory keywords
TRADE_KEYWORDS_MAP: Dict[str, List[str]] = {
    # Documentation
    "bill of entry": ["bill of entry", "boe", "customs declaration"],
    "commercial invoice": ["commercial invoice", "invoice", "cif value"],
    "packing list": ["packing list", "package breakdown"],
    "bill of lading": ["bill of lading", "bol", "ocean bill", "shipping document"],
    "airway bill": ["airway bill", "awb", "air freight"],
    "certificate of origin": ["certificate of origin", "coo", "preferential trade agreement"],
    "bis": ["bis certificate", "bureau of indian standards", "mandatory certification"],
    "fssai": ["fssai", "food safety", "food import clearance"],
    "epr": ["epr certificate", "extended producer responsibility", "e-waste", "plastic waste"],
    "scomet": ["scomet", "dual-use goods", "export authorization", "special chemicals"],
    "dgft": ["dgft", "directorate general of foreign trade", "import export code", "iec"],
    "msds": ["msds", "material safety data sheet", "hazardous chemicals"],
    # Duties & Tariffs
    "basic customs duty": ["basic customs duty", "bcd", "tariff rate"],
    "igst": ["integrated goods and services tax", "igst", "import gst"],
    "anti-dumping duty": ["anti-dumping duty", "add", "safeguard duty"],
    "social welfare surcharge": ["social welfare surcharge", "sws"],
    "duty drawback": ["duty drawback", "dbk", "export incentive"],
    # Customs Workflow & Processing
    "faceless assessment": ["faceless assessment", "fag", "anonymized assessment"],
    "out of charge": ["out of charge", "ooc", "gate pass"],
    "first check": ["first check examination", "physical examination"],
    "second check": ["second check examination", "standard clearance"],
    "icegate": ["icegate", "customs portal", "electronic filing"],
}

# Domain expansion terms by commodity category
CATEGORY_EXPANSIONS: Dict[str, List[str]] = {
    "machinery": ["machinery", "capital goods", "mechanical appliances", "chapter 84", "bis certification", "inspection"],
    "chemical": ["chemicals", "organic chemicals", "inorganic chemicals", "chapter 28", "chapter 29", "msds", "hazmat"],
    "electronics": ["electronics", "integrated circuits", "semiconductors", "chapter 85", "bis crs", "wpc certificate"],
    "textiles": ["textiles", "garments", "fabric", "chapter 61", "chapter 62", "textile committee testing"],
    "food": ["food products", "agricultural imports", "fssai clearance", "phytosanitary certificate", "quarantine"],
    "dual-use": ["dual-use goods", "scomet license", "end-user certificate", "munitions list"],
    "pharmaceutical": ["pharmaceuticals", "drugs", "active pharmaceutical ingredients", "cdsco noc", "adc clearance"],
}


class QueryReformulator:
    """
    Intelligent Query Reformulator implementing Section IV-B.1 of ICCA-RAG.

    Translates raw conversational queries q_input into clean, expanded,
    concept-focused queries q_query optimized for hybrid retrieval.
    """

    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming

    def clean_noise(self, query: str) -> str:
        """Strip greetings, conversational fillers, and leading/trailing punctuation."""
        cleaned = query.strip()
        
        # Apply regex filters for conversational patterns
        for pattern in CONVERSATIONAL_FILLERS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
            
        # Clean redundant punctuation but preserve dots in HS codes (e.g., 8429.52)
        cleaned = re.sub(r"[?!;:,]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def extract_hs_codes(self, text: str) -> List[str]:
        """
        Extract Harmonized System (HS) codes and chapter references.
        Detects 2-digit chapters, 4-digit headings, 6-digit subheadings, and 8-digit tariff lines.
        """
        hs_codes: List[str] = []

        # Explicit HS code mentions (e.g., 'HS 8429', 'HS Code 8429.52', 'HSCode: 8517.62.00')
        explicit_matches = re.findall(
            r"(?:hs\s*(?:code)?\s*[:#-]?\s*|tariff\s+heading\s*[:#-]?\s*)(\d{2,4}(?:\.\d{2}){0,2})",
            text,
            re.IGNORECASE
        )
        for m in explicit_matches:
            if m not in hs_codes:
                hs_codes.append(m)

        # Chapter mentions (e.g., 'Chapter 84', 'Ch 29')
        chapter_matches = re.findall(
            r"\b(?:chapter|ch\.?)\s*(\d{1,2})\b",
            text,
            re.IGNORECASE
        )
        for ch in chapter_matches:
            formatted_ch = f"Chapter {ch.zfill(2)}"
            if formatted_ch not in hs_codes:
                hs_codes.append(formatted_ch)

        # Standalone dotted codes (e.g., '8429.52' or '8429.52.00')
        dotted_matches = re.findall(r"\b(\d{4}\.\d{2}(?:\.\d{2})?)\b", text)
        for dm in dotted_matches:
            if dm not in hs_codes:
                hs_codes.append(dm)

        # 4-digit code if preceded by trade keywords (avoid capturing years like 2024)
        four_digit_matches = re.findall(r"\b(\d{4})\b", text)
        for fdm in four_digit_matches:
            # Check not a year (1900-2099) unless explicitly tagged
            val = int(fdm)
            if 1000 <= val <= 9999 and not (1990 <= val <= 2030):
                if fdm not in hs_codes:
                    hs_codes.append(fdm)

        return hs_codes

    def extract_trade_entities(self, text: str) -> List[str]:
        """Extract recognized trade, compliance, and customs entities."""
        entities: List[str] = []
        text_lower = text.lower()

        for entity_name, syn_list in TRADE_KEYWORDS_MAP.items():
            for syn in syn_list:
                if re.search(r"\b" + re.escape(syn) + r"\b", text_lower):
                    if entity_name.upper() not in entities:
                        entities.append(entity_name.upper())
                    break

        return entities

    def detect_query_intent(self, text: str) -> str:
        """
        Classify the user's primary regulatory intent.
        
        Intents:
        - CLEARANCE_DOCUMENTS: Questions about mandatory forms, permits, certificates.
        - TARIFF_AND_DUTY: Questions about BCD, IGST, cess, valuation, payment.
        - REGULATORY_RESTRICTION: Questions about SCOMET, EPR, BIS, hazmat, bans.
        - HS_CLASSIFICATION: Questions identifying the code or chapter of a good.
        - GENERAL_PROCEDURE: General customs import/export workflow questions.
        """
        text_lower = text.lower()

        if any(w in text_lower for w in ["document", "paperwork", "form", "certificate", "license", "licence", "noc", "bill of entry"]):
            return "CLEARANCE_DOCUMENTS"
        elif any(w in text_lower for w in ["duty", "tariff", "tax", "igst", "bcd", "rate", "cost", "anti-dumping", "surcharge"]):
            return "TARIFF_AND_DUTY"
        elif any(w in text_lower for w in ["scomet", "dual-use", "restriction", "prohibited", "banned", "epr", "hazardous", "hazmat"]):
            return "REGULATORY_RESTRICTION"
        elif any(w in text_lower for w in ["classify", "classification", "hs code", "chapter", "heading"]):
            return "HS_CLASSIFICATION"
        else:
            return "GENERAL_PROCEDURE"

    def expand_terms(self, text: str, entities: List[str], intent: str) -> List[str]:
        """Generate domain expansions based on commodity category and intent."""
        expansions: Set[str] = set()
        text_lower = text.lower()

        # Category detection
        for category, terms in CATEGORY_EXPANSIONS.items():
            if any(term in text_lower for term in terms):
                expansions.update(terms[:3])

        # Intent-driven expansions
        if intent == "CLEARANCE_DOCUMENTS":
            expansions.update(["Bill of Entry", "Commercial Invoice", "Packing List", "Compliance Certificate"])
        elif intent == "TARIFF_AND_DUTY":
            expansions.update(["Basic Customs Duty", "IGST", "Customs Tariff Schedule"])
        elif intent == "REGULATORY_RESTRICTION":
            expansions.update(["DGFT Import Policy", "Restricted / Prohibited Schedule", "NOC"])

        return sorted(list(expansions))

    def reformulate(
        self,
        query: str,
        use_llm: bool = False,
        llm_client: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Reformulate raw user query into an optimized retrieval query.

        Args:
            query: Raw user query string q_input.
            use_llm: Whether to invoke an LLM for semantic query expansion.
            llm_client: Optional UnifiedLLMClient instance.

        Returns:
            Dictionary containing:
            - 'original_query': raw input
            - 'cleaned_query': noise-stripped query
            - 'reformulated_query': final search query q_query
            - 'keywords': extracted technical terms Q
            - 'hs_codes': detected HS codes / chapters
            - 'trade_entities': recognized customs entities
            - 'intent': detected regulatory intent
            - 'expansion_terms': domain expansion keywords
            - 'method': 'llm' or 'deterministic_rule_based'
        """
        if not query or not query.strip():
            return {
                "original_query": "",
                "cleaned_query": "",
                "reformulated_query": "",
                "keywords": [],
                "hs_codes": [],
                "trade_entities": [],
                "intent": "GENERAL_PROCEDURE",
                "expansion_terms": [],
                "method": "deterministic_rule_based"
            }

        # Step 1: Strip conversational noise
        cleaned_text = self.clean_noise(query)
        if not cleaned_text:
            cleaned_text = query.strip()

        # Step 2: Extract technical concepts Q
        hs_codes = self.extract_hs_codes(query)
        trade_entities = self.extract_trade_entities(query)
        intent = self.detect_query_intent(query)
        expansions = self.expand_terms(query, trade_entities, intent)

        # Collect core keywords
        keywords_set: Set[str] = set()
        for code in hs_codes:
            keywords_set.add(code)
        for ent in trade_entities:
            keywords_set.add(ent)
        for word in re.findall(r"\b[A-Za-z0-9\.\-_]{3,}\b", cleaned_text):
            if word.lower() not in {"what", "which", "when", "where", "how", "with", "from", "that", "this", "these", "those"}:
                keywords_set.add(word)

        keywords = sorted(list(keywords_set))

        # Step 3: LLM-assisted reformulation if requested and client available
        if use_llm and llm_client is not None:
            try:
                llm_prompt = (
                    f"Rewrite the following customs query into a concise, keyword-rich search query "
                    f"for retrieving customs regulations, HS tariff schedules, and required clearance documents.\n"
                    f"Input Query: {query}\n"
                    f"Detected Entities: {', '.join(keywords)}\n"
                    f"Optimized Search Query:"
                )
                llm_out = llm_client.generate(llm_prompt, max_tokens=60, temperature=0.0).strip()
                # Clean any markdown or quotes
                llm_out = re.sub(r'["\']', '', llm_out).strip()
                if len(llm_out) > 5:
                    return {
                        "original_query": query,
                        "cleaned_query": cleaned_text,
                        "reformulated_query": llm_out,
                        "keywords": keywords,
                        "hs_codes": hs_codes,
                        "trade_entities": trade_entities,
                        "intent": intent,
                        "expansion_terms": expansions,
                        "method": "llm"
                    }
            except Exception:
                pass  # Fall back to deterministic

        # Step 4: Deterministic Synthesis (Equation 3)
        # q_query = Concat(Cleaned Core) + HS Codes + Core Trade Entities
        components: List[str] = []
        if hs_codes:
            components.append(" ".join(f"HS {c}" if not c.lower().startswith("chapter") and not c.lower().startswith("hs") else c for c in hs_codes))
        
        components.append(cleaned_text)
        
        # Add high-priority expansions matching the intent
        if trade_entities:
            components.append(" ".join(trade_entities[:3]))

        synthesized_query = " ".join(components)
        # Deduplicate redundant words preserving order
        seen_words = set()
        dedup_tokens = []
        for token in synthesized_query.split():
            tok_norm = token.lower().strip(".,;:()")
            if tok_norm not in seen_words:
                seen_words.add(tok_norm)
                dedup_tokens.append(token)
        reformulated_query = " ".join(dedup_tokens)

        return {
            "original_query": query,
            "cleaned_query": cleaned_text,
            "reformulated_query": reformulated_query,
            "keywords": keywords,
            "hs_codes": hs_codes,
            "trade_entities": trade_entities,
            "intent": intent,
            "expansion_terms": expansions,
            "method": "deterministic_rule_based"
        }
