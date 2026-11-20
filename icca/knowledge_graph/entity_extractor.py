"""
Customs Entity and Relation Extractor for ICCA-RAG.

Performs high-precision extraction of customs entities and semantic relations from:
- Regulatory guidelines and circulars
- Commercial invoices and shipping documents
- Natural language compliance queries

Features:
- Regex-based high-precision extraction for HS codes, statutory documents,
  regulatory acts, inspection authorities, and product categories.
- Heuristic relation binding for compliance triples.
- Optional LLM-based entity-relation extraction fallback with structured JSON schema.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from icca.knowledge_graph.schema import (
    CustomsEntity,
    CustomsRelation,
    EntityType,
    RelationType,
)

logger = logging.getLogger(__name__)


class CustomsEntityExtractor:
    """
    Extracts customs domain entities and relationships from text.
    Combines rule-based regex patterns with optional LLM fallback.
    """

    # -------------------------------------------------------------------------
    # Regex Patterns for Customs Entities
    # -------------------------------------------------------------------------

    # HS Code: e.g. "8429", "8471.30", "8429.11.00", "HS 8542.31"
    HS_CODE_REGEX = re.compile(
        r"(?:(?:HS|Heading|ITC(?:[ -]?\(?HS\)?)?|Subheading)\s*[:#\.-]?\s*)?"
        r"\b([0-9]{4}(?:\.[0-9]{2}(?:\.[0-9]{2})?)?)\b",
        re.IGNORECASE
    )

    # Years to avoid treating as HS codes when un-prefixed
    EXCLUDED_YEARS = {
        "1962", "1968", "1985", "1986", "1989", "1992", "2000", "2003",
        "2010", "2014", "2016", "2017", "2018", "2019", "2020", "2021",
        "2022", "2023", "2024", "2025", "2026", "2030"
    }

    # Document Catalogs: Pattern -> (Entity ID, Standard Name, Description, Issuing Agency)
    DOCUMENT_PATTERNS = [
        (
            r"\b(?:Bill\s+of\s+Entry(?:\s+Form\s+[I|II|III])?|BOE)\b",
            "DOC_BILL_OF_ENTRY",
            "Bill of Entry",
            "Statutory declaration filed by importer/customs broker for clearance under Customs Act 1962",
            "AGENCY_CBIC"
        ),
        (
            r"\bCommercial\s+Invoice\b",
            "DOC_COMMERCIAL_INVOICE",
            "Commercial Invoice",
            "Itemized seller invoice specifying value, Incoterms, currency, and complete product description",
            "AGENCY_EXPORTER"
        ),
        (
            r"\bPacking\s+List\b",
            "DOC_PACKING_LIST",
            "Packing List",
            "Container-wise breakdown of gross weight, net weight, serial numbers, and dimensions",
            "AGENCY_EXPORTER"
        ),
        (
            r"\b(?:(?:Master|House)\s+)?Bill\s+of\s+Lading\b|\bMBL\b|\bHBL\b|\bB/L\b",
            "DOC_BILL_OF_LADING",
            "Bill of Lading",
            "Maritime transport document and document of title issued by carrier",
            "AGENCY_SHIPPING_LINE"
        ),
        (
            r"\b(?:Airway\s+Bill|Air\s+Waybill|MAWB|HAWB)\b",
            "DOC_AIRWAY_BILL",
            "Airway Bill",
            "Air cargo transport document issued by airline",
            "AGENCY_AIRLINE"
        ),
        (
            r"\b(?:Certificate\s+of\s+Origin|COO)\b",
            "DOC_CERTIFICATE_OF_ORIGIN",
            "Certificate of Origin",
            "Preferential or Non-Preferential origin certificate verifying country of manufacturing",
            "AGENCY_CHAMBER_COMMERCE"
        ),
        (
            r"\b(?:Material\s+Safety\s+Data\s+Sheet|Safety\s+Data\s+Sheet|MSDS|SDS)\b",
            "DOC_MSDS",
            "Material Safety Data Sheet (MSDS/SDS)",
            "16-section standard chemical safety declaration conforming to GHS mandates",
            "AGENCY_MANUFACTURER"
        ),
        (
            r"\b(?:Dangerous\s+Goods\s+Declaration|Hazmat\s+DGD|DGD)\b",
            "DOC_DANGEROUS_GOODS_DECLARATION",
            "Dangerous Goods Declaration (IMO/ICAO DGD)",
            "Mandatory shipper declaration for hazardous maritime or air consignments",
            "AGENCY_SHIPPER"
        ),
        (
            r"\b(?:Chartered\s+Engineer(?:\s+Inspection)?\s+Certificate|CEC)\b",
            "DOC_CHARTERED_ENGINEER_CERT",
            "Chartered Engineer Certificate (CEC)",
            "Mandatory technical appraisal certifying residual life (>= 5 yrs) and value for used machinery",
            "AGENCY_CHARTERED_ENGINEER"
        ),
        (
            r"\b(?:BIS(?:\s+Registration|\s+CRS|\s+CRO)?\s+Certificate|BIS\s+Quality\s+Order|IS\s+13252)\b",
            "DOC_BIS_CERTIFICATE",
            "BIS Registration Certificate (CRO/CRS)",
            "Conformity registration under Compulsory Registration Scheme of Bureau of Indian Standards",
            "AGENCY_BIS"
        ),
        (
            r"\b(?:End-User\s+Certificate|EUC|Appendix\s+2S)\b",
            "DOC_END_USER_CERTIFICATE",
            "End-User Certificate (EUC - Appendix 2S)",
            "Statutory commitment that dual-use SCOMET goods will not be diverted to military or WMD applications",
            "AGENCY_END_USER"
        ),
        (
            r"\b(?:SCOMET(?:\s+Export|\s+Import)?\s+Authorization|SCOMET\s+License|DGFT\s+Export\s+Authorization)\b",
            "DOC_SCOMET_LICENSE",
            "SCOMET Export/Import Authorization",
            "Strategic export/import license issued by DGFT for dual-use technologies and materials",
            "AGENCY_DGFT"
        ),
        (
            r"\b(?:CDSCO(?:\s+Import)?\s+License|Form\s+MD-15|Form\s+MD-14)\b",
            "DOC_CDSCO_LICENSE",
            "CDSCO Medical Device Import License (Form MD-15)",
            "Statutory import license for medical devices granted under Medical Devices Rules 2017",
            "AGENCY_CDSCO"
        ),
        (
            r"\b(?:Free\s+Sale\s+Certificate|FSC)\b",
            "DOC_FREE_SALE_CERTIFICATE",
            "Free Sale Certificate (FSC)",
            "Regulatory certificate confirming medical device or product is freely marketed in originating country",
            "AGENCY_NATIONAL_HEALTH_AUTHORITY"
        ),
        (
            r"\b(?:Phytosanitary\s+Certificate|NPPO\s+Certificate)\b",
            "DOC_PHYTOSANITARY_CERTIFICATE",
            "Phytosanitary Certificate",
            "Plant health certificate issued by National Plant Protection Organization of exporting nation",
            "AGENCY_NPPO"
        ),
        (
            r"\b(?:Plant\s+Quarantine\s+(?:Import\s+)?Clearance|PQ\s+Order\s+Form|PQ\s+Clearance)\b",
            "DOC_PLANT_QUARANTINE_CLEARANCE",
            "Plant Quarantine Import Clearance",
            "Biosecurity clearance issued by Directorate of Plant Protection, Quarantine & Storage",
            "AGENCY_DPPQS"
        ),
        (
            r"\b(?:FSSAI(?:\s+Import)?\s+NOC|Food\s+Import\s+Clearance(?:\s+System)?)\b",
            "DOC_FSSAI_NOC",
            "FSSAI Import NOC / Clearance Certificate",
            "Food safety clearance certificate verifying compliance with statutory FSSAI import standards",
            "AGENCY_FSSAI"
        ),
        (
            r"\b(?:SPCB\s+NOC|State\s+Pollution\s+Control\s+Board\s+Consignment\s+Authorization|CPCB\s+Registration)\b",
            "DOC_POLLUTION_CONTROL_NOC",
            "Pollution Control Board Import NOC / Authorization",
            "Environmental clearance for import of hazardous chemicals and wastes",
            "AGENCY_SPCB"
        ),
        (
            r"\b(?:WPC\s+ETA|Equipment\s+Type\s+Approval|Wireless\s+Planning\s+and\s+Coordination)\b",
            "DOC_WPC_ETA",
            "WPC Equipment Type Approval (ETA)",
            "Wireless frequency clearance for equipment incorporating Wi-Fi, Bluetooth, or RF transmitters",
            "AGENCY_WPC"
        ),
        (
            r"\b(?:Certificate\s+of\s+Analysis|COA)\b",
            "DOC_CERTIFICATE_OF_ANALYSIS",
            "Certificate of Analysis (COA)",
            "Manufacturer chemical assay verifying composition, purity, and batch compliance",
            "AGENCY_MANUFACTURER"
        ),
    ]

    # Regulations: Pattern -> (Entity ID, Standard Name, Description)
    REGULATION_PATTERNS = [
        (
            r"\b(?:Customs\s+Act(?:,?\s*1962)?)\b",
            "REG_CUSTOMS_ACT_1962",
            "Customs Act, 1962",
            "Primary Indian statute governing import and export procedures, valuation, assessment, and duties"
        ),
        (
            r"\b(?:Foreign\s+Trade\s+Policy(?:,?\s*2023)?|FTP\s*2023|FTP)\b",
            "REG_FTP_2023",
            "Foreign Trade Policy (FTP 2023)",
            "Statutory policy formulated by DGFT governing trade facilitation, export promotion, and restrictions"
        ),
        (
            r"\b(?:DGFT\s+Notification(?:s)?(?:\s+No\.?\s*[0-9]+/[0-9]+)?)\b",
            "REG_DGFT_NOTIFICATIONS",
            "DGFT Regulatory Notifications",
            "Executive foreign trade orders regulating item-wise import licensing and restrictions"
        ),
        (
            r"\b(?:Manufacture,?\s+Storage\s+and\s+Import\s+of\s+Hazardous\s+Chemical(?:s)?\s+Rules(?:,?\s*1989)?|MSIHC\s+Rules)\b",
            "REG_MSIHC_RULES_1989",
            "MSIHC Rules, 1989",
            "Mandates safety protocols, threshold inventories, and port safety for toxic and hazardous chemicals"
        ),
        (
            r"\b(?:Hazardous\s+and\s+Other\s+Wastes(?:\s*\(Management\s+and\s+Transboundary\s+Movement\)\s*)?\s+Rules(?:,?\s*2016)?|HOWM\s+Rules)\b",
            "REG_HOWM_RULES_2016",
            "Hazardous and Other Wastes Rules, 2016",
            "Regulates transboundary movement, import authorization, and recycling of hazardous waste"
        ),
        (
            r"\b(?:SCOMET(?:\s+List)?|Wassenaar\s+Arrangement)\b",
            "REG_SCOMET_EXPORT_CONTROL",
            "SCOMET Strategic Dual-Use Control Framework",
            "Dual-use control framework under Appendix 3 Schedule 2 of ITC(HS) aligning with Wassenaar regime"
        ),
        (
            r"\b(?:Medical\s+Devices\s+Rules(?:,?\s*2017)?|MDR-2017)\b",
            "REG_MDR_2017",
            "Medical Devices Rules, 2017",
            "Comprehensive regulatory framework under Drugs and Cosmetics Act for medical device registration"
        ),
        (
            r"\b(?:Plant\s+Quarantine(?:\s*\(Regulation\s+of\s+Import\s+into\s+India\)\s*)?\s+Order(?:,?\s*2003)?|PQ\s+Order\s*2003)\b",
            "REG_PLANT_QUARANTINE_ORDER_2003",
            "Plant Quarantine Order 2003",
            "Biosecurity order preventing introduction of exotic pests and diseases through imported agricultural goods"
        ),
        (
            r"\b(?:Food\s+Safety\s+and\s+Standards(?:\s*\(Import\)\s*)?\s+Regulations(?:,?\s*2017)?|FSSAI\s+Import\s+Regulations)\b",
            "REG_FSSAI_IMPORT_REGS",
            "Food Safety and Standards (Import) Regulations 2017",
            "Standards and testing regimes for food imports, labeling, shelf-life, and testing protocols"
        ),
        (
            r"\b(?:Bureau\s+of\s+Indian\s+Standards\s+Act(?:,?\s*2016)?|BIS\s+Act|BIS\s+Quality\s+Control\s+Order)\b",
            "REG_BIS_CONFORMITY_ORDER",
            "BIS Quality Control Order (QCO)",
            "Statutory conformity scheme requiring mandatory BIS certification marks on specified goods"
        ),
        (
            r"\b(?:IMDG\s+Code|International\s+Maritime\s+Dangerous\s+Goods\s+Code)\b",
            "REG_IMDG_CODE",
            "IMDG Maritime Safety Code",
            "International maritime framework governing safe sea carriage and segregation of dangerous goods"
        ),
        (
            r"\b(?:ICAO\s+Technical\s+Instructions|IATA\s+Dangerous\s+Goods\s+Regulations|IATA\s+DGR)\b",
            "REG_ICAO_IATA_DGR",
            "ICAO / IATA Dangerous Goods Regulations",
            "International aviation standards governing packaging, net quantities, and air carriage of hazmat"
        ),
    ]

    # Agencies: Pattern -> (Entity ID, Standard Name, Description)
    AGENCY_PATTERNS = [
        (
            r"\b(?:CBIC|Central\s+Board\s+of\s+Indirect\s+Taxes\s+and\s+Customs|Indian\s+Customs|Customs\s+Department|ICEGATE)\b",
            "AGENCY_CBIC",
            "CBIC / Indian Customs",
            "Nodal customs authority under Ministry of Finance responsible for border clearance, valuation, and revenue"
        ),
        (
            r"\b(?:DGFT|Directorate\s+General\s+of\s+Foreign\s+Trade)\b",
            "AGENCY_DGFT",
            "Directorate General of Foreign Trade (DGFT)",
            "Formulates Foreign Trade Policy, grants SCOMET and import licenses, and manages IEC numbers"
        ),
        (
            r"\b(?:BIS|Bureau\s+of\s+Indian\s+Standards)\b",
            "AGENCY_BIS",
            "Bureau of Indian Standards (BIS)",
            "National standards body administering Compulsory Registration Scheme (CRS) and ISI certification"
        ),
        (
            r"\b(?:CDSCO|Central\s+Drugs\s+Standard\s+Control\s+Organization)\b",
            "AGENCY_CDSCO",
            "CDSCO",
            "National regulatory authority for pharmaceuticals, medical devices, and clinical trials in India"
        ),
        (
            r"\b(?:FSSAI|Food\s+Safety\s+and\s+Standards\s+Authority\s+of\s+India)\b",
            "AGENCY_FSSAI",
            "FSSAI",
            "Statutory food authority monitoring sanitary standards, food imports, and laboratory testing"
        ),
        (
            r"\b(?:PESO|Petroleum\s+and\s+Explosives\s+Safety\s+Organization)\b",
            "AGENCY_PESO",
            "PESO",
            "Safety regulator under DPIIT controlling manufacture, storage, and import of explosives and compressed gases"
        ),
        (
            r"\b(?:CPCB|Central\s+Pollution\s+Control\s+Board)\b",
            "AGENCY_CPCB",
            "Central Pollution Control Board (CPCB)",
            "National statutory board under MoEFCC regulating pollution prevention and hazardous waste import quotas"
        ),
        (
            r"\b(?:SPCB|State\s+Pollution\s+Control\s+Board)\b",
            "AGENCY_SPCB",
            "State Pollution Control Board (SPCB)",
            "State environmental regulatory agency issuing consignment consents and site storage authorizations"
        ),
        (
            r"\b(?:DPPQS|Plant\s+Quarantine\s+(?:Station|Division)?|Directorate\s+of\s+Plant\s+Protection)\b",
            "AGENCY_DPPQS",
            "DPPQS (Plant Quarantine Authority)",
            "Biosecurity authority responsible for inspecting, testing, and clearing imported plant materials"
        ),
        (
            r"\b(?:WPC(?:\s+Wing)?|Wireless\s+Planning\s+and\s+Coordination)\b",
            "AGENCY_WPC",
            "WPC Wing (Ministry of Communications)",
            "National radio regulatory authority granting Equipment Type Approvals for wireless transmitters"
        ),
        (
            r"\b(?:TEC|Telecommunication\s+Engineering\s+Centre)\b",
            "AGENCY_TEC",
            "Telecommunication Engineering Centre (TEC)",
            "Administers Mandatory Testing and Certification of Telecom Equipment (MTCTE) in India"
        ),
        (
            r"\b(?:MEA|Ministry\s+of\s+External\s+Affairs)\b",
            "AGENCY_MEA",
            "Ministry of External Affairs (MEA)",
            "Participates in inter-ministerial screening of strategic dual-use SCOMET authorizations"
        ),
    ]

    # Product Categories: Pattern -> (Entity ID, Standard Name, Description)
    PRODUCT_CATEGORY_PATTERNS = [
        (
            r"\b(?:Industrial\s+Machinery|Earth[- ]moving\s+Machinery|Bulldozers?|Excavators?|Loaders?|Road\s+Rollers?)\b",
            "CAT_INDUSTRIAL_MACHINERY",
            "Industrial Machinery & Earthmoving Equipment",
            "Heavy civil engineering capital equipment, earthmovers, and automated mechanical equipment"
        ),
        (
            r"\b(?:Computing\s+Hardware|Automatic\s+Data\s+Processing|Laptops?|Notebooks?|Servers?|Data\s+Storage|Microcomputers?)\b",
            "CAT_COMPUTING_HARDWARE",
            "Computing Hardware & Data Processing Equipment",
            "Electronic automatic data processing machines, commercial servers, notebooks, and workstations"
        ),
        (
            r"\b(?:Hazardous\s+Chemicals?|Toxic\s+Chemicals?|Organic\s+Chemicals?|Acyclic\s+Hydrocarbons?|Ethylene|Propylene|Chlorine|Chemical\s+Precursors?)\b",
            "CAT_HAZARDOUS_CHEMICALS",
            "Hazardous & Industrial Chemicals",
            "Chemical compounds, organic and inorganic reagents, industrial hydrocarbons, and gases subject to hazmat rules"
        ),
        (
            r"\b(?:Dual[- ]Use\s+(?:Goods|Electronics|Items|Technology)|Semiconductors?|Integrated\s+Circuits?|Microprocessors?|FPGAs?|Cryptographic\s+Equipment|Telecom\s+Routers?)\b",
            "CAT_DUAL_USE_ELECTRONICS",
            "Dual-Use Strategic Electronics & Telecom",
            "High-performance computing units, cryptography gear, and integrated circuits governed by SCOMET controls"
        ),
        (
            r"\b(?:Medical\s+Devices?|Surgical\s+Instruments?|Diagnostic\s+Apparatus|MRI\s+Scanners?|Surgical\s+Robots?|Electromedical\s+Equipment)\b",
            "CAT_MEDICAL_DEVICES",
            "Medical Devices & Diagnostic Instruments",
            "Electromedical apparatus, diagnostic equipment, surgical devices, and implants regulated by CDSCO"
        ),
        (
            r"\b(?:Agricultural\s+Imports?|Wheat|Meslin|Durum\s+Wheat|Grains?|Agricultural\s+Produce|Cereals?)\b",
            "CAT_AGRICULTURAL_IMPORTS",
            "Agricultural Produce & Grains",
            "Bulk food grains, cereals, and agricultural commodities subject to plant quarantine and FSSAI clearance"
        ),
    ]

    # Inspection Rules: Pattern -> (Entity ID, Standard Name, Description)
    INSPECTION_RULE_PATTERNS = [
        (
            r"\b(?:Chartered\s+Engineer(?:\s+Inspection)?|Residual\s+Life\s*(?:>=?|at\s+least)?\s*5\s*years|Second[- ]hand\s+machinery\s+inspection)\b",
            "RULE_SECOND_HAND_INSPECTION",
            "Second-Hand Machinery Pre-Shipment Inspection Rule",
            "Mandatory physical inspection certifying minimum 5-year residual life and absence of ozone-depleting substances"
        ),
        (
            r"\b(?:BIS\s+CRS\s+Marking|IS\s+13252\s+Conformity|Standard\s+Mark\s+Affixation)\b",
            "RULE_BIS_MARKING_VERIFICATION",
            "BIS CRS Mark Port Verification Rule",
            "Customs port officer verification of embossed BIS registration number and standard label on computing units"
        ),
        (
            r"\b(?:48\s*hours?\s*(?:prior\s+to\s+berthing|advance\s+manifest)|Dangerous\s+Cargo\s+Manifest\s+Filing)\b",
            "RULE_HAZMAT_ADVANCE_MANIFEST",
            "Hazmat Port 48-Hour Advance Manifest Rule",
            "Carrier obligation to transmit Dangerous Cargo Manifest to customs and port health 48 hours prior to berthing"
        ),
        (
            r"\b(?:SCOMET\s+Catch[- ]All\s+Scrutiny|End[- ]User\s+Verification|Appendix\s+2S\s+Scrutiny)\b",
            "RULE_SCOMET_SCREENING",
            "SCOMET Catch-All and End-User Scrutiny Rule",
            "Customs screening of dual-use parameters, technical specifications, and end-user non-proliferation declarations"
        ),
        (
            r"\b(?:Fumigation\s+Inspection|Weed\s+Seed\s+Testing|Plant\s+Quarantine\s+Visual\s+Check)\b",
            "RULE_PLANT_QUARANTINE_INSPECTION",
            "Plant Quarantine Biosecurity Inspection Rule",
            "Port quarantine officer visual draw-sample and laboratory testing for quarantine weed seeds, fungi, and live insects"
        ),
        (
            r"\b(?:Medical\s+Device\s+Shelf[- ]Life\s+Rule|60%\s+Remaining\s+Shelf[- ]Life)\b",
            "RULE_MEDICAL_SHELF_LIFE",
            "Medical Device Shelf-Life & Labeling Rule",
            "Requirement that imported sterile medical goods possess at least 60% residual shelf-life at arrival"
        ),
    ]

    # Tariff Rates: Pattern -> (Entity ID, Standard Name, Description, Properties)
    TARIFF_PATTERNS = [
        (
            r"\b(?:BCD|Basic\s+Customs\s+Duty)\s*[:=]?\s*(7\.5%|7\.5\s*percent)\b",
            "TARIFF_MACHINERY_DEFAULT",
            "Standard Machinery Tariff (BCD: 7.5%, IGST: 18%)",
            "Standard industrial machinery tariff under Chapter 84",
            {"basic_customs_duty": "7.5%", "igst": "18%", "social_welfare_surcharge": "10%"}
        ),
        (
            r"\b(?:(?:BCD|Basic\s+Customs\s+Duty)\s*[:=]?\s*(?:Free|0%)\b|(?:ITA[- ]1|Information\s+Technology\s+Agreement))\b",
            "TARIFF_ITA1_DUTY_FREE",
            "ITA-1 Duty-Free IT Tariff (BCD: 0%, IGST: 18%)",
            "Concessional zero-duty customs schedule for qualifying information technology and computing hardware",
            {"basic_customs_duty": "0%", "igst": "18%", "social_welfare_surcharge": "0%"}
        ),
        (
            r"\b(?:BCD|Basic\s+Customs\s+Duty)\s*[:=]?\s*(?:40%|40\s*percent)\b",
            "TARIFF_AGRICULTURAL_WHEAT",
            "Protective Wheat Tariff (BCD: 40%, IGST: Exempt)",
            "Protective agricultural duty applied on wheat imports under Chapter 10",
            {"basic_customs_duty": "40%", "igst": "Exempt", "social_welfare_surcharge": "10%"}
        ),
    ]

    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize CustomsEntityExtractor.
        
        Args:
            llm_provider: LLM backend ('gemini', 'openai', or 'offline').
                          If offline or None, uses high-precision regex extraction.
        """
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "offline")

    # -------------------------------------------------------------------------
    # Rule-Based Regex Entity Extraction
    # -------------------------------------------------------------------------

    def extract_entities(self, text: str) -> List[CustomsEntity]:
        """
        Extract structured CustomsEntity nodes from input text using high-precision regex.
        
        Args:
            text: Raw customs document text, SOP paragraph, or search query.
            
        Returns:
            List of detected CustomsEntity instances (de-duplicated by ID).
        """
        if not text:
            return []

        extracted_entities: Dict[str, CustomsEntity] = {}

        # 1. Extract HS Codes
        for match in self.HS_CODE_REGEX.finditer(text):
            code_str = match.group(1).strip()
            # Clean and validate
            clean_digits = code_str.replace(".", "")
            if clean_digits in self.EXCLUDED_YEARS and "." not in code_str:
                continue

            # Valid HS headings are 4 digits (e.g. 8429, 8471, 2901) or 6/8 digits
            chapter = int(clean_digits[:2]) if len(clean_digits) >= 2 else 0
            if 1 <= chapter <= 99 and (len(clean_digits) >= 4 or "." in code_str):
                # Standardize ID: e.g. "HS_8429", "HS_8471.30", "HS_8429.11.00"
                entity_id = f"HS_{code_str}"
                name = f"HS Code {code_str}"
                desc = self._lookup_hs_description(code_str)

                extracted_entities[entity_id] = CustomsEntity(
                    id=entity_id,
                    type=EntityType.HS_CODE.value,
                    name=name,
                    description=desc,
                    properties={"hs_code": code_str, "chapter": f"{chapter:02d}"}
                )

        # 2. Extract Required Documents
        for pattern, doc_id, doc_name, doc_desc, issuing_agency in self.DOCUMENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                extracted_entities[doc_id] = CustomsEntity(
                    id=doc_id,
                    type=EntityType.REQUIRED_DOCUMENT.value,
                    name=doc_name,
                    description=doc_desc,
                    properties={"issuing_agency": issuing_agency}
                )

        # 3. Extract Regulations
        for pattern, reg_id, reg_name, reg_desc in self.REGULATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                extracted_entities[reg_id] = CustomsEntity(
                    id=reg_id,
                    type=EntityType.REGULATION.value,
                    name=reg_name,
                    description=reg_desc,
                    properties={}
                )

        # 4. Extract Government Agencies
        for pattern, agency_id, agency_name, agency_desc in self.AGENCY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                extracted_entities[agency_id] = CustomsEntity(
                    id=agency_id,
                    type=EntityType.GOVERNMENT_AGENCY.value,
                    name=agency_name,
                    description=agency_desc,
                    properties={}
                )

        # 5. Extract Product Categories
        for pattern, cat_id, cat_name, cat_desc in self.PRODUCT_CATEGORY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                extracted_entities[cat_id] = CustomsEntity(
                    id=cat_id,
                    type=EntityType.PRODUCT_CATEGORY.value,
                    name=cat_name,
                    description=cat_desc,
                    properties={}
                )

        # 6. Extract Inspection Rules
        for pattern, rule_id, rule_name, rule_desc in self.INSPECTION_RULE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                extracted_entities[rule_id] = CustomsEntity(
                    id=rule_id,
                    type=EntityType.INSPECTION_RULE.value,
                    name=rule_name,
                    description=rule_desc,
                    properties={}
                )

        # 7. Extract Tariff Rates
        for pattern, tariff_id, tariff_name, tariff_desc, props in self.TARIFF_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                extracted_entities[tariff_id] = CustomsEntity(
                    id=tariff_id,
                    type=EntityType.TARIFF_RATE.value,
                    name=tariff_name,
                    description=tariff_desc,
                    properties=props
                )

        return list(extracted_entities.values())

    def _lookup_hs_description(self, hs_code: str) -> str:
        """Lookup canonical description for primary HS codes."""
        code_prefix = hs_code.split(".")[0]
        descriptions = {
            "8429": "Self-propelled bulldozers, angledozers, graders, excavators, loaders, and road rollers",
            "8471": "Automatic data processing machines, computing units, microcomputers, laptops, and servers",
            "8479": "Machines and mechanical appliances having individual functions (industrial robotics, assembly equipment)",
            "8504": "Electrical transformers, static converters, inverters, and rectifiers",
            "8517": "Telephone sets, transmission and switching apparatus for telecommunication and digital networks",
            "8528": "Monitors, projectors, and reception apparatus for television",
            "8541": "Diodes, transistors, photosensitive semiconductor devices, and photovoltaic solar cells",
            "8542": "Electronic integrated circuits (processors, controllers, memories, and amplifiers)",
            "8544": "Insulated wire, cable, and optical fibre cables",
            "2901": "Acyclic hydrocarbons (ethylene, propylene, butene, butadiene)",
            "2804": "Hydrogen, rare gases, and other non-metals (chlorine, silicon)",
            "9018": "Instruments and appliances used in medical, surgical, dental or veterinary sciences",
            "1001": "Wheat and meslin (durum wheat, organic wheat)",
        }
        return descriptions.get(code_prefix, f"Commodity classified under Harmonized System heading {hs_code}")

    # -------------------------------------------------------------------------
    # Heuristic Relation Extraction
    # -------------------------------------------------------------------------

    def extract_relations(
        self,
        text: str,
        entities: Optional[List[CustomsEntity]] = None
    ) -> List[CustomsRelation]:
        """
        Infer semantic relationships between extracted entities based on
        regulatory co-occurrence and domain constraints.
        
        Args:
            text: Raw input text.
            entities: Optional list of pre-extracted entities; if None, runs extract_entities().
            
        Returns:
            List of CustomsRelation instances.
        """
        if entities is None:
            entities = self.extract_entities(text)

        if not entities or len(entities) < 2:
            return []

        # Group entities by type
        by_type: Dict[str, List[CustomsEntity]] = {}
        for e in entities:
            by_type.setdefault(e.type, []).append(e)

        relations: List[CustomsRelation] = []
        seen_edges: Set[Tuple[str, str, str]] = set()

        def add_rel(src_id: str, tgt_id: str, r_type: str, weight: float = 1.0, props: Optional[Dict] = None):
            key = (src_id, tgt_id, r_type)
            if key not in seen_edges and src_id != tgt_id:
                seen_edges.add(key)
                relations.append(CustomsRelation(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=r_type,
                    properties=props or {},
                    weight=weight
                ))

        hs_nodes = by_type.get(EntityType.HS_CODE.value, [])
        cat_nodes = by_type.get(EntityType.PRODUCT_CATEGORY.value, [])
        doc_nodes = by_type.get(EntityType.REQUIRED_DOCUMENT.value, [])
        reg_nodes = by_type.get(EntityType.REGULATION.value, [])
        rule_nodes = by_type.get(EntityType.INSPECTION_RULE.value, [])
        agency_nodes = by_type.get(EntityType.GOVERNMENT_AGENCY.value, [])
        tariff_nodes = by_type.get(EntityType.TARIFF_RATE.value, [])

        # 1. Product Category -> CLASSIFIED_AS -> HS Code
        for cat in cat_nodes:
            for hs in hs_nodes:
                add_rel(cat.id, hs.id, RelationType.CLASSIFIED_AS.value, weight=1.0)

        # 2. HS Code -> REQUIRES_DOC -> Required Document
        # Also Product Category -> REQUIRES_DOC if no HS Code
        subject_nodes = hs_nodes if hs_nodes else cat_nodes
        for subj in subject_nodes:
            for doc in doc_nodes:
                add_rel(subj.id, doc.id, RelationType.REQUIRES_DOC.value, weight=1.0)

        # 3. HS Code -> REGULATED_BY -> Regulation
        for subj in subject_nodes:
            for reg in reg_nodes:
                add_rel(subj.id, reg.id, RelationType.REGULATED_BY.value, weight=1.0)

        # 4. HS Code -> SUBJECT_TO -> Inspection Rule
        for subj in subject_nodes:
            for rule in rule_nodes:
                add_rel(subj.id, rule.id, RelationType.SUBJECT_TO.value, weight=1.0)

        # 5. HS Code -> HAS_TARIFF -> Tariff Rate
        for hs in hs_nodes:
            for tariff in tariff_nodes:
                add_rel(hs.id, tariff.id, RelationType.HAS_TARIFF.value, weight=1.0)

        # 6. Required Document -> ISSUED_BY -> Government Agency
        for doc in doc_nodes:
            target_agency = doc.properties.get("issuing_agency")
            if target_agency:
                add_rel(doc.id, target_agency, RelationType.ISSUED_BY.value, weight=1.0)
            else:
                # If agency co-occurs in document text
                for agency in agency_nodes:
                    add_rel(doc.id, agency.id, RelationType.ISSUED_BY.value, weight=0.8)

        return relations

    # -------------------------------------------------------------------------
    # Optional LLM-Based Entity Extraction Fallback
    # -------------------------------------------------------------------------

    def extract_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Use an LLM (Gemini / OpenAI) to perform open-vocabulary customs entity
        and relation extraction.
        
        Falls back to rule-based extraction if offline or if API calls fail.
        """
        # If offline or provider not set, return rule-based extraction
        if self.llm_provider == "offline" or not self.llm_provider:
            entities = self.extract_entities(text)
            relations = self.extract_relations(text, entities=entities)
            return {
                "source": "rule_based_regex",
                "entities": [e.to_dict() for e in entities],
                "relations": [r.to_dict() for r in relations],
            }

        prompt = f"""You are a specialized customs knowledge graph engineer.
Extract all customs entities and relations from the text below according to this schema:

Entity Types:
- HS_CODE (e.g. HS 8429, 8471.30)
- PRODUCT_CATEGORY (e.g. Industrial Machinery, Hazardous Chemicals)
- REGULATION (e.g. Customs Act 1962, Foreign Trade Policy)
- REQUIRED_DOCUMENT (e.g. Bill of Entry, Commercial Invoice, MSDS, BIS Certificate)
- GOVERNMENT_AGENCY (e.g. CBIC, DGFT, BIS, CDSCO, FSSAI)
- INSPECTION_RULE (e.g. Pre-shipment Chartered Engineer inspection)
- TARIFF_RATE (e.g. BCD 7.5%, IGST 18%)

Relation Types:
- REQUIRES_DOC (HS_CODE / PRODUCT_CATEGORY -> REQUIRED_DOCUMENT)
- REGULATED_BY (HS_CODE -> REGULATION)
- HAS_TARIFF (HS_CODE -> TARIFF_RATE)
- CLASSIFIED_AS (PRODUCT_CATEGORY -> HS_CODE)
- SUBJECT_TO (HS_CODE -> INSPECTION_RULE)
- ISSUED_BY (REQUIRED_DOCUMENT -> GOVERNMENT_AGENCY)

Return ONLY valid JSON matching this schema:
{{
  "entities": [
    {{"id": "...", "type": "...", "name": "...", "description": "..."}}
  ],
  "relations": [
    {{"source_id": "...", "target_id": "...", "relation_type": "..."}}
  ]
}}

Input Text:
\"\"\"{text[:4000]}\"\"\"
"""
        try:
            raw_response = self._call_llm(prompt)
            # Parse JSON from response
            cleaned_json_str = self._clean_json_markdown(raw_response)
            parsed = json.loads(cleaned_json_str)

            # Reconstruct typed instances to validate
            valid_entities = [CustomsEntity.from_dict(e).to_dict() for e in parsed.get("entities", [])]
            valid_relations = [CustomsRelation.from_dict(r).to_dict() for r in parsed.get("relations", [])]

            return {
                "source": f"llm_{self.llm_provider}",
                "entities": valid_entities,
                "relations": valid_relations,
            }
        except Exception as e:
            logger.warning(f"LLM entity extraction failed ({e}); falling back to rule-based regex extraction.")
            entities = self.extract_entities(text)
            relations = self.extract_relations(text, entities=entities)
            return {
                "source": "rule_based_fallback",
                "entities": [e.to_dict() for e in entities],
                "relations": [r.to_dict() for r in relations],
            }

    def _call_llm(self, prompt: str) -> str:
        """Helper to invoke configured LLM provider."""
        if self.llm_provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        elif self.llm_provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            client = OpenAI(api_key=api_key)
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    @staticmethod
    def _clean_json_markdown(text: str) -> str:
        """Strip markdown fences (```json ... ```) from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    # -------------------------------------------------------------------------
    # Unified Public Extraction Interface
    # -------------------------------------------------------------------------

    def extract_all(self, text: str, use_llm: bool = False) -> Dict[str, Any]:
        """
        Unified extraction pipeline returning entities and relations.
        
        Args:
            text: Raw input text.
            use_llm: Whether to attempt LLM extraction before rule-based fallback.
            
        Returns:
            Dictionary with 'entities', 'relations', and extraction 'source'.
        """
        if use_llm and self.llm_provider != "offline":
            return self.extract_with_llm(text)

        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities=entities)

        return {
            "source": "rule_based_regex",
            "entities": [e.to_dict() for e in entities],
            "relations": [r.to_dict() for r in relations],
        }
