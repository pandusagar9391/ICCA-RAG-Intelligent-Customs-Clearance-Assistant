"""
Seed Customs Knowledge Graph Generator for ICCA-RAG.

Generates `data/knowledge_graph/customs_kg.json` covering:
1. Industrial Machinery (HS 8429, 8471, 8479)
2. Hazardous Chemicals (HS 2901, 2804)
3. Dual-Use Electronics (HS 8542, 8517)
4. Medical Devices (HS 9018)
5. Agricultural Imports (HS 1001)
"""

import json
from pathlib import Path

from icca.knowledge_graph.schema import (
    EntityType,
    RelationType,
    CustomsEntity,
    CustomsRelation,
)
from icca.knowledge_graph.graph_store import CustomsGraphStore


def build_seed_customs_kg() -> CustomsGraphStore:
    store = CustomsGraphStore(name="ICCA_Comprehensive_Customs_KG")

    # =========================================================================
    # 1. PRODUCT CATEGORIES
    # =========================================================================
    categories = [
        CustomsEntity(
            id="CAT_INDUSTRIAL_MACHINERY",
            type=EntityType.PRODUCT_CATEGORY,
            name="Industrial Machinery & Earthmoving Equipment",
            description="Self-propelled earthmoving, excavating, and civil construction capital equipment",
            properties={"sector": "Heavy Engineering", "chapters": ["84"]}
        ),
        CustomsEntity(
            id="CAT_COMPUTING_HARDWARE",
            type=EntityType.PRODUCT_CATEGORY,
            name="Computing Hardware & Data Processing Equipment",
            description="Automatic data processing machines, enterprise servers, laptops, and data storage arrays",
            properties={"sector": "Information Technology", "chapters": ["84"]}
        ),
        CustomsEntity(
            id="CAT_AUTOMATED_MACHINERY",
            type=EntityType.PRODUCT_CATEGORY,
            name="Specialized Industrial & Robotics Machinery",
            description="Robotics, automated assembly systems, and machines having individual functions",
            properties={"sector": "Manufacturing Automation", "chapters": ["84"]}
        ),
        CustomsEntity(
            id="CAT_HAZARDOUS_CHEMICALS",
            type=EntityType.PRODUCT_CATEGORY,
            name="Hazardous & Industrial Chemical Substances",
            description="Toxic chemicals, acyclic hydrocarbons, industrial gases, and hazmat compounds",
            properties={"sector": "Chemicals & Petrochemicals", "chapters": ["28", "29"]}
        ),
        CustomsEntity(
            id="CAT_DUAL_USE_ELECTRONICS",
            type=EntityType.PRODUCT_CATEGORY,
            name="Dual-Use Strategic Electronics & Telecom",
            description="High-performance processors, cryptographic hardware, and telecommunication switches",
            properties={"sector": "Defense & Strategic Technologies", "chapters": ["85"]}
        ),
        CustomsEntity(
            id="CAT_MEDICAL_DEVICES",
            type=EntityType.PRODUCT_CATEGORY,
            name="Medical Devices & Diagnostic Instruments",
            description="Surgical apparatus, electro-diagnostic systems, MRI machines, and implants",
            properties={"sector": "Healthcare & Pharmaceuticals", "chapters": ["90"]}
        ),
        CustomsEntity(
            id="CAT_AGRICULTURAL_IMPORTS",
            type=EntityType.PRODUCT_CATEGORY,
            name="Agricultural Produce, Grains & Cereals",
            description="Bulk food grains, wheat, durum wheat, and agricultural commodities",
            properties={"sector": "Agriculture & Food Security", "chapters": ["10"]}
        ),
    ]
    for c in categories:
        store.add_entity(c)

    # =========================================================================
    # 2. HS CODES
    # =========================================================================
    hs_codes = [
        # Machinery (8429, 8471, 8479)
        CustomsEntity(
            id="HS_8429",
            type=EntityType.HS_CODE,
            name="HS 8429 - Self-Propelled Earthmoving Machinery",
            description="Bulldozers, angledozers, graders, levellers, scrapers, mechanical shovels, and excavators",
            properties={"chapter": "84", "heading": "8429", "duty_type": "ad_valorem"}
        ),
        CustomsEntity(
            id="HS_8429.11",
            type=EntityType.HS_CODE,
            name="HS 8429.11 - Track Laying Bulldozers and Angledozers",
            description="Crawler and track-laying heavy earthmoving bulldozers and angledozers",
            properties={"chapter": "84", "heading": "8429", "subheading": "8429.11"}
        ),
        CustomsEntity(
            id="HS_8429.52",
            type=EntityType.HS_CODE,
            name="HS 8429.52 - Hydraulic Excavators with 360° Revolving Superstructure",
            description="Heavy tracked or wheeled hydraulic excavators with full revolving superstructure",
            properties={"chapter": "84", "heading": "8429", "subheading": "8429.52"}
        ),
        CustomsEntity(
            id="HS_8471",
            type=EntityType.HS_CODE,
            name="HS 8471 - Automatic Data Processing Machines & Units",
            description="Digital automatic data processing machines, computing hardware, microcomputers, and servers",
            properties={"chapter": "84", "heading": "8471", "duty_type": "ita1_concessional"}
        ),
        CustomsEntity(
            id="HS_8471.30",
            type=EntityType.HS_CODE,
            name="HS 8471.30 - Portable Data Processing Machines (Laptops & Notebooks)",
            description="Portable computers weighing <= 10 kg consisting of at least a central processing unit, keyboard, and display",
            properties={"chapter": "84", "heading": "8471", "subheading": "8471.30"}
        ),
        CustomsEntity(
            id="HS_8471.50",
            type=EntityType.HS_CODE,
            name="HS 8471.50 - Enterprise Digital Processing Units and Servers",
            description="Enterprise computing processing units, rack servers, and high-performance compute clusters",
            properties={"chapter": "84", "heading": "8471", "subheading": "8471.50"}
        ),
        CustomsEntity(
            id="HS_8479",
            type=EntityType.HS_CODE,
            name="HS 8479 - Machines & Mechanical Appliances Having Individual Functions",
            description="Industrial robotics, automated manufacturing equipment, mixing, and dosing systems",
            properties={"chapter": "84", "heading": "8479"}
        ),

        # Hazardous Chemicals (2901, 2804)
        CustomsEntity(
            id="HS_2901",
            type=EntityType.HS_CODE,
            name="HS 2901 - Acyclic Hydrocarbons",
            description="Saturated and unsaturated acyclic hydrocarbons (ethylene, propylene, butene, butadiene)",
            properties={"chapter": "29", "heading": "2901", "chemical_class": "Organic Petrochemicals"}
        ),
        CustomsEntity(
            id="HS_2804",
            type=EntityType.HS_CODE,
            name="HS 2804 - Hydrogen, Rare Gases and Other Non-Metals",
            description="Industrial gases, hydrogen, argon, chlorine, and electronic-grade silicon",
            properties={"chapter": "28", "heading": "2804", "chemical_class": "Inorganic Non-Metals"}
        ),

        # Dual-Use Electronics (8542, 8517)
        CustomsEntity(
            id="HS_8542",
            type=EntityType.HS_CODE,
            name="HS 8542 - Electronic Integrated Circuits",
            description="Monolithic integrated circuits, microprocessors, microcontrollers, memories, and amplifiers",
            properties={"chapter": "85", "heading": "8542", "dual_use": True}
        ),
        CustomsEntity(
            id="HS_8517",
            type=EntityType.HS_CODE,
            name="HS 8517 - Telecommunications Apparatus & Network Switches",
            description="Switching and routing apparatus, carrier-grade transmission equipment, and digital modems",
            properties={"chapter": "85", "heading": "8517", "dual_use": True}
        ),

        # Medical Devices (9018)
        CustomsEntity(
            id="HS_9018",
            type=EntityType.HS_CODE,
            name="HS 9018 - Medical, Surgical & Veterinary Instruments",
            description="Diagnostic imaging systems, MRI scanners, surgical robotics, catheters, and ophthalmic instruments",
            properties={"chapter": "90", "heading": "9018", "regulated_by": "CDSCO"}
        ),

        # Agricultural Imports (1001)
        CustomsEntity(
            id="HS_1001",
            type=EntityType.HS_CODE,
            name="HS 1001 - Wheat and Meslin",
            description="Durum wheat, seed wheat, and common food wheat grain",
            properties={"chapter": "10", "heading": "1001", "commodity": "Food Grain"}
        ),
    ]
    for h in hs_codes:
        store.add_entity(h)

    # =========================================================================
    # 3. REQUIRED DOCUMENTS
    # =========================================================================
    documents = [
        # Statutory Core Documents
        CustomsEntity(
            id="DOC_BILL_OF_ENTRY_FORM_I",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Bill of Entry Form I (Home Consumption)",
            description="Statutory legal declaration filed under Section 46 of Customs Act 1962 for customs assessment and direct out-of-charge",
            properties={"mandatory": True, "issuing_agency": "AGENCY_CBIC", "portal": "ICEGATE"}
        ),
        CustomsEntity(
            id="DOC_COMMERCIAL_INVOICE",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Commercial Invoice",
            description="Itemized commercial sales document establishing transaction value, Incoterms 2020, currency, and product specification",
            properties={"mandatory": True, "issuing_agency": "AGENCY_EXPORTER"}
        ),
        CustomsEntity(
            id="DOC_PACKING_LIST",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Packing List",
            description="Detailed shipping specification detailing crate/container breakdown, net/gross weights, and part serial numbers",
            properties={"mandatory": True, "issuing_agency": "AGENCY_EXPORTER"}
        ),
        CustomsEntity(
            id="DOC_BILL_OF_LADING",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Bill of Lading / Airway Bill",
            description="Contract of carriage and document of title issued by carrier (MBL/HBL for sea freight, MAWB/HAWB for air)",
            properties={"mandatory": True, "issuing_agency": "AGENCY_CARRIER"}
        ),
        CustomsEntity(
            id="DOC_CERTIFICATE_OF_ORIGIN",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Certificate of Origin (COO)",
            description="Document certifying country of manufacturing; required for non-preferential check or preferential trade duty concession",
            properties={"mandatory": True, "issuing_agency": "AGENCY_CHAMBER_COMMERCE"}
        ),

        # Specialized Machinery Documents
        CustomsEntity(
            id="DOC_CHARTERED_ENGINEER_CERT",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Chartered Engineer Inspection Certificate (CEC)",
            description="Mandatory overseas technical certification for second-hand machinery validating minimum 5-year residual life, depreciated valuation, and absence of CFCs",
            properties={"mandatory": False, "condition": "Applicable for used/refurbished capital machinery", "issuing_agency": "AGENCY_INSPECTION_AGENCY"}
        ),
        CustomsEntity(
            id="DOC_BIS_CRS_CERTIFICATE",
            type=EntityType.REQUIRED_DOCUMENT,
            name="BIS Registration Certificate (CRS / CRO)",
            description="Mandatory certificate under Bureau of Indian Standards Compulsory Registration Scheme for electronics and computing equipment under IS 13252",
            properties={"mandatory": True, "issuing_agency": "AGENCY_BIS"}
        ),
        CustomsEntity(
            id="DOC_WPC_ETA",
            type=EntityType.REQUIRED_DOCUMENT,
            name="WPC Equipment Type Approval (ETA)",
            description="Wireless regulatory certificate required for equipment incorporating Wi-Fi, Bluetooth, or RF transmitters in de-licensed frequency bands",
            properties={"mandatory": False, "condition": "Applicable for wireless/RF equipped electronics", "issuing_agency": "AGENCY_WPC"}
        ),

        # Hazardous Chemicals Documents
        CustomsEntity(
            id="DOC_MSDS_GHS",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Material Safety Data Sheet (16-Section GHS Format)",
            description="Manufacturer-certified chemical safety dossier detailing CAS number, UN number, hazardous decomposition, toxicity LD50, and emergency measures",
            properties={"mandatory": True, "issuing_agency": "AGENCY_MANUFACTURER", "format": "GHS 16-Section"}
        ),
        CustomsEntity(
            id="DOC_DANGEROUS_GOODS_DECLARATION",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Dangerous Goods Declaration (IMO/ICAO Hazmat DGD)",
            description="Signed multimodal dangerous goods declaration specifying UN hazard class, packing group (I/II/III), and flash point",
            properties={"mandatory": True, "issuing_agency": "AGENCY_SHIPPER"}
        ),
        CustomsEntity(
            id="DOC_SPCB_NOC",
            type=EntityType.REQUIRED_DOCUMENT,
            name="State Pollution Control Board (SPCB) Consignment NOC",
            description="Environmental clearance authorization for consignment storage, processing, and effluent handling of hazardous chemicals",
            properties={"mandatory": True, "issuing_agency": "AGENCY_SPCB"}
        ),
        CustomsEntity(
            id="DOC_CPCB_REGISTRATION",
            type=EntityType.REQUIRED_DOCUMENT,
            name="CPCB Hazardous Waste & Chemical Import Registration",
            description="National registration with Central Pollution Control Board for hazardous chemicals and electronic waste management",
            properties={"mandatory": True, "issuing_agency": "AGENCY_CPCB"}
        ),
        CustomsEntity(
            id="DOC_CERTIFICATE_OF_ANALYSIS",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Certificate of Analysis (COA)",
            description="Batch-specific chemical laboratory assay certifying purity percentage, volatile organic compounds, and test parameters matching commercial invoice",
            properties={"mandatory": True, "issuing_agency": "AGENCY_MANUFACTURER"}
        ),

        # Dual-Use Strategic Goods Documents
        CustomsEntity(
            id="DOC_SCOMET_LICENSE",
            type=EntityType.REQUIRED_DOCUMENT,
            name="DGFT SCOMET Export/Import Authorization License",
            description="Special statutory clearance issued by DGFT for items falling under SCOMET categories 7 and 8 (dual-use electronics, crypto, and computing)",
            properties={"mandatory": True, "issuing_agency": "AGENCY_DGFT"}
        ),
        CustomsEntity(
            id="DOC_END_USER_CERTIFICATE",
            type=EntityType.REQUIRED_DOCUMENT,
            name="End-User Certificate (EUC - Appendix 2S)",
            description="Statutory non-proliferation covenant verifying specific civilian facility installation and prohibiting re-export or diversion to military/WMD programs",
            properties={"mandatory": True, "issuing_agency": "AGENCY_END_USER", "form": "Appendix 2S(i) / 2S(ii)"}
        ),
        CustomsEntity(
            id="DOC_TECHNICAL_SPEC_SHEET",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Technical Specifications & Cryptographic Architecture Sheet",
            description="Detailed technical dossier indicating microprocessor performance (TFLOPS), bus width, key length, and encryption algorithms",
            properties={"mandatory": True, "issuing_agency": "AGENCY_MANUFACTURER"}
        ),

        # Medical Devices Documents
        CustomsEntity(
            id="DOC_CDSCO_IMPORT_LICENSE",
            type=EntityType.REQUIRED_DOCUMENT,
            name="CDSCO Medical Device Import License (Form MD-15)",
            description="Statutory import license granted by Central Drugs Standard Control Organization under Medical Devices Rules 2017 for Class A/B/C/D devices",
            properties={"mandatory": True, "issuing_agency": "AGENCY_CDSCO"}
        ),
        CustomsEntity(
            id="DOC_FREE_SALE_CERTIFICATE",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Free Sale Certificate (FSC)",
            description="Certificate issued by national competent health authority confirming medical device is legally marketed without restriction in country of origin",
            properties={"mandatory": True, "issuing_agency": "AGENCY_HEALTH_AUTHORITY"}
        ),
        CustomsEntity(
            id="DOC_ISO_13485_CERT",
            type=EntityType.REQUIRED_DOCUMENT,
            name="ISO 13485 Quality Management System Certificate",
            description="Accredited certification verifying medical device manufacturer compliance with ISO 13485 standards for design and manufacturing quality",
            properties={"mandatory": True, "issuing_agency": "AGENCY_ACCREDITED_REGISTRAR"}
        ),

        # Agricultural Produce Documents
        CustomsEntity(
            id="DOC_PHYTOSANITARY_CERT",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Phytosanitary Certificate",
            description="Official biosecurity certificate issued by National Plant Protection Organization (NPPO) verifying freedom from quarantine pests and weed seeds",
            properties={"mandatory": True, "issuing_agency": "AGENCY_NPPO"}
        ),
        CustomsEntity(
            id="DOC_PLANT_QUARANTINE_CLEARANCE",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Plant Quarantine Import Clearance (PQ Order Form)",
            description="Border biosecurity inspection clearance issued by Indian Directorate of Plant Protection, Quarantine & Storage prior to customs out-of-charge",
            properties={"mandatory": True, "issuing_agency": "AGENCY_DPPQS"}
        ),
        CustomsEntity(
            id="DOC_FSSAI_IMPORT_NOC",
            type=EntityType.REQUIRED_DOCUMENT,
            name="FSSAI Import Food Safety Clearance NOC",
            description="Food safety inspection certificate and visual/lab testing clearance granted through FSSAI Food Import Clearance System (FICS)",
            properties={"mandatory": True, "issuing_agency": "AGENCY_FSSAI"}
        ),
        CustomsEntity(
            id="DOC_NON_GMO_CERTIFICATE",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Non-GMO Certificate for Agricultural Grain Consignments",
            description="Accredited laboratory certificate declaring that imported grain does not contain genetically modified traits above statutory threshold (<= 1%)",
            properties={"mandatory": True, "issuing_agency": "AGENCY_TESTING_LAB"}
        ),
        CustomsEntity(
            id="DOC_FUMIGATION_CERTIFICATE",
            type=EntityType.REQUIRED_DOCUMENT,
            name="Fumigation Treatment Certificate",
            description="Official certification of methyl bromide or phosphine quarantine fumigation treatment applied prior to shipment",
            properties={"mandatory": True, "issuing_agency": "AGENCY_FUMIGATION_AGENCY"}
        ),
    ]
    for d in documents:
        store.add_entity(d)

    # =========================================================================
    # 4. STATUTORY REGULATIONS
    # =========================================================================
    regulations = [
        CustomsEntity(
            id="REG_CUSTOMS_ACT_1962",
            type=EntityType.REGULATION,
            name="Customs Act, 1962",
            description="Primary statute establishing statutory clearance mechanisms, entry examination, assessment, valuation, and penalty provisions",
            properties={"jurisdiction": "India", "enforcing_ministry": "Ministry of Finance"}
        ),
        CustomsEntity(
            id="REG_FOREIGN_TRADE_POLICY_2023",
            type=EntityType.REGULATION,
            name="Foreign Trade Policy (FTP 2023)",
            description="National regulatory framework formulated by DGFT governing trade facilitation, EPCG scheme, import licensing, and ITC(HS) classifications",
            properties={"jurisdiction": "India", "enforcing_ministry": "Ministry of Commerce & Industry"}
        ),
        CustomsEntity(
            id="REG_BIS_ACT_2016",
            type=EntityType.REGULATION,
            name="Bureau of Indian Standards Act, 2016 & Electronics CRO",
            description="Statutory conformity mandate requiring registration and ISI/CRS marking for designated electronics and industrial equipment",
            properties={"standard": "IS 13252 (Part 1): 2010", "scheme": "Compulsory Registration Scheme"}
        ),
        CustomsEntity(
            id="REG_MACHINERY_SAFETY_TECHNICAL_REG",
            type=EntityType.REGULATION,
            name="Machinery and Electrical Equipment Safety Technical Regulations",
            description="Mandatory technical safety standards governing heavy capital goods, earthmoving equipment, and operator safety cabins",
            properties={"sector": "Industrial Safety"}
        ),
        CustomsEntity(
            id="REG_MSIHC_RULES_1989",
            type=EntityType.REGULATION,
            name="Manufacture, Storage and Import of Hazardous Chemical (MSIHC) Rules, 1989",
            description="Environmental statute under EPA 1986 mandating emergency plans, threshold disclosures, and MSDS for toxic and hazardous chemicals",
            properties={"parent_statute": "Environment (Protection) Act, 1986"}
        ),
        CustomsEntity(
            id="REG_HOWM_RULES_2016",
            type=EntityType.REGULATION,
            name="Hazardous and Other Wastes Rules, 2016",
            description="Statute governing transboundary movement, waste tracking, recycling authorization, and port contamination controls",
            properties={"parent_statute": "Environment (Protection) Act, 1986"}
        ),
        CustomsEntity(
            id="REG_IMDG_CODE",
            type=EntityType.REGULATION,
            name="International Maritime Dangerous Goods (IMDG) Code",
            description="IMO international convention regulating maritime stowage, classification, hazardous segregation, and container placarding",
            properties={"organization": "International Maritime Organization (IMO)"}
        ),
        CustomsEntity(
            id="REG_ICAO_DGR",
            type=EntityType.REGULATION,
            name="ICAO Technical Instructions / IATA Dangerous Goods Regulations",
            description="International aviation standards defining permissible net quantities per packaging unit and aircraft compatibility for hazmat",
            properties={"organization": "ICAO / IATA"}
        ),
        CustomsEntity(
            id="REG_SCOMET_FRAMEWORK",
            type=EntityType.REGULATION,
            name="SCOMET Strategic Dual-Use Export/Import Control Framework",
            description="Schedule 2 Appendix 3 of ITC(HS) regulating munitions, nuclear components, advanced cryptography, and high-performance electronics",
            properties={"aligned_regimes": ["Wassenaar Arrangement", "MTCR", "NSG", "Australia Group"]}
        ),
        CustomsEntity(
            id="REG_WMD_ACT_2005",
            type=EntityType.REGULATION,
            name="Weapons of Mass Destruction Act, 2005 (Catch-All Controls)",
            description="Section 13 statutory Catch-All provision empowering customs to detain unlisted goods if suspected of WMD delivery proliferation",
            properties={"mechanism": "Intent-based catch-all screening"}
        ),
        CustomsEntity(
            id="REG_MEDICAL_DEVICES_RULES_2017",
            type=EntityType.REGULATION,
            name="Medical Devices Rules, 2017 (MDR-2017)",
            description="Comprehensive regulatory code framed under Drugs and Cosmetics Act 1940 governing risk-based classification (A, B, C, D) and import licensing",
            properties={"parent_statute": "Drugs and Cosmetics Act, 1940"}
        ),
        CustomsEntity(
            id="REG_PLANT_QUARANTINE_ORDER_2003",
            type=EntityType.REGULATION,
            name="Plant Quarantine (Regulation of Import into India) Order, 2003",
            description="Biosecurity regulation issued under Destructive Insects and Pests Act 1914 prohibiting introduction of invasive pests and quarantine weed seeds",
            properties={"parent_statute": "Destructive Insects and Pests Act, 1914"}
        ),
        CustomsEntity(
            id="REG_FSSAI_IMPORT_REGS_2017",
            type=EntityType.REGULATION,
            name="Food Safety and Standards (Import) Regulations, 2017",
            description="National regulatory standards governing food safety sampling, laboratory testing, proprietary food clearance, and labeling",
            properties={"parent_statute": "Food Safety and Standards Act, 2006"}
        ),
    ]
    for r in regulations:
        store.add_entity(r)

    # =========================================================================
    # 5. GOVERNMENT AGENCIES
    # =========================================================================
    agencies = [
        CustomsEntity(
            id="AGENCY_CBIC",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Central Board of Indirect Taxes and Customs (CBIC) / Indian Customs",
            description="Principal revenue and border enforcement body administering customs clearance at all air cargo complexes, seaports, and ICDs",
            properties={"ministry": "Ministry of Finance"}
        ),
        CustomsEntity(
            id="AGENCY_DGFT",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Directorate General of Foreign Trade (DGFT)",
            description="Commerce department authority administering foreign trade policies, SCOMET licenses, EPCG authorizations, and import-export quotas",
            properties={"ministry": "Ministry of Commerce and Industry"}
        ),
        CustomsEntity(
            id="AGENCY_BIS",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Bureau of Indian Standards (BIS)",
            description="National standardization body running the Compulsory Registration Scheme (CRS) for electronics, IT goods, and steel products",
            properties={"ministry": "Ministry of Consumer Affairs, Food & Public Distribution"}
        ),
        CustomsEntity(
            id="AGENCY_WPC",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Wireless Planning & Coordination (WPC) Wing",
            description="National radio frequency licensing authority issuing Equipment Type Approvals for wireless and telecommunication products",
            properties={"ministry": "Ministry of Communications"}
        ),
        CustomsEntity(
            id="AGENCY_PESO",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Petroleum and Explosives Safety Organization (PESO)",
            description="Industrial safety agency supervising transportation, storage, and clearance of compressed gases and hazardous chemical compounds",
            properties={"ministry": "Ministry of Commerce and Industry (DPIIT)"}
        ),
        CustomsEntity(
            id="AGENCY_CPCB",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Central Pollution Control Board (CPCB)",
            description="Apex environmental authority managing chemical risk thresholds, hazardous waste trade permissions, and EPR obligations",
            properties={"ministry": "Ministry of Environment, Forest and Climate Change"}
        ),
        CustomsEntity(
            id="AGENCY_SPCB",
            type=EntityType.GOVERNMENT_AGENCY,
            name="State Pollution Control Board (SPCB)",
            description="State-level environmental board granting destination consent-to-establish (CTE) and consignment movement authorizations",
            properties={"jurisdiction": "State Governments"}
        ),
        CustomsEntity(
            id="AGENCY_CDSCO",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Central Drugs Standard Control Organization (CDSCO)",
            description="National regulatory authority evaluating medical device safety, efficacy, manufacturing standards, and granting import licenses",
            properties={"ministry": "Ministry of Health and Family Welfare"}
        ),
        CustomsEntity(
            id="AGENCY_DPPQS",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Directorate of Plant Protection, Quarantine & Storage (DPPQS)",
            description="National biosecurity agency inspecting agricultural imports, testing grain samples, and enforcing quarantine pest protocols",
            properties={"ministry": "Ministry of Agriculture and Farmers Welfare"}
        ),
        CustomsEntity(
            id="AGENCY_FSSAI",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Food Safety and Standards Authority of India (FSSAI)",
            description="National food regulator inspecting and clearing commercial food and edible grain imports via accredited testing laboratories",
            properties={"ministry": "Ministry of Health and Family Welfare"}
        ),
        CustomsEntity(
            id="AGENCY_MEA",
            type=EntityType.GOVERNMENT_AGENCY,
            name="Ministry of External Affairs (MEA)",
            description="Foreign affairs ministry evaluating strategic non-proliferation covenants and SCOMET end-user authorizations",
            properties={"role": "Strategic Export Control Review"}
        ),
    ]
    for a in agencies:
        store.add_entity(a)

    # =========================================================================
    # 6. INSPECTION RULES
    # =========================================================================
    inspection_rules = [
        CustomsEntity(
            id="RULE_SECOND_HAND_MACHINERY_INSPECTION",
            type=EntityType.INSPECTION_RULE,
            name="Second-Hand Machinery Pre-Shipment Inspection Rule",
            description="Mandatory verification by Chartered Engineer at export origin certifying >= 5 years residual life, depreciated valuation, and no CFC gases",
            properties={"trigger": "Used or refurbished machinery import", "target_headings": ["8429", "8479"]}
        ),
        CustomsEntity(
            id="RULE_BIS_CRS_MARKING_VERIFICATION",
            type=EntityType.INSPECTION_RULE,
            name="BIS Standard Mark & CRO Registration Port Verification Rule",
            description="Customs dock officer visual examination verifying embossed standard mark and valid BIS registration number (R-XXXXXXXX) on computing hardware",
            properties={"trigger": "Arrival of computing hardware", "target_headings": ["8471"]}
        ),
        CustomsEntity(
            id="RULE_HAZMAT_ADVANCE_MANIFEST_48H",
            type=EntityType.INSPECTION_RULE,
            name="Hazardous Cargo 48-Hour Advance Manifest Filing Mandate",
            description="Carrier requirement to transmit Dangerous Cargo Manifest (DCM) to port health and customs boarding officers 48 hours prior to berthing",
            properties={"trigger": "Maritime carriage of IMDG dangerous goods", "target_headings": ["2901", "2804"]}
        ),
        CustomsEntity(
            id="RULE_UN_PACKAGING_VERIFICATION",
            type=EntityType.INSPECTION_RULE,
            name="UN Specification Packaging & Marking Verification",
            description="Verification of embossed UN specification code on packaging drums confirming pressure, drop, and leakproof test compliance",
            properties={"trigger": "Chemical container packaging", "target_headings": ["2901", "2804"]}
        ),
        CustomsEntity(
            id="RULE_SCOMET_CATCH_ALL_SCRUTINY",
            type=EntityType.INSPECTION_RULE,
            name="SCOMET Catch-All Proliferation & End-User Scrutiny",
            description="Customs appraising evaluation of high-performance microprocessors and encrypted hardware against Wassenaar performance thresholds",
            properties={"trigger": "Dual-use technology import/export", "target_headings": ["8542", "8517"]}
        ),
        CustomsEntity(
            id="RULE_MEDICAL_DEVICE_SHELF_LIFE",
            type=EntityType.INSPECTION_RULE,
            name="Medical Device Residual Shelf-Life Rule (>= 60%)",
            description="Statutory import prerequisite requiring sterile medical consumables and diagnostic kits to have at least 60% shelf life remaining at entry",
            properties={"trigger": "Medical device border clearance", "target_headings": ["9018"]}
        ),
        CustomsEntity(
            id="RULE_PLANT_QUARANTINE_LAB_TESTING",
            type=EntityType.INSPECTION_RULE,
            name="Plant Quarantine Draw-Sample & Weed Seed Laboratory Testing Rule",
            description="Plant quarantine officer visual draw-sample inspection and laboratory assay for exotic weed seeds (Ambrosia, Striga) and aflatoxin levels",
            properties={"trigger": "Grain consignment clearance", "target_headings": ["1001"]}
        ),
    ]
    for r in inspection_rules:
        store.add_entity(r)

    # =========================================================================
    # 7. TARIFF RATES
    # =========================================================================
    tariff_rates = [
        CustomsEntity(
            id="TARIFF_8429_DEFAULT",
            type=EntityType.TARIFF_RATE,
            name="Earthmoving Machinery Tariff (BCD: 7.5%, IGST: 18%)",
            description="Standard customs duty schedule for self-propelled bulldozers, excavators, and earthmoving capital equipment",
            properties={"basic_customs_duty": "7.5%", "igst": "18%", "social_welfare_surcharge": "10%", "aidc": "0%"}
        ),
        CustomsEntity(
            id="TARIFF_8471_ITA1",
            type=EntityType.TARIFF_RATE,
            name="ITA-1 Duty-Free IT Tariff (BCD: 0%, IGST: 18%)",
            description="Concessional duty-free schedule under Information Technology Agreement (ITA-1) for laptops, microcomputers, and data servers",
            properties={"basic_customs_duty": "0%", "igst": "18%", "social_welfare_surcharge": "0%", "concession": "ITA-1"}
        ),
        CustomsEntity(
            id="TARIFF_8479_ROBOTICS",
            type=EntityType.TARIFF_RATE,
            name="Industrial Robotics Tariff (BCD: 7.5%, IGST: 18%)",
            description="Customs duty schedule for automated industrial robotics and individual function mechanical appliances",
            properties={"basic_customs_duty": "7.5%", "igst": "18%", "social_welfare_surcharge": "10%"}
        ),
        CustomsEntity(
            id="TARIFF_2901_HYDROCARBONS",
            type=EntityType.TARIFF_RATE,
            name="Acyclic Hydrocarbons Chemical Tariff (BCD: 2.5%, IGST: 18%)",
            description="Concessional duty rate for primary petrochemical building blocks (ethylene, propylene)",
            properties={"basic_customs_duty": "2.5%", "igst": "18%", "social_welfare_surcharge": "10%"}
        ),
        CustomsEntity(
            id="TARIFF_2804_INDUSTRIAL_GASES",
            type=EntityType.TARIFF_RATE,
            name="Industrial Non-Metals & Gas Tariff (BCD: 5.0%, IGST: 18%)",
            description="Standard duty rate for chlorine, industrial non-metals, and rare gases",
            properties={"basic_customs_duty": "5.0%", "igst": "18%", "social_welfare_surcharge": "10%"}
        ),
        CustomsEntity(
            id="TARIFF_8542_SEMICONDUCTORS",
            type=EntityType.TARIFF_RATE,
            name="Semiconductor Integrated Circuit Tariff (BCD: 0%, IGST: 18%)",
            description="Duty-free tariff under ITA-1 for electronic processors, microcontrollers, and memories",
            properties={"basic_customs_duty": "0%", "igst": "18%", "social_welfare_surcharge": "0%", "concession": "ITA-1"}
        ),
        CustomsEntity(
            id="TARIFF_8517_TELECOM",
            type=EntityType.TARIFF_RATE,
            name="Telecom Network Apparatus Tariff (BCD: 10.0%, IGST: 18%)",
            description="Standard customs duty for carrier-grade routers, digital switches, and transmission units",
            properties={"basic_customs_duty": "10.0%", "igst": "18%", "social_welfare_surcharge": "10%"}
        ),
        CustomsEntity(
            id="TARIFF_9018_MEDICAL_DEVICES",
            type=EntityType.TARIFF_RATE,
            name="Medical Equipment Tariff (BCD: 7.5%, IGST: 12%, Health Cess: 5%)",
            description="Duty schedule for medical diagnostic equipment, surgical devices, and electromedical apparatus",
            properties={"basic_customs_duty": "7.5%", "igst": "12%", "health_cess": "5%", "social_welfare_surcharge": "10%"}
        ),
        CustomsEntity(
            id="TARIFF_1001_PROTECTIVE_WHEAT",
            type=EntityType.TARIFF_RATE,
            name="Protective Wheat Tariff (BCD: 40.0%, IGST: Exempt)",
            description="Protective domestic customs tariff levied on wheat and meslin imports to shield domestic farmers",
            properties={"basic_customs_duty": "40.0%", "igst": "Exempt", "social_welfare_surcharge": "10%"}
        ),
    ]
    for t in tariff_rates:
        store.add_entity(t)

    # =========================================================================
    # 8. COUNTRIES
    # =========================================================================
    countries = [
        CustomsEntity(id="COUNTRY_IN", type=EntityType.COUNTRY, name="India", description="Importing Sovereign Jurisdiction"),
        CustomsEntity(id="COUNTRY_US", type=EntityType.COUNTRY, name="United States", description="Exporting Trading Jurisdiction"),
        CustomsEntity(id="COUNTRY_DE", type=EntityType.COUNTRY, name="Germany", description="Exporting Trading Jurisdiction"),
        CustomsEntity(id="COUNTRY_JP", type=EntityType.COUNTRY, name="Japan", description="Exporting Trading Jurisdiction"),
        CustomsEntity(id="COUNTRY_CN", type=EntityType.COUNTRY, name="China", description="Exporting Trading Jurisdiction"),
    ]
    for cn in countries:
        store.add_entity(cn)

    # =========================================================================
    # 9. RELATIONSHIPS (EDGES)
    # =========================================================================
    relations = [
        # --- Category -> HS Code (CLASSIFIED_AS) ---
        CustomsRelation("CAT_INDUSTRIAL_MACHINERY", "HS_8429", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_INDUSTRIAL_MACHINERY", "HS_8429.11", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_INDUSTRIAL_MACHINERY", "HS_8429.52", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_COMPUTING_HARDWARE", "HS_8471", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_COMPUTING_HARDWARE", "HS_8471.30", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_COMPUTING_HARDWARE", "HS_8471.50", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_AUTOMATED_MACHINERY", "HS_8479", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_HAZARDOUS_CHEMICALS", "HS_2901", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_HAZARDOUS_CHEMICALS", "HS_2804", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_DUAL_USE_ELECTRONICS", "HS_8542", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_DUAL_USE_ELECTRONICS", "HS_8517", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_MEDICAL_DEVICES", "HS_9018", RelationType.CLASSIFIED_AS),
        CustomsRelation("CAT_AGRICULTURAL_IMPORTS", "HS_1001", RelationType.CLASSIFIED_AS),

        # --- Industrial Machinery (HS 8429, 8471, 8479) Required Documents ---
        CustomsRelation("HS_8429", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8429", "DOC_COMMERCIAL_INVOICE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8429", "DOC_PACKING_LIST", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8429", "DOC_BILL_OF_LADING", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8429", "DOC_CERTIFICATE_OF_ORIGIN", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8429", "DOC_CHARTERED_ENGINEER_CERT", RelationType.REQUIRES_DOC),

        CustomsRelation("HS_8471", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471", "DOC_COMMERCIAL_INVOICE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471", "DOC_PACKING_LIST", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471", "DOC_BILL_OF_LADING", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471", "DOC_CERTIFICATE_OF_ORIGIN", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471", "DOC_BIS_CRS_CERTIFICATE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471.30", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471.30", "DOC_COMMERCIAL_INVOICE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471.30", "DOC_PACKING_LIST", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471.30", "DOC_BIS_CRS_CERTIFICATE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471.30", "DOC_WPC_ETA", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471.50", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8471.50", "DOC_BIS_CRS_CERTIFICATE", RelationType.REQUIRES_DOC),

        CustomsRelation("HS_8429.11", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8429.11", "DOC_CHARTERED_ENGINEER_CERT", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8429.52", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8429.52", "DOC_CHARTERED_ENGINEER_CERT", RelationType.REQUIRES_DOC),

        CustomsRelation("HS_8479", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8479", "DOC_COMMERCIAL_INVOICE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8479", "DOC_PACKING_LIST", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8479", "DOC_CERTIFICATE_OF_ORIGIN", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8479", "DOC_CHARTERED_ENGINEER_CERT", RelationType.REQUIRES_DOC),

        # --- Hazardous Chemicals (HS 2901, 2804) Required Documents ---
        CustomsRelation("HS_2901", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2901", "DOC_COMMERCIAL_INVOICE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2901", "DOC_PACKING_LIST", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2901", "DOC_MSDS_GHS", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2901", "DOC_DANGEROUS_GOODS_DECLARATION", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2901", "DOC_SPCB_NOC", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2901", "DOC_CPCB_REGISTRATION", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2901", "DOC_CERTIFICATE_OF_ANALYSIS", RelationType.REQUIRES_DOC),

        CustomsRelation("HS_2804", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2804", "DOC_MSDS_GHS", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2804", "DOC_DANGEROUS_GOODS_DECLARATION", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2804", "DOC_SPCB_NOC", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_2804", "DOC_CERTIFICATE_OF_ANALYSIS", RelationType.REQUIRES_DOC),

        # --- Dual-Use Electronics (HS 8542, 8517) Required Documents ---
        CustomsRelation("HS_8542", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8542", "DOC_COMMERCIAL_INVOICE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8542", "DOC_SCOMET_LICENSE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8542", "DOC_END_USER_CERTIFICATE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8542", "DOC_TECHNICAL_SPEC_SHEET", RelationType.REQUIRES_DOC),

        CustomsRelation("HS_8517", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8517", "DOC_COMMERCIAL_INVOICE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8517", "DOC_SCOMET_LICENSE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8517", "DOC_END_USER_CERTIFICATE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_8517", "DOC_WPC_ETA", RelationType.REQUIRES_DOC),

        # --- Medical Devices (HS 9018) Required Documents ---
        CustomsRelation("HS_9018", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_9018", "DOC_COMMERCIAL_INVOICE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_9018", "DOC_PACKING_LIST", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_9018", "DOC_CDSCO_IMPORT_LICENSE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_9018", "DOC_FREE_SALE_CERTIFICATE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_9018", "DOC_ISO_13485_CERT", RelationType.REQUIRES_DOC),

        # --- Agricultural Imports (HS 1001) Required Documents ---
        CustomsRelation("HS_1001", "DOC_BILL_OF_ENTRY_FORM_I", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_1001", "DOC_COMMERCIAL_INVOICE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_1001", "DOC_PHYTOSANITARY_CERT", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_1001", "DOC_PLANT_QUARANTINE_CLEARANCE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_1001", "DOC_FSSAI_IMPORT_NOC", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_1001", "DOC_NON_GMO_CERTIFICATE", RelationType.REQUIRES_DOC),
        CustomsRelation("HS_1001", "DOC_FUMIGATION_CERTIFICATE", RelationType.REQUIRES_DOC),

        # --- Applicable Regulations (REGULATED_BY) ---
        CustomsRelation("HS_8429", "REG_CUSTOMS_ACT_1962", RelationType.REGULATED_BY),
        CustomsRelation("HS_8429", "REG_FOREIGN_TRADE_POLICY_2023", RelationType.REGULATED_BY),
        CustomsRelation("HS_8429", "REG_MACHINERY_SAFETY_TECHNICAL_REG", RelationType.REGULATED_BY),

        CustomsRelation("HS_8471", "REG_CUSTOMS_ACT_1962", RelationType.REGULATED_BY),
        CustomsRelation("HS_8471", "REG_FOREIGN_TRADE_POLICY_2023", RelationType.REGULATED_BY),
        CustomsRelation("HS_8471", "REG_BIS_ACT_2016", RelationType.REGULATED_BY),

        CustomsRelation("HS_8479", "REG_CUSTOMS_ACT_1962", RelationType.REGULATED_BY),
        CustomsRelation("HS_8479", "REG_FOREIGN_TRADE_POLICY_2023", RelationType.REGULATED_BY),

        CustomsRelation("HS_2901", "REG_MSIHC_RULES_1989", RelationType.REGULATED_BY),
        CustomsRelation("HS_2901", "REG_HOWM_RULES_2016", RelationType.REGULATED_BY),
        CustomsRelation("HS_2901", "REG_IMDG_CODE", RelationType.REGULATED_BY),

        CustomsRelation("HS_2804", "REG_MSIHC_RULES_1989", RelationType.REGULATED_BY),
        CustomsRelation("HS_2804", "REG_ICAO_DGR", RelationType.REGULATED_BY),

        CustomsRelation("HS_8542", "REG_SCOMET_FRAMEWORK", RelationType.REGULATED_BY),
        CustomsRelation("HS_8542", "REG_WMD_ACT_2005", RelationType.REGULATED_BY),
        CustomsRelation("HS_8517", "REG_SCOMET_FRAMEWORK", RelationType.REGULATED_BY),
        CustomsRelation("HS_8517", "REG_WMD_ACT_2005", RelationType.REGULATED_BY),

        CustomsRelation("HS_9018", "REG_MEDICAL_DEVICES_RULES_2017", RelationType.REGULATED_BY),
        CustomsRelation("HS_9018", "REG_CUSTOMS_ACT_1962", RelationType.REGULATED_BY),

        CustomsRelation("HS_1001", "REG_PLANT_QUARANTINE_ORDER_2003", RelationType.REGULATED_BY),
        CustomsRelation("HS_1001", "REG_FSSAI_IMPORT_REGS_2017", RelationType.REGULATED_BY),
        CustomsRelation("HS_1001", "REG_CUSTOMS_ACT_1962", RelationType.REGULATED_BY),

        # --- Special Inspection Mandates (SUBJECT_TO) ---
        CustomsRelation("HS_8429", "RULE_SECOND_HAND_MACHINERY_INSPECTION", RelationType.SUBJECT_TO),
        CustomsRelation("HS_8479", "RULE_SECOND_HAND_MACHINERY_INSPECTION", RelationType.SUBJECT_TO),
        CustomsRelation("HS_8471", "RULE_BIS_CRS_MARKING_VERIFICATION", RelationType.SUBJECT_TO),
        CustomsRelation("HS_2901", "RULE_HAZMAT_ADVANCE_MANIFEST_48H", RelationType.SUBJECT_TO),
        CustomsRelation("HS_2901", "RULE_UN_PACKAGING_VERIFICATION", RelationType.SUBJECT_TO),
        CustomsRelation("HS_2804", "RULE_HAZMAT_ADVANCE_MANIFEST_48H", RelationType.SUBJECT_TO),
        CustomsRelation("HS_8542", "RULE_SCOMET_CATCH_ALL_SCRUTINY", RelationType.SUBJECT_TO),
        CustomsRelation("HS_8517", "RULE_SCOMET_CATCH_ALL_SCRUTINY", RelationType.SUBJECT_TO),
        CustomsRelation("HS_9018", "RULE_MEDICAL_DEVICE_SHELF_LIFE", RelationType.SUBJECT_TO),
        CustomsRelation("HS_1001", "RULE_PLANT_QUARANTINE_LAB_TESTING", RelationType.SUBJECT_TO),

        # --- Tariffs (HAS_TARIFF) ---
        CustomsRelation("HS_8429", "TARIFF_8429_DEFAULT", RelationType.HAS_TARIFF),
        CustomsRelation("HS_8429.11", "TARIFF_8429_DEFAULT", RelationType.HAS_TARIFF),
        CustomsRelation("HS_8429.52", "TARIFF_8429_DEFAULT", RelationType.HAS_TARIFF),
        CustomsRelation("HS_8471", "TARIFF_8471_ITA1", RelationType.HAS_TARIFF),
        CustomsRelation("HS_8471.30", "TARIFF_8471_ITA1", RelationType.HAS_TARIFF),
        CustomsRelation("HS_8471.50", "TARIFF_8471_ITA1", RelationType.HAS_TARIFF),
        CustomsRelation("HS_8479", "TARIFF_8479_ROBOTICS", RelationType.HAS_TARIFF),
        CustomsRelation("HS_2901", "TARIFF_2901_HYDROCARBONS", RelationType.HAS_TARIFF),
        CustomsRelation("HS_2804", "TARIFF_2804_INDUSTRIAL_GASES", RelationType.HAS_TARIFF),
        CustomsRelation("HS_8542", "TARIFF_8542_SEMICONDUCTORS", RelationType.HAS_TARIFF),
        CustomsRelation("HS_8517", "TARIFF_8517_TELECOM", RelationType.HAS_TARIFF),
        CustomsRelation("HS_9018", "TARIFF_9018_MEDICAL_DEVICES", RelationType.HAS_TARIFF),
        CustomsRelation("HS_1001", "TARIFF_1001_PROTECTIVE_WHEAT", RelationType.HAS_TARIFF),

        # --- Document Issuing Agencies (ISSUED_BY) ---
        CustomsRelation("DOC_BILL_OF_ENTRY_FORM_I", "AGENCY_CBIC", RelationType.ISSUED_BY),
        CustomsRelation("DOC_BIS_CRS_CERTIFICATE", "AGENCY_BIS", RelationType.ISSUED_BY),
        CustomsRelation("DOC_WPC_ETA", "AGENCY_WPC", RelationType.ISSUED_BY),
        CustomsRelation("DOC_SPCB_NOC", "AGENCY_SPCB", RelationType.ISSUED_BY),
        CustomsRelation("DOC_CPCB_REGISTRATION", "AGENCY_CPCB", RelationType.ISSUED_BY),
        CustomsRelation("DOC_SCOMET_LICENSE", "AGENCY_DGFT", RelationType.ISSUED_BY),
        CustomsRelation("DOC_CDSCO_IMPORT_LICENSE", "AGENCY_CDSCO", RelationType.ISSUED_BY),
        CustomsRelation("DOC_PLANT_QUARANTINE_CLEARANCE", "AGENCY_DPPQS", RelationType.ISSUED_BY),
        CustomsRelation("DOC_FSSAI_IMPORT_NOC", "AGENCY_FSSAI", RelationType.ISSUED_BY),

        # --- Tariff Exemptions / Concessions (EXEMPT_UNDER) ---
        CustomsRelation("TARIFF_8471_ITA1", "REG_FOREIGN_TRADE_POLICY_2023", RelationType.EXEMPT_UNDER),
        CustomsRelation("TARIFF_8542_SEMICONDUCTORS", "REG_FOREIGN_TRADE_POLICY_2023", RelationType.EXEMPT_UNDER),
    ]
    for rel in relations:
        store.add_relation(rel)

    return store


if __name__ == "__main__":
    kg = build_seed_customs_kg()
    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "customs_kg.json"
    kg.save_json(out_json)
    print(f"Customs KG successfully built and saved to {out_json}")
    print("Graph Statistics:")
    print(kg.get_stats())
