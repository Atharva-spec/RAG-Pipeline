# 🌿 AYUSH IPR & Regulatory Intelligence Engine: Architecture, Corpus & Pivot Roadmap

---

## 🎯 1. Executive Summary & Purpose Pivot

This document details the transition of the Conversational RAG pipeline from a generic document assistant into a specialized, high-fidelity **Ayurvedic IPR, Regulatory, and ABS (Access and Benefit Sharing) Intelligence Engine**.

The objective is **not generic SaaS automation**, but solving the acute systemic challenges faced by the Indian AYUSH ecosystem (Vaidyas, startups, MSMEs, researchers, and cultivators):
1. **Preventing Misappropriation Abroad**: Grounding claims in codified traditional knowledge, defending against bio-piracy via TKDL alignment and WIPO GRATK (2024) compliance.
2. **Preventing Rejection at Home**: Navigating Section 3(p) (traditional knowledge bar) and Section 3(e) (mere admixture bar) of the Indian Patents Act with inventive-step / synergy assessment.
3. **Regulatory Clarity Before IP**: Integrating the drug/food regulatory framework (Drugs and Cosmetics Act, FSSAI Ayurveda-Aahar, Phytopharmaceuticals) directly into the IP assessment.
4. **Demystifying ABS Duties**: Untangling the Biological Diversity (Amendment) Act 2023 and 2024 Rules (who pays ABS fees, who is exempt, and when NBA Form III is mandatory).
5. **Traceable Grounding & Zero Fabrication**: Every recommendation is anchored in official gazettes, section/rule citations, and authoritative databases, with a strict "Information, Not Legal Advice" posture.

---

## 🏛️ 2. Authoritative Open Public Corpus Matrix

The knowledge base is constructed exclusively from verified, version-tracked public legal and scientific repositories.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             OPEN AUTHORITATIVE REPOSITORIES                              │
├──────────────────────────┬──────────────────────────┬────────────────────────────────────┤
│ REPOSITORY & SOURCE      │ CANONICAL URL            │ DATA EXTRACTED & LEGAL SIGNIFICANCE│
├──────────────────────────┼──────────────────────────┼────────────────────────────────────┤
│ 1. Traditional Knowledge │ https://tkdl.res.in      │ • Prior-art formulations from 54   │
│    Digital Library       │                          │   First Schedule classical texts   │
│    (TKDL) - CSIR/AYUSH   │                          │ • International patent revocation  │
│                          │                          │   case studies & TK classification │
├──────────────────────────┼──────────────────────────┼────────────────────────────────────┤
│ 2. India Code            │ https://indiacode.nic.in │ • Patents Act, 1970 (amended 2024) │
│    (Legislative Dept,    │                          │ • Biological Diversity Act (2023)  │
│    Ministry of Law)      │                          │ • Drugs & Cosmetics Act, 1940      │
│                          │                          │ • Trademarks Act 1999, GI Act 1999 │
│                          │                          │ • DMR(OA) Act 1954 (Schedule J)    │
│                          │                          │ • PPV&FR Act 2001 (Plant Varieties)│
├──────────────────────────┼──────────────────────────┼────────────────────────────────────┤
│ 3. IP India Public       │ https://ipindia.gov.in   │ • InPASS Indian Patent Database    │
│    Registries            │                          │ • Manual of Patent Practice (TK)   │
│    (CGPDTM)              │                          │ • Trademark Search (Classes 5,30,3)│
│                          │                          │ • Registered GI of Goods Registry  │
│                          │                          │ • Designs Registry database        │
├──────────────────────────┼──────────────────────────┼────────────────────────────────────┤
│ 4. National Biodiversity │ https://nbaindia.org     │ • Biological Diversity Rules 2024  │
│    Authority (NBA) & ABS │                          │ • ABS Guidelines & Benefit Formulas│
│    e-Licensing Portal    │                          │ • NBA Forms I, II, III templates   │
│                          │                          │ • Exempted Cultivated Species lists│
├──────────────────────────┼──────────────────────────┼────────────────────────────────────┤
│ 5. Pharmacopoeia Comm.   │ https://pcimh.gov.in     │ • Ayurvedic Pharmacopoeia of India │
│    for Indian Medicine   │                          │   (API) Vol I-IX (Monographs)      │
│    (PCIM&H / AYUSH)      │                          │ • Ayurvedic Formulary of India(AFI)│
├──────────────────────────┼──────────────────────────┼────────────────────────────────────┤
│ 6. Food Safety (FSSAI)   │ https://fssai.gov.in     │ • FSSAI (Ayurveda-Aahar) 2022      │
│                          │                          │ • Labelling & Display Regulations  │
│                          │                          │ • Prohibited therapeutic claims    │
├──────────────────────────┼──────────────────────────┼────────────────────────────────────┤
│ 7. International Treaties│ https://wipo.int         │ • WIPO GRATK Treaty (May 2024)     │
│    (WIPO & CBD)          │ https://cbd.int          │ • Nagoya Protocol on ABS           │
│                          │                          │ • PCT Guidelines & Budapest Treaty │
└──────────────────────────┴──────────────────────────┴────────────────────────────────────┘
```

---

## 💡 3. Key Improvisations Designed for the AYUSH Ecosystem

| # | Improvisation Feature | Problem It Solves | Technical Implementation |
| :--- | :--- | :--- | :--- |
| **1** | **AyurTaxa Canonical Entity Resolver** | A user says *"Giloy"*, API says *"Guduchi"*, classical texts say *"Amrita"*, botany says *"Tinospora cordifolia"*. Standard search misses prior art. | Pre-retrieval polyherbal synonym dictionary mapping vernacular, Sanskrit, trade names to canonical Taxon IDs & API monographs. |
| **2** | **Section 3(p) & 3(e) Patentability Analyzer** | Over 85% of AYUSH patents are rejected as traditional knowledge or mere admixtures. | Evaluates if formulation is verbatim classical vs modified; calculates required synergy index (Chou-Talalay CI < 1.0) or pivots to novel extraction/process patent. |
| **3** | **Dynamic ABS Compliance Calculator** | The 2023 Biodiversity Amendment exempted AYUSH practitioners and cultivated plants from ABS fees, but foreign-equity companies remain liable under Sec 3(2). | Interactive corporate profiler (Foreign shareholding check + Cultivated vs Wild-harvested sourcing check) to output exact NBA approval roadmap. |
| **4** | **360° Ayur-IP Portfolio Architect** | Startups give up if told they cannot patent a classical formulation. | Formulates alternative holistic protection: Trade Secrets on extraction SOPs, Multi-Class Trademarks (Class 5, 30, 3), GIs, and Registered Designs. |
| **5** | **DMR-OA & Schedule J Advertising Guard** | Many AYUSH brands face criminal notices or product seizures for claiming cures for diabetes, hypertension, or cancer. | Pre-submission label and trademark check against the 51 prohibited conditions under Schedule J and the Drugs & Magic Remedies Act. |
| **6** | **WIPO GRATK Treaty (2024) Export Checker** | New mandatory international disclosure of genetic resources risks invalidation of PCT foreign patents if Indian NBA clearances are absent. | Maps foreign PCT/Paris filings to Indian Section 10(4)(ii)(D) biological origin disclosures and target country herbal standards (US DSHEA / EU THMPD). |
| **7** | **Zero-Retention Blind Evaluation Mode** | Vaidyas and MSMEs fear sharing secret herb ratios, extraction temperatures, or proprietary processing steps. | Client-side formula abstraction that masks exact ratios into generic categorical brackets before sending to retrieval/LLM, compliant with DPDP Act 2023. |
| **8** | **Auto-Drafted Statutory Forms** | Innovators struggle with bureaucratic forms. | Generates pre-populated drafts of NBA Form I (Access), NBA Form III (IPR), and Patent Form 1/2 biological disclosure clauses. |

---

---

## 🔄 4. Definitive System Architecture: IP-SAKTI Sahayak (16:9 Slide Blueprint)

```mermaid
flowchart LR
    %% Slide Color Palette matching your exact slide
    classDef titleBanner fill:#0B1E3B,stroke:#38BDF8,stroke-width:2px,color:#FFFFFF;
    classDef col1 fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC;
    classDef col2 fill:#064E3B,stroke:#34D399,stroke-width:2px,color:#F8FAFC;
    classDef diamond fill:#047857,stroke:#6EE7B7,stroke-width:2px,color:#FFFFFF;
    classDef col3 fill:#2E1065,stroke:#C084FC,stroke-width:2px,color:#F8FAFC;
    classDef col4 fill:#78350F,stroke:#FBBF24,stroke-width:2px,color:#F8FAFC;
    classDef col5 fill:#0C4A6E,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC;

    %% TITLE BANNER
    Title["<b>IP-SAKTI Sahayak</b><br><br>Multilingual AI<br>Assistant for<br>Ayurveda IP &<br>Regulatory<br>Guidance"]
    class Title titleBanner;

    %% STAGE 1: INTAKE & TOGGLE
    subgraph S1 ["1. SECURE INTAKE & TOGGLE"]
        direction TB
        In1["<b>User / AYUSH Innovator Query</b><br>• Bhashini Indic voice/text ingress"]
        In2["<b>Blind Recipe Redactor</b><br>• Zero Trade Secret Leakage<br>• DPDP Act 2023 Compliant"]
        In3["<b>Jurisdiction Switch</b><br>• 🇮🇳 India vs 🌐 International"]
        In1 --> In2 --> In3
    end
    class In1,In2,In3 S1 col1;

    %% STAGE 2: REGULATORY CLASSIFIER
    subgraph S2 ["2. REGULATORY CLASSIFIER"]
        direction TB
        C1["<b>Classification Wizard</b><br>• 1st Schedule Texts vs Novel<br>• CITES / Negative Species Screen<br>• Rule 158B Solvent Check"]
        C2{"Category<br>Identified"}
        C3["<b>• Classical Medicine</b><br><b>• Proprietary / P&P Formulation</b><br><b>• Phytopharmaceutical</b><br><b>• Ayurveda-Aahar / Cosmetic</b>"]
        C1 --> C2 --> C3
    end
    class C1,C3 S2 col2;
    class C2 diamond;

    %% STAGE 3: KNOWLEDGE & RETRIEVAL CORE
    subgraph S3 ["3. KNOWLEDGE & RETRIEVAL CORE"]
        direction TB
        K1["<b>AyurTaxa Canonical Resolver</b><br>Vernacular ↔ Sanskrit ↔ Botanical"]
        K2["<b>GraphRAG + pgvector Hybrid</b><br>API Monograph ↔ TKDL ↔ Statutes"]
        K3["<b>ONNX Quantized Reranker</b><br>Evidence Relevance Scoring<br>Sub-45ms CPU • Gate > 0.35"]
        K1 --> K2 --> K3
    end
    class K1,K2,K3 S3 col3;

    %% STAGE 4: COMPLIANCE & IP REASONING
    subgraph S4 ["4. COMPLIANCE & IP REASONING"]
        direction TB
        A1["<b>Section 3(p) & 3(e) Patent Analyzer</b><br>TK Bar & Synergy Index Test (CI < 1.0)"]
        A2["<b>BDA 2023 ABS Calculator</b><br>Cultivated vs Wild / Foreign Equity"]
        A3["<b>DMR-OA & Schedule J Guard</b><br>Misleading Claims & Ad Audit"]
        A1 ~~~ A2 ~~~ A3
    end
    class A1,A2,A3 S4 col4;

    %% STAGE 5: ACTIONABLE DELIVERABLES
    subgraph S5 ["5. ACTIONABLE DELIVERABLES"]
        direction TB
        O1["<b>Verifiable Source Citations</b><br>Exact Sections, Gazette & PDF Links"]
        O2["<b>Auto-Drafted Statutory Forms</b><br>NBA Form I/III & Patent Form 1/2"]
        O3["<b>360° IP Portfolio Bundle</b><br>Trade Secrets, TM (Cl. 5/30), GI, Designs"]
        O4["<b>Human Escalation Desk</b><br>Connect to AYUSH Patent Facilitator"]
        O1 ~~~ O2 ~~~ O3 ~~~ O4
    end
    class O1,O2,O3,O4 S5 col5;

    %% CONNECTIONS
    Title ==> S1
    S1 ==> C1
    C3 ==> K1
    K3 ==> A1 & A2 & A3
    A1 & A2 & A3 ==> O1 & O2 & O3 & O4
```

### 🎤 Executive 60-Second Slide Pitch Script
1. **Title & Mission**: **IP-SAKTI Sahayak** is India's dedicated AI assistant helping AYUSH researchers, startups, and Vaidyas protect legitimate innovations while shielding codified traditional knowledge from global biopiracy.
2. **Stage 1 (Secure Intake & Toggle)**: Users query in any Indian language via Bhashini; our client-side Blind Recipe Redactor abstracts proprietary ratios under DPDP Act standards, and a strict Jurisdiction Switch prevents Indian laws from conflating with foreign export rules.
3. **Stage 2 (Regulatory Classifier)**: Classifies formulations across 5 statutory tiers (Classical, P&P, Phytopharmaceutical, Ayurveda-Aahar, Cosmetic) while actively screening for CITES endangered species and auditing extraction solvents under Rule 158B.
4. **Stage 3 (Knowledge & Retrieval Core)**: AyurTaxa resolves complex botanical taxonomy (*Haridra ↔ Haldi ↔ Curcuma longa*); our hybrid GraphRAG engine pairs a PostgreSQL Knowledge Graph with pgvector HNSW and GIN full-text search, reranked under 45ms with a zero-hallucination gate (>0.35).
5. **Stage 4 (Compliance & IP Reasoning)**: Evaluates Section 3(p) TK bars, tests for Chou-Talalay synergy ($CI < 1.0$) for Section 3(e), calculates BDA 2023 ABS liabilities (foreign equity triggers vs. cultivated waivers), and audits marketing claims against Schedule J.
6. **Stage 5 (Actionable Deliverables)**: Generates verifiable gazette citations, auto-drafts NBA Forms I/III and Patent Form 1/2 disclosure clauses, prescribes a 360° IP bundle (Trade Secrets, Trademarks, GIs, Designs), and provides a 1-click bridge to an empaneled patent facilitator.


---

## 🛠️ 5. Required Technical Updates to the Codebase

### 5.1 Database & Schema Updates (`config/`, `core/`, `retrieval/`)
* **Hierarchical Statutory Schema**: Upgrade the PostgreSQL `pgvector` chunk metadata to support legal taxonomy:
  ```sql
  ALTER TABLE document_chunks ADD COLUMN jurisdiction VARCHAR(32) DEFAULT 'INDIA';
  ALTER TABLE document_chunks ADD COLUMN legal_regime VARCHAR(64); -- 'PATENT', 'BIODIVERSITY', 'DRUG_REGULATORY', 'TRADEMARK', 'GI', 'FOOD_FSSAI'
  ALTER TABLE document_chunks ADD COLUMN statute_name VARCHAR(255);
  ALTER TABLE document_chunks ADD COLUMN section_or_rule VARCHAR(128);
  ALTER TABLE document_chunks ADD COLUMN official_url TEXT;
  ALTER TABLE document_chunks ADD COLUMN gazette_date DATE;
  ALTER TABLE document_chunks ADD COLUMN is_repealed BOOLEAN DEFAULT FALSE;
  
  -- Compound index for fast partitioned queries
  CREATE INDEX idx_chunks_jurisdiction ON document_chunks (jurisdiction, legal_regime);
  ```
* **Strict SQL Jurisdiction Partitioning**: Update `retrieval/vector_store.py` so queries under the `INDIA` jurisdiction filter strictly enforce `WHERE jurisdiction = :jurisdiction` to prevent legal conflation.

### 5.2 Ingestion & Semantic Parsing Updates (`ingestion/`)
* **Legal Section-Aware Chunking**: Replace character-only chunking with statutory structural chunking:
  * Splitting by **Act ➔ Chapter ➔ Section ➔ Sub-Section ➔ Proviso / Explanation**.
  * Splitting pharmacopoeia entries by **Monograph ➔ Botanical Name ➔ Synonyms ➔ Classical Formulations ➔ Therapeutic Properties**.
* **Open Source Scrapers / Ingestors**:
  * Ingestor for India Code PDFs / HTML (Patents Act, BDA, DCA, DMR-OA).
  * Ingestor for PCIM&H Pharmacopoeial monographs (API & AFI).
  * Ingestor for NBA Guidelines & Form templates.

### 5.3 PostgreSQL-Native Knowledge Graph & GraphRAG Layer (`core/graph/`)
To eliminate hallucinations on multi-hop regulatory logic, we implement a relational Knowledge Graph directly in PostgreSQL:

```sql
-- 1. Knowledge Graph Nodes (Herbs, Formulations, Texts, Sections, Forms)
CREATE TABLE IF NOT EXISTS ayur_nodes (
    id VARCHAR(128) PRIMARY KEY,      -- e.g. 'HERB_HARIDRA', 'TEXT_CHARAKA_SAMHITA', 'STATUTE_S3P'
    label VARCHAR(64) NOT NULL,        -- 'HERB', 'FORMULATION', 'CLASSICAL_TEXT', 'PATENT_SECTION', 'NBA_FORM'
    canonical_name VARCHAR(255) NOT NULL,
    synonyms TEXT[],                   -- Sanskrit, Vernacular, Botanical names
    properties JSONB DEFAULT '{}'      -- Taxon family, therapeutic indications, schedule listings
);

-- 2. Knowledge Graph Edges (Directed Relationships)
CREATE TABLE IF NOT EXISTS ayur_edges (
    source_id VARCHAR(128) REFERENCES ayur_nodes(id) ON DELETE CASCADE,
    target_id VARCHAR(128) REFERENCES ayur_nodes(id) ON DELETE CASCADE,
    relationship VARCHAR(64) NOT NULL, -- 'CONTAINS_HERB', 'DOCUMENTED_IN', 'TRIGGERS_BAR', 'REQUIRES_FORM'
    properties JSONB DEFAULT '{}',
    PRIMARY KEY (source_id, target_id, relationship)
);

CREATE INDEX idx_edges_source ON ayur_edges(source_id);
CREATE INDEX idx_edges_target ON ayur_edges(target_id);
```

* **GraphRAG Dual-Pass Retrieval Flow**:
  1. **Pass 1 (Structured Subgraph Traversal)**: Extracts entities from the query, traverses 1-3 hops in PostgreSQL using a recursive CTE (`WITH RECURSIVE graph_path AS (...)`), and compiles a deterministic fact chain (e.g. *Polyherbal Formulation ➔ First Schedule Text ➔ Section 3(p) Bar ➔ Mandatory NBA Form III*).
  2. **Pass 2 (Unstructured Statutory Retrieval)**: Executes PostgreSQL `pgvector` HNSW + GIN full-text search to retrieve exact paragraphs of the cited statutes.
  3. **Pass 3 (Unified Synthesis)**: Injects both the graph relation chain and the statutory text chunks into the LLM context.

### 5.4 Specialized Reasoning & Verification Engines (`generation/`)
* **`FormulationClassifier` Module**: Interactive multi-turn question flow determining formulation category (Classical, P&P, Phytopharmaceutical, Aahar, Cosmetic).
* **`PatentabilityAnalyzer` Module**: Section 3(p) Traditional Knowledge audit and Section 3(e) synergism requirements.
* **`ABSComplianceCalculator` Module**: Biological Diversity Act 2023 checks (Section 3(2) foreign control, Section 7 domestic intimation, cultivated vs wild sourcing exemptions).
* **`DMRAdvertisingGuard` Module**: Real-time regex and semantic check against Schedule J prohibited disease claims.

### 5.5 Privacy, Security & Disclaimers (`api/`, `core/guardrails/`)
* **Blind Mode Masking**: Client-side / pre-LLM regex redacting exact formula percentages (`XX%`, `YY mg/kg`) and proprietary extraction temperature profiles before embedding.
* **Standing Statutory Disclaimer**: Mandatory injection in every API response:
  > *"Notice: This tool provides research and regulatory information grounded in public statutes and does not constitute formal legal counsel. For formal patent filing or NBA representation, consult an empaneled patent attorney."*
* **DPDP Compliance**: No caching or logging of user formulation prompts in permanent storage.

---

## 💻 6. Technology Stack & Tooling Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 TECHNOLOGY STACK                                │
├──────────────────────────┬──────────────────────┬───────────────────────────────┤
│ LAYER                    │ TECHNOLOGY / TOOL    │ ROLE & SPECIFIC FUNCTION      │
├──────────────────────────┼──────────────────────┼───────────────────────────────┤
│ 1. Vector & Relational   │ PostgreSQL 16 +      │ • HNSW Cosine vector indexing │
│    Database Core         │ pgvector extension   │   (384 dimensions, M=16)      │
│                          │                      │ • GIN tsvector full-text index│
│                          │                      │ • Statutory metadata filtering│
├──────────────────────────┼──────────────────────┼───────────────────────────────┤
│ 2. Embeddings & Search   │ • all-MiniLM-L6-v2   │ • Lightweight 384d embeddings │
│    Fusion                │ • Reciprocal Rank    │ • Parallel dense + lexical    │
│                          │   Fusion (RRF)       │   search fusion (RRF_K=60)    │
├──────────────────────────┼──────────────────────┼───────────────────────────────┤
│ 3. Quantized Reranking   │ • ms-marco-MiniLM-   │ • 3–4× faster CPU latency via │
│    & Gate                │   L-6-v2 Cross-Enc.  │   ONNX Runtime quantization   │
│                          │ • ONNX Runtime       │ • Sigmoid confidence gating   │
│                          │                      │   (Threshold = 0.35)          │
├──────────────────────────┼──────────────────────┼───────────────────────────────┤
│ 4. Deterministic LLM     │ Groq LPU             │ • Sub-second cloud inference  │
│    Generation Core       │ (Llama-3.3-70b /     │ • Temperature = 0.0 for       │
│                          │  gpt-oss-120b)       │   verifiable statutory output │
├──────────────────────────┼──────────────────────┼───────────────────────────────┤
│ 5. API & Backend         │ • Python 3.10+       │ • Async streaming endpoints   │
│    Orchestration         │ • FastAPI & Uvicorn  │ • Multi-turn session states   │
│                          │ • Redis Cache        │ • Semantic response caching   │
├──────────────────────────┼──────────────────────┼───────────────────────────────┤
│ 6. Multilingual &        │ Bhashini API         │ • Indian language translation │
│    Voice Engine          │ (GoI ULCA framework) │   (Hindi, Tamil, Sanskrit)    │
├──────────────────────────┼──────────────────────┼───────────────────────────────┤
│ 7. Privacy & DPDP        │ Client-Side Regex +  │ • Masks confidential recipes  │
│    Compliance            │ Session Ephemerality │ • Zero prompt-retention logs  │
└──────────────────────────┴──────────────────────┴───────────────────────────────┘
```

---

## ⚡ 7. Comprehensive List of Technical, Legal & Domain Challenges

1. **The "Taxonomy & Synonym Trap"**:
   * A single herb can have dozens of Sanskrit synonyms (*Haridra, Nisha, Gauri*), vernacular trade names (*Haldi, Manjal*), and botanical revisions (*Curcuma longa L.*). Standard semantic search fails if the query uses a colloquial name while the statute or pharmacopoeia monograph uses the botanical taxon.
2. **Polyherbal Complexity (*Kalpanas* & Multi-Herbs)**:
   * Classical Ayurveda uses polyherbal blends (*Trikatu, Dashamula, Triphala*). Tracing prior art requires decomposing compounds into constituent species and cross-referencing against the 54 First Schedule texts.
3. **The Section 3(p) vs. 3(e) Patent Hurdles**:
   * **Section 3(p)** bars patenting of traditional knowledge.
   * **Section 3(e)** bars mere admixtures aggregating known properties.
   * *Challenge*: The assistant must know how to guide an innovator to legally patentable subject matter (e.g., proving synergism via Chou-Talalay combination index, novel delivery nano-carriers, or novel extraction processes) without making unauthorized legal guarantees.
4. **Temporal Law & the 2023/2024 Amendments**:
   * The **Biological Diversity (Amendment) Act 2023** exempted registered AYUSH practitioners and cultivated medicinal plants from ABS fees, but retained strict prior-approval duties under Section 3(2) for foreign-invested companies. The engine must determine applicability based on filing dates and corporate equity structures.
5. **Cross-Border Legal Conflation**:
   * US FDA (DSHEA / Botanical Drug Guidance), EU (THMPD), and Indian (Drugs and Cosmetics Act / FSSAI) treat herbal products under completely contradictory categories. Cross-jurisdictional leakage in RAG retrieval must be strictly guarded against.
6. **DMR-OA & Advertising Liability (Schedule J)**:
   * Startups frequently attempt to trademark brand names or write labels claiming to "treat or cure" conditions prohibited under the Drugs and Magic Remedies Act (e.g., diabetes, hypertension, cancer).
7. **Statutory Structural Integrity**:
   * Standard character-based chunking breaks legal provisos, exceptions, and schedules in half. Chunking must strictly respect statutory syntax (Act ➔ Chapter ➔ Section ➔ Proviso).
8. **Trade Secret Preservation (Zero-Leakage)**:
   * Vaidyas and MSMEs will not adopt an AI if their proprietary extraction temperatures, herb ratios, and purification (*Shodhana*) steps are sent to third-party model providers.

---

## 📚 8. Authoritative References & Academic/Legal Research Work

### A. Primary Statutes, Rules & Treaties
1. **The Patents Act, 1970** (as amended up to the **Patents (Amendment) Rules, 2024**), Ministry of Commerce and Industry, Government of India.
   * *Key Provisions*: Section 3(p) [Traditional Knowledge bar], Section 3(e) [Mere admixture bar], Section 10(4)(ii)(D) [Mandatory disclosure of biological source].
2. **The Biological Diversity Act, 2002 & The Biological Diversity (Amendment) Act, 2023**, with **Biological Diversity Rules, 2024**, National Biodiversity Authority (NBA), India.
   * *Key Provisions*: Section 3, 4, 6 approvals; NBA Forms I, II, III; exemptions for cultivated medicinal plants and codified practitioners.
3. **The Drugs and Cosmetics Act, 1940 & Rules, 1945**, Ministry of Health and Family Welfare, Government of India.
   * *Key Provisions*: First Schedule (54 authoritative texts), Rule 158B (licensing of AYUSH medicines), Schedule T (Good Manufacturing Practices).
4. **WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge (GRATK Treaty)**, adopted by the WIPO Diplomatic Conference on May 24, 2024, Geneva.
   * *Key Provisions*: Mandatory patent applicant disclosure of origin for genetic resources and traditional knowledge worldwide.
5. **Nagoya Protocol on Access to Genetic Resources and the Fair and Equitable Sharing of Benefits (ABS)** to the Convention on Biological Diversity (CBD), United Nations (2010).
6. **Food Safety and Standards (Ayurveda Aahar) Regulations, 2022**, Food Safety and Standards Authority of India (FSSAI).
7. **The Drugs and Magic Remedies (Objectionable Advertisements) Act, 1954**, and **Schedule J**, Ministry of Health and Family Welfare.

### B. Pharmacopoeias & Classical Codices
8. **The Ayurvedic Pharmacopoeia of India (API)**, Part I (Vols I–IX, Single Drugs) & Part II (Vols I–IV, Formulations), Pharmacopoeia Commission for Indian Medicine & Homoeopathy (PCIM&H), Ministry of AYUSH.
9. **The Ayurvedic Formulary of India (AFI)**, Parts I, II & III, Ministry of AYUSH, Government of India.
10. **The Classical First Schedule Texts**: *Charaka Samhita*, *Sushruta Samhita*, *Ashtanga Hridaya*, *Bhaishajya Ratnavali*, *Sahasrayogam*, *Sharngadhara Samhita*.

### C. Key Research Papers & Guidelines
11. **Indian Patent Office (CGPDTM)** (2012). *Guidelines for Examination of Patent Applications relating to Traditional Knowledge and Biological Material*. Office of the Controller General of Patents, Designs and Trade Marks, Mumbai.
12. **Gupta, V. K.** (2011). *Protecting Indian Traditional Knowledge from Biopiracy: The Traditional Knowledge Digital Library (TKDL) Experience*. Council of Scientific and Industrial Research (CSIR) / WIPO.
13. **Mashelkar, R. A.** (2001). *Intellectual Property Rights and the Third World*. *Current Science*, 81(8), 955–965.
14. **Chou, T. C.** (2006). *Theoretical Basis, Experimental Design, and Computerized Simulation of Synergism and Antagonism in Drug-Combination Studies*. *Pharmacological Reviews*, 58(3), 621–681.
15. **Patwardhan, B., & Vaidya, A. D.** (2010). *Natural Products and Drug Discovery: Concept of Reverse Pharmacology and Synergism*. *Current Bioactive Compounds*, 6(3), 181–190.
16. **Dutfield, G.** (2017). *Traditional Knowledge, Intellectual Property and Alternative Approaches*. In *Research Handbook on Intellectual Property and the Life Sciences*, Edward Elgar Publishing.

---

## 📈 9. Scale & Vector Capacity Analysis (Can Current RAG Handle It?)

### 9.1 Data Volume Estimation
When assembling all 4 open repositories:
* **India Code** (Patents, BDA, DCA, DMR-OA, GI, TM): ~15,000 chunks (~15 MB)
* **National Biodiversity Authority (NBA)** (Rules 2024, Guidelines, Forms, NTC): ~5,000 chunks (~5 MB)
* **IP India** (Patent Manual, GI Register, Examination Guidelines): ~10,000 chunks (~10 MB)
* **TKDL Public & Classical Formulations** (Curated summaries): ~50,000 – 100,000 chunks (~75 MB)
* **Total Knowledge Base**: **~80,000 to 130,000 chunks (~100 – 120 MB)**

### 9.2 Capacity Assessment
* **PostgreSQL + `pgvector` HNSW**:
  * 100,000 vectors of 384 dimensions require:
    $$\approx 100,000 \times 384 \times 4 \text{ bytes} \approx 153 \text{ MB of RAM}$$
  * PostgreSQL `pgvector` easily scales up to **10M+ vectors**. Search latency remains **<15ms**.
* **PostgreSQL GIN Full-Text Index (`tsv`)**:
  * GIN handles hundreds of thousands of document terms effortlessly. Exact keyword search (*"Section 3(p)"*, *"Form III"*) completes in **<5ms**.
* **ONNX Reranker**:
  * Only evaluates `TOP_K = 15` candidates; independent of database size. Rerank latency remains **~30–50ms on CPU**.

### 9.3 Required Ingestion Optimizations
1. **Streaming Ingestion Batch**: Refactor `add_chunks()` in `retrieval/vector_store.py` to stream in batches of 500 records instead of loading all 100,000 texts into Python RAM at once.
2. **PostgreSQL Maintenance Memory**: Increase `maintenance_work_mem` to `512MB` during bulk HNSW index creation.
3. **Composite B-tree Index**: Add `(jurisdiction, legal_regime)` index to accelerate filtered queries.

---

## ⏱️ 10. Latency Budget & Sub-250ms p99 SLA

To deliver instantaneous regulatory intelligence without lag, the system enforces a strict **sub-250ms p99 Time-To-First-Token (TTFT)**.

### 10.1 Stage-by-Stage Latency Budget

```
┌───────────────────────────────────────────────────────────────────┬──────────────┬──────────────┐
│ PIPELINE STAGE                                                    │ TYPICAL (p50)│ WORST (p99)  │
├───────────────────────────────────────────────────────────────────┼──────────────┼──────────────┤
│ 1. Guardrails, Blind Mode & AyurTaxa Entity Lookup               │ 2 ms         │ 5 ms         │
│    (In-memory trie / B-Tree indexed lookup)                       │              │              │
├───────────────────────────────────────────────────────────────────┼──────────────┼──────────────┤
│ 2. PostgreSQL Knowledge Graph Traversal (Pass 1)                  │ 3 ms         │ 6 ms         │
│    (Indexed 1-3 hop recursive CTE on ayur_nodes / ayur_edges)     │              │              │
├───────────────────────────────────────────────────────────────────┼──────────────┼──────────────┤
│ 3. Query Embedding (all-MiniLM-L6-v2 ONNX)                        │ 8 ms         │ 14 ms        │
│    (384-dimensional vector inference on CPU)                      │              │              │
├───────────────────────────────────────────────────────────────────┼──────────────┼──────────────┤
│ 4. PostgreSQL Hybrid Search (Pass 2)                              │ 10 ms        │ 18 ms        │
│    • pgvector HNSW cosine search (ef_search=40): ~8ms             │              │              │
│    • GIN tsvector lexical full-text search: ~5ms                  │              │              │
│    • Reciprocal Rank Fusion (RRF) in Python: ~0.5ms               │              │              │
├───────────────────────────────────────────────────────────────────┼──────────────┼──────────────┤
│ 5. ONNX Quantized Cross-Encoder Reranker                          │ 30 ms        │ 45 ms        │
│    (ms-marco-MiniLM-L-6-v2 INT8 scoring top 15 candidate pairs)   │              │              │
├───────────────────────────────────────────────────────────────────┼──────────────┼──────────────┤
│ ⏩ TOTAL SEARCH & RETRIEVAL ENGINE LATENCY (Context Ready)        │ ~53 ms       │ ~88 ms       │
├───────────────────────────────────────────────────────────────────┼──────────────┼──────────────┤
│ 6. Groq LPU Cloud Time-To-First-Token (TTFT)                      │ 110 ms       │ 145 ms       │
│    (High-speed custom LPU silicon)                                │              │              │
├───────────────────────────────────────────────────────────────────┼──────────────┼──────────────┤
│ 🏁 END-TO-END STREAMING TIME-TO-FIRST-TOKEN (TTFT p99)            │ ~163 ms      │ ~233 ms ✅   │
└───────────────────────────────────────────────────────────────────┴──────────────┴──────────────┘
```
*(Queries hitting Redis Cache return complete answers in **< 15ms**).*

### 10.2 The 4 Architectural Levers Enforcing the SLA
1. **ONNX INT8 Quantization**: Reduces Cross-Encoder CPU inference from ~180ms to **<45ms**.
2. **PostgreSQL Composite Indexing**: Enforces `(jurisdiction, legal_regime)` index filtering during HNSW traversal, keeping vector retrieval under **12ms**.
3. **In-Database Recursive Graph CTE**: Runs multi-hop relationship lookups directly in PostgreSQL memory in **3–6ms**, avoiding external graph database network latency.
4. **Groq LPU Acceleration**: Generates output at 350–500 tokens/second, beginning SSE stream delivery within **~210–235ms**.

---

## 🏗️ 11. Granular Implementation Phases & Work Breakdown

Below is the concrete, phase-by-phase engineering execution plan with explicit deliverables and files to modify.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PHASED IMPLEMENTATION TIMELINE                                  │
├──────────────┬──────────────────────────────────────────┬─────────────────────────────────────────┤
│ PHASE        │ CORE MILESTONE                           │ ESTIMATED TIMELINE                      │
├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────┤
│ Phase 1      │ Database Schema, Streaming & Settings    │ Week 1                                  │
│ Phase 2      │ Corpus Ingestion & Legal Chunker         │ Week 2                                  │
│ Phase 3      │ PostgreSQL Knowledge Graph & GraphRAG    │ Week 3                                  │
│ Phase 4      │ Regulatory Logic, S.3(p)/3(e) & ABS      │ Week 4                                  │
│ Phase 5      │ Latency Optimization & Redis Caching     │ Week 5                                  │
│ Phase 6      │ UI Wizard, Form Drafter & Verification   │ Week 6                                  │
└──────────────┴──────────────────────────────────────────┴─────────────────────────────────────────┘
```

### 🔹 Phase 1: Database Migration, Settings Lock & Streaming Ingestion
* **Goal**: Establish the legal database schema and optimize ingestion for bulk data without memory exhaustion.
* **Tasks & Files to Touch**:
  1. [`config/settings.py`](file:///Users/atharva/RAG-pipeline/config/settings.py):
     - Lock `LLM_TEMPERATURE = 0.0`.
     - Calibrate `GUARDRAIL_CONFIDENCE_THRESHOLD = 0.35`.
     - Expand retrieval depth: `TOP_K = 15`, `RERANK_TOP_N = 5`.
     - Switch `CHUNKING_STRATEGY = "statutory_section"`.
  2. [`retrieval/vector_store.py`](file:///Users/atharva/RAG-pipeline/retrieval/vector_store.py):
     - Add database migration: add columns `jurisdiction`, `legal_regime`, `statute_name`, `section_or_rule`, `official_url`, `gazette_date` to `document_chunks`.
     - Add composite B-tree index on `(jurisdiction, legal_regime)`.
     - Refactor `add_chunks()` into a **streaming generator batch** (processing 500 chunks per iteration to eliminate Python RAM spikes).
     - Add `jurisdiction` and `regime` filters to `vector_search()`, `lexical_search()`, and `hybrid_search()`.

---

### 🔹 Phase 2: Authoritative Corpus Curation & Structural Legal Chunker
* **Goal**: Ingest core public legal texts from India Code, NBA, and PCIM&H with zero broken clauses.
* **Tasks & Files to Touch**:
  1. [`ingestion/chunker.py`](file:///Users/atharva/RAG-pipeline/ingestion/chunker.py):
     - Implement `StatutorySectionChunker`: regex-based splitting on statutory boundaries:
       `r'(?=(?:Section|Rule|Article)\s+\d+[A-Z]?(?:\.\d+)?|CHAPTER\s+[IVXLCDM]+|MONOGRAPH:\s+)'`.
     - Generate hierarchical parent context (Entire Section: 1500–3000 chars) and child snippets (Sub-sections/Provisos: 500–800 chars).
  2. `ingestion/legal_parsers/`:
     - Create parsers for India Code Gazette HTML/PDFs (*Patents Act 1970/2024*, *BDA 2002/2023*, *DCA 1940 First Schedule*, *DMR-OA 1954*).
     - Create parsers for PCIM&H Pharmacopoeial Monographs (API/AFI).
     - Create parsers for NBA Guidelines & Form templates (Forms I, II, III).
  3. `scripts/seed_statutes.py`:
     - CLI command to parse and batch-ingest all seed statutes into PostgreSQL.

---

### 🔹 Phase 3: PostgreSQL-Native Knowledge Graph & GraphRAG Core
* **Goal**: Build relational tables and recursive SQL traversal linking herbs, texts, patent bars, and ABS rules.
* **Tasks & Files to Touch**:
  1. `core/graph/schema.sql`:
     - Create `ayur_nodes` (Herbs, Formulations, Texts, Sections, Forms).
     - Create `ayur_edges` (Directed relationships: `CONTAINS`, `DOCUMENTED_IN`, `TRIGGERS_S3P`, `REQUIRES_NBA`).
  2. `core/graph/graph_retriever.py`:
     - Implement recursive SQL traversal:
       `WITH RECURSIVE graph_path AS (SELECT ... FROM ayur_edges WHERE source_id = :entity ...)`
     - Extract entities from query via AyurTaxa matcher and fetch 1–3 hop subgraphs in <6ms.
  3. [`retrieval/retriever.py`](file:///Users/atharva/RAG-pipeline/retrieval/retriever.py):
     - Combine **Pass 1 (Graph Relational Subgraph)** with **Pass 2 (Hybrid pgvector/GIN search)** into a unified context payload.

---

### 🔹 Phase 4: Regulatory Reasoning & Specialized Compliance Modules
* **Goal**: Implement deterministic legal rule engines to evaluate formulation types, patentability, and ABS.
* **Tasks & Files to Touch**:
  1. `core/compliance/formulation_classifier.py`:
     - Interactive multi-turn state machine: determines Classical Medicine vs. P&P vs. Phytopharmaceutical vs. Ayurveda-Aahar vs. Cosmetic.
  2. `core/compliance/patentability_analyzer.py`:
     - Section 3(p) Traditional Knowledge audit (verbatim match against First Schedule texts).
     - Section 3(e) Synergistic Efficacy audit (evaluates combination index $CI < 1.0$ data or redirects to process patents).
  3. `core/compliance/abs_calculator.py`:
     - Checks corporate structure (Section 3(2) foreign shareholding triggers).
     - Sourcing check: Cultivated medicinal plant exemption vs. wild forest harvest under BDA 2023 Amendment.
  4. `core/compliance/dmr_ad_guard.py`:
     - Regex & semantic auditor against the 51 prohibited disease conditions in **Schedule J** of the DMR(OA) Act.
  5. `core/guardrails/blind_mode.py`:
     - Client-side and ingress formula redactor: abstracts exact percentages and extraction temperatures into categorical brackets before sending to LLM.

---

### 🔹 Phase 5: Latency Optimization, ONNX Benchmarking & Redis Caching
* **Goal**: Enforce strict p99 < 250ms SLA across all endpoints.
* **Tasks & Files to Touch**:
  1. [`retrieval/reranker.py`](file:///Users/atharva/RAG-pipeline/retrieval/reranker.py):
     - Ensure ONNX INT8 quantized model runs under 45ms for 15 candidate pairs.
     - Calibrate zero-hallucination score threshold at `0.35`.
  2. `core/cache/redis_manager.py`:
     - Cache frequent statutory queries and graph lookups with 24-hour TTL (<15ms p99 response).
  3. `tests/benchmark_latency.py`:
     - Automated Locust / pytest-benchmark suite verifying p50, p95, and p99 latencies under concurrent load.

---

### 🔹 Phase 6: Form Drafter, Human Escalation & Web Interface
* **Goal**: Deliver actionable outputs (pre-filled government forms, citations, patent agent escalation).
* **Tasks & Files to Touch**:
  1. `generation/form_drafter.py`:
     - Auto-generate pre-filled drafts for **NBA Form I** (Access), **NBA Form III** (IPR approval), and **Patent Form 1/2** Section 10(4)(ii)(D) biological disclosure clauses.
  2. `generation/prompts/ayush_prompts.py`:
     - Prompt templates enforcing mandatory statutory citations, official URL references, and standing "Information, Not Legal Advice" disclaimers.
  3. `api/routes/ayush_routes.py`:
     - Endpoints for `/api/v1/classify-formulation`, `/api/v1/abs-calculate`, `/api/v1/query`, and `/api/v1/escalate-facilitator`.
  4. `frontend/` (or UI templates):
     - Modern responsive UI with Jurisdiction Toggle (`🇮🇳 India` vs `🌐 International`), Formulation Wizard, and Citation Side-Drawer.

---

## 🚫 12. Negative Species & CITES Protection Filter

Under **Section 38 of the Biological Diversity Act, 2002** and **Schedule VI of the Wildlife (Protection) Act, 1972**, wild collection, trading, or patenting of critically endangered medicinal plants is strictly prohibited or regulated under criminal law.

### 12.1 High-Risk Protected Species Registry

The assistant maintains an active negative screening table:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CRITICALLY PROTECTED AYURVEDIC SPECIES                          │
├──────────────────────┬────────────────────────┬─────────────────────┬──────────────────┤
│ BOTANICAL TAXON      │ SANSKRIT / COMMON NAME │ STATUTORY STATUS    │ COMPLIANCE DUTY  │
├──────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Nardostachys         │ Jatamansi              │ CITES Appendix II / │ Cultivation proof│
│ jatamansi            │ (Spikenard)            │ BDA Section 38      │ & Forest Permit  │
├──────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Saussurea costus     │ Kuth / Kushta          │ CITES Appendix I /  │ Banned from wild │
│ (Saussurea lappa)    │                        │ WPA Schedule VI     │ harvest; DGFT lic│
├──────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Picrorhiza kurroa    │ Kutki                  │ CITES Appendix II / │ Cultivation cer- │
│                      │                        │ Negative export list│ tificate mandatory│
├──────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Pterocarpus          │ Raktachandana          │ CITES Appendix II / │ Non-bailable wild│
│ santalinus           │ (Red Sanders)          │ WPA Schedule VI     │ harvest offenses │
├──────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Aquilaria agallocha  │ Agaru                  │ CITES Appendix II / │ Strict transit & │
│ (Aquilaria malaccensis) (Agarwood)            │ DGFT Export Restrict│ CITES export cert│
├──────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Taxus wallichiana    │ Talispatra             │ CITES Appendix II / │ Wild extraction  │
│                      │ (Himalayan Yew)        │ BDA Section 38      │ prohibited       │
├──────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Aconitum             │ Ativisha / Vatsanabha  │ Negative Plant List │ Wild harvesting  │
│ heterophyllum        │                        │ State Biodiversity  │ audit required   │
└──────────────────────┴────────────────────────┴─────────────────────┴──────────────────┘
```

### 12.2 Automated Statutory Safeguard
* When any formulation contains these botanical taxa, the assistant **immediately raises a Legal Alert Flag**:
  > ⚠️ **STATUTORY PROTECTION NOTICE**: Wild harvesting of *{Botanical_Name}* is prohibited under Schedule VI of the Wildlife Protection Act and CITES. To commercialize or patent this formulation, you must possess a verifiable **Certificate of Artificial Propagation / Cultivation** issued by the State Forest Department and comply with DGFT export licensing requirements.

---

## ⚡ 13. The Complete AYUSH Legal Edge Cases & Remediation Matrix

The system actively guards against the 9 most dangerous regulatory and intellectual property edge cases:

| # | Edge Case Scenario | Legal / Regulatory Pitfall | Assistant Remediation Strategy |
| :--- | :--- | :--- | :--- |
| **1** | **1% Foreign Shareholding (The FDI Trap)** | Indian AYUSH startup takes foreign seed investment (e.g. Y-Combinator, US Angel). They assume they are a domestic Indian entity. | Under **Section 3(2) of BDA**, ANY non-Indian equity or foreign director triggers foreign status. Assistant mandates **NBA Form I** before accessing biological resources, warning that domestic SBB intimation is legally insufficient. |
| **2** | **The "Solvent Shift" Trap (Rule 158B)** | Innovator takes a classical First Schedule formulation, but extracts herbs using ethanol or supercritical $\text{CO}_2$ instead of water (*Kwatha*). | Reclassified under **Rule 158B(IV)** as a "New Ayurvedic Drug" requiring animal acute/sub-acute toxicity studies and clinical trials. Assistant flags clinical trial requirement before commercial licensing. |
| **3** | **Normally Traded Commodities (Section 40 BDA)** | Exporter claims exemption because herbs (e.g. *Pippali*, *Senna*, *Isabgol*) are on the Section 40 NTC list of 421 commodities. | Assistant warns: *NTC exemption applies strictly to conventional agricultural commodity export for consumption. It is VOID if the plant is accessed for research, bio-survey, or IPR/patenting.* |
| **4** | **The 15-Year EU THMPD Trap** | Startup seeks to export classical Ayurvedic medicines to the European Union under Directive 2004/24/EC. | Assistant evaluates requirement: Must prove 30 years of traditional medicinal use, with **at least 15 years documented within the EU**. Suggests alternative pathways (Food Supplement / Cosmetic) if 15-year EU retail proof is absent. |
| **5** | **The "Mere Admixture" Bar (Section 3(e))** | Startup combines classical herbs (*Ashwagandha + Haridra*) claiming "synergistic immunity boosting". | Automatically rejected by IPO under Section 3(e). Assistant checks if user has quantitative combination index ($CI < 1.0$) data via Chou-Talalay method; otherwise recommends process/extraction patent or trade secret. |
| **6** | **Fermentation & Budapest Treaty Strain Deposit** | Formulating *Asava* / *Arishta* using an isolated, genetically characterized novel yeast/microbe strain rather than wild *Woodfordia* flowers. | Mandates microbial deposit at an **International Depositary Authority (IDA)** (e.g. MTCC Chandigarh) under the Budapest Treaty, and verifies citation of deposit accession number in Form 1/2. |
| **7** | **The Ayurveda-Aahar Advertising Trap** | Brand registers an *Ayurveda-Aahar* (FSSAI) dietary product, but advertises it to "treat or manage Diabetes / Blood Sugar". | Flags immediate dual violation: FSSAI Ayurveda-Aahar regulations prohibit medicinal cure claims, and DMR(OA) Act strictly bars advertising cures for **Schedule J** conditions. |
| **8** | **Classical Text Edition Discrepancies** | Drug Controller rejects licensing because the cited shloka is in an unauthorized or unapproved regional commentary. | Cross-references against the authoritative editions recognized by the Ministry of AYUSH in the First Schedule (e.g. Chaukhambha Sanskrit Sansthan / Orient Longman publications). |
| **9** | **Cultivated vs. Wild Traceability under BDA 2023** | Manufacturer claims ABS fee exemption under 2023 Amendment for using cultivated medicinal plants. | Assistant warns that exemption requires verifiable **traceability documentation (Gram Panchayat / BMC / Forest Range Cultivation Certificates)** to survive State Biodiversity Board audits. |

---

## 🧪 14. Legal Ground-Truth Benchmarking & Testing Protocol

To ensure 100% legal accuracy, zero hallucination, and safe abstention, the pipeline undergoes continuous automated benchmarking against real-world jurisprudence.

### 14.1 Ground-Truth Benchmark Datasets

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               BENCHMARK EVALUATION SUITES                              │
├──────────────────────────┬──────────────┬──────────────────────────────────────────────┤
│ BENCHMARK SUITE          │ SAMPLE SIZE  │ TEST CRITERIA & REAL-WORLD SOURCE            │
├──────────────────────────┼──────────────┼──────────────────────────────────────────────┤
│ 1. IPO FER Herbal        │ 100 Cases    │ Real Indian Patent Office First Examination  │
│    Rejections Suite      │              │ Reports (FERs) under Section 3(p) and 3(e)   │
├──────────────────────────┼──────────────┼──────────────────────────────────────────────┤
│ 2. TKDL Opposition &     │ 30 Cases     │ Historic international patent oppositions    │
│    Revocation Victories  │              │ won by CSIR-TKDL (Neem, Turmeric, Nutmeg)    │
├──────────────────────────┼──────────────┼──────────────────────────────────────────────┤
│ 3. NBA Form Compliance & │ 50 Cases     │ Published National Biodiversity Authority    │
│    ABS Scenarios         │              │ orders, Form I/III approvals & penalty cases │
├──────────────────────────┼──────────────┼──────────────────────────────────────────────┤
│ 4. Out-of-Scope & Safe   │ 50 Cases     │ Ambiguous / speculative legal queries testing│
│    Abstention Gates      │              │ safe refusal and human facilitator referral  │
└──────────────────────────┴──────────────┴──────────────────────────────────────────────┘
```

### 14.2 Target Performance Metrics

| Evaluation Metric | Minimum Acceptable Threshold | Measurement Tool |
| :--- | :--- | :--- |
| **Citation Precision** | $\ge 98.5\%$ | Exact Section, Rule, and Gazette reference matching |
| **Section 3(p) Detection Recall**| $\ge 99.0\%$ | Flags classical traditional knowledge prior art |
| **Hallucination Rate** | $\le 0.5\%$ | Zero fabrication of non-existent sections or cases |
| **Safe Abstention Accuracy** | $\ge 96.0\%$ | Refuses speculative advice and directs to human facilitator |
| **Time-to-First-Token (p99)** | $\le 235 \text{ ms}$ | Automated Locust concurrent load testing |


