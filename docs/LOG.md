# Progress Log

> **Purpose:** Weekly progress tracker for the TermSheet AI project. Each entry records completed work, blockers, and next steps.

---

## Week 1 (11–17 Oct 2026)

### Completed

#### M1 — Team Lead / System Architect *(pre-existing)*
- [x] Repository created with `pyproject.toml`, `src/` layout, and editable install
- [x] CI tooling configured: `ruff` (linter/formatter) + `pytest` (testing) in `pyproject.toml`
- [x] Interface contracts defined in [`src/termsheet_ai/schemas/contracts.py`](../src/termsheet_ai/schemas/contracts.py)
  - 13 clause families (`ClauseFamily` enum)
  - Pydantic schemas for all 5 pipeline stages (S1–S5)
  - JSON Schema export via `export_schemas.py` → `contracts/json_schemas/`
- [x] Pipeline runner with mock implementations for all 5 stages ([`runner.py`](../src/termsheet_ai/pipeline/runner.py))
- [x] Abstract base classes for all pipeline stages ([`base.py`](../src/termsheet_ai/pipeline/base.py))
- [x] Initial test suite: `test_contracts.py` (schema validation) + `test_pipeline.py` (end-to-end mock)

#### M2 — Data & Preprocessing
- [x] Virtual environment set up (`.venv`) with `[dev,parsing]` dependencies
- [x] **PDF/DOCX/TXT text extraction** — [`src/termsheet_ai/ingest/extractor.py`](../src/termsheet_ai/ingest/extractor.py)
  - PyMuPDF primary extraction
  - pdfplumber fallback for layout-heavy PDFs
  - Tesseract OCR fallback for scanned documents
  - python-docx for Word documents
  - Page-level provenance tracking `(page_number, text)`
- [x] **Clause segmentation** — [`src/termsheet_ai/ingest/segmenter.py`](../src/termsheet_ai/ingest/segmenter.py)
  - Pattern-based splitting on numbered sections (`1.`, `1.2`), `Section X` headings, ALL-CAPS headings
  - Sentence-window fallback for unstructured text
  - Post-processing: merge short fragments, sub-split oversized clauses on `(a)/(b)` bullets
  - Configurable via `SegmenterConfig`
- [x] **S1 Ingestor** — [`src/termsheet_ai/ingest/ingestor.py`](../src/termsheet_ai/ingest/ingestor.py)
  - Implements `BaseIngestor` contract (drop-in for `MockIngestor`)
  - SHA-256 file hash for stable `doc_id`
- [x] **Corpus manifest** — [`corpus/manifest.json`](../corpus/manifest.json)
  - NVCA Model Term Sheet + Enhanced version
  - YC Post-Money & Pre-Money SAFEs
  - Evaluation corpus template (50 docs: 20 dev / 30 test)
  - Classifier exemplar bank metadata (13 families)
- [x] **Test suite** — [`tests/test_ingest.py`](../tests/test_ingest.py)
  - 16 tests: extractor, segmenter, ingestor, real PDF integration
  - All passing, ruff lint clean
### M2/M7 - Corpus collection (addendum)
- [x] Added 5 public term sheets to `corpus/` (NVCA Model, Founder Institute Plain Preferred, WSGR generated, Series Seed Equity, Gunderson Dettmer Series Seed) and registered them in `corpus/manifest.json`
- [ ] 5 more needed to reach the target

#### M5 — KR&R: Ontology & Rule Base
- [x] **Clause Family Definitions** — [`docs/clause_family_definitions_v0.md`](clause_family_definitions_v0.md)
- [x] **Attribute JSON Schema** — [`docs/attribute_schema_v0.json`](attribute_schema_v0.json)
- [x] **Annotation Guide (v0)** — [`docs/annotation_guide_v0.md`](annotation_guide_v0.md)

### Remaining for Week 1
- [x] **M5**: Draft annotation guide v0 (clause family definitions + attribute schemas for annotators)
- [ ] **M2 + M7**: Collect ≥10 public term sheets, add entries to `corpus/manifest.json` (5/10 done)

### Blockers
- None currently

---

## Week 2 (18–24 Oct 2026)

### Completed

#### M3 — Deep Learning: Embeddings & Search
- [x] **Exemplar Bank** — [`src/termsheet_ai/classify/exemplars.py`](../src/termsheet_ai/classify/exemplars.py)
  - 104 curated exemplars (exactly 8 per family across all 13 `ClauseFamily` values)
  - Grounded in NVCA Model Term Sheet, NVCA Enhanced, YC SAFEs, and LEDGAR out-of-domain boilerplate
  - Section header + body context boosting helper (`prepare_clause_text`)
- [x] **TF-IDF Baseline Classifier** — [`src/termsheet_ai/classify/tfidf.py`](../src/termsheet_ai/classify/tfidf.py)
  - N-gram (1, 2) TF-IDF vectorizer + balanced Logistic Regression
  - Calibrated probability outputs and confidence thresholding (`is_unclassified=True`, `ClauseFamily.OTHER`)
  - Out-of-distribution & noise length guard
- [x] **SBERT + ChromaDB k-NN + LR Classifier** — [`src/termsheet_ai/classify/sbert.py`](../src/termsheet_ai/classify/sbert.py)
  - `all-MiniLM-L6-v2` 384-d normalized embeddings with `@functools.lru_cache` singleton loading
  - Isolated in-memory `chromadb.EphemeralClient` vector collection with cosine space
  - Three scoring modes: ChromaDB cosine $k$-NN, Logistic Regression head, and Hybrid weighted blend
  - Built-in `retrieve_nearest_exemplars` helper for Stage S5 RAG summaries
  - Pydantic v2 immutability preserving input spans
- [x] **Classification Metrics & Evaluation Benchmark** — [`src/termsheet_ai/classify/metrics.py`](../src/termsheet_ai/classify/metrics.py)
  - Precision, Recall, Macro-F1, Weighted-F1, and Confusion Matrix computation
  - 5-fold stratified cross-validation benchmark comparing TF-IDF vs. SBERT variants
- [x] **Test Suite & CI Configuration** — [`tests/test_classify.py`](../tests/test_classify.py) and [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
  - 29 unit and integration tests covering all components, modes, edge cases, and end-to-end `TermSheetPipeline`
  - All 51 repository tests passing; zero lint errors

#### M5 — KR&R: Ontology & Rule Base
- [x] **Ontology v1 Graph Construction** — [`src/termsheet_ai/ontology.py`](../src/termsheet_ai/ontology.py)
  - Directed NetworkX semantic network encoding `SourceDoc`, `ClauseFamily`, `Attribute`, and `MarketNorm` nodes
- [x] **Initial YAML Rule Base (15 Rules)** — [`docs/rules_v1.yaml`](rules_v1.yaml)
  - First-order Horn-clause rules covering liquidation preference, board control, vesting, anti-dilution, protective provisions, drag-along, option pool, redemption, exclusivity, and dividends

### Planned
- **M6**: Implement forward-chaining engine v0 with justification records
- **M3**: Completed Week 2 deliverables; ready for S1 -> S2 -> S3 end-to-end integration in Week 3
- **M4**: Develop LLM extraction prompts, JSON schema, span validator
- **M2**: Continue corpus collection, begin synthetic term-sheet generator

---

*Last updated: 2026-10-24 by M3 & M5*


## Week 3 (25–31 Oct 2026)

### Completed

#### M3 — Deep Learning: Embeddings & Search
- [x] **Pipeline Verification & SBERT Production Default** — [`src/termsheet_ai/pipeline/runner.py`](../src/termsheet_ai/pipeline/runner.py)
  - Updated production pipeline (`TermSheetPipeline.create_production` and default constructor) to cleanly default Stage S2 to `SbertClassifier(mode="hybrid")`.
  - Verified contract compatibility with S1 `Ingestor` output and S3 `FactExtractor` input across all 13 clause families.
- [x] **Stage S5 Retrieval Pairing for Market Phrasing** — [`src/termsheet_ai/explain/explainer.py`](../src/termsheet_ai/explain/explainer.py)
  - Connected `retrieve_nearest_exemplars()` from Stage S2 SBERT ChromaDB index into `PlainEnglishExplainer`.
  - Enabled plain-English summaries to cite standard market phrasing directly from NVCA/YC exemplars.
  - Hardened `NumberConsistencyChecker` integration by passing exemplar citation numbers via `allowed_extras` to preserve zero-hallucination guarantees.
  - Added unit test `test_explainer_with_sbert_retriever_cites_market_exemplar` in [`tests/test_consistency.py`](../tests/test_consistency.py).
- [x] **Synthetic Corpus Benchmark v0** — [`scripts/benchmark_synthetic.py`](../scripts/benchmark_synthetic.py)
  - Benchmarked out-of-fold generalization across all 30 synthetic term sheets in `corpus/synthetic/` (15 standard with 221 clauses, 15 adversarial with 202 clauses; 423 total clauses).
  - Evaluated SBERT Hybrid vs. TF-IDF Baseline across overall, standard, and adversarial subsets:
    - Overall SBERT Accuracy: **71.63%**, Weighted F1: **0.7457**, Macro Precision: **0.7996**, Macro Recall: **0.8515**.
    - Standard Documents: Accuracy **73.76%**, Weighted F1: **0.7643**.
    - Adversarial Documents: Accuracy **69.31%**, Weighted F1: **0.7254**.
  - Verified adversarial clause robustness:
    - 3x+ / Participating Liquidation: **100.0%** (9/9)
    - Full-Ratchet Anti-Dilution: **100.0%** (9/9)
    - Hostile Founder Vesting: **100.0%** (3/3)
    - Investor-Controlled Board: **66.7%** (2/3)
    - Total Adversarial Clauses Correct: **23/24 (95.8%)**

#### M5 — KR&R: Ontology & Rule Base
- [x] **Expanded YAML Rule Base (31 Rules Across All 12 Canonical Families + Dividends)** — [`docs/rules_v1.yaml`](rules_v1.yaml)
  - Expanded rule base to 31 rules with full coverage of all 12 canonical `ClauseFamily` categories (`liquidation_preference`, `anti_dilution`, `board_composition`, `founder_vesting`, `protective_provisions`, `information_rights`, `rofr_co_sale`, `drag_along`, `option_pool`, `exclusivity_no_shop`, `conversion_rights`, `redemption_rights`) plus `dividends`.
  - Added derived `conclusion` predicates and `norm_id` links on all rules, including multi-step chained rule `R-CT-05` (`InvestorControl == true and VestingReset == true` $\Rightarrow$ `FounderRemovableWithEquityLoss`) chaining off `R-CT-02` (`InvestorControl`).
- [x] **Typed Semantic Network Completion with `Rule` Nodes & `deviatesFrom` Edges** — [`src/termsheet_ai/ontology.py`](../src/termsheet_ai/ontology.py)
  - Integrated `Rule` nodes and `appliesTo`, `deviatesFrom`, `conformsTo`, `citedIn`, and `dependsOn` edges into `build_ontology_graph()` (103 nodes, 202 directed edges).
  - Added `MARKET_NORMS` and `FAMILY_ATTRIBUTES` across all 12 canonical clause families and aliases, plus `get_rule_provenance()`, `get_rules_for_family()`, `get_family_norm()`, and `get_ontology_summary()`.
- [x] **Comprehensive Rule & Ontology Unit Test Suite** — [`tests/test_ontology.py`](../tests/test_ontology.py) and [`tests/test_reasoning.py`](../tests/test_reasoning.py)
  - Parameterized unit tests covering positive trigger and negative boundary conditions for all 31 YAML rules, ontology graph structure, AST syntax validation, and attribute grounding.

### Planned
- **M3**: Tune SBERT similarity and OOD thresholds based on synthetic benchmark error analysis; support gold annotation evaluation across the 50-document corpus.
- **M4**: Complete LLM attribute extraction benchmarking and prompt caching for S3.
- **M5**: Calibrate severity weights and market-norm thresholds against the synthetic/dev gold corpus; finalize full 12-family attribute schema and annotation guide for Week 4 gold labels.
- **M7**: Complete full Streamlit UI integration with trace visualizer.

---

## Week 4 (1–7 Nov 2026)

### Completed

#### M5 — KR&R: Ontology & Rule Base
- [x] **Multi-Step Forward-Chaining Proof Tree Support (`R-CT-02` $\rightarrow$ `R-CT-05`)** — [`src/termsheet_ai/reasoning/engine.py`](../src/termsheet_ai/reasoning/engine.py)
  - Upgraded `ForwardChainingEngine.infer()` to execute multi-pass fixpoint inference asserting derived `conclusion` predicates into `WorkingMemory` (`InvestorControl` from `R-CT-02` chaining into `R-CT-05`), enabling depth-2 proof trees in the Streamlit UI trace visualizer (`build_trace_tree`).
  - Added `ProtectiveProvisionsFact` and `VestingReset` working-memory grounding.
- [x] **Severity Weight & Market Norm Calibration** — [`src/termsheet_ai/ontology.py`](../src/termsheet_ai/ontology.py) and [`scripts/calibrate_norms.py`](../scripts/calibrate_norms.py)
  - Implemented `load_severity_weights_from_yaml()`, `compute_weighted_risk_score()`, `evaluate_norm_calibration()`, and `calibrate_severity_weights()`.
  - Verified **100.0% precision, 100.0% recall, 1.0000 F1, and 100.0% exact match** on core gold `expected_flags` across all 30 synthetic term sheets in `corpus/synthetic/`.
  - Confirmed calibrated severity weights (`CRITICAL: 40, HIGH: 25, MEDIUM: 10, LOW: 5, INFO: 0`) separate clean NVCA baseline term sheets (`10.0/100`) from adversarial founder-hostile term sheets (`100.0/100`, `+90.0` baseline-to-adversarial gap).
- [x] **Full 12-Family Attribute Schema & Annotation Guide Update** — [`docs/attribute_schema_v0.json`](attribute_schema_v0.json), [`docs/clause_family_definitions_v0.md`](clause_family_definitions_v0.md), and [`docs/annotation_guide_v0.md`](annotation_guide_v0.md)
  - Expanded JSON schema and annotation documentation to cover all 12 canonical clause families + dividends, rule IDs, and severity weights for Week 4 gold evaluation labelling.

---

*Last updated: 2026-11-11 by M5*
