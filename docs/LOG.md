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

### Remaining for Week 1
- [ ] **M5**: Draft annotation guide v0 (clause family definitions + attribute schemas for annotators)
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

### Planned
- **M5**: Build ontology v1 (NetworkX graph) + author first 15 rules in YAML
- **M6**: Implement forward-chaining engine v0 with justification records
- **M3**: Completed Week 2 deliverables; ready for S1 -> S2 -> S3 end-to-end integration in Week 3
- **M4**: Develop LLM extraction prompts, JSON schema, span validator
- **M2**: Continue corpus collection, begin synthetic term-sheet generator

---

*Last updated: 2026-10-24 by M3*


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

### Planned
- **M3**: Tune SBERT similarity and OOD thresholds based on synthetic benchmark error analysis; support gold annotation evaluation across the 50-document corpus.
- **M4**: Complete LLM attribute extraction benchmarking and prompt caching for S3.
- **M7**: Complete full Streamlit UI integration with trace visualizer.

---

*Last updated: 2026-10-31 by M3*

## Week 4 (1–7 Nov 2026)

### Completed

#### M2 — Data & Preprocessing
- [x] **Annotation guide v0** — [`docs/annotation_guide_v0.md`](annotation_guide_v0.md)
  - Documented annotation rules, literal source-span requirements, null handling,
    expected flags, and gold-file format
  - Added family-specific guidance and attributes for all 13 clause families
- [x] **Synthetic evaluation registration** — [`corpus/manifest.json`](../corpus/manifest.json)
  - Registered all 30 synthetic text/gold pairs as auto-labeled entries
  - Maintained a 20-document dev split with the currently available corpus
- [x] **SEC EDGAR Form D lookup client** — [`src/termsheet_ai/edgar/client.py`](../src/termsheet_ai/edgar/client.py)
  - Added required User-Agent support, rate limiting, response caching, and
    found/not-found/ambiguous result handling
  - Ensured lookup failures degrade to `not_found` without surfacing false risk flags
- [x] **EDGAR mocked test suite** — [`tests/test_edgar.py`](../tests/test_edgar.py)
  - Covers positive, empty, ambiguous, failed, cached, batched, rate-limited,
    and invalid-input lookups

### Validation
- [x] Targeted EDGAR tests pass in the repository virtual environment
- [x] Manifest JSON and synthetic text/gold registrations validated

### Remaining
- [ ] Gold annotations for the real documents require independent human labeling
  by Annotators A and B before reconciliation and Cohen's kappa reporting.

## Week 5 (8–14 Nov 2026)

### In progress

#### M2 — Data & Preprocessing
- [x] **Corpus evaluation harness** — [`src/termsheet_ai/evaluation/corpus.py`](../src/termsheet_ai/evaluation/corpus.py)
  - Validates manifest paths, gold sidecars, duplicate clause IDs, and literal source spans
  - Computes family precision, recall, F1, confusion matrices, and expected-flag counts
  - Emits explicit missing-file and missing-annotation diagnostics
- [x] **Evaluation CLI and focused tests** — [`scripts/evaluate_corpus.py`](../scripts/evaluate_corpus.py) and [`tests/test_evaluation.py`](../tests/test_evaluation.py)
  - Supports validation-only audits and prediction-vs-gold evaluation
  - Covers valid and invalid spans, missing files, duplicate IDs, metrics, and malformed predictions
- [x] **Corpus audit**
  - Synthetic sidecars are available for the 30 registered synthetic documents
  - Real-document gold sidecars remain blocked on independent Annotator A/B labeling
