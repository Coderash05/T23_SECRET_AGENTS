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
