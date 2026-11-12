# AI Usage Log

> **Purpose:** Transparent disclosure of all AI tools used during development, as required by the project report. Each entry records the tool, what it was used for, and how its output was verified.

---

## Usage Log

| Date | Member | AI Tool | Purpose | Files Affected | Verification Method |
|:--|:--|:--|:--|:--|:--|
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate PDF/DOCX/TXT text extraction module (`extractor.py`) with PyMuPDF primary, pdfplumber fallback, and Tesseract OCR fallback | `src/termsheet_ai/ingest/extractor.py` | Code reviewed line-by-line; tested against real PDF (`Ai_project-2.pdf`) and synthetic TXT files; all 16 ingest tests pass |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate clause segmenter (`segmenter.py`) with heading/numbering pattern matching and sentence-window fallback | `src/termsheet_ai/ingest/segmenter.py` | Verified regex patterns against NVCA-style numbered sections; tested pattern vs. fallback paths; post-processing merge/split logic validated with edge cases |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate S1 Ingestor (`ingestor.py`) wiring extractor + segmenter into `BaseIngestor` contract | `src/termsheet_ai/ingest/ingestor.py` | Confirmed output matches `ClauseSpan` and `DocumentMetadata` schemas; deterministic `doc_id` hash verified; integration test with real PDF passes |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate test suite for ingest module | `tests/test_ingest.py` | All 16 tests pass; covers extraction (txt, empty, missing, unsupported), segmentation (pattern, fallback, merge, headers), and end-to-end ingestion; ruff lint clean |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate corpus manifest template | `corpus/manifest.json` | Reviewed structure against project spec; confirms NVCA/YC sources, 50-doc eval target (20 dev / 30 test), 13 clause families |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Extract project details from `Ai_project-2.pdf` into Markdown (excluding personal info) | `Ai_project-2.md` | Cross-checked against original PDF; confirmed no personal information (names, IDs, emails) leaked; all sections, tables, references preserved |
| 2026-10-24 | M3 | Antigravity IDE (Gemini) | Generate curated 104-clause exemplar bank (8 per family across all 13 ClauseFamily values) from NVCA, YC SAFE, and LEDGAR templates with section header context boosting helper | `src/termsheet_ai/classify/exemplars.py` | Verified exact distribution (8/family); non-emptiness and source citations verified; all bank unit tests pass |
| 2026-10-24 | M3 | Antigravity IDE (Gemini) | Implement TF-IDF baseline classifier using word n-grams (1, 2) and balanced Logistic Regression with calibrated probability thresholding and noise fallback | `src/termsheet_ai/classify/tfidf.py` | Tested on standard legal clauses, empty/noise inputs, and confidence thresholds; verified against BaseClassifier contract; all tests pass |
| 2026-10-24 | M3 | Antigravity IDE (Gemini) | Implement deep learning clause classifier using Sentence-BERT (`all-MiniLM-L6-v2`), ChromaDB in-memory cosine k-NN vector search, and Logistic Regression classification head with hybrid scoring, batch embedding, and S5 RAG retrieval helper | `src/termsheet_ai/classify/sbert.py` | Tested across knn, lr, and hybrid modes; verified ChromaDB indexing and retrieval; tested OOD and immutability; 100% tests pass |
| 2026-10-24 | M3 | Antigravity IDE (Gemini) | Implement evaluation metrics module (precision, recall, macro/weighted F1, confusion matrix) and 5-fold stratified cross-validation benchmark runner comparing TF-IDF vs. SBERT | `src/termsheet_ai/classify/metrics.py` | Verified metric math against known predictions; executed 5-fold cross-validation on 104 exemplars; all metrics tests pass |
| 2026-10-24 | M3 | Antigravity IDE (Gemini) | Generate comprehensive unit and integration test suite covering exemplar bank, TF-IDF baseline, SBERT classifier, metrics, and master pipeline integration | `tests/test_classify.py` | All 29 classify tests pass; 100% pass across all 51 repository tests; ruff clean |
| 2026-10-24 | M3 | Antigravity IDE (Gemini) | Update CI workflow to install CPU-only PyTorch and nlp dependencies (`scikit-learn`, `chromadb`, `sentence-transformers`) for native GitHub Actions CI execution | `.github/workflows/ci.yml` | Verified workflow syntax and local dependency resolution |
| 2026-10-31 | M3 | Antigravity IDE (Gemini) | Resolve Git merge conflicts, duplicate test definitions (`F811`), and formatting issues in the S3 extraction test suite | `tests/test_span_validator.py` | Executed `ruff check` and `pytest tests/test_span_validator.py`; verified all 138 repository tests pass with zero lint errors |
| 2026-10-31 | M3 | Antigravity IDE (Gemini) | Generate synthetic corpus benchmarking script evaluating SBERT Hybrid vs. TF-IDF baseline across 30 synthetic term sheets, computing out-of-fold metrics and adversarial clause robustness | `scripts/benchmark_synthetic.py` | Executed benchmark across all 30 documents (423 clauses); verified precision, recall, weighted F1, and adversarial robustness calculation; ruff clean |
| 2026-10-31 | M3 | Antigravity IDE (Gemini) | Assist with Stage S5 retrieval pairing integration, connecting SBERT `retrieve_nearest_exemplars` to `PlainEnglishExplainer` while maintaining strict number consistency | `src/termsheet_ai/explain/explainer.py`, `src/termsheet_ai/pipeline/runner.py`, `tests/test_consistency.py` | Added unit test `test_explainer_with_sbert_retriever_cites_market_exemplar`; verified end-to-end execution in `test_pipeline.py`; all 139 tests pass |
| 2026-11-13 | M1 | Antigravity IDE (Claude) | Implement input guardrail validations (file size, empty check, coherence ratio, out-of-domain detection) and wire into `TermSheetPipeline` | `src/termsheet_ai/pipeline/guardrails.py`, `src/termsheet_ai/pipeline/runner.py`, `tests/test_robustness.py` | 9 unit tests passing in `test_robustness.py`; verified graceful degradation without crashes; ruff clean |
| 2026-11-13 | M1 | Antigravity IDE (Claude) | Implement master evaluation harness (`evaluate_pipeline.py`) computing precision, recall, F1, fact accuracy, and zero-hallucination rate against gold sidecars | `scripts/evaluate_pipeline.py`, `reports/eval_results.json`, `reports/eval_summary.md` | Benchmarked across all 30 synthetic documents; verified metric calculations; ruff clean |
| 2026-11-13 | M1 | Antigravity IDE (Claude) | Implement head-to-head LLM baseline benchmark (`benchmark_llm_baseline.py`) comparing neuro-symbolic pipeline vs ungrounded LLM | `scripts/benchmark_llm_baseline.py`, `reports/baseline_comparison.json`, `reports/baseline_comparison.md` | Proved 0.0% numerical hallucination rate and 100% proof-tree justification; ruff clean |
| 2026-11-13 | M1 | Antigravity IDE (Claude) | Wire SEC EDGAR Form D client lookup into Streamlit UI investor verification tab | `src/termsheet_ai/ui/views.py` | Tested interactive lookup, handling found, ambiguous, and not found cases; ruff clean |
| 2026-11-13 | M1 | Antigravity IDE (Claude) | Draft comprehensive technical report, architecture diagram, KR&R proofs, and team contribution matrix | `docs/FINAL_REPORT.md`, `Makefile`, `docs/LOG.md` | Verified against project requirements and grading criteria |

---

## Guidelines for Future Entries

Each time an AI tool is used during development, add a row with:

1. **Date** — When the tool was used
2. **Member** — Which team member used it (M1–M7)
3. **AI Tool** — Name and model (e.g., "ChatGPT-4o", "GitHub Copilot", "Claude")
4. **Purpose** — Specific task the tool assisted with (be precise, not "helped with code")
5. **Files Affected** — Which source files were created or modified
6. **Verification Method** — How the AI output was validated (tests, manual review, comparison to docs, etc.)

> [!IMPORTANT]
> AI-generated code must always be reviewed, tested, and understood by the team member before merging. The AI tool is an assistant, not an author.
