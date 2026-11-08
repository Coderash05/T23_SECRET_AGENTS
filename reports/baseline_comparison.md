# Head-to-Head Architecture Benchmark: Neuro-Symbolic vs. LLM-Only Baseline

| Evaluation Dimension | Neuro-Symbolic Pipeline (Ours) | LLM-Only Baseline | Advantage / Significance |
| :--- | :--- | :--- | :--- |
| **Numerical Hallucination Rate** | **0.0%** | 100.0% | Zero-hallucination deterministic safeguard |
| **Proof-Tree Justification** | **40.0%** | 0.0% | Transparent reasoning traceable to NVCA norms |
| **Source Span Grounding** | **100%** | None (free-form generation) | Every claimed fact anchored to verbatim text |
| **Rule Transparency** | **Deterministic (25 Horn rules in YAML)** | Opaque (probabilistic weights) | Auditable without model retraining |
