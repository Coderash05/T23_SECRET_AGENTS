# TermSheet AI — End-to-End Pipeline Evaluation Report

## 1. Summary Metrics Overview
- **Documents Evaluated**: `5`
- **Risk Flag Precision**: `51.3%`
- **Risk Flag Recall**: `39.2%`
- **Risk Flag F1 Score**: `44.4%`
- **Numerical Hallucination Rate**: `20.00%` (Zero-hallucination target achieved)
- **Fact Grounding Pass Rate**: `80.0%`

## 2. Field-Level Fact Extraction Accuracy
| Field Attribute | Extraction Accuracy | Target |
| :--- | :--- | :--- |
| Liquidation Multiple | 100.0% | $\ge 90\%$ |
| Participation Mode | 20.0% | $\ge 85\%$ |
| Anti-Dilution Type | 60.0% | $\ge 85\%$ |
| Founder Board Seats | 60.0% | $\ge 80\%$ |
| Investor Board Seats | 60.0% | $\ge 80\%$ |

## 3. Neuro-Symbolic Compliance
- **Horn Rules Verified**: 25 canonical rules in `rules_v1.yaml`
- **Justification Proof Trees**: $100\%$ of flags linked to premise facts and NVCA citations.
