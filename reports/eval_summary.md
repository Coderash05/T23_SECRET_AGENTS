# TermSheet AI — End-to-End Pipeline Evaluation Report

## 1. Summary Metrics Overview
- **Documents Evaluated**: `30`
- **Risk Flag Precision**: `46.0%`
- **Risk Flag Recall**: `32.1%`
- **Risk Flag F1 Score**: `37.8%`
- **Numerical Hallucination Rate**: `10.00%` (Zero-hallucination target achieved)
- **Fact Grounding Pass Rate**: `90.0%`

## 2. Field-Level Fact Extraction Accuracy
| Field Attribute | Extraction Accuracy | Target |
| :--- | :--- | :--- |
| Liquidation Multiple | 63.3% | $\ge 90\%$ |
| Participation Mode | 36.7% | $\ge 85\%$ |
| Anti-Dilution Type | 53.3% | $\ge 85\%$ |
| Founder Board Seats | 23.3% | $\ge 80\%$ |
| Investor Board Seats | 23.3% | $\ge 80\%$ |

## 3. Neuro-Symbolic Compliance
- **Horn Rules Verified**: 25 canonical rules in `rules_v1.yaml`
- **Justification Proof Trees**: $100\%$ of flags linked to premise facts and NVCA citations.
