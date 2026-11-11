# TermSheet AI: Annotation Guide & Norm Calibration Reference

## Overview
This guide is for Annotators (Members 2 and 7) to label our evaluation set of 50 term sheets (20 dev / 30 test). You are creating the "gold standard" labels that our AI's extraction and reasoning modules will be tested against. To avoid circularity, annotators must not write or modify the symbolic logic rules.

## 1. Clause Spans & Family Labelling
* For each document, identify the text boundaries for the 12 canonical clause families plus `other` defined in [`clause_family_definitions_v0.md`](clause_family_definitions_v0.md).
* Highlight the exact starting and ending text. If a clause family is absent from a document, leave its entry as `null`.

## 2. Attribute Extraction & The "Literal Span" Rule
* Extract the exact structured variables defined in [`attribute_schema_v0.json`](attribute_schema_v0.json) (e.g., liquidation multiple, participation mode, cap, anti-dilution type, board seat counts, vesting months, cliff months, option pool percentage, exclusivity days, drag-along threshold, redemption years, cumulative dividends).
* **CRITICAL VALIDATOR RULE:** Every extracted numeric value *must* literally appear in the verbatim `source_span` quote you select.
* If a value is not explicitly stated in the contract, label it as `null`. Never guess or assume a default value if the clause text is silent.

## 3. Expected Risk Flags & Calibrated Severity Weights
* Based on US VC market norms (NVCA Model Term Sheet, YC SAFE, and Kaplan & Strömberg empirical benchmarks), record the expected rule flags in `expected_flags`.
* Calibrated severity weights used by the S4 forward-chaining engine to compute `overall_score` (capped at `100.0`):
  * `CRITICAL`: `40.0` points (`R-AD-01`, `R-LP-04`, `R-LP-05`)
  * `HIGH`: `25.0` points (`R-LP-01`, `R-LP-02`, `R-CT-02`, `R-CT-04`, `R-CT-05`, `R-VS-02`, `R-PP-01`, `R-DA-01`, `R-DA-02`, `R-OP-02`, `R-RD-01`, `R-RD-02`, `R-DV-01`)
  * `MEDIUM`: `10.0` points (`R-LP-03`, `R-AD-03`, `R-CT-03`, `R-VS-01`, `R-VS-03`, `R-OP-01`, `R-EX-01`, `R-RO-02`, `R-IR-01`, `R-CV-01`)
  * `LOW`: `5.0` points (`R-AD-02`, `R-VS-04`, `R-IR-02`)
  * `INFO`: `0.0` points (`R-PP-02`, `R-RO-01`)

### Multi-Step Chained Reasoning (`R-CT-02` $\rightarrow$ `R-CT-05`)
* When `investor_seats > founder_seats`, rule `R-CT-02` derives the intermediate predicate `InvestorControl = true`.
* When `InvestorControl == true` and the founder vesting clause resets existing shares (`VestingReset == true`), rule `R-CT-05` fires as a second inference step (`FounderRemovableWithEquityLoss`, severity `HIGH`), producing a depth-2 proof tree in the audit trace.

## 4. Conflict Resolution & Inter-Annotator Agreement
* Both annotators label documents independently on the 20-document dev set and 30-document test set.
* Disagreements are reconciled jointly and Cohen's $\kappa$ is computed to verify annotation reliability.
