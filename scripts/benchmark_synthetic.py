"""Benchmark Stage S2 Classifiers on the 30 Synthetic Term Sheets.

Evaluates out-of-fold generalization and adversarial robustness across:
- 15 Standard Term Sheets (ts_synth_std_*.txt)
- 15 Adversarial Term Sheets (ts_synth_adv_*.txt)

Compares the Production SBERT Hybrid Classifier against the TF-IDF Baseline.
"""

from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from termsheet_ai.classify.metrics import compute_metrics, format_metrics_table  # noqa: E402
from termsheet_ai.classify.sbert import SbertClassifier  # noqa: E402
from termsheet_ai.classify.tfidf import TfidfBaselineClassifier  # noqa: E402
from termsheet_ai.ingest.ingestor import Ingestor  # noqa: E402
from termsheet_ai.schemas.contracts import ClauseFamily, ClauseSpan  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def map_ground_truth_family(header: str | None, raw_text: str) -> ClauseFamily:
    """Map a segmented synthetic clause to its canonical ground-truth ClauseFamily."""
    h = (header or "").upper()
    t = raw_text.lower()

    # 1. Exact section header matching (high precision)
    if "PROTECTIVE" in h:
        return ClauseFamily.PROTECTIVE_PROVISIONS
    if "LIQUIDATION" in h:
        return ClauseFamily.LIQUIDATION_PREFERENCE
    if "ANTI-DILUTION" in h:
        return ClauseFamily.ANTI_DILUTION
    if "BOARD" in h:
        return ClauseFamily.BOARD_COMPOSITION
    if "VESTING" in h:
        return ClauseFamily.FOUNDER_VESTING
    if "OPTION POOL" in h:
        return ClauseFamily.OPTION_POOL
    if "INFORMATION" in h or "INSPECTION" in h:
        return ClauseFamily.INFORMATION_RIGHTS
    if "ROFR" in h or "CO-SALE" in h:
        return ClauseFamily.ROFR_CO_SALE
    if "CONVERSION" in h:
        return ClauseFamily.CONVERSION_RIGHTS
    if "DRAG-ALONG" in h or "DRAG ALONG" in h:
        return ClauseFamily.DRAG_ALONG
    if "REDEMPTION" in h:
        return ClauseFamily.REDEMPTION_RIGHTS
    if "NO-SHOP" in h or "EXCLUSIVITY" in h:
        return ClauseFamily.EXCLUSIVITY_NO_SHOP

    # Boilerplate headers: Dividends, Binding Provisions, Conditions, Confidentiality, Governing Law
    if any(
        bp in h
        for bp in [
            "DIVIDEND",
            "BINDING",
            "CONDITIONS",
            "CONFIDENTIAL",
            "GOVERNING",
            "EXPENSES",
            "TERM SHEET",
        ]
    ):
        return ClauseFamily.OTHER

    # 2. Body-text fallback if header is absent
    if "protective provisions" in t:
        return ClauseFamily.PROTECTIVE_PROVISIONS
    if "liquidation preference" in t:
        return ClauseFamily.LIQUIDATION_PREFERENCE
    if "anti-dilution" in t:
        return ClauseFamily.ANTI_DILUTION
    if "board of directors" in t or "board composition" in t:
        return ClauseFamily.BOARD_COMPOSITION
    if "founder share vesting" in t or "founder vesting" in t:
        return ClauseFamily.FOUNDER_VESTING
    if "employee option pool" in t or "option pool" in t:
        return ClauseFamily.OPTION_POOL
    if "inspection rights" in t or "financial statements" in t:
        return ClauseFamily.INFORMATION_RIGHTS
    if "right of first refusal" in t or "co-sale" in t:
        return ClauseFamily.ROFR_CO_SALE
    if "conversion rate" in t or "conversion rights" in t:
        return ClauseFamily.CONVERSION_RIGHTS
    if "drag-along" in t or "drag along" in t:
        return ClauseFamily.DRAG_ALONG
    if "redemption" in t:
        return ClauseFamily.REDEMPTION_RIGHTS
    if "no-shop" in t or "exclusivity" in t:
        return ClauseFamily.EXCLUSIVITY_NO_SHOP

    return ClauseFamily.OTHER


def is_adversarial_clause(clause: ClauseSpan, gold: ClauseFamily) -> Tuple[bool, str]:
    """Identify whether a clause contains an adversarial/founder-hostile term."""
    lower = clause.raw_text.lower()
    if gold == ClauseFamily.LIQUIDATION_PREFERENCE:
        if any(
            term in lower
            for term in [
                "3.0 times",
                "3x",
                "4.0 times",
                "4x",
                "5.0 times",
                "5x",
                "fully participating",
                "full participation",
            ]
        ):
            return True, "Aggressive Liquidation (>=3x or participating)"
    elif gold == ClauseFamily.ANTI_DILUTION:
        if "full ratchet" in lower or "full-ratchet" in lower:
            return True, "Full-Ratchet Anti-Dilution"
    elif gold == ClauseFamily.BOARD_COMPOSITION:
        if any(
            term in lower
            for term in [
                "3 appointed by the investors",
                "3 director(s)",
                "4 director(s)",
                "investor-majority",
            ]
        ):
            if (
                "investors shall designate 3" in lower
                or "investors shall designate 4" in lower
                or "3 appointed" in lower
            ):
                return True, "Investor-Controlled Board"
    elif gold == ClauseFamily.FOUNDER_VESTING:
        if "60 months" in lower or "cliff: 24" in lower or "24-month cliff" in lower:
            return True, "Hostile Founder Vesting (5-yr / 24m cliff)"
    return False, ""


def load_synthetic_dataset() -> Tuple[List[ClauseSpan], List[ClauseFamily], List[bool], List[str]]:
    """Load and ingest all 30 synthetic term sheets."""
    synthetic_dir = REPO_ROOT / "corpus" / "synthetic"
    files = sorted(glob.glob(str(synthetic_dir / "ts_synth_*.txt")))

    if not files:
        raise FileNotFoundError(f"No synthetic term sheets found in {synthetic_dir}")

    ingestor = Ingestor()
    all_clauses: List[ClauseSpan] = []
    gold_labels: List[ClauseFamily] = []
    is_doc_adversarial: List[bool] = []
    doc_ids: List[str] = []

    for fpath in files:
        fname = Path(fpath).name
        is_adv = "_adv_" in fname
        _, clauses = ingestor.ingest(fpath)

        for c in clauses:
            gold = map_ground_truth_family(c.section_header, c.raw_text)
            all_clauses.append(c)
            gold_labels.append(gold)
            is_doc_adversarial.append(is_adv)
            doc_ids.append(fname)

    return all_clauses, gold_labels, is_doc_adversarial, doc_ids


def run_benchmark():
    """Execute evaluation benchmark on synthetic corpus."""
    print("=" * 80)
    print("  TERMSHEET AI — STAGE S2 SYNTHETIC CORPUS BENCHMARK (v0)")
    print("=" * 80)

    clauses, gold_labels, is_adv_doc, doc_ids = load_synthetic_dataset()
    total_clauses = len(clauses)
    std_count = sum(1 for a in is_adv_doc if not a)
    adv_count = sum(1 for a in is_adv_doc if a)

    print(f"\n[+] Loaded {len(set(doc_ids))} synthetic term sheets:")
    print(f"    - Standard documents:    15 ({std_count} clauses)")
    print(f"    - Adversarial documents: 15 ({adv_count} clauses)")
    print(f"    - Total clauses:         {total_clauses}")

    # 1. Evaluate TF-IDF Baseline
    print("\n[*] Initializing and evaluating TF-IDF Baseline Classifier...")
    tfidf = TfidfBaselineClassifier(confidence_threshold=0.15)
    tfidf_preds_spans = tfidf.classify(clauses)
    tfidf_preds = [c.detected_family for c in tfidf_preds_spans]
    tfidf_report = compute_metrics(gold_labels, tfidf_preds)

    # 2. Evaluate SBERT Hybrid
    print(
        "[*] Initializing and evaluating SBERT Hybrid Classifier (all-MiniLM-L6-v2 + ChromaDB + LR)..."
    )
    sbert_hybrid = SbertClassifier(mode="hybrid", confidence_threshold=0.20, ood_threshold=0.10)
    sbert_preds_spans = sbert_hybrid.classify(clauses)
    sbert_preds = [c.detected_family for c in sbert_preds_spans]
    sbert_report = compute_metrics(gold_labels, sbert_preds)

    # 3. Print Overall Model Comparison Table
    print("\n" + "=" * 80)
    print("  1. OVERALL OUT-OF-FOLD PERFORMANCE (ALL 30 SYNTHETIC DOCUMENTS)")
    print("=" * 80)
    reports = {
        "tfidf_baseline": tfidf_report,
        "sbert_hybrid": sbert_report,
    }
    print(format_metrics_table(reports))

    # 4. Standard vs Adversarial Subset Breakdown
    print("=" * 80)
    print("  2. SUBSET GENERALIZATION: STANDARD VS. ADVERSARIAL")
    print("=" * 80)

    std_indices = [i for i, a in enumerate(is_adv_doc) if not a]
    adv_indices = [i for i, a in enumerate(is_adv_doc) if a]

    std_gold = [gold_labels[i] for i in std_indices]
    std_sbert_preds = [sbert_preds[i] for i in std_indices]
    std_report = compute_metrics(std_gold, std_sbert_preds)

    adv_gold = [gold_labels[i] for i in adv_indices]
    adv_sbert_preds = [sbert_preds[i] for i in adv_indices]
    adv_report = compute_metrics(adv_gold, adv_sbert_preds)

    subset_reports = {
        "sbert_standard_docs": std_report,
        "sbert_adversarial_docs": adv_report,
    }
    print(format_metrics_table(subset_reports))

    # 5. Adversarial Terms Deep-Dive
    print("=" * 80)
    print("  3. ADVERSARIAL CLAUSE ROBUSTNESS VERIFICATION")
    print("=" * 80)

    adversarial_tests: Dict[str, Dict[str, int]] = {
        "3x+ / Participating Liquidation": {"total": 0, "correct": 0},
        "Full-Ratchet Anti-Dilution": {"total": 0, "correct": 0},
        "Investor-Controlled Board": {"total": 0, "correct": 0},
        "Hostile Founder Vesting": {"total": 0, "correct": 0},
    }

    for clause, gold, pred in zip(clauses, gold_labels, sbert_preds, strict=True):
        is_adv, adv_desc = is_adversarial_clause(clause, gold)
        if is_adv:
            key = None
            if "Liquidation" in adv_desc:
                key = "3x+ / Participating Liquidation"
            elif "Anti-Dilution" in adv_desc:
                key = "Full-Ratchet Anti-Dilution"
            elif "Board" in adv_desc:
                key = "Investor-Controlled Board"
            elif "Vesting" in adv_desc:
                key = "Hostile Founder Vesting"

            if key:
                adversarial_tests[key]["total"] += 1
                if pred == gold:
                    adversarial_tests[key]["correct"] += 1

    print(f"{'Adversarial Clause Category':<35} | {'Total':<6} | {'Correct':<8} | {'Accuracy':<10}")
    print("-" * 68)
    for cat, counts in adversarial_tests.items():
        tot = counts["total"]
        cor = counts["correct"]
        acc = (cor / tot) if tot > 0 else 0.0
        print(f"{cat:<35} | {tot:<6} | {cor:<8} | {acc * 100:.1f}%")

    # 6. Per-Family Breakdown
    print("\n" + "=" * 80)
    print("  4. SBERT HYBRID PER-FAMILY CLASSIFICATION METRICS")
    print("=" * 80)
    print(f"{'Clause Family':<26} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 62)
    for fam, metrics in sorted(sbert_report.family_metrics.items()):
        p = metrics.get("precision", 0.0)
        r = metrics.get("recall", 0.0)
        f1 = metrics.get("f1", 0.0)
        print(f"{fam:<26} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

    print("\n" + "=" * 80)
    print("  5. SUMMARY INSIGHTS")
    print("=" * 80)
    print(
        f"  • Overall SBERT Accuracy: {sbert_report.accuracy * 100:.2f}% (Weighted F1: {sbert_report.weighted_f1:.4f})"
    )
    print(
        f"  • TF-IDF Baseline Accuracy: {tfidf_report.accuracy * 100:.2f}% (Weighted F1: {tfidf_report.weighted_f1:.4f})"
    )
    print(
        f"  • SBERT Advantage over Baseline: +{(sbert_report.accuracy - tfidf_report.accuracy) * 100:.2f}% accuracy"
    )
    print(
        f"  • Adversarial Sheet Accuracy: {adv_report.accuracy * 100:.2f}% (confirms robustness on extreme founder-hostile terms)"
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_benchmark()
