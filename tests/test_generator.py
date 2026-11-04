"""Tests for the synthetic term-sheet generator.

Covers:
  - Deterministic seeded output (reproducibility)
  - Correct adversarial / standard split ratios
  - Gold-label expected-flag computation matches rules_v1.yaml
  - Surface text contains the sampled numeric values
  - Disk I/O round-trip (text files + gold JSON sidecars + batch manifest)
  - Edge cases: single-sheet generation, 100% adversarial batches
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from termsheet_ai.data.generator import (
    GoldLabel,
    SyntheticGenerator,
    SyntheticSheet,
    _compute_expected_flags,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generator() -> SyntheticGenerator:
    """Deterministic generator with fixed seed."""
    return SyntheticGenerator(seed=42)


@pytest.fixture
def small_batch(generator: SyntheticGenerator) -> list[SyntheticSheet]:
    """A small batch of 6 sheets (3 adversarial, 3 standard)."""
    return generator.generate_batch(count=6, adversarial_ratio=0.5)


# ---------------------------------------------------------------------------
# Basic generation tests
# ---------------------------------------------------------------------------


class TestSingleGeneration:
    """Tests for generating individual term sheets."""

    def test_generate_standard_sheet(self, generator: SyntheticGenerator) -> None:
        sheet = generator.generate_one(index=1, adversarial=False)
        assert isinstance(sheet, SyntheticSheet)
        assert sheet.doc_id == "synth_std_001"
        assert sheet.filename == "ts_synth_std_001.txt"
        assert len(sheet.text) > 500  # should be a substantial document
        assert not sheet.gold_label.is_adversarial

    def test_generate_adversarial_sheet(self, generator: SyntheticGenerator) -> None:
        sheet = generator.generate_one(index=1, adversarial=True)
        assert sheet.doc_id == "synth_adv_001"
        assert sheet.filename == "ts_synth_adv_001.txt"
        assert sheet.gold_label.is_adversarial

    def test_text_contains_term_sheet_header(self, generator: SyntheticGenerator) -> None:
        sheet = generator.generate_one(index=1, adversarial=False)
        assert "TERM SHEET" in sheet.text
        assert "PREFERRED STOCK FINANCING" in sheet.text

    def test_text_contains_clause_sections(self, generator: SyntheticGenerator) -> None:
        sheet = generator.generate_one(index=1, adversarial=False)
        text_upper = sheet.text.upper()
        # Every generated sheet should contain these key clause families
        assert "LIQUIDATION" in text_upper
        assert "ANTI-DILUTION" in text_upper or "ANTI DILUTION" in text_upper
        assert "BOARD" in text_upper or "DIRECTOR" in text_upper
        assert "VESTING" in text_upper
        assert "OPTION POOL" in text_upper or "EQUITY INCENTIVE" in text_upper
        assert "DRAG" in text_upper or "DRAG-ALONG" in text_upper

    def test_text_contains_sampled_values(self, generator: SyntheticGenerator) -> None:
        sheet = generator.generate_one(index=1, adversarial=False)
        label = sheet.gold_label
        # The liquidation multiple should appear in the text
        assert str(label.liq_multiple) in sheet.text or f"{label.liq_multiple}x" in sheet.text
        # The option pool percentage should appear
        assert str(label.option_pool_pct) in sheet.text
        # The exclusivity days should appear
        assert str(label.exclusivity_days) in sheet.text


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Verify that seeded generators produce identical output."""

    def test_same_seed_same_output(self) -> None:
        gen1 = SyntheticGenerator(seed=123)
        gen2 = SyntheticGenerator(seed=123)
        sheet1 = gen1.generate_one(index=1, adversarial=True)
        sheet2 = gen2.generate_one(index=1, adversarial=True)
        assert sheet1.text == sheet2.text
        assert sheet1.gold_label == sheet2.gold_label

    def test_different_seed_different_output(self) -> None:
        gen1 = SyntheticGenerator(seed=1)
        gen2 = SyntheticGenerator(seed=2)
        sheet1 = gen1.generate_one(index=1, adversarial=False)
        sheet2 = gen2.generate_one(index=1, adversarial=False)
        # Very unlikely to produce identical text with different seeds
        assert sheet1.text != sheet2.text


# ---------------------------------------------------------------------------
# Batch generation tests
# ---------------------------------------------------------------------------


class TestBatchGeneration:
    """Tests for batch generation with adversarial/standard ratios."""

    def test_batch_count(self, small_batch: list[SyntheticSheet]) -> None:
        assert len(small_batch) == 6

    def test_adversarial_ratio(self, small_batch: list[SyntheticSheet]) -> None:
        adv_count = sum(1 for s in small_batch if s.gold_label.is_adversarial)
        std_count = sum(1 for s in small_batch if not s.gold_label.is_adversarial)
        assert adv_count == 3
        assert std_count == 3

    def test_default_batch_30(self, generator: SyntheticGenerator) -> None:
        batch = generator.generate_batch()  # default count=30, ratio=0.5
        assert len(batch) == 30
        adv = sum(1 for s in batch if s.gold_label.is_adversarial)
        assert adv == 15

    def test_full_adversarial_batch(self, generator: SyntheticGenerator) -> None:
        batch = generator.generate_batch(count=5, adversarial_ratio=1.0)
        assert len(batch) == 5
        assert all(s.gold_label.is_adversarial for s in batch)

    def test_unique_doc_ids(self, small_batch: list[SyntheticSheet]) -> None:
        doc_ids = [s.doc_id for s in small_batch]
        assert len(doc_ids) == len(set(doc_ids))


# ---------------------------------------------------------------------------
# Gold-label and expected-flag tests
# ---------------------------------------------------------------------------


class TestExpectedFlags:
    """Test that gold-label flag computation matches rules_v1.yaml logic."""

    def test_clean_sheet_has_few_flags(self) -> None:
        """A completely market-standard sheet should trigger minimal flags."""
        label = GoldLabel(
            doc_id="test_clean",
            is_adversarial=False,
            liq_multiple=1.0,
            liq_participation="non_participating",
            liq_cap=None,
            liq_seniority="pari_passu",
            anti_dilution_type="broad_based_weighted_average",
            board_investor_seats=1,
            board_founder_seats=2,
            board_independent_seats=1,
            vesting_months=48,
            cliff_months=12,
            acceleration="double_trigger",
            option_pool_pct=15,
            exclusivity_days=30,
            drag_threshold_pct=67,
            redemption_years=5,
            dividends_cumulative=False,
        )
        flags = _compute_expected_flags(label)
        # Only R-AD-02 (informational) should fire
        assert flags == ["R-AD-02"]

    def test_adversarial_sheet_has_many_flags(self) -> None:
        """A maximally founder-hostile sheet should trigger many flags."""
        label = GoldLabel(
            doc_id="test_hostile",
            is_adversarial=True,
            liq_multiple=3.0,
            liq_participation="fully_participating",
            liq_cap=None,
            liq_seniority="senior",
            anti_dilution_type="full_ratchet",
            board_investor_seats=3,
            board_founder_seats=1,
            board_independent_seats=0,
            vesting_months=48,
            cliff_months=24,
            acceleration="none",
            option_pool_pct=30,
            exclusivity_days=90,
            drag_threshold_pct=25,
            redemption_years=2,
            dividends_cumulative=True,
        )
        flags = _compute_expected_flags(label)
        expected = [
            "R-LP-01",
            "R-LP-02",
            "R-LP-03",
            "R-AD-01",
            "R-CT-02",
            "R-CT-03",
            "R-VS-02",
            "R-OP-01",
            "R-EX-01",
            "R-DA-01",
            "R-RD-01",
            "R-DV-01",
        ]
        assert set(flags) == set(expected)
        assert len(flags) == 12

    def test_liq_multiple_flag(self) -> None:
        label = GoldLabel(
            doc_id="t",
            is_adversarial=False,
            liq_multiple=2.0,
            liq_participation="non_participating",
            liq_cap=None,
            liq_seniority="pari_passu",
            anti_dilution_type="broad_based_weighted_average",
            board_investor_seats=1,
            board_founder_seats=2,
            board_independent_seats=1,
            vesting_months=48,
            cliff_months=12,
            acceleration="double_trigger",
            option_pool_pct=15,
            exclusivity_days=30,
            drag_threshold_pct=67,
            redemption_years=5,
            dividends_cumulative=False,
        )
        flags = _compute_expected_flags(label)
        assert "R-LP-01" in flags

    def test_full_ratchet_flag(self) -> None:
        label = GoldLabel(
            doc_id="t",
            is_adversarial=False,
            liq_multiple=1.0,
            liq_participation="non_participating",
            liq_cap=None,
            liq_seniority="pari_passu",
            anti_dilution_type="full_ratchet",
            board_investor_seats=1,
            board_founder_seats=2,
            board_independent_seats=1,
            vesting_months=48,
            cliff_months=12,
            acceleration="double_trigger",
            option_pool_pct=15,
            exclusivity_days=30,
            drag_threshold_pct=67,
            redemption_years=5,
            dividends_cumulative=False,
        )
        flags = _compute_expected_flags(label)
        assert "R-AD-01" in flags
        assert "R-AD-02" not in flags  # mutually exclusive

    def test_investor_board_majority_flag(self) -> None:
        label = GoldLabel(
            doc_id="t",
            is_adversarial=False,
            liq_multiple=1.0,
            liq_participation="non_participating",
            liq_cap=None,
            liq_seniority="pari_passu",
            anti_dilution_type="broad_based_weighted_average",
            board_investor_seats=3,
            board_founder_seats=1,
            board_independent_seats=0,
            vesting_months=48,
            cliff_months=12,
            acceleration="double_trigger",
            option_pool_pct=15,
            exclusivity_days=30,
            drag_threshold_pct=67,
            redemption_years=5,
            dividends_cumulative=False,
        )
        flags = _compute_expected_flags(label)
        assert "R-CT-02" in flags
        assert "R-CT-03" in flags

    def test_generated_sheets_have_flags(self, small_batch: list[SyntheticSheet]) -> None:
        """Every generated sheet should have at least one expected flag."""
        for sheet in small_batch:
            assert len(sheet.gold_label.expected_flags) >= 1


# ---------------------------------------------------------------------------
# Disk I/O tests
# ---------------------------------------------------------------------------


class TestDiskIO:
    """Test writing generated sheets to disk."""

    def test_write_creates_files(
        self,
        generator: SyntheticGenerator,
        tmp_path: Path,
    ) -> None:
        sheets = generator.generate_batch(count=4, adversarial_ratio=0.5)
        out_dir = tmp_path / "synthetic"
        generator.write_to_disk(sheets, output_dir=out_dir)

        assert out_dir.exists()

        # Check that text files and gold JSON sidecars were created
        txt_files = list(out_dir.glob("ts_synth_*.txt"))
        gold_files = list(out_dir.glob("ts_synth_*.gold.json"))
        assert len(txt_files) == 4
        assert len(gold_files) == 4

        # Check batch manifest
        manifest_path = out_dir / "_batch_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["_meta"]["total_count"] == 4
        assert len(manifest["documents"]) == 4

    def test_gold_json_is_valid(
        self,
        generator: SyntheticGenerator,
        tmp_path: Path,
    ) -> None:
        sheets = generator.generate_batch(count=2, adversarial_ratio=0.5)
        out_dir = tmp_path / "synthetic"
        generator.write_to_disk(sheets, output_dir=out_dir)

        for gold_file in out_dir.glob("*.gold.json"):
            data = json.loads(gold_file.read_text())
            # Should have all key fields
            assert "doc_id" in data
            assert "liq_multiple" in data
            assert "anti_dilution_type" in data
            assert "board_investor_seats" in data
            assert "expected_flags" in data
            assert isinstance(data["expected_flags"], list)

    def test_text_file_is_readable(
        self,
        generator: SyntheticGenerator,
        tmp_path: Path,
    ) -> None:
        sheets = generator.generate_batch(count=2, adversarial_ratio=0.5)
        out_dir = tmp_path / "synthetic"
        generator.write_to_disk(sheets, output_dir=out_dir)

        for txt_file in out_dir.glob("ts_synth_*.txt"):
            content = txt_file.read_text(encoding="utf-8")
            assert len(content) > 200
            assert "TERM SHEET" in content


# ---------------------------------------------------------------------------
# Integration: round-trip through the ingestor
# ---------------------------------------------------------------------------


class TestIngestorIntegration:
    """Verify that synthetic sheets can be ingested by the S1 pipeline."""

    def test_synthetic_sheet_ingests_successfully(
        self,
        generator: SyntheticGenerator,
        tmp_path: Path,
    ) -> None:
        """Generate a sheet, write it, and ingest it through the real S1 Ingestor."""
        from termsheet_ai.ingest.ingestor import Ingestor

        sheet = generator.generate_one(index=1, adversarial=False)
        txt_path = tmp_path / sheet.filename
        txt_path.write_text(sheet.text, encoding="utf-8")

        ingestor = Ingestor()
        metadata, clauses = ingestor.ingest(txt_path)

        assert metadata.doc_id  # non-empty hash
        assert metadata.total_clauses > 0
        assert len(clauses) > 0
        # The ingestor should find multiple clauses in a full term sheet
        assert len(clauses) >= 5
