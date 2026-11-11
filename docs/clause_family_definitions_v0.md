# TermSheet AI: Canonical Clause Family Definitions & Market Norms

1. **Liquidation Preference (`liquidation_preference`)**: Determines the payout priority and multiple investors receive before common stockholders upon a liquidation, dissolution, or deemed liquidation event (M&A).
   - *Market Norm*: `1.0x` non-participating preferred stock, *pari passu* across series (NVCA Model Term Sheet §3).
   - *Associated Rules*: `R-LP-01` (>1x multiple, HIGH), `R-LP-02` (participating, HIGH), `R-LP-03` (senior preference, MEDIUM), `R-LP-04` (>=3x multiple, CRITICAL), `R-LP-05` (uncapped participation, CRITICAL).

2. **Anti-Dilution (`anti_dilution`)**: Protects preferred investors from conversion-price dilution if the company issues equity in a subsequent down round below the original purchase price.
   - *Market Norm*: Broad-based weighted average anti-dilution adjustment (NVCA Model Term Sheet §4).
   - *Associated Rules*: `R-AD-01` (full ratchet, CRITICAL), `R-AD-02` (broad-based weighted average, LOW), `R-AD-03` (narrow-based weighted average, MEDIUM).

3. **Board Composition (`board_composition` / `board_of_directors`)**: Defines the size and seat allocation of the company's Board of Directors across founders, investors, and independent directors.
   - *Market Norm*: Balanced or founder-controlled board in Seed/Series A with at least one mutually agreed independent seat (NVCA Model Term Sheet §2; Kaplan & Strömberg).
   - *Associated Rules*: `R-CT-02` (investor majority -> derives `InvestorControl`, HIGH), `R-CT-03` (zero independent seats, MEDIUM), `R-CT-04` (<=1 founder seat on >=5-seat board, HIGH).

4. **Protective Provisions (`protective_provisions`)**: Grants preferred stockholders class-veto rights over enumerated corporate actions (charter amendments, new senior stock, merger/sale, board size changes).
   - *Market Norm*: Preferred majority vote required for structural corporate actions only; no veto over day-to-day operational management (NVCA Model Term Sheet §3).
   - *Associated Rules*: `R-PP-01` (operational day-to-day veto, HIGH), `R-PP-02` (standard preferred majority consent, INFO).

5. **Founder Vesting (`founder_vesting` / `vesting`)**: Dictates the schedule over which founders and key employees earn their equity and whether unvested shares accelerate upon acquisition.
   - *Market Norm*: 4-year (48-month) monthly vesting with a 1-year (12-month) cliff and double-trigger acceleration upon Change of Control + termination without Cause (YC SAFE / NVCA Model Term Sheet §5).
   - *Associated Rules*: `R-VS-01` (<48 months, MEDIUM), `R-VS-02` (>12-month cliff, HIGH), `R-VS-03` (no acceleration, MEDIUM), `R-VS-04` (double-trigger acceleration, LOW), `R-CT-05` (`InvestorControl` + `VestingReset` multi-step chain, HIGH).

6. **Information Rights (`information_rights`)**: Guarantees qualifying investors access to periodic financial statements, budgets, and inspection rights.
   - *Market Norm*: Annual audited (within 120 days) and quarterly unaudited (within 45 days) statements restricted to Major Investors (NVCA Model Term Sheet §4).
   - *Associated Rules*: `R-IR-01` (no Major Investor threshold, MEDIUM), `R-IR-02` (mandatory monthly reporting, LOW).

7. **ROFR / Co-Sale (`rofr_co_sale`)**: Gives the company a primary Right of First Refusal (ROFR) and investors a secondary ROFR and pro-rata Co-Sale (tag-along) right on founder secondary share transfers.
   - *Market Norm*: Standard company-then-investor ROFR with pro-rata co-sale rights (NVCA Model Term Sheet §4).
   - *Associated Rules*: `R-RO-01` (standard co-sale right, INFO), `R-RO-02` (outright founder transfer lockup beyond ROFR, MEDIUM).

8. **Drag-Along (`drag_along`)**: Compels minority shareholders to vote in favor of an approved sale or liquidation of the company.
   - *Market Norm*: Requires approval by preferred majority, common/founder majority, and the Board of Directors (NVCA Model Term Sheet §4).
   - *Associated Rules*: `R-DA-01` (<50% threshold, HIGH), `R-DA-02` (<60% threshold without founder consent, HIGH).

9. **Option Pool (`option_pool`)**: The percentage of fully diluted equity reserved for future employee and advisor stock option grants.
   - *Market Norm*: 10%–20% of post-money fully diluted capitalization (NVCA Enhanced Model Term Sheet).
   - *Associated Rules*: `R-OP-01` (>20% pool, MEDIUM), `R-OP-02` (>15% pre-money pool carveout, HIGH).

10. **Exclusivity / No-Shop (`exclusivity_no_shop` / `exclusivity`)**: Binding lockup prohibiting the company and founders from soliciting competing financing offers while definitive documents are negotiated.
    - *Market Norm*: 30–45 calendar days from term-sheet execution (NVCA Model Term Sheet §6).
    - *Associated Rules*: `R-EX-01` (>45 days exclusivity, MEDIUM).

11. **Conversion Rights (`conversion_rights`)**: Governs optional 1:1 conversion of preferred stock into common stock and mandatory automatic conversion upon a Qualified IPO.
    - *Market Norm*: 1:1 initial conversion rate; reasonable Qualified IPO gross proceeds threshold (NVCA Model Term Sheet §4).
    - *Associated Rules*: `R-CV-01` (Qualified IPO threshold > $100M, MEDIUM).

12. **Redemption Rights (`redemption_rights` / `redemption`)**: Gives preferred investors the right to demand that the company repurchase their shares after a specified maturity horizon.
    - *Market Norm*: No mandatory redemption, or exercisable only after >= 5 years from closing (NVCA Model Term Sheet §4).
    - *Associated Rules*: `R-RD-01` (<5 years redemption horizon, HIGH), `R-RD-02` (>10% punitive redemption interest rate, HIGH).

13. **Other & Dividends (`other` / `dividends`)**: Dividend provisions and general boilerplate (governing law, legal expenses, confidentiality, conditions precedent).
    - *Market Norm*: Non-cumulative dividends payable only when and if declared by the Board (NVCA Model Term Sheet §3).
    - *Associated Rules*: `R-DV-01` (cumulative dividends, HIGH).
