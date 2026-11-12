# TermSheet AI: Annotation Guide v0

This guide defines the labels used by Annotators A and B for the evaluation
corpus. Annotators must work independently and must not infer terms from
market practice when the document is silent.

## General annotation rules

### Clause spans

Create one annotation for each identifiable clause. `clause_id` should match
the ingestor's clause ID (for example, `c001`). Assign exactly one
`clause_family`; use `other` for text that does not fit a family.

### Literal source spans

`source_span` must be an exact, contiguous substring of the clause text,
including the number or term that supports the extracted attributes. Preserve
the document's spelling, punctuation, and units. Every numeric attribute must
literally occur in its source span; use `null` when the document does not
state a value.

### Attributes and flags

Record only attributes that are explicitly extractable from the clause.
`expected_flags` contains rule IDs from `docs/rules_v1.yaml` that should fire
for the labeled attributes; it is an empty array when no rule applies.

## Clause-family instructions

### `liquidation_preference`

Identify the priority and payout available to preferred holders on a
liquidation, sale, or similar event. Include the preference multiple,
participation mechanics, cap, and seniority when stated.

Attributes: `liq_multiple`, `liq_participation`, `liq_cap`, `liq_seniority`.

### `anti_dilution`

Identify price-protection provisions that adjust conversion or ownership after
a later issuance at a lower price. Distinguish broad-based weighted average,
narrow-based weighted average, full ratchet, no protection, and unknown.

Attributes: `anti_dilution_type`, `broad_based`, `narrow_based`, `adjustment_trigger`,
`excluded_issuances`.

### `board_composition`

Identify the board size, appointment rights, and the seats allocated to
founders, investors, and independent directors. Count only seats explicitly
described in the clause.

Attributes: `total_seats`, `investor_seats`, `founder_seats`,
`independent_seats`, `appointment_rights`.

### `protective_provisions`

Identify investor consent or veto rights over specified company actions, such
as issuing senior securities, changing the charter, incurring debt, or selling
the company. Record whether consent is required from a preferred majority and
whether the provision reaches ordinary operations.

Attributes: `consent_threshold`, `requires_series_majority`,
`veto_day_to_day_operations`, `protected_actions`.

### `founder_vesting`

Identify the schedule under which founder or employee equity is earned,
including the total period and any cliff. Record acceleration only when the
clause expressly describes single-trigger, double-trigger, or no acceleration.

Attributes: `vesting_months`, `cliff_months`, `acceleration`,
`repurchase_right`, `applies_to`.

### `information_rights`

Identify rights to receive financial statements, budgets, cap tables, or
operating reports from the company. Record the delivery frequency and any
eligibility threshold when stated.

Attributes: `report_types`, `frequency`, `inspection_right`,
`eligibility_threshold`.

### `rofr_co_sale`

Identify a right of first refusal that lets a holder purchase shares before a
third-party sale, and a co-sale or tag-along right that lets the holder sell
alongside a founder. Label each mechanism separately when both appear.

Attributes: `investor_rofr`, `investor_co_sale`, `covered_holders`,
`notice_period_days`, `transfer_exceptions`.

### `drag_along`

Identify provisions that require minority holders to join an approved sale or
transfer. Record the approval threshold and whether founder or preferred
consent is required.

Attributes: `approval_threshold`, `requires_founder_consent`,
`requires_preferred_consent`, `sale_scope`.

### `option_pool`

Identify the equity reserved for future employee or advisor options and state
whether the pool is calculated before or after the financing. Capture the
explicit percentage and the parties bearing the dilution.

Attributes: `pool_percentage`, `pre_money_pool`, `post_money_pool`,
`dilution_holder`, `pool_purpose`.

### `exclusivity_no_shop`

Identify a no-shop, exclusivity, or non-solicitation commitment that restricts
fundraising or discussions with other parties. Record the duration in days
only when the document states it or provides enough explicit dates to compute
it without assumptions.

Attributes: `duration_days`, `restricted_activities`, `starts_on`,
`termination_event`, `exceptions`.

### `conversion_rights`

Identify rights or mandatory events that convert preferred shares into common
shares, including optional conversion and automatic conversion on an IPO or
approved sale. Record the conversion ratio or trigger when stated.

Attributes: `conversion_ratio`, `optional_conversion`, `automatic_conversion`,
`conversion_triggers`, `majority_threshold`.

### `redemption_rights`

Identify an investor right to require the company to repurchase preferred
shares after a specified period or event. Record the waiting period, payment
terms, and required approval.

Attributes: `redemption_years`, `redemption_price`, `installments`,
`approval_threshold`, `redemption_trigger`.

### `other`

Use `other` for boilerplate, definitions, or substantive provisions that do
not fit one of the twelve core families. Preserve useful text and attributes
without forcing it into a more specific family.

Attributes: `topic`, `parties`, `effective_date`, `amounts`, `conditions`.

## Gold-file format

Save one file per source document at
`corpus/gold/<document-filename>.gold.json`:

```json
{
  "doc_id": "eval-001",
  "clauses": [
    {
      "clause_id": "c001",
      "clause_family": "liquidation_preference",
      "source_span": "1x the Original Purchase Price",
      "attributes": {
        "liq_multiple": 1.0,
        "liq_participation": "non_participating",
        "liq_cap": null,
        "liq_seniority": "pari_passu"
      },
      "expected_flags": []
    }
  ]
}
```

If a family is absent, do not create an empty clause. If an attribute is not
explicitly stated, set it to `null`; do not fill it from a market norm.

## Conflict resolution

Annotators must label the same source independently. After both annotations
are complete, reconcile disagreements using the exact source text and report
the pre-reconciliation agreement, including Cohen's kappa where applicable.
