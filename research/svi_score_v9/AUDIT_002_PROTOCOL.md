# V9 sealed audit 002 — outcome-blind abandonment receipt

## Governance

Audit 002 was frozen and opened for truth-blind payload preparation on
2026-09-04. No SVI posterior fit was launched, no hidden economic truth was
opened, and no selector or endpoint was evaluated.

During preparation, duplicate progress for worker shard 1 exposed a parser
defect: explicit `--shard=0` was coerced to 1 by a generic positive-integer
helper. Preparation was interrupted immediately. The partial frozen cohort was
moved intact to
`.flux-artifacts/svi-score-v9/abandoned-audits/audit-002-20260904/`.

Because repairing a frozen source in place would invalidate the protocol hash,
audit 002 is permanently retired. Audit 003 uses a new seed window and includes
a regression-tested zero-based parser. Audit 002 is not an additional chance
to inspect or choose a favorable statistical result.
