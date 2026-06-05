# CLAIM_LEDGER.md — claim ledger (Stage 2 output)

Single source of truth mapping each experimental claim to its evidence and verdict. Stage 3 (`paper-claim-audit`) checks the manuscript against this file.

| id | claim | verdict | integrity_status | evidence (file paths) | note |
|----|-------|---------|------------------|------------------------|------|
| claim-001 | Method X improves accuracy by 3.2pts over baseline on CIFAR-100 | supported | pass | results/cifar100/seed{0,1,2}.json | mean 3.2 ± 0.4 over 3 seeds |
| claim-002 | X is faster at inference | partially_supported | warn | results/timing.json | holds at batch=1 only; not tested at batch>1 (scope) |
| claim-003 | X generalizes to all vision tasks | invalidated | fail | — | only CIFAR tested; scope inflation, no evidence |

**Verdicts:** `supported` | `partially_supported` | `invalidated`.
**integrity_status** (propagated from Stage 1 `EXPERIMENT_AUDIT.json`): `pass` | `warn` | `fail`.
**Rule:** a claim with `integrity_status: fail` may not be `supported` until the integrity issue is resolved.

Per-claim detail (optional, one block each):

## claim-001
- **Statement:** ...
- **Evidence for:** results/.../seedN.json (metric=accuracy)
- **Evidence against / qualifying:** none
- **Verdict:** supported
- **Wiki edges:** exp-012 supports claim-001
