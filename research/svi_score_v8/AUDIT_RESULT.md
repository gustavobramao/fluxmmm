# V8 one-time sealed audit receipt

The audit was opened only after the development selector, V5D-SVI comparator,
136-token registry, four economic scenarios, abstention policy, endpoints,
and scorer code had been hashed into the protocol freeze. The frozen hashes
still match after scoring. No NUTS draw was used.

| Endpoint | Frozen V5D-SVI | V8 | Result |
| --- | ---: | ---: | --- |
| Primary policy objective | 2.8605 | 1.9631 | Pass |
| Mean policy loss | 1.9965 | 1.2353 | Pass |
| P90 policy loss | 4.4651 | 3.3148 | Pass |
| P95 policy loss | 6.3945 | 5.3634 | Pass |
| Dangerous false champions | 50.0% | 40.4% | Pass |
| Promotion coverage | 100% | 65% | Pass (floor: 60%) |
| Selected-model bound violations | Not optimized | 0 | Pass |

The V8 selector abstained for 35% of audit businesses and charged the
predeclared abstention cost in policy loss. Exact economic-oracle recovery was
19.2% among promoted V8 decisions versus 1.25% for V5D-SVI, but exact oracle
recovery is secondary: multiple candidate MMMs can induce nearly equivalent
budget actions. The primary target is avoided economic loss.

The audit contains 80 untouched adversarial simulator businesses and 3,840
SVI candidate fits. It tests two deliberately difficult families:
wrong external evidence and swapped channel mechanics. This is independent
simulator evidence, not validation on historical advertiser experiments.

Method changes made after this result may not be evaluated against the same
audit as confirmatory evidence. A new method requires a newly generated and
sealed cohort. The machine-readable source of truth is
`artifacts/svi-score-v8-sealed-audit.json`.

Execution note: one parallel worker stopped once on the frozen-label equality
guard. A read-only replay of the implicated record showed exact agreement
among the stored label, historical evaluator, accelerated evaluator, and exact
fallback. The identical hashed worker was resumed successfully without any
code, model, endpoint, or threshold change. No audit performance result was
available before the resume.
