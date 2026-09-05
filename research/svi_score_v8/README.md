# FluxMMM V8 selection-aware SVI research

V8 is the internal development name for the next FluxMMM selector. It does
not designate the eventual paper title. The research pipeline is SVI-only:
all candidate posteriors, observable posterior tokens, scenario actions, and
economic labels use the frozen FullRankADVI contract. NUTS remains an optional
open-source product capability and is outside this study.

## What changes from V5D-SVI

V5D-SVI predicts candidate-level economic loss well enough to improve average
pair ordering, but its top-ranked candidate can still be a dangerous false
champion. V8 trains for the actual action: selecting one candidate from the 48
available to an unseen advertiser.

V8 adds:

1. Four separately retained economic decision-scenario losses.
2. Mean, P90, per-scenario, and dangerous-champion learning heads.
3. Every valid oracle-versus-candidate comparison, weighted by the exact
   predeclared capped normalized economic gap.
4. Nested business-grouped model and policy selection.
5. Ensemble disagreement, a dangerous-champion penalty, and explicit
   abstention with both cost and coverage reported.
6. Hard gates only for invalid artifacts. Identification and evidence concerns
   remain observable continuous risk tokens.

## Development sequence

```bash
npm run research:svi-score-v8:scenario-targets
npm run research:svi-score-v8:prepare
npm run research:svi-score-v8:baseline
npm run research:svi-score-v8:train
npm run research:svi-score-v8:verify-freeze
npm run test:svi-score-v8
```

The first command reconstructs the four scenario losses from existing aligned
SVI decision draws. It does not refit any MMM and dereferences zero audit
posterior files. The 420 development businesses produce 20,160 rows. The 80
audit businesses remain sealed.

After the development checks pass, the one-time audit sequence is:

```bash
npm run research:svi-score-v8:freeze-comparator
npm run research:svi-score-v8:audit-freeze
npm run research:svi-score-v8:audit-build
npm run research:svi-score-v8:audit-score
```

The protocol freeze hashes the development artifact, both frozen selectors,
the endpoint contract, and the exact audit builder/scorer before an audit row
is selected. The opening receipt is exclusive. Once it exists, any changed
method requires a new sealed cohort rather than a repair on these businesses.

## Interpretation boundary

Cross-fitting estimates performance on unseen development advertisers while
all tuning remains inside development. It does not establish external
generalization. The sealed audit may be opened exactly once only after the
complete simulator, token, learner, policy, abstention, and endpoint contract
has been frozen and the development checks pass.

The paper-safe inference statement is:

> V8 uses scalable variational posterior inference during research and
> candidate evaluation. Its posterior-token and decision-risk framework is
> inference-engine agnostic in principle and can be applied to NUTS posterior
> draws.

This is an architectural statement, not a claim that the learned SVI selector
has already been validated under NUTS.

## Frozen V8 result

V8 passed every predeclared development check and the once-opened 80-business
adversarial audit. Relative to the frozen V5D-SVI comparator on the audit, the
primary policy objective decreased from 2.8605 to 1.9631, mean policy loss
from 1.9965 to 1.2353, and P90 policy loss from 4.4651 to 3.3148. Promotion
coverage was 65%. The dangerous false-champion share among promoted models
decreased from 50.0% to 40.4%, with zero theorem-bound violations.

These results establish simulator-held-out evidence under the declared
wrong-evidence and swapped-channel-mechanics adversarial families. They do not
establish external real-world generalization, production validity, or a SOTA
claim. Further selector tuning requires a newly sealed cohort. The complete
receipt is stored in
`artifacts/svi-score-v8-sealed-audit.json`.
