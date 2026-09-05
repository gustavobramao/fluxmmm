# Frozen validation protocol

Status: **predeclared; validation outcomes unopened**

Protocol version: `flux-svi-score-v3-validation-v2-powered-screen`

This protocol evaluates whether the frozen SVI-trained score ranks candidates
better than the currently active V6 diagnostic ranker. It does not refit a
weight, change a gate, select a threshold, or inspect the 80-business sealed
audit.

## Cohort and estimand

The validation cohort contains 120 businesses: 60 delayed-TV worlds and 60
correlated-planning worlds. Each business exposes the same 48 candidate MMMs.
All candidates must have a finite frozen SVI label and newly computed
truth-blind diagnostics before evaluation can begin.

Within the highest available immutable eligibility tier for business \(b\), let
\(c_b^*\) be the candidate with the lowest SVI posterior-expected decision loss.
For method \(m\), define uncapped excess loss

\[
R_{m,b}=L_{m,b}-L_{c_b^*,b}.
\]

The economic label averages the four predeclared decisions: spend reduction,
fixed-budget reallocation, spend increase, and the supported economic ceiling.

## Primary comparison

The paired business-level effect is

\[
D_b=R_{new,b}-R_{V6,b}.
\]

Negative values favor the new score. A deterministic 20,000-replicate paired
bootstrap resamples businesses within generator family and calculates the
one-sided 95% upper confidence bound for \(E[D_b]\).

The primary progression screen requires:

1. the observed mean difference is below zero; and
2. the mean regret-weighted oracle-misranking bound from the theorem is below
   active V6.

The bootstrap interval is still reported. An upper bound no greater than 2% of
the comparator mean supports statistical non-inferiority, and an upper bound
below zero supports superiority. Neither confidence classification is used as
an automatic progression gate because the predeclared validation cohort is not
large enough to resolve the expected small effect reliably.

## Prospective power check

Before opening validation, nuisance quantities were estimated using only the
300 training businesses. The paired standard deviation of uncapped excess-loss
differences was 3.144 economic-loss units and active V6 mean excess loss was
2.548. With 120 validation businesses, a one-sided 5% test has approximately
80% power only for an absolute effect of 0.714, or roughly 28% of the comparator
mean. The observed training difference was much smaller. Treating a 2%
confidence margin as a mandatory gate would therefore turn inadequate sample
size into an almost certain rejection rather than a scientifically useful
validation.

This power result was calculated before any validation outcome was evaluated.
It changes the protocol from a confirmatory test into a transparent progression
screen. A future superiority claim requires a larger independent cohort.

## Predeclared safety gates

All gates must pass:

- P90 excess loss no more than 2% above active V6;
- P95 excess loss no more than 2% above active V6;
- mean excess loss in each of the two validation families no more than 2%
  above active V6;
- lowest-loss selection rate no more than two percentage points below active
  V6; and
- regret-weighted oracle-misranking bound lower than active V6; and
- exactly the same decision-grade business availability under both rankers.

Decision-grade availability is an integrity check, not evidence that the score
is better: rank weights are not allowed to change eligibility.

## Secondary reporting

The report will disclose mean, P90 and P95 uncapped excess loss, lowest-loss
selection rate, family-specific effects, capped normalized regret, ROI error,
contribution error, and the theorem-bound receipt. Secondary endpoints cannot
rescue a failed primary or safety gate. Only the paired mean comparison carries
an inferential claim; the safety endpoints are predeclared operational gates.

## Governance

- If every validation gate passes, the method may proceed to the sealed
  80-business adversarial audit. Passing means “safe to challenge,” not
  “statistically proven superior.”
- If a gate fails, the audit remains sealed and the frozen score is not
  activated.
- Any methodological change after viewing validation results creates a new
  research version. The same 120 businesses cannot be relabelled as untouched
  validation for that revised method.
- Audit results are never used for corrective action. They are reported as the
  final adversarial assessment of this frozen method.
