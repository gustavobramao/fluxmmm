# Published V11 AMSS audit artifacts

These compact artifacts support the external synthetic audit claims in the
paper and product without shipping multi-gigabyte posterior checkpoints.

- `result.json` — accepted confirmatory endpoints, paired comparisons,
  evidence-group results, decision rule, and provenance hashes.
- `candidate-losses.csv` — frozen candidate-level economic losses for all 100
  businesses and 4,800 SVI candidates.
- `frozen-selections.json` — V11 and prediction-only selections committed
  before truth was opened.
- `secondary-baselines.json` — post-truth, no-refit comparisons with
  prediction-only and the conventional Pareto selector.
- `cloud-truth-retrieval-receipt.json` — immutable cloud execution and object
  retrieval receipt.

The audit used 4,800 FullRankADVI fits and 19,200 frozen decision actions. Raw
posterior checkpoints and generated business panels are intentionally omitted
from Git because of their size; the protocol, candidate specifications,
selector, source code, manifests, hashes, and compact outcomes are versioned.

The conventional Pareto comparison is a clearly labelled secondary analysis.
It changed no candidate posterior, V11 selection, or confirmatory decision.
