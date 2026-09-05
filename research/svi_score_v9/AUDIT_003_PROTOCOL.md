# V9 sealed audit 003 execution protocol

## Governance

Two earlier cohorts were abandoned without opening or scoring their hidden
economic outcomes. Audit 001 produced 171 posterior envelopes before its
quota-constrained orchestration was stopped. Audit 002 produced no posterior
envelopes: its local preparation was stopped after a zero-based shard-index
defect duplicated shard 1 and omitted shard 0. Both local cohorts are retained
under `.flux-artifacts/svi-score-v9/abandoned-audits/`, and audit 001's cloud
objects remain under `v9-sealed-audit-001/` for traceability. Neither cohort is
an input to audit 003.

Audit 003 uses a new, disjoint seed-index window (90,000--90,019), a new remote
prefix (`v9-sealed-audit-003/`), and a new one-time protocol freeze. The shard
parser now has an isolated regression test proving that explicit shard index
zero remains zero. No hidden truth from audits 001, 002, or 003 was
dereferenced while repairing or benchmarking the orchestration.

## Root-cause findings

The abandoned cloud design combined four infrastructure problems:

1. It requested more Compute Engine CPU than the project's 32-vCPU global
   Batch quota could supply.
2. It allocated oversized VMs to mostly single-threaded candidate fits.
3. Candidate processes repeatedly compiled PyTensor graphs. Sharing one
   compiler directory across concurrent processes introduced a compiler lock,
   so a larger VM serialized important portions of the work instead of scaling
   them.
4. The container lacked a system BLAS link.

The replacement image links OpenBLAS and holds it to one thread per candidate,
avoiding nested oversubscription. Each candidate now receives an independent
small task, eliminating cross-candidate compiler-lock contention.

## Frozen execution shape

- Platform: Google Cloud Run Jobs, `europe-west1`.
- Work: 6,720 SVI candidate fits, one immutable fingerprint per task.
- Parallelism: 180 one-vCPU tasks against a verified regional allowance of 200
  vCPU.
- Memory: 2 GiB per task.
- Scientific runtime: unchanged `scripts/svi_batch.py` C/VM backend, with the
  original FullRankADVI contract and random seeds.
- Image: `svi-cvm-one-candidate` at digest
  `sha256:3276faca285998efdf40db81ec8b881d4102ab5b3eb528e4dba529e53c9e893e`.
- Results: content-addressed GCS objects created with generation-match zero;
  retries cannot overwrite a retained result.
- Retry policy: at most one infrastructure retry under the identical payload,
  image, inference contract, and seeds.
- Per-task timeout: 30 minutes. Operational wall-clock stop: four hours.

## Outcome-blind benchmark

Before freezing audit 003, 20 fingerprint-selected audit-001 payloads were run
as 20 independent one-vCPU Cloud Run tasks. This used posterior payloads only;
no synthetic economic truth or selector outcome was read.

- Successful fits: 20/20; errors or reviews: 0.
- Scientific runtime: mean 217.0 s, median 192.0 s, P90 288.8 s, P95 297.7 s,
  maximum 502.7 s.
- End-to-end 20-task execution: 9 minutes 24 seconds, including initial image
  import and the single slow-tail fit.
- Numerical equivalence: for a matching historical fingerprint, every
  non-runtime field had the same structure and categorical value; the maximum
  floating-point difference was `5.59e-9`, attributable to the linked BLAS
  implementation.

At 180-way dynamic parallelism, the benchmark implies about 2.25 hours of pure
SVI work at the observed mean. Allowing for container startup, scheduling, and
the observed tail gives a predeclared forecast of 2.5--3.7 hours.

## Cost forecast

Cloud Run's published Tier-1 job rates are $0.000018 per vCPU-second and
$0.000002 per GiB-second. At one vCPU and 2 GiB per task, the observed mean SVI
runtime implies about $32.1 of gross scientific compute. Container startup and
scheduling yield a conservative gross forecast of $32--$42. The monthly free
tier, if still available on the billing account, can reduce this by about
$5.22. The operational cost stop is $45; the job is not allowed to continue
beyond four hours without a new explicit decision.

These forecasts are operational declarations, not research endpoints, and may
not be changed after the audit freeze in response to audit outcomes.
