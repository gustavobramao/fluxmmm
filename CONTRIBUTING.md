# Contributing to FluxMMM

Thank you for helping make marketing measurement more transparent.

## Before opening a change

1. Search existing issues and discussions.
2. Describe the statistical or product claim the change affects.
3. For model changes, identify the data-generating assumptions and expected
   failure modes—not only the desired score improvement.
4. Never attach proprietary advertiser data, credentials, or unredacted logs.

## Local checks

```bash
pnpm install --frozen-lockfile
pnpm lint
pnpm exec tsc --noEmit
pnpm test
```

Sampling changes also require:

```bash
pnpm mcmc:setup
pnpm mcmc:test
```

## Pull-request expectations

- Keep evidence, parameter bounds, validation gates, and inference contracts
  explicit.
- Add regression coverage for numerical or statistical behavior.
- Preserve the distinction between predictive performance, structural
  adequacy, causal credibility, and ROI decision coherence.
- Document any new prior, benchmark, likelihood, optimization objective, or
  approximation.
- Do not claim causality, convergence, or global optimality beyond what the
  implementation and tests establish.

By contributing, you agree that your contribution is licensed under the MIT
License included in this repository.
