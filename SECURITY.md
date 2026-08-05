# Security and data privacy

## Supported version

Security fixes are currently applied to the latest `main` branch and v1.x
releases.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting or a private Security Advisory for
this repository. Please do not open a public issue containing exploit details,
credentials, advertiser data, or sensitive logs.

Include the affected version, reproduction steps, impact, and the smallest safe
test case. Replace any proprietary dataset with a synthetic fixture.

## Local-data boundary

The open-source product is local-first. Uploaded datasets, local databases, and
model artifacts are stored under ignored project directories. They must never be
committed to a fork or attached to an issue. Review `git status` before every
push and keep `.env*`, `.wrangler`, `.flux-artifacts`, and `.venv` ignored.

FluxMMM does not require an OpenAI key, hosted login, or third-party model API.
