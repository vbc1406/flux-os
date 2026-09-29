# Security Policy

## Supported versions

flux-os is pre-1.0 (`0.x`). Security fixes are made against the latest release on `main`; older
tags are not separately maintained.

## Reporting a vulnerability

Please report security vulnerabilities privately, not in a public issue or pull request.

- Preferred: use [GitHub private vulnerability reporting](https://github.com/vbc1406/flux-os/security/advisories/new)
  for this repository.
- Alternative: email vandith@flux-llm.com with a description of the issue, steps to reproduce, and
  its impact.

We aim to acknowledge new reports within 5 business days and to provide an initial assessment
(severity, whether it's accepted, and a rough timeline for a fix) within 10 business days of
acknowledgment. Coordinated disclosure timing is worked out with the reporter case by case.

## Scope

flux-os is a routing proxy with no dashboard, database, or telemetry, and no built-in rate
limiting or spend caps — see the Security and Scope sections of the [README](README.md) for what
it deliberately does not do. Reports about the absence of those features are not vulnerabilities;
reports that flux-os could be tricked into something it doesn't document (e.g. request forgery,
authentication bypass, leaking one caller's data to another) are in scope.
