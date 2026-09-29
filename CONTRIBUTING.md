# Contributing

Thanks for considering a contribution to flux-os.

## Getting set up

```bash
git clone https://github.com/vbc1406/flux-os && cd flux-os
pip install -e ".[dev,frameworks]"
pytest -q
ruff check . && ruff format --check .
```

Please keep `pytest` and `ruff` green before opening a pull request.

## Developer Certificate of Origin (DCO)

All contributions must be signed off under the [Developer Certificate of Origin](https://developercertificate.org/).
By signing off, you certify that you wrote the contribution yourself, or otherwise have the right
to submit it under the project's license (MIT).

Sign off every commit with `git commit -s`, which appends a line like:

```
Signed-off-by: Your Name <you@example.com>
```

to the commit message, using your real name and a working email address. Pull requests with
unsigned commits will be asked to amend them before merge (`git commit --amend -s`, or
`git rebase --signoff <base>` for multiple commits).

## Pull requests

- Keep changes focused; unrelated cleanup makes review harder.
- Add or update tests for behavior changes.
- Describe what changed and why in the PR description.

## Reporting bugs and compatibility issues

Open a GitHub issue. Compatibility reports for clients and frameworks not yet listed in the
README's "Works with" table are especially welcome, as are quality ratings backed by data.

## Reporting security issues

Do not open a public issue for security vulnerabilities — see [SECURITY.md](SECURITY.md).
