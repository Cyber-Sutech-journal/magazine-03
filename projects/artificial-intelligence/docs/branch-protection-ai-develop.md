# Branch protection for `ai-develop` (T21 / §15.3)

`gh` is not required in the project tree. A repo admin should require these
**GitHub Actions check names** (the job `name:` fields) before merge:

1. `Lint and format (Ruff)`
2. `Type check (mypy)`
3. `Unit tests (pytest)`
4. `Docker CPU integration test`

Settings path: GitHub → repository → Settings → Branches → `ai-develop`
→ Require status checks to pass before merging → add the four names above
after they have appeared at least once on a PR.

Do **not** allow bypassing for administrators if you want the same rule for
everyone. The workflow file is `.github/workflows/ai-ci.yml` at the
magazine-03 root.
