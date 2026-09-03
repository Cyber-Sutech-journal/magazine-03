# AI pipeline CI — how to read a red X (T21, §12.4, §15.3, §15.4)

GitHub Actions **does not run workflows from this subdirectory**. The live
workflow file is:

```
.github/workflows/ai-ci.yml
```

at the **magazine-03 repository root** (this project lives in
`projects/artificial-intelligence/`).

## Jobs (each is independent)

| Job name on GitHub | What it runs | How you fix a failure |
|---|---|---|
| Lint and format (Ruff) | `ruff check` then `ruff format --check` | From this folder: `ruff check --fix .` and `ruff format .` |
| Type check (mypy) | `mypy src/mot_counting` | Read the file:line in the log; add a type hint or fix the mismatch. Config is `[tool.mypy]` in `pyproject.toml`. |
| Unit tests (pytest) | `pytest tests/unit/` | Run the same command locally with `pip install -e ".[dev]"`. |
| Docker CPU integration test | Build CPU image, run `configs/ci.yaml` | Pipeline must finish; `outputs/annotated.mp4` and `outputs/events.csv` must exist and be non-empty. GPU is **not** in CI. |

Type checker choice: **mypy** (pinned in `pyproject.toml` optional `dev` extra), not pyright.

## Clip used in Docker CI

See [`docs/ci-sample-clip.md`](../docs/ci-sample-clip.md). Synthetic / no real people.
