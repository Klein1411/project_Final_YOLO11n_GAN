# Repository Guidelines

## First Rule
Read `task.md` first. Do not touch code, docs, or cleanup decisions before that step.

`task.md` is the source of truth for project state, folder meaning, environment details, current progress, and known issues. `AGENTS.md` only defines behavior rules.

## Project Structure & Workflow
Follow the notebook-first workflow. Use `notebooks/` for experiment logic, training flow, and evaluation review.

Treat `data/`, `models/`, `runs/`, `docs/`, and `archive/` exactly as documented in `task.md`.
Do not modify GAN-related assets unless the current task explicitly includes them.
Keep helper scripts minimal. Do not add new utility files without a clear reason.

## Build, Check, and Development Commands
Use lightweight inspection commands first:

```powershell
git status --short
rg --files
python -m py_compile path\to\file.py
```

Use syntax checks and reference scans to verify logic flow.
Do not auto-run notebooks, long preprocessing, training, or heavy dataset jobs.
The user runs notebooks manually in VSCode to control outputs, cache, and progress.

## Coding Style & Naming Conventions
Match existing repository style.
Use 4-space indentation in Python and `snake_case` for files and functions.
Keep names short and descriptive.
Do not refactor unrelated code.
Keep changes small, traceable, and scoped to the request.
Prefer editing existing files over adding abstractions.

## Testing Guidelines
There is no guaranteed formal test suite.
Validate changes with static review, path consistency checks, config cross-checks, and lightweight syntax checks where safe.
If a notebook or training flow is required to fully verify behavior, stop at logic validation and state the limitation explicitly.

## Commit & Pull Request Guidelines
Use short, concrete commit messages.
Prefer prefixes such as `feat:` and `docs:` when they add clarity.
In PRs, state scope, affected paths, config or dataset assumptions, and whether notebook execution was intentionally skipped.

## Environment & Safety Notes
Assume Windows + VSCode as the primary runtime.
Respect resource limits: the working setup uses a laptop GPU.
Avoid unnecessary duplicated datasets, cached outputs, or heavy reruns unless explicitly requested.
