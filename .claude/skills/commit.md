# Skill: commit

Stage and commit changes following this project's conventions.

## Steps

1. Run in parallel:
   - `git status` — see what's changed and untracked
   - `git diff` — see unstaged changes
   - `git log --oneline -5` — check recent commit style

2. Analyse what changed and why — draft a commit message:
   - One concise subject line (≤72 chars)
   - Focus on the *why*, not the *what*
   - Use: `Add`, `Fix`, `Update`, `Remove`, `Refactor` as appropriate
   - No `Co-Authored-By:` lines — ever

3. Stage specific files (never `git add -A` or `git add .` blindly):
   ```bash
   git add <specific files>
   ```

4. Commit — the pre-commit hooks will run automatically:
   - ruff lint
   - black format check
   - isort import check
   - mypy type check
   - pytest (all 68 tests)

   If any hook fails, fix the issue and re-stage before retrying. Never use `--no-verify`.

5. If black or isort fails:
   ```bash
   make format
   git add <affected files>
   # then retry the commit
   ```

6. Report: commit hash, subject line, files changed.

## Constraints

- **Never** add `Co-Authored-By:` lines — only the repo owner is the commit author
- **Never** use `--no-verify` to skip hooks
- **Never** commit `mlflow.db`, `*.pt`, `*.pth`, `.env`, or anything in `outputs/` or `data/`
- Prefer specific `git add <file>` over `git add .`
- Do not push unless the user explicitly asks
