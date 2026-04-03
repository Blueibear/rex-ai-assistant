# Contributing to AskRex Assistant

## Branch Strategy

The canonical primary branch is **`master`**.

- All feature work is done on short-lived feature branches cut from `master`.
- Pull requests must target `master`.
- `claude/**` branches are AI-generated and follow the same PR process as human branches.
- Do not merge directly to `master` — always open a PR so CI runs first.

## Commit Message Format

All commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

**Format:** `<type>[(<scope>)]: <short description>`

**Allowed types:** `feat` | `fix` | `test` | `docs` | `refactor` | `chore` | `perf` | `ci`

**Examples:**
```
feat: add wake-word sensitivity setting
fix(tts): handle empty transcript gracefully
chore(deps): bump openai to 1.30.0
docs: update installation instructions
```

## Installing Git Hooks

A `commit-msg` hook is included in `.githooks/` to enforce the above format locally before commits are recorded.

For formatting and linting hooks, install `pre-commit` and register the default Git hooks:

```bash
pip install pre-commit
pre-commit install
```

Install the Conventional Commits `commit-msg` hook separately with:

```bash
cp .githooks/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

After installation, any commit message that does not match the Conventional Commits pattern will be rejected with a clear error.
