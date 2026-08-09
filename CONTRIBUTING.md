# Contributing to AskRex Assistant

Thanks for your interest in contributing to AskRex Assistant.

AskRex is an open-source, local-first AI assistant focused on voice control, Home Assistant integration, smart home automation, local/cloud LLM support, and privacy-friendly personal assistant features.

The project is still early, so documentation, testing, cleanup, UX improvements, bug reports, and small fixes are all useful.

## Good places to start

If you are new to the project, start with issues labeled:

- `good first issue`
- `help wanted`
- `documentation`

You can also open a GitHub Discussion if you are not sure where to begin.

## First-time contributor checklist

New to the project? Follow these steps from fork to pull request. Each step links to the section below with more detail.

1. **Fork the repository.** On GitHub, click **Fork** at the top right of the [AskRex Assistant repo](https://github.com/Blueibear/AskRex-Assistant) to create a copy under your account, then clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/AskRex-Assistant.git
   cd AskRex-Assistant
   ```

2. **Create a branch from `master`.** Use a short, descriptive name. See [Branch strategy](#branch-strategy) and [Commit message format](#commit-message-format) for naming and commit conventions.

   ```bash
   git checkout master
   git pull
   git checkout -b docs/short-description
   ```

3. **Install dependencies.** AskRex targets Python 3.11. Create a virtual environment and install the project, plus the dev tools used by tests and linters. See the [Quick Start in the README](README.md#quick-start) for full setup options (CPU/GPU stacks, GUI, voice).

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip setuptools wheel
   pip install .
   pip install -r requirements-dev.txt
   ```

4. **Run basic checks.** Make sure the test suite, linters, and the built-in health check pass before opening a PR:

   ```bash
   pytest -q
   ruff check .
   black --check .
   python -m rex doctor
   ```

5. **Commit your changes.** Use [Conventional Commits](#commit-message-format) (for example `docs: clarify install steps`). Push the branch to your fork:

   ```bash
   git push -u origin docs/short-description
   ```

6. **Open a pull request against `master`.** From your fork on GitHub, click **Compare & pull request**. Target the `master` branch of `Blueibear/AskRex-Assistant`. The PR template will guide you; see also the [Pull request checklist](#pull-request-checklist) below.

### What to include in the PR description

The PR template covers most of this, but at a minimum your description should have:

- **Summary** — one or two sentences explaining what the change does.
- **Type** — feat, fix, docs, refactor, perf, or chore.
- **Verification** — which checks you ran (for example `pytest -q`, `ruff check`, `black --check`).
- **How you tested it** — manual steps, sample commands, or screenshots if the change is user-facing.
- **Related issue** — link to the issue this PR closes, if any (for example `Closes #123`).

If you get stuck on any step, open a GitHub Discussion — questions are welcome.

## Ways to contribute

Useful contributions include:

- Improving setup documentation
- Testing the install process on Windows, macOS, or Linux
- Improving Home Assistant setup instructions
- Improving the desktop GUI
- Fixing bugs
- Writing tests
- Improving error messages
- Suggesting better project structure
- Reviewing pull requests
- Improving accessibility and beginner usability

## Before opening a pull request

Please try to:

1. Check whether an issue already exists for the change.
2. Keep the pull request focused on one improvement.
3. Explain what changed and why.
4. Mention how you tested it.
5. Avoid committing secrets, tokens, passwords, private URLs, or local config files.

## Branch strategy

The canonical primary branch is `master`.

All feature work should be done on short-lived feature branches cut from `master`.

Pull requests must target `master`.

`claude/**` branches are AI-generated and follow the same PR process as human branches.

Do not merge directly to `master`. Always open a pull request so CI runs first. Do not merge a PR until every required GitHub check is green on the exact head being merged.

## Commit message format

All commits must follow the Conventional Commits specification.

Format:

```text
<type>[(<scope>)]: <short description>
```

Allowed types:

```text
feat | fix | test | docs | refactor | chore | perf | ci
```

Examples:

```text
feat: add wake-word sensitivity setting
fix(tts): handle empty transcript gracefully
chore(deps): bump openai to 1.30.0
docs: update installation instructions
```

## Installing Git hooks

A `commit-msg` hook is included in `.githooks/` to enforce the commit format locally before commits are recorded.

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

## Pull request checklist

Before submitting a pull request, please confirm:

- The change is focused and easy to review.
- Documentation was updated if needed.
- Existing behavior was not intentionally broken.
- Any new user-facing text is clear and beginner-friendly.
- You described how the change was tested.

## Code style

AskRex is primarily a Python project with a desktop GUI.

Please keep changes readable, practical, and well-scoped. Avoid large rewrites unless they are discussed first.

## Security

Do not commit secrets, API keys, tokens, passwords, local configuration files, or private URLs.

If you find a security issue, please do not open a public issue with exploit details. Start a private conversation with the maintainer instead.

## Questions

If you are interested in helping but are not sure where to start, open a GitHub Discussion and ask.

Maintainer: @Blueibear
