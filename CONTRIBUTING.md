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

Do not merge directly to `master`. Always open a pull request so CI runs first.

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
