# Contributing to RAG Zero

Thanks for your interest! Issues and pull requests are welcome in **English or Portuguese**.

## Setup

```bash
git clone https://github.com/BlackYuriJDU/rag-zero && cd rag-zero
uv sync --dev                 # lightweight install (BM25 only)
uv sync --dev --extra dense   # also installs fastembed (dense / hybrid retrieval)
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

## Before you open a PR

Run the same checks as CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest        # no network and no model download needed
```

## What makes a good contribution

- **Small and focused.** One problem per PR, with a description of what changed and why.
- **Tested.** New behaviour comes with a test; bug fixes come with a test that fails
  without the fix. The default test suite must keep running offline and without the model.
- **Measured.** This project decides by numbers. If a change affects retrieval or
  answers, run `rag-zero evals run` and include the before/after table in the PR.
- **Honest.** If something is a limitation, say so in the README instead of hiding it.

Good places to start:

- Add questions to the golden set — especially paraphrases **without proper nouns** (see
  the "Honesty about the golden set" note in the README).
- Tests for the PNCP client and edge cases of the notice parser.
- Items from the roadmap and the Limitations section of the README.
- Documentation and translations.

## Using AI assistants

AI-assisted contributions are fine, but you are responsible for them: say so in the PR
description, make sure you understand every line, and run the checks above yourself.

## Pull request flow

1. Fork, then create a branch from `main` (`fix/…`, `feat/…`, `docs/…`).
2. Commit with a clear message (Conventional Commits style is welcome but not required).
3. Open the PR and fill in the template. CI must be green.
4. Maintainers aim to respond within a few days; this is a small project, so please be patient.

## Reporting bugs and security issues

Use the issue templates for bugs and feature requests. For security problems, **do not**
open a public issue — see [SECURITY.md](SECURITY.md).
