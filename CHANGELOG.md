# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added
- English README (`README.md`); the Portuguese one moved to `README.pt-BR.md`.
- `CONTRIBUTING.md`, `SECURITY.md`, issue and pull request templates.
- Optional `dense` extra: `pip install "rag-zero[dense]"`.
- CI: Python 3.11–3.13 matrix and a packaging job (build, `twine check`, clean-venv install).

### Changed
- `fastembed` is no longer a required dependency; dense and hybrid modes need the `dense`
  extra, and the CLI explains how to install it instead of failing with a traceback.
- Lean source distribution (no lockfile or eval artifacts).

## [0.0.1]

### Added
- PNCP client, idempotent ingestion, chunking with Portuguese stemming, FTS5/BM25 and dense
  retrieval with RRF, extractive answers with citations and refusal, golden set and metrics.
