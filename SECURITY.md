# Security policy

## Supported versions

Only the latest release (and `main`) receives fixes.

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Use GitHub's private vulnerability reporting: open the repository's **Security** tab and
choose **Report a vulnerability**. Include what you found, how to reproduce it and the
impact you expect. You should get an answer within a few days.

## Scope notes

RAG Zero ingests public data from the PNCP API and stores it in a local SQLite file.
Reports about prompt injection through ingested text, unsafe handling of the database
path, or dependency vulnerabilities are especially welcome.
