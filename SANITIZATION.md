# Public export and reproducibility boundaries

This repository was exported from an allowlist into a new directory with no original Git history. The original working repository was not given a remote and was not pushed.

Included: experiment implementations, scientific plans and amendments, architecture/results documents, numerical analysis outputs and per-question final-answer/grade records. Derived per-question grading files, when supplied, were produced offline with the original scorer. No new inference was run for this export.

Excluded: unrelated private integration projects, personal note/application data, original credential integration, credentials and configuration files, local storage and spend journals, agent/conversation transcripts, full call/message streams, scratch/build logs, large model checkpoints and activation caches. Raw generated-text dumps are also omitted from the small-model study; archived per-question scores and fresh-test repetition flags remain.

Copies of plans/results may have local paths and unrelated contact addresses redacted. Such copies are identified as sanitized, not presented as byte-identical private originals. No reported accuracy, budget, question count or statistical result was changed by privacy sanitization.

## Export-only code changes

The private platform credential provider is replaced by an explicitly configured environment-only provider. Profile selection and budget guards remain; profiles do not fall back to each other's values. Four offline safety tests cover missing credentials, secret wrapping, profile isolation and invalid profiles. This adapter change is not a new experiment or a change to the model prompts, grader, gate/index logic or analysis equations.

The optional personal-note application package and its console entry point are not part of the public research export. Its integration tests and the original platform credential tests are omitted. Some archived documentation describes the original implementation; its credential-specific wording is redacted. Local directory defaults in the retained code are portability-sanitized. The complete original integration test suite is not claimed to be reproducible from this minimal export.

`runs/v4/per-question/` contains the scorer's final per-item grading/analysis rows, split by model. Use `tools/reproduce_v4.py` to recompute aggregate statistics without the source corpus or paid model calls. Raw final-answer records, question records and corpus manifests are supplied for auditing; full source-corpus blobs and call transcripts are deliberately not distributed.

## Release checks

Before the initial public commit, all included text/JSON records were checked for private usernames, absolute home-directory paths, unrelated integration references, unexpected email addresses and structured credential values. Gitleaks 8.30.1 was run with full redaction; no secret findings remained. Matches for `token` in the numerical records are token counts/budgets, not authentication values. Credential-variable names and literal redaction/detection prefixes in safety code are configuration or protection logic, not credentials.

The fresh release commit identifies the distributed files; the maintainer keeps per-file SHA-256 checksums outside the public tree. Third-party notices and license text are preserved. Validation/scan reports with local execution paths are kept outside this public tree.
