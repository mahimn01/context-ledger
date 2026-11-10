# Initial release checks

- Gitleaks 8.30.1: no detected secrets in the release tree; full redaction enabled.
- Personal home paths, private usernames/integration references, unrelated contact addresses and nonempty structured authentication fields: none retained in release files.
- Large pretrained checkpoints, personal application storage and conversation/call transcripts: not exported. Test/runtime caches are ignored and excluded from the release index.
- Configuration variable names, numerical token counts and secret-detection prefixes are not credentials; retained safety-code matches were reviewed.
- Original local repository history was not copied. This release starts with one fresh sanitized commit.
- Original source files selected for the export retain their private-original hashes in the maintainer's audit; sanitized-file hashes are kept in the separate private release audit.
