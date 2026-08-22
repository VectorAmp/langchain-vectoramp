# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.

## [Unreleased]

### Changed

- **Breaking:** `VectorAmpIntelligence` scopes questions with `dataset_ids` instead of the retired
  `dataset_id`. `POST /intelligence/query` answers any request carrying the singular field with a
  400 naming the replacement, so an `ask(..., dataset_id=...)` override now raises a `TypeError`
  naming `dataset_ids` rather than reaching the API.
- An unscoped instance no longer sends `dataset_id="all"`. The sentinel is retired: omitting
  `dataset_ids` is how the API says "every dataset the caller can see", so the field is left off
  the request entirely. Empty and `"all"`-only scopes are omitted the same way.

### Added

- `dataset_ids=` and `dataset_names=` constructor arguments, and a per-call `dataset_ids=` override,
  for scoping one question to several datasets. Both stay optional; the singular `dataset_id=` /
  `dataset_name=` forms still work and merge into the same scope.

## [0.2.0] - 2026-07-14

### Added

- Add sync and async vector deletion support through the LangChain VectorStore API.
- Document `delete` and `adelete` usage for deleting vectors by id.

## [0.1.0] - 2026-07-02

### Added

- Initial public-ready package baseline for VectorAmp SDK/CLI migration to GitHub.
- GitHub Actions CI workflow.
