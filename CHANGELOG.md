# Changelog

All notable changes to `unju` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- Memory API path corrected: `/v1/memories/` → `/v1/memory/` (affects all `memory.*` methods)

### Added
- Comprehensive memory integration test suite

---

## [0.1.0] — 2026-02-15

### Added
- `unju.memory` — add, search, list, get, delete, delete_all (mem0-compatible API)
- `unju.agents` — list, get, connect, card, trust
- `unju.credits` — balance, usage, yield_info
- Sync (`Unju`) and async (`AsyncUnju`) clients
- `api_version` parameter on both clients to pin the API version (default: `"v1"`)
- `py.typed` marker — full mypy/pyright support out of the box
- Zero heavy dependencies — httpx only

---

## Deprecation Policy

When a method, parameter, or behaviour is scheduled for removal:

1. It will be marked deprecated for **at least one minor version** before removal.
2. A `DeprecationWarning` will be emitted at call time.
3. The CHANGELOG entry will note the version in which it was deprecated and the version it will be removed.
4. Breaking changes are reserved for major version bumps (`1.0.0`, `2.0.0`, …).

While the SDK is in `v0.x.x` (pre-1.0 alpha/beta), minor versions may include breaking changes — these will always be called out explicitly in the `### Breaking` section of the relevant CHANGELOG entry.

---

[Unreleased]: https://github.com/unju-ai/unju-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/unju-ai/unju-python/releases/tag/v0.1.0
