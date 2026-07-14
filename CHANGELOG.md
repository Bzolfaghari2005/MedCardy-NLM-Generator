# Changelog

All notable changes to NLM are documented here.

## 1.0.0 - 2026-07-14

### Added

- Release verification script for dependencies, FFmpeg, syntax, and all tests.
- Production NotebookLM dependency and Windows Edge/cookie login support.
- Shared-source ordering regression tests and release service tests.
- Log redaction for email addresses, bearer tokens, and secret query values.

### Changed

- Shared sources are fully ready before the main chunk source is uploaded.
- Audio generation starts only after every source is ready and a 60-second
  production settle period has elapsed.
- Project start now validates account authentication and allocation capacity.
- Retry, resume, pause, rate-limit, and runner lifecycle behavior hardened.
- FFmpeg temporary MP3 files retain a recognizable media extension.
- Windows project cleanup retries transient file locks.

### Security

- Runtime logging sanitizes personal email addresses and common credentials.
- Runtime `data/`, local profiles, logs, and `.env` files remain excluded from
  source releases.
