# NLM release checklist

## Automated gate

- [ ] Install a clean Python 3.11 or 3.12 environment.
- [ ] Run `pip install -r requirements-dev.txt`.
- [ ] Run `powershell -ExecutionPolicy Bypass -File scripts\verify_release.ps1`.
- [ ] Confirm no important test was skipped and every command exited with zero.

## Real NotebookLM smoke test

- [ ] Login with a dedicated NotebookLM profile and confirm `ACTIVE` in Accounts.
- [ ] Create one project containing one PDF chunk and one small shared source.
- [ ] Confirm logs show shared ready, main ready, 60-second settle, generation,
      completed artifact, and M4A download.
- [ ] Test Pause/Resume and Retry once without duplicate notebooks or audio.
- [ ] Confirm optional MP3 conversion and transcription when enabled.

## Other product features

- [ ] Open all 13 pages without an exception.
- [ ] Convert a short audio file with FFmpeg.
- [ ] Transcribe a short audio file locally.
- [ ] Run AI Folder with its fake/offline test suite.
- [ ] Build and open a Word booklet, then update its TOC in Word.
- [ ] Export a project ZIP and delete a stopped project's files.

## Distribution hygiene

- [ ] `VERSION` and `CHANGELOG.md` match the release tag.
- [ ] `data/`, `.env`, browser profiles, logs, generated media, and API keys are
      absent from the source archive.
- [ ] Search the archive for email addresses, bearer tokens, and private keys.
- [ ] Include `LICENSE`, `README.md`, requirements files, and this checklist.
