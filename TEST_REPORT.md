# Validation report

Current run: Python 3.12, Streamlit 1.66.0, torch 2.14.1+cpu, timm 1.0.30.

`/workspace/rocole-venv/bin/python -m pytest -q tests`: **10 passed**.

- SQLite account isolation: another owner cannot read profiles, list scans, retrieve images or delete records. Reopened database retains the owner's result and PNG. Owner deletion succeeds.
- PDF generation from synthetic scan/image: valid PDF structure; probability indicators and invalid/non-finite input rejection checked.
- Repeated navigation through all seven pages and switching to light theme: no application exceptions, selected page remains stable. This is a Streamlit AppTest check, not visual/browser verification.
- Protected original model functions/constants and Coffee AI prompt/request functions: AST comparison against the retained upload passes.
- Feedback: configured SQLite storage and mocked SMTP both attempted; STARTTLS and send invoked; safe authentication failure status checked while preserving successful storage.
- Voice: mocked Gemini response transcription, missing credential and provider rejection checked. No real recording or live transcription request validated.
- Experimental Grad-CAM: synthetic spatial classifier only; finite nonconstant map, unchanged weights, hook cleanup including failure. No research checkpoint or ConvNeXtV2 explanation validation.
- Seeded session history with image: PDF/JSON controls render. Camera selection creates a camera control. Real capture is not tested.

`python -m pip check` in the venv: **No broken requirements found**.
Server restarted successfully; local `/_stcore/health` returned `ok`.

Full browser test **blocked**: Playwright browser was absent, and Chromium download from cdn.playwright.dev returned HTTP 403 `Domain forbidden`. No visual contrast, desktop/mobile, camera/microphone permission or actual device tests completed. AppTest does not substitute for those checks.

Not run / externally blocked: actual OIDC login and logout with a provider, authenticated UI with real identities, real SMTP delivery, live Gemini transcription and language quality, real trained-checkpoint inference and Grad-CAM, held-out evaluation, calibration, severity validation, and durable-volume recovery/backups. No performance or accuracy metrics generated. Mocked external calls are not live integration evidence.

No Git push or deployment performed. Environment setup/startup instructions were updated in the draft; publication and new-task restoration remain unverified.

Portable path update: two additional tests verify nutrient/disease directory discovery and process overrides, invalid task rejection, and exact checkpoint filename resolution from a portable task folder. The path fixture is not model weights; no inference validation is claimed.
