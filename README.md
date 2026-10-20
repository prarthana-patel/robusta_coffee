# RoCoLe development setup

Python 3.12; CPU runtime is supported. The original uploaded source is retained outside the checkout in `/workspace/attachments/`. No changes were made to the model architectures, checkpoint selection/loading, preprocessing, class mappings, ensemble calculation, or Coffee AI prompt/request functions.

## Install and run

Use the existing checkout; do not create a worktree unless requested.

```bash
cd /workspace/robusta_coffee
python3 -m venv /workspace/rocole-venv
/workspace/rocole-venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
/workspace/rocole-venv/bin/python -m pytest -q tests
/workspace/rocole-venv/bin/python -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false --theme.base=dark
```

`requirements.txt` lists direct dependencies; `requirements-lock.txt` captures this tested environment. Do not use these CPU torch pins for GPU deployment without choosing a compatible official torch/torchvision pair. The training-time timm version is unknown, so real checkpoint compatibility remains unverified.

## Configuration

Use process variables or private `.streamlit/secrets.toml`. `.streamlit/secrets.toml.example` contains placeholders only. Do not deploy it unchanged. Never commit real credentials.

- **Coffee AI:** `GEMINI_API_KEY`; optional `GEMINI_MODEL`, `GEMINI_BACKUP_MODEL`. Existing prompts and response flow are preserved.
- **Voice:** native microphone inside `st.chat_input`, opt-in Gemini transcription, transcript returned to that same textbox for editing, then explicit Send. `GEMINI_TRANSCRIPTION_MODEL` can select an available audio-capable model. Submitting an audio recording sends it to Gemini for transcription; the transcript is not sent to Coffee AI until submitted again as text. Browser microphone needs user permission and a secure origin. No speech output or offline speech recognition is implemented.
- **Camera:** browser camera permission and a secure origin. Upload remains available. Capture uses the same original image preprocessing as uploads.
- **Navigation:** Home, Analyze Leaf, Coffee AI, History, Settings. Account and SYSTEM pages have been removed.
- **History:** anonymous scans and images remain session-only. Previous owner-scoped storage helpers remain available for compatibility, but this version has no account sign-in/profile page.
- **Feedback:** optional `FEEDBACK_DB_PATH` plus `FEEDBACK_DB_PERSISTENT=true` only after independently verifying storage durability. Its parent directory must already exist. Optional `SMTP_HOST`, `SMTP_PORT` (587 STARTTLS or 465 TLS), `SMTP_USER`, `SMTP_PASSWORD`, `FEEDBACK_TO_EMAIL`. Both configured backends are attempted. Safe status distinguishes storage, configuration, auth, recipient, connection, TLS and server failures. SMTP acceptance is not verified inbox delivery. SMTP credentials require actual server-side secret injection; HTTPS proxy placeholders are not SMTP passwords.
- **Market:** `MARKET_API_KEY` with licensed Barchart Robusta data. Provider connectivity and entitlement are separate from app startup.
- **Inference:** provide only trusted trained checkpoints in `models/nutrient` and `models/disease` (see `models/README.md` for filenames, Windows paths and environment overrides), and verified training class mapping metadata/sidecars. Checkpoint configuration is server-side; no SYSTEM page is exposed. Never substitute random weights. The original unverified-class-order fallback remains explicit.

## Reports, guidance and research limits

Each result and stored image can be exported as JSON/PDF. PDF includes model statuses, class mapping sources, probabilities, image, and non-diagnostic care guidance. Current labels are English; ReportLab's built-in font is not a general multilingual font solution.

Condition-specific guidance suggests inspections and confirmation, not chemical doses. Probability margin and normalized entropy are descriptive; neither provides calibration, validated severity, diagnostic certainty or accuracy evidence.

`explain.py` implements experimental spatial Grad-CAM for a separately supplied model and explicit NCHW target layer. Synthetic tests cover finite maps, unchanged weights and hook cleanup. **It is not connected to the app UI and has not been validated for ConvNeXtV2 or the other research checkpoints.** Real weights, training configuration, target-layer verification and explanation checks are needed first. Severity stays unavailable; no segmentation model or validated severity scale was supplied. No held-out evaluation or accuracy result was produced.

## Delivery

See `CHANGELOG.md` and `TEST_REPORT.md`. Changes are local and have not been committed, pushed or deployed. Origin was empty; preserve these files through publication or normal version control. This environment cannot demonstrate new-task snapshot restoration.

## Portable model paths

The configured Windows folders are `D:\rocole_github_deploy\models\nutrient` and `D:\rocole_github_deploy\models\disease`. Local checkout `models/<task>` folders are preferred when present. `ROCOLE_NUTRIENT_DIR` and `ROCOLE_DISEASE_DIR` process variables override discovery. Existing Colab folders remain a fallback. This only configures locations; model files must exist on the actual machine running Streamlit. See [model placement](models/README.md).

## Automatic Google Drive checkpoint fetching

When RUN NEURAL SCAN detects missing checkpoint files for the selected task, it automatically attempts to fetch that task's four exact filenames from the supplied Drive folder. There is no download button. Both disease and nutrient sets are supported. Existing local checkpoints are preserved. The folder must allow downloads without interactive sign-in. Downloaded artifacts stay on the running machine; restarts/redeployments that discard local storage require fetching them again. TLS verification remains enabled. Acquired SHA-256 hashes are recorded, but no original-computer hash comparison has been made. Model compatibility is still checked by the original strict loader.

`python checkpoint_download.py` can prefetch all eight files during operator setup. It is not automatically executed by merely pushing this repository. This development environment's proxy returned 403 for live Drive access; no actual model download or inference has been verified here.

The app reads GEMINI_API_KEY from Streamlit Secrets or process configuration. Set it once in the deployed Streamlit app's Secrets settings. The developer environment's secret bindings are separate from Streamlit Cloud; API credentials are never committed into application source.

## Coffee AI reliability update

The primary remains `gemini-3.1-flash-lite`. The default backup is `gemini-3.5-flash-lite`; both produced valid live text responses during this update using the development environment's existing credential. This does not establish the deployed Streamlit app's key, quota or network status. `GEMINI_BACKUP_MODEL` overrides the backup; an explicit empty value disables it.

Temporary provider errors, timeouts and connection failures switch to the backup after the first failed primary attempt. Without a distinct backup, a transient primary failure is retried once. Invalid-key, permission and quota errors return specific safe messages without retrying or exposing provider response bodies. Prompts, language checks, scan context and JSON response schema are unchanged. An existing explicit GEMINI_MODEL or GEMINI_BACKUP_MODEL setting is preserved; remove an unwanted override in Streamlit Secrets to use these defaults.

## Voice request recovery

Voice transcription now defaults independently to gemini-3.5-flash-lite, which accepted a synthetic WAV request in a live API check. The old gemini-3.1-flash-lite voice request timed out in that check; text chat success does not establish audio availability. Explicit GEMINI_TRANSCRIPTION_MODEL settings are preserved. An optional GEMINI_TRANSCRIPTION_BACKUP_MODEL overrides/disables fallback; otherwise a distinct backup is chosen between these two models. Text chat settings and prompts are unchanged.

Voice requests use canonical inlineData/mimeType JSON fields, try a backup for unsupported/unavailable models or temporary transport/provider failures, and stop with distinct safe authentication, permissions or quota messages when appropriate. Never expose raw provider responses or credentials. The live test used synthetic silence and validates audio request acceptance only; no real-user microphone or multilingual recognition quality claim is made.

## Short voice questions and coffee quote panels

Voice records are limited to 30 seconds; 5–15 seconds is recommended. Conversation options now has an independent Speech language setting, including Kannada / Kanglish, Kannada script, English, Hindi, Tulu, Tamil and Malayalam. Auto remains the default. Explicit hints preserve code switching and prevent answering the audio instead of transcribing it. This is not evidence of improved recognition across all languages.

Transcription has a 12-second total async deadline and 10-second read timeout. A timed-out model is not followed by another long model request. Fallback remains available for prompt model/request/provider failures within the total deadline. Slow requests fail clearly instead of leaving users waiting through two 30-second requests. The transcript remains editable before sending.

Home now embeds provider-hosted TradingView / ICE continuous Robusta and Arabica quote panels, with units and source links. These load in the browser and do not need a MARKET_API_KEY in this app. Market data can be delayed, region restricted, or unavailable; check the provider timestamp/delay indicator. No prices are generated by the app. Robusta is USD/tonne; Arabica is US cents/lb, and neither is a local farm-gate offer.

The separate Barchart API snapshot now has a shorter request timeout and 60-second cache/refresh. Its current configured credential returned HTTP 401 during validation; a licensed valid key is still needed for that API feed. That does not establish TradingView availability. Browser quote rendering could not be verified from this environment because its proxy blocked the widget endpoint.

Latest voice default: after the deadline/prompt update, a five-second English speech fixture transcribed correctly with gemini-3.1-flash-lite in 1.66 seconds, versus 8.54 seconds with gemini-3.5-flash-lite. The default is therefore gemini-3.1-flash-lite again; existing explicit GEMINI_TRANSCRIPTION_MODEL settings still override it. This is one sample, not a latency guarantee or multilingual evaluation.
