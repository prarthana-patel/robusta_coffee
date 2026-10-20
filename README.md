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
- **Voice:** audio recording through `st.audio_input`, opt-in Gemini transcription, editable transcript, explicit Send. `GEMINI_TRANSCRIPTION_MODEL` can select an available audio-capable model. Recording is sent to Gemini only after Transcribe. Browser microphone needs user permission and a secure origin. No speech output or offline speech recognition is implemented.
- **Camera:** browser camera permission and a secure origin. Upload remains available. Capture uses the same original image preprocessing as uploads.
- **Accounts:** Streamlit OIDC `[auth]` configuration (provider discovery URL, client ID/secret, redirect URI, cookie secret). Use a trusted provider that supplies verified `iss` and `sub` claims. Streamlit/Authlib validates identity; account keys derive from issuer and subject, never a supplied username or email. The login page stays disabled until configured. Actual provider login has not been tested.
- **Profiles/history:** set absolute `ROCOLE_DB_PATH` to an operator-managed database path. Signed-in users can save display name, farm name, region, variety, notes, and future scans with PNG images. History is paginated and owner-filtered for every query and deletion. Anonymous history is session-only, without the previous 20-record truncation. Anonymous scans are not silently migrated into an account. Clear Session does not delete account data. Individual account scan deletion requires confirmation. Persistence across local SQLite connections is tested; a path or flag does not prove volume durability. Configure backups, access permissions and retention for the deployment.
- **Feedback:** optional `FEEDBACK_DB_PATH` plus `FEEDBACK_DB_PERSISTENT=true` only after independently verifying storage durability. Its parent directory must already exist. Optional `SMTP_HOST`, `SMTP_PORT` (587 STARTTLS or 465 TLS), `SMTP_USER`, `SMTP_PASSWORD`, `FEEDBACK_TO_EMAIL`. Both configured backends are attempted. Safe status distinguishes storage, configuration, auth, recipient, connection, TLS and server failures. SMTP acceptance is not verified inbox delivery. SMTP credentials require actual server-side secret injection; HTTPS proxy placeholders are not SMTP passwords.
- **Market:** `MARKET_API_KEY` with licensed Barchart Robusta data. Provider connectivity and entitlement are separate from app startup.
- **Inference:** provide only trusted trained checkpoints in `models/nutrient` and `models/disease` (see `models/README.md` for filenames, Windows paths and environment overrides), and verified training class mapping metadata/sidecars. SYSTEM is now reachable for runtime diagnostics and manual selections. Never substitute random weights. The original unverified-class-order fallback remains explicit.

## Reports, guidance and research limits

Each result and stored image can be exported as JSON/PDF. PDF includes model statuses, class mapping sources, probabilities, image, and non-diagnostic care guidance. Current labels are English; ReportLab's built-in font is not a general multilingual font solution.

Condition-specific guidance suggests inspections and confirmation, not chemical doses. Probability margin and normalized entropy are descriptive; neither provides calibration, validated severity, diagnostic certainty or accuracy evidence.

`explain.py` implements experimental spatial Grad-CAM for a separately supplied model and explicit NCHW target layer. Synthetic tests cover finite maps, unchanged weights and hook cleanup. **It is not connected to the app UI and has not been validated for ConvNeXtV2 or the other research checkpoints.** Real weights, training configuration, target-layer verification and explanation checks are needed first. Severity stays unavailable; no segmentation model or validated severity scale was supplied. No held-out evaluation or accuracy result was produced.

## Delivery

See `CHANGELOG.md` and `TEST_REPORT.md`. Changes are local and have not been committed, pushed or deployed. Origin was empty; preserve these files through publication or normal version control. This environment cannot demonstrate new-task snapshot restoration.

## Portable model paths

The configured Windows folders are `D:\rocole_github_deploy\models\nutrient` and `D:\rocole_github_deploy\models\disease`. Local checkout `models/<task>` folders are preferred when present. `ROCOLE_NUTRIENT_DIR` and `ROCOLE_DISEASE_DIR` process variables override discovery. Existing Colab folders remain a fallback. This only configures locations; model files must exist on the actual machine running Streamlit. See [model placement](models/README.md).
