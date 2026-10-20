# Function-level changes

- `initialize_state`: image archive and history UI state.
- `go`, `main`: stable keyed navigation, reachable Account/SYSTEM, light-theme color overrides.
- `scan_page`: camera/upload selection; stores image and result in session, attempts owner-scoped account storage after inference; removes last-20 truncation.
- `render_result`: PDF download, associated image, condition-specific guidance, margin/entropy, explicit severity and explanation availability.
- `submit_feedback`, `settings_page`: independent SQLite/SMTP attempts; safe per-backend diagnostics; accurate acceptance wording.
- `chat_page`: opt-in voice recording/transcription, editable draft and explicit send into the existing Coffee AI question flow.
- `current_owner`: stable identity from verified OIDC issuer/subject.
- `account_page`: provider sign-in/out and farm-profile form.
- `archive_page`: account pagination, images, owner-scoped deletion, session-only fallback.
- `features.Store`: parameterized owner-scoped SQLite profiles and scan/image storage; transactional writes and connection cleanup.
- `features.account_id`, `png_bytes`, `guidance`, `uncertainty`, `pdf_report`: identity key, image encoding, class-specific inspection advice, descriptive indicators, PDF creation.
- `voice.transcribe`: bounded audio payload, configured Gemini transcription, response/error handling.
- `explain.spatial_gradcam`: experimental explicit-layer spatial Grad-CAM; isolated from app inference, not checkpoint validated.

Protected model functions/constants and Coffee AI prompt/request functions are AST-identical to the original upload; verified by regression test. Added modules, requirements, example secrets, setup guide and tests. No model checkpoint, calibration, severity model or research metrics were fabricated.

- `model_paths.model_directory`: server-side per-task process overrides, portable checkout folders, supplied Windows folders, and existing Colab fallback. Original checkpoint filenames, loading checks and classifier behavior remain unchanged.

## Chat composer simplification

- Removed Account and SYSTEM navigation and page functions; resets stale navigation selections to Home.
- `chat_page`: native inline microphone via `st.chat_input(accept_audio=True)`; removed separate VOICE QUESTION expander, recording widget and transcript form. Recorded audio is transcribed into the composer for review before sending. Typed questions use the same protected Coffee AI functions.
- Composer text/background colors follow the selected theme. History no longer directs anonymous users to a removed sign-in page.

## Automatic checkpoint retrieval

- `checkpoint_download.download_checkpoints`: download only the eight explicit filenames from the supplied Drive folder, preserve existing files, reject missing/duplicate matches, use verified TLS, atomic writes and a process lock, record acquired hashes, and fetch unambiguous class JSON sidecars. No trained files are bundled.
- `scan_page`: attempt automatic retrieval of the selected task's missing models before running the original inference function. No download button.
- Added gdown and pinned its dependencies. API keys remain server-side configuration.
