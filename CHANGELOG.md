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
