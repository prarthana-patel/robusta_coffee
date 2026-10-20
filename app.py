"""ROCOLE // NEURAL LAB. Run: streamlit run app.py --theme.base=dark

Credentials: GEMINI_API_KEY environment variable or Streamlit secrets.
GEMINI_MODEL / GEMINI_BACKUP_MODEL: server-configured Gemini text models.
Original training roots are preserved; ambiguous checkpoints require selection.
"""
import asyncio
import base64
import hashlib
import html
import io
import json
import os
import re
import time
import smtplib
import ssl
import sqlite3
import logging
import math
import socket
from model_paths import model_directory
from contextlib import closing
from features import Store, account_id, png_bytes, guidance, uncertainty, pdf_report
import uuid
from email.message import EmailMessage
from datetime import datetime, timezone
from pathlib import Path

import httpx
import streamlit as st
from PIL import Image, ImageOps, ImageStat

logger = logging.getLogger("rocole.services")
# HTTP debug logging can expose authentication headers; keep transport logs off.
logging.getLogger("httpx").disabled = True
logging.getLogger("httpcore").disabled = True

# ========================= CONFIGURATION =========================
PROJECT_DIR = model_directory("nutrient")
ROCOLE_PHASE2_DIR = model_directory("disease")
ROCOLE_EXTRACTED_DIR = "/content/drive/MyDrive/RoCoLe_Extracted"
ROOTS = {
    "Disease": [
        ROCOLE_PHASE2_DIR + "/checkpoints", ROCOLE_PHASE2_DIR,
        ROCOLE_EXTRACTED_DIR + "/checkpoints", ROCOLE_EXTRACTED_DIR,
    ],
    "Nutrient": [PROJECT_DIR + "/checkpoints", PROJECT_DIR],
}

EXACT_CHECKPOINTS = {
    "Disease": {
        "MaxViT Tiny": ROCOLE_PHASE2_DIR + "/checkpoints/maxvit_tiny_tf_224_best.pth",
        "Swin Tiny": ROCOLE_PHASE2_DIR + "/checkpoints/swin_tiny_patch4_window7_224_best.pth",
        "ConvNeXtV2 Tiny": ROCOLE_PHASE2_DIR + "/checkpoints/convnextv2_tiny_best.pth",
        "MambaOut Tiny": ROCOLE_PHASE2_DIR + "/checkpoints/mambaout_tiny_best.pth",
    },
    "Nutrient": {
        "MaxViT Tiny": PROJECT_DIR + "/checkpoints/maxvit_tiny_recovered.pth",
        "Swin Tiny": PROJECT_DIR + "/checkpoints/swin_best.pth",
        "ConvNeXtV2 Tiny": PROJECT_DIR + "/checkpoints/convnextv2_tiny_final_checkpoint.pth",
        "MambaOut Tiny": PROJECT_DIR + "/checkpoints/mambaout_tiny_final_checkpoint.pth",
    },
}

MODEL_ARCHS = {"MaxViT Tiny": "maxvit_tiny_tf_224",
               "Swin Tiny": "swin_tiny_patch4_window7_224",
               "ConvNeXtV2 Tiny": "convnextv2_tiny", "MambaOut Tiny": "mambaout_tiny"}
ALIASES = {"MaxViT Tiny": ("maxvit",), "Swin Tiny": ("swin",),
           "ConvNeXtV2 Tiny": ("convnext",), "MambaOut Tiny": ("mambaout",)}
DISEASE_CLASSES = ["Healthy", "Red Spider Mite", "Rust"]
NUTRIENT_CLASSES = ["Healthy", "Boron (B) Deficiency", "Calcium (Ca) Deficiency",
                   "Iron (Fe) Deficiency", "Magnesium (Mg) Deficiency",
                   "Manganese (Mn) Deficiency", "Nitrogen (N) Deficiency",
                   "Phosphorus (P) Deficiency", "Potassium (K) Deficiency"]
CLASSES = {"Disease": DISEASE_CLASSES, "Nutrient": NUTRIENT_CLASSES}
PAGES = ["Home", "Analyze Leaf", "Coffee AI", "History", "Settings"]
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
AI_DEADLINE = None  # no artificial total response timer
SYSTEM_PROMPT = """You are RoCoLe, a friendly, knowledgeable coffee assistant.
Answer coffee questions about plants, Robusta, Arabica, varieties, diseases, pests,
nutrient deficiencies, cultivation, harvest, processing, beans, roasting, brewing,
research and markets. Usually answer directly in 2–5 short, clear sentences.

Detect the language AND writing style of EACH new message independently. Match
English, Kannada script, Kanglish (Kannada in English letters), Hindi script,
Romanized Hindi, Tulu, Romanized Tulu, other supported languages and mixed styles.
The user's explicit request for a language overrides automatic matching. The
response-language selector is a preference only; a request in the message wins.
Do not translate in parentheses unless asked. Do not comment on spelling.
For Kanglish use everyday spoken Kannada in English letters with familiar English
coffee terms. Avoid formal transliteration, invented words and forced translations.
Example user: coffee andre enu
Example answer: Coffee andre coffee gidada beans inda tayarisuva ondu popular drink.
Usually beans-na roast maadi, powder maadi, hot water jothe brew maadtare. Coffee
alli caffeine irutte, adikke kudidaga swalpa energetic mattu alert feel agabahudu.
This example demonstrates tone, not a template for unrelated questions.
For Tulu/Romanized Tulu, answer naturally only if you understand the meaning and
can express it reliably. Never invent Tulu words or randomly blend Kannada and
Tulu. If uncertain, ask one short clarification; use the language you can identify
reliably, or simple English if none is clear. Never pretend you understood.

A greeting alone gets one short friendly response, e.g. 'Hey! 👋 How can I help you
with coffee today?' Match the greeting's language/style where identifiable.
Never invent context for ambiguous input. '7yo' alone does not establish whether
it means a person, a plant, an age or anything else: ask what the user means.
Use previous messages only when there is an obvious connection. The attached
scan must not turn an unrelated question or short unclear message into a diagnosis.
Treat older assistant messages as fallible; do not repeat their language mistakes.

CURRENT SCAN is application model output, not a confirmed agricultural diagnosis.
You cannot see the leaf image or Grad-CAM. Never invent symptoms, predictions,
confidence, research metrics, citations or sources. If asked about a specific leaf
without a scan, invite the user to Analyze Leaf. Give general explanations as such.
Do not prescribe pesticide/fertilizer doses, concentrations or mixing ratios;
suggest local agricultural advice and soil/leaf testing where appropriate.
You have no live web or market feed in this chat. Never invent current prices;
for current prices direct the user to the Home page Robusta Market card and its
source/timestamp. General market explanations are fine. For unrelated subjects,
briefly invite a coffee question. Do not imitate the wording or language mistakes of earlier assistant replies.
Tulu and Kannada are distinct languages. Kannada written in English letters is
not Tulu. Never make a Tulu answer by adding Tulu endings to Kannada sentences.
For each message choose one target language/style based on the current input or
its explicit language request. The previous turn's language is not binding.
Common English coffee terms are allowed in Romanized speech, but do not switch
between unrelated regional languages. If unable to understand the input OR write
a coherent answer in the target language, admit uncertainty instead of inventing.
If the user explicitly selects Tulu, do not silently answer in Kannada.

Return a JSON object with answer, language_uncertain, and language_consistent.
These flags concern your ability to understand and respond in the requested
language, not uncertainty about a coffee fact. Set language_uncertain=true when
unsure which language is intended or unsure how to express the answer naturally.
Set language_consistent=false if the answer mixes unrelated languages or uses
invented regional words. Check this before returning; do not provide reasoning.
For unclear meaning in an otherwise known language, put a short clarification
question in answer. For '7yo' alone, ask what the user means; do not assume age.
The answer field contains only the user-facing text, with no JSON explanation.
"""

def setting(name, default=""):
    value = os.getenv(name)
    if value is not None:
        return value.strip()
    try:
        return str(st.secrets.get(name, default)).strip()
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default

def safe_detail(value):
    text = str(value)
    for secret_name in ("GEMINI_API_KEY", "MARKET_API_KEY", "SMTP_PASSWORD", "SMTP_USER"):
        key = setting(secret_name)
        if key:
            text = text.replace(key, "[REDACTED]")
    return re.sub(r"AIza[\w-]+", "[REDACTED]", text)[:18000]

def esc(value):
    return html.escape(str(value), quote=True)

# ========================= MODEL SYSTEM =========================
@st.cache_resource(show_spinner=False)
def ml_runtime():
    # Import only when a scan/diagnostic is requested; chat never loads torch.
    import torch
    import timm
    from torchvision import transforms
    return torch, timm, transforms

@st.cache_data(ttl=60, show_spinner=False)
def checkpoint_candidates(task):
    found = {name: [] for name in MODEL_ARCHS}
    paths = set()
    for root in ROOTS[task]:
        if Path(root).is_dir():
            for folder, _, files in os.walk(root):
                for filename in files:
                    if filename.lower().endswith((".pth", ".pt", ".ckpt")):
                        paths.add(str(Path(folder) / filename))
    for path in sorted(paths):
        filename = Path(path).name.lower()
        if any(token in filename for token in ("optimizer", "scheduler", "scaler")):
            continue
        for name, aliases in ALIASES.items():
            if any(alias in filename for alias in aliases):
                found[name].append(path)
    return found

def resolve_checkpoint(task, name):
    """Resolve only the intended checkpoint filename; never choose a random alias match."""
    selected = st.session_state.path_overrides.get(task + "/" + name)
    if selected:
        if Path(selected).is_file():
            return selected
        raise FileNotFoundError(f"Selected checkpoint not found: {selected}")

    preferred = Path(EXACT_CHECKPOINTS[task][name])
    if preferred.is_file():
        return str(preferred)

    # The disease project has existed under both RoCoLe_Phase2_Project and
    # RoCoLe_Extracted. Search for the SAME trained filename in configured roots.
    # This avoids the old bug where any file containing 'swin'/'maxvit' became a candidate.
    basename = preferred.name
    exact_name_matches = []
    for root in ROOTS[task]:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        direct = root_path / basename
        if direct.is_file():
            exact_name_matches.append(str(direct))
        for match in root_path.rglob(basename):
            if match.is_file():
                exact_name_matches.append(str(match))

    exact_name_matches = list(dict.fromkeys(exact_name_matches))
    if exact_name_matches:
        # ROOTS is deliberately ordered: current project first, legacy project second.
        return exact_name_matches[0]

    raise FileNotFoundError(
        f"Intended checkpoint '{basename}' was not found for {task}/{name}. "
        f"Searched: {ROOTS[task]}. Open SYSTEM to select a checkpoint manually if your trained file has a different name."
    )

def extract_state_dict(checkpoint, torch):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dictionary or a dictionary containing one.")
    for name in ("state_dict", "model_state_dict", "model", "net", "weights", "state_dict_ema"):
        child = checkpoint.get(name)
        if isinstance(child, dict):
            try:
                return extract_state_dict(child, torch)
            except (TypeError, ValueError):
                pass
    tensors = {k: v for k, v in checkpoint.items() if isinstance(k, str) and torch.is_tensor(v)}
    if not tensors:
        raise ValueError("No tensor state_dict found in supported nested checkpoint containers.")
    return tensors

def align_state_dict(state, expected):
    prefixes = ("module.", "_orig_mod.", "model.", "net.")
    cleaned = {}
    for key, value in state.items():
        variants = [key]
        while any(variants[-1].startswith(p) for p in prefixes):
            current = variants[-1]
            prefix = next(p for p in prefixes if current.startswith(p))
            variants.append(current[len(prefix):])
        matched = next((v for v in variants if v in expected), variants[-1])
        if matched in cleaned:
            raise ValueError(f"Prefix normalization creates duplicate key: {matched}")
        cleaned[matched] = value
    return cleaned

def metadata_classes(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    for key in ("class_to_idx", "classes", "class_names", "idx_to_class"):
        if key in checkpoint:
            return checkpoint[key]
    for key in ("metadata", "config", "model", "net"):
        found = metadata_classes(checkpoint.get(key))
        if found is not None:
            return found
    return None

def canonical_class(label, task):
    normalized = re.sub(r"[^a-z]", "", str(label).lower())

    # Disease checkpoints were trained with PlantVillage-style names:
    # coffee___healthy, coffee___red_spider_mite, coffee___rust.
    if task == "Disease":
        aliases = {
            "coffeehealthy": "Healthy",
            "coffeeredspidermite": "Red Spider Mite",
            "coffeeredspidermites": "Red Spider Mite",
            "coffeerust": "Rust",
            "healthy": "Healthy",
            "redspidermite": "Red Spider Mite",
            "redspidermites": "Red Spider Mite",
            "redmite": "Red Spider Mite",
            "rust": "Rust",
        }
        if normalized in aliases:
            return aliases[normalized]

    # Nutrient checkpoints have appeared with labels such as boron-B, calcium-Ca,
    # healthy, iron-Fe, magnesium-Mg, manganese-Mn, nitrogen-N, phosphorus-P,
    # potasium-K / potassium-K. Match the nutrient name rather than punctuation.
    if task == "Nutrient":
        nutrient_aliases = {
            "healthy": "Healthy",
            "boron": "Boron (B) Deficiency",
            "calcium": "Calcium (Ca) Deficiency",
            "iron": "Iron (Fe) Deficiency",
            "magnesium": "Magnesium (Mg) Deficiency",
            "manganese": "Manganese (Mn) Deficiency",
            "nitrogen": "Nitrogen (N) Deficiency",
            "phosphorus": "Phosphorus (P) Deficiency",
            "potassium": "Potassium (K) Deficiency",
            "potasium": "Potassium (K) Deficiency",
        }
        for prefix, target in nutrient_aliases.items():
            if normalized == prefix or normalized.startswith(prefix):
                return target

    for target in CLASSES[task]:
        target_normalized = re.sub(r"[^a-z]", "", target.lower())
        if normalized == target_normalized:
            return target

    raise ValueError(f"Unrecognized training class {label!r} for {task}.")

def parse_classes(raw, task):
    if isinstance(raw, dict):
        for key in ("class_to_idx", "classes", "class_names", "idx_to_class"):
            if key in raw:
                return parse_classes(raw[key], task)
        if all(isinstance(v, int) for v in raw.values()):
            if sorted(raw.values()) != list(range(len(raw))):
                raise ValueError("class_to_idx indices must be consecutive, starting at zero.")
            raw = [k for k, _ in sorted(raw.items(), key=lambda x: x[1])]
        elif all(str(k).isdigit() for k in raw):
            if sorted(int(k) for k in raw) != list(range(len(raw))):
                raise ValueError("idx_to_class indices must be consecutive, starting at zero.")
            raw = [v for k, v in sorted(raw.items(), key=lambda x: int(x[0]))]
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Class mapping must be a list or an indexed dictionary.")
    result = [canonical_class(label, task) for label in raw]
    if len(result) != len(CLASSES[task]) or set(result) != set(CLASSES[task]):
        raise ValueError(f"Training classes {result} do not match the {task} class set.")
    return result

@st.cache_resource(show_spinner=False, max_entries=8)
def load_model_checked(name, task, path, stamp, mapping_path, mapping_stamp, device):
    torch, timm, _ = ml_runtime()
    architecture = MODEL_ARCHS[name]
    # Only trusted, locally configured research checkpoints are loaded.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint, torch)
    model = timm.create_model(architecture, pretrained=False, num_classes=len(CLASSES[task]))
    expected = model.state_dict()
    state = align_state_dict(state, expected)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatches = [f"{k}: expected {tuple(expected[k].shape)}, checkpoint {tuple(state[k].shape)}"
                  for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape]
    if missing or unexpected or mismatches:
        raise ValueError(f"STATE DICT ERROR / architecture {architecture}\n"
                         f"Shape mismatches: {mismatches}\nMissing keys: {missing}\n"
                         f"Unexpected keys: {unexpected}\n"
                         "Use the architecture/timm version from training. A Base checkpoint "
                         "cannot load into a Tiny model. No partial/random weights were accepted.")
    model.load_state_dict(state, strict=True)
    raw = metadata_classes(checkpoint)
    source = "checkpoint metadata"
    if mapping_path:
        raw = json.loads(Path(mapping_path).read_text())
        source = mapping_path
    elif raw is None:
        for candidate in (Path(path).with_suffix(".classes.json"), Path(path).parent / "classes.json"):
            if candidate.is_file():
                raw = json.loads(candidate.read_text())
                source = str(candidate)
                break
    if raw is None:
        order = CLASSES[task][:]
        source = "UNVERIFIED: original app order; confirm against training class_to_idx"
    else:
        order = parse_classes(raw, task)
    model.eval().to(device)
    return model, order, source

# ========================= INFERENCE =========================
def run_scan(image, task, image_hash, update):
    results, diagnostics = [], []

    # SPEED: preprocess the uploaded image only once for all four models.
    # All four classifiers use the same 224x224 ImageNet preprocessing.
    torch, _, transforms = ml_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    if device == "cuda":
        tensor = tensor.pin_memory().to(device, non_blocking=True)
    else:
        tensor = tensor.to(device)

    for name in MODEL_ARCHS:
        record = {"model": name, "task": task, "state": "LOADING", "steps": []}
        update(name, "LOADING")
        try:
            path = resolve_checkpoint(task, name)
            record.update(path=path, architecture=MODEL_ARCHS[name])
            record["steps"].append("CHECKPOINT FOUND")
            mapping = st.session_state.mapping_overrides.get(task + "/" + name, "")
            sidecars = [Path(path).with_suffix(".classes.json"), Path(path).parent / "classes.json"]
            stamps = tuple((str(p), p.stat().st_mtime_ns) for p in sidecars if p.is_file())
            if mapping:
                stamps += ((mapping, Path(mapping).stat().st_mtime_ns),)
            model, order, source = load_model_checked(name, task, path, Path(path).stat().st_mtime_ns,
                                                      mapping, stamps, device)
            record["steps"] += ["ARCHITECTURE LOADED", "STATE DICT LOADED", "READY"]
            record.update(device=device, class_order=order, class_source=source)
            update(name, "INFERENCE")
            # Tensor is shared across all models; preprocessing is identical to training.
            with torch.inference_mode():
                logits = model(tensor)
                if not torch.is_tensor(logits) or tuple(logits.shape) != (1, len(order)):
                    raise ValueError(f"Unexpected model output shape: {getattr(logits, 'shape', type(logits))}")
                if not torch.isfinite(logits).all():
                    raise ValueError("Model output contains NaN/Inf; inference rejected.")
                p = torch.softmax(logits, dim=1)[0].cpu().tolist()
            aligned = [p[order.index(label)] for label in CLASSES[task]]
            index = max(range(len(aligned)), key=aligned.__getitem__)
            results.append({"model": name, "label": CLASSES[task][index],
                            "confidence": aligned[index] * 100, "probabilities": aligned,
                            "class_source": source})
            record["state"] = "COMPLETE"
        except Exception as exc:
            record.update(state="ERROR", error=safe_detail(f"{type(exc).__name__}: {exc}"))
        diagnostics.append(record)
        update(name, record["state"])
    st.session_state.model_diagnostics = diagnostics
    if not results:
        return None
    # Preserve latest app's arithmetic mean of all successful probability vectors.
    means = [sum(r["probabilities"][i] for r in results) / len(results) for i in range(len(CLASSES[task]))]
    index = max(range(len(means)), key=means.__getitem__)
    label = CLASSES[task][index]
    votes = sum(r["label"] == label for r in results)
    return {"task": task, "label": label, "confidence": means[index] * 100,
            "votes": votes, "agreement": votes / len(results) * 100,
            "predictions": results, "probabilities": means, "classes": CLASSES[task],
            "failed_models": [r["model"] for r in diagnostics if r["state"] == "ERROR"],
            "time": datetime.now(timezone.utc).isoformat(), "image_hash": image_hash,
            "method": "Mean of successful model probability vectors"}

def quality_messages(image):
    warnings = []
    if min(image.size) < 224:
        warnings.append("Image is smaller than 224 pixels on one side.")
    brightness = ImageStat.Stat(image.convert("L")).mean[0]
    if brightness < 45:
        warnings.append("Low mean brightness; photograph the leaf in better light.")
    if brightness > 235:
        warnings.append("High mean brightness; check for glare or overexposure.")
    return warnings

# ========================= COFFEE AI =========================
class AIError(Exception):
    def __init__(self, category, message, status=None):
        super().__init__(message)
        self.category, self.status = category, status

def check_http(response):
    if response.status_code < 400:
        return
    code = response.status_code
    category = {400: "INVALID_REQUEST", 401: "AUTHENTICATION", 403: "PERMISSION",
                404: "MODEL_UNAVAILABLE", 429: "QUOTA"}.get(code, "PROVIDER")
    messages = {
        400: "Coffee AI rejected the request. Check that the configured key and model support Gemini text generation.",
        401: "Coffee AI could not authenticate. The operator needs to update GEMINI_API_KEY in Streamlit Secrets.",
        403: "Coffee AI access was denied. Check the key's API permissions and project restrictions.",
        404: "The configured Coffee AI model is unavailable for this key. Check GEMINI_MODEL in Streamlit Secrets.",
        429: "Coffee AI quota or rate limit reached. Wait before retrying; the operator may need to check the Gemini project's quota and billing.",
    }
    if code == 400:
        try:
            details = response.json().get("error", {}).get("details", [])
            if any(isinstance(item, dict) and item.get("reason") in ("API_KEY_INVALID", "API_KEY_EXPIRED") for item in details):
                category = "AUTHENTICATION"
                messages[400] = "The Coffee AI API key is invalid or expired. Update GEMINI_API_KEY in Streamlit Secrets."
        except (ValueError, AttributeError, TypeError):
            pass
    raise AIError(category, messages.get(code, "The Coffee AI provider is temporarily unavailable. Please try again shortly."), code)

async def discover_models(client, key):
    models, token = [], None
    for _ in range(5):
        params = {"pageSize": 1000}
        if token:
            params["pageToken"] = token
        response = await client.get(API_ROOT + "/models", params=params, headers={"x-goog-api-key": key})
        check_http(response)
        data = response.json()
        models.extend(m for m in data.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", [])
                      and re.fullmatch(r"models/gemini-[a-zA-Z0-9._-]+", m.get("name", ""))
                      and not any(x in m["name"] for x in ("image", "tts", "audio", "robotics")))
        token = data.get("nextPageToken")
        if not token:
            break
    if not models:
        raise AIError("MODEL_UNAVAILABLE", "No compatible Gemini text models were listed for this key.")
    return models

def choose_model(models, preferred=""):
    if preferred:
        requested = preferred if preferred.startswith("models/") else "models/" + preferred
        if requested not in [m["name"] for m in models]:
            raise AIError("MODEL_UNAVAILABLE", "GEMINI_MODEL is not in the API's supported model list.")
        return requested
    # Choose only names actually returned by the provider; prefer stable Flash-Lite.
    def rank(m):
        n = m["name"]
        return (any(x in n for x in ("preview", "exp", "latest")),
                "flash-lite" not in n, "flash" not in n, not n.endswith("flash-lite"), n)
    return sorted(models, key=rank)[0]["name"]

def scan_context(result):
    if not result:
        return {"status": "No completed scan. Do not invent a leaf prediction."}
    return {k: result[k] for k in ("task", "label", "confidence", "votes", "agreement", "time", "failed_models")} | {
        "predictions": [{k: p[k] for k in ("model", "label", "confidence")} for p in result["predictions"]]}

def build_contents(turns, question):
    contents = []
    # Store successful visible pairs, not UI errors or rejected language drafts.
    for turn in turns[-5:]:
        contents.append({"role": "user", "parts": [{"text": turn["question"]}]})
        # Only visible answers are context; rejected mixed-language drafts must not recur.
        contents.append({"role": "model", "parts": [{"text": turn["answer"]}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    return contents

async def ai_exchange(key, models, preferred, turns, question, language, context, transport=None, telemetry=None):
    async with httpx.AsyncClient(timeout=httpx.Timeout(25, connect=5), transport=transport,
                                 follow_redirects=False) as client:
        # The normal chat path never calls models.list.
        model = preferred if preferred.startswith("models/") else "models/" + preferred
        if not re.fullmatch(r"models/gemini-[a-zA-Z0-9._-]+", model):
            raise AIError("MODEL_CONFIGURATION", "Coffee AI is temporarily unavailable. Please try again later.")
        if telemetry is not None:
            telemetry["model"] = model
        prompt = SYSTEM_PROMPT + "\nResponse-language preference (auto means detect THIS message): " + language + "\nCURRENT SCAN: " + json.dumps(context)
        config = {"maxOutputTokens": 1024, "responseMimeType": "application/json",
                  "responseSchema": {"type": "OBJECT", "properties": {
                      "language_uncertain": {"type": "BOOLEAN"},
                      "language_consistent": {"type": "BOOLEAN"},
                      "answer": {"type": "STRING"}},
                      "required": ["language_uncertain", "language_consistent", "answer"]}}
        # Explicit supported non-thinking setting only for the 2.5 Flash family.
        if model.startswith("models/gemini-2.5-flash"):
            config["thinkingConfig"] = {"thinkingBudget": 0}
        payload = {"systemInstruction": {"parts": [{"text": prompt}]},
                   "contents": build_contents(turns, question), "generationConfig": config}
        response = await client.post(API_ROOT + "/" + model + ":generateContent",
                                     headers={"x-goog-api-key": key}, json=payload)
        check_http(response)
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise AIError("EMPTY_RESPONSE", "Gemini returned no answer. Rephrase the question and retry.", 200)
        candidate = candidates[0]
        content = candidate.get("content", {})
        answer = "".join(p.get("text", "") for p in content.get("parts", []) if not p.get("thought")).strip()
        if not answer:
            raise AIError("EMPTY_RESPONSE", "Coffee AI could not produce an answer. Please try rephrasing your question.", 200)
        if candidate.get("finishReason") not in (None, "STOP", "MAX_TOKENS"):
            raise AIError("BLOCKED_RESPONSE", "Gemini could not complete that answer. Please rephrase.", 200)
        try:
            result = json.loads(answer)
            if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
                raise ValueError("Invalid answer object")
            if type(result.get("language_uncertain")) is not bool or type(result.get("language_consistent")) is not bool:
                raise ValueError("Invalid language flags")
            if result["language_uncertain"] or not result["language_consistent"]:
                answer = ("I'm not confident I understood the language correctly. "
                          "Could you rephrase that or tell me which language you're using?")
            else:
                answer = result["answer"].strip()
            if not answer:
                raise ValueError("Empty answer")
        except (ValueError, TypeError):
            raise AIError("INVALID_RESPONSE", "Coffee AI could not complete that answer. Please try again.") from None
        # Keep original provider parts/signatures for valid Gemini conversation history.
        return answer, content, model, models, False

def ask_ai(question, language):
    start = time.monotonic()
    diag = {"status": "DEGRADED", "http": None, "model": None,
            "last_request": datetime.now(timezone.utc).isoformat(), "attempts": []}
    key = setting("GEMINI_API_KEY")
    primary = st.session_state.get("preferred_model") or setting("GEMINI_MODEL", "gemini-3.1-flash-lite")
    backup = setting("GEMINI_BACKUP_MODEL", "gemini-3.5-flash-lite")
    primary = primary.removeprefix("models/")
    backup = backup.removeprefix("models/")

    async def request_model(model):
        provider_model = {"name": model, "supportedGenerationMethods": ["generateContent"]}
        return await ai_exchange(
            key, [provider_model], model, st.session_state.ai_turns,
            question, language, scan_context(st.session_state.analysis_result),
            telemetry=diag
        )

    async def request_with_recovery():
        # The default backup was checked with a real generation request.
        # An explicit empty GEMINI_BACKUP_MODEL disables fallback.
        transient = (500, 502, 503, 504)
        last_error = None
        attempt = 0
        configured = [primary] + ([backup] if backup and backup != primary else [])
        for model_index, model in enumerate(configured):
            for retry in range(2 if model_index == 0 else 1):
                if retry:
                    await asyncio.sleep(0.9)
                attempt += 1
                try:
                    # Bound the entire attempt, not only each socket read.
                    result = await asyncio.wait_for(request_model(model), timeout=26)
                    diag["attempts"].append({"attempt": attempt, "model": model, "http": 200})
                    return result
                except (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError):
                    last_error = AIError("TIMEOUT", "Coffee AI took too long to respond. Please try again.")
                except httpx.RequestError:
                    last_error = AIError("NETWORK", "Could not reach Coffee AI. Please try again shortly.")
                except AIError as exc:
                    last_error = exc
                diag["attempts"].append({"attempt": attempt, "model": model,
                                         "http": last_error.status, "cause": last_error.category})
                logger.warning("Coffee AI attempt=%s model=%s cause=%s http=%s",
                               attempt, model if re.fullmatch(r"(?:models/)?gemini-[a-zA-Z0-9._-]+", model) else "INVALID_CONFIG",
                               last_error.category, last_error.status)
                if last_error.status == 404:
                    break
                if model_index == 0 and len(configured) > 1 and (last_error.status in transient or last_error.category in ("TIMEOUT", "NETWORK")):
                    # Try the available backup promptly instead of repeating a failing primary.
                    break
                if last_error.status not in transient and last_error.category not in ("TIMEOUT", "NETWORK"):
                    raise last_error
        raise last_error

    try:
        if not key:
            raise AIError("NOT_CONFIGURED", "Coffee AI is not configured. Add GEMINI_API_KEY once in the deployed app’s Streamlit Secrets.")

        answer, content, used_model, _, truncated = asyncio.run(request_with_recovery())
        st.session_state.ai_turns.append({"question": question, "answer": answer,
                                          "provider_content": content})
        st.session_state.ai_turns = st.session_state.ai_turns[-20:]
        diag.update(status="ONLINE", http=200, model=used_model,
                    cause="SUCCESS", truncated=truncated)
        return answer, None, truncated

    except AIError as exc:
        diag.update(cause=exc.category, http=exc.status)
        if exc.status in (500, 502, 503, 504):
            return None, "The Coffee AI provider is temporarily unavailable after retrying. Please try again shortly.", False
        return None, str(exc), False
    except (httpx.ConnectTimeout, httpx.ReadTimeout, asyncio.TimeoutError, TimeoutError):
        diag["cause"] = "TIMEOUT"
        return None, "Coffee AI took too long to respond. Please try again.", False
    except httpx.RequestError:
        diag["cause"] = "NETWORK"
        return None, "Could not reach Coffee AI. Please check the connection and try again.", False
    except ImportError:
        diag["cause"] = "MISSING_DEPENDENCY"
        return None, "Coffee AI is temporarily unavailable. Please try again later.", False
    except (ValueError, KeyError, TypeError, AttributeError):
        diag["cause"] = "INVALID_RESPONSE"
        return None, "Coffee AI received an unexpected response. Please try again.", False
    except Exception as exc:
        # Do not log exception text, response bodies, headers or credentials.
        diag["cause"] = type(exc).__name__
        return None, "Coffee AI is temporarily unavailable. Please try again later.", False
    finally:
        logger.info("Coffee AI status=%s cause=%s http=%s attempts=%s",
                    diag["status"], diag.get("cause"), diag.get("http"),
                    len(diag["attempts"]))
        if diag["status"] != "ONLINE":
            logger.warning("Coffee AI failed: cause=%s http=%s", diag.get("cause"), diag.get("http"))
        diag["latency_seconds"] = round(time.monotonic() - start, 2)
        st.session_state.ai_diagnostic = diag

# ========================= STYLE =========================
STYLE = """
<style>
:root{--bg:#07110d;--surface:#0e1b15;--surface2:#13251c;--line:#234232;--text:#f5f3ea;--muted:#a7b5aa;--accent:#74c365;--accent2:#a7e38b;--shadow:0 18px 45px rgba(0,0,0,.18)}
[data-theme="light"]{--bg:#f5f3e9;--surface:#fffdf7;--surface2:#edf3e8;--line:#d9e2d5;--text:#17241c;--muted:#68776d;--accent:#3f7e4d;--accent2:#6cae6f;--shadow:0 16px 40px rgba(38,63,45,.10)}
.stApp{background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
[data-testid="stHeader"]{background:transparent!important}
.block-container{max-width:none!important;width:100%!important;padding:2rem clamp(1rem,3vw,3.5rem) 5rem!important}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp li{color:var(--text)}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--line);width:255px!important}
[data-testid="stSidebarCollapsedControl"]{background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:12px!important}
.brand{font-size:25px;font-weight:800;letter-spacing:-.7px;margin:10px 0 4px}.brand-sub{font-size:12px;color:var(--muted);margin-bottom:28px}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--accent2);margin-bottom:10px}
.title{font-size:clamp(36px,5vw,66px);font-weight:750;letter-spacing:-.055em;line-height:1;margin:8px 0 24px}.title em{font-style:normal;color:var(--accent)}
.hero-card{background:linear-gradient(135deg,var(--surface),var(--surface2));border:1px solid var(--line);border-radius:24px;padding:clamp(24px,5vw,58px);box-shadow:var(--shadow);position:relative;overflow:hidden;margin-bottom:24px}
.hero-card:after{content:'☘';position:absolute;right:5%;top:5%;font-size:180px;opacity:.055;transform:rotate(-12deg)}
.hero-card h1{font-size:clamp(42px,6vw,76px);letter-spacing:-.06em;line-height:.98;margin:8px 0 18px!important}.hero-card p{max-width:650px;color:var(--muted);font-size:16px;line-height:1.7}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0 32px}.stat{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:var(--shadow)}.stat strong{display:block;font-size:28px;margin-bottom:6px}.stat small{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}
.stButton button,.stDownloadButton button{border:1px solid var(--line)!important;border-radius:12px!important;background:var(--surface2)!important;color:var(--text)!important;min-height:45px;box-shadow:none!important}.stButton button:hover,.stDownloadButton button:hover{border-color:var(--accent)!important;transform:translateY(-1px)}.stButton button[kind="primary"]{background:var(--accent)!important;border-color:var(--accent)!important;color:white!important}
[data-testid="stFileUploaderDropzone"]{background:var(--surface)!important;border:1.5px dashed var(--line)!important;border-radius:18px!important;min-height:180px;padding:28px}[data-testid="stFileUploaderDropzone"] *{color:var(--text)!important}
.scanner{background:var(--surface);padding:14px;border:1px solid var(--line);border-radius:20px;margin:14px 0;box-shadow:var(--shadow)}.scanner img{display:block;width:100%;max-height:430px;object-fit:contain;border-radius:14px}
.pipeline{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:16px 0}.pipeline>div{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px;font-size:12px}.pipeline strong{display:block;margin-top:8px;color:var(--accent2);font-size:11px}.pipeline .err{color:#e78373}.pipeline .busy{animation:pulse 1s infinite}@keyframes pulse{50%{opacity:.45}}
.result{background:var(--surface);border:1px solid var(--line);border-radius:22px;padding:28px;margin-top:28px;box-shadow:var(--shadow)}.result h2{font-size:clamp(34px,4vw,54px);letter-spacing:-.04em;margin:.3rem 0}.result .score{font-size:44px;color:var(--accent);font-weight:750}
.record{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:20px;margin:12px 0}.strip{border-top:1px solid var(--line);padding:16px 0;color:var(--muted);font-size:11px;line-height:1.7}
div[data-baseweb="select"]>div,div[data-baseweb="input"],.stTextInput input,.stTextArea textarea{background:var(--surface)!important;color:var(--text)!important;border-color:var(--line)!important;border-radius:12px!important}[data-baseweb="popover"],[role="listbox"]{background:var(--surface)!important;color:var(--text)!important}
[data-testid="stExpander"],[data-testid="stAlert"]{background:var(--surface);border:1px solid var(--line);border-radius:14px!important}
[data-testid="stChatMessage"]{background:var(--surface)!important;border:1px solid var(--line);border-radius:16px!important;padding:18px!important;margin:10px 0}[data-testid="stBottom"],[data-testid="stBottomBlockContainer"]{background:var(--bg)!important}[data-testid="stChatInput"]{border:1px solid var(--line)!important;border-radius:14px!important;background:var(--surface)!important}[data-testid="stChatInput"] textarea{background:var(--surface)!important;color:var(--text)!important}
.settings-card{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:22px;margin:12px 0 22px;box-shadow:var(--shadow)}
.feature-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0 28px}.feature{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:22px;min-height:150px}.feature .icon{font-size:24px;margin-bottom:20px}.feature h3{font-size:17px;margin:0 0 8px!important}.feature p{font-size:13px;color:var(--muted);line-height:1.6;margin:0}
.section-head{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:34px 0 14px}.section-head h2{font-size:26px;letter-spacing:-.03em;margin:0!important}.section-head span{font-size:12px;color:var(--muted)}
.workflow{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:18px;overflow:hidden;margin:16px 0 30px}.workflow>div{background:var(--surface);padding:22px}.workflow b{display:block;color:var(--accent);font-size:11px;letter-spacing:.1em;margin-bottom:10px}.workflow strong{display:block;font-size:15px;margin-bottom:6px}.workflow span{font-size:12px;color:var(--muted);line-height:1.5}
.tip{background:linear-gradient(120deg,var(--surface2),var(--surface));border:1px solid var(--line);border-radius:18px;padding:22px;margin:18px 0}.tip strong{font-size:14px}.tip p{color:var(--muted);font-size:13px;margin:7px 0 0;line-height:1.6}
@media(max-width:850px){.stats,.feature-grid{grid-template-columns:repeat(2,1fr)}.pipeline,.workflow{grid-template-columns:1fr}.block-container{padding:1.3rem 1rem 4rem!important}}
</style>"""

INTERACTIVE_STYLE = r"""
<style>
html{scroll-behavior:smooth}
.stApp{background:radial-gradient(circle at 88% 8%,color-mix(in srgb,var(--accent) 12%,transparent) 0,transparent 24rem),radial-gradient(circle at 8% 90%,color-mix(in srgb,var(--accent2) 7%,transparent) 0,transparent 26rem),var(--bg)}
.block-container{animation:pageIn .42s ease both}@keyframes pageIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.hero-card{min-height:320px;display:flex;flex-direction:column;justify-content:center}
.hero-card:before{content:"";position:absolute;width:360px;height:360px;border-radius:50%;right:-110px;bottom:-170px;background:radial-gradient(circle,color-mix(in srgb,var(--accent) 24%,transparent),transparent 68%);pointer-events:none}
.hero-card:after{font-size:230px;opacity:.065;transition:transform .6s ease,opacity .6s ease}.hero-card:hover:after{transform:rotate(-6deg) scale(1.04);opacity:.09}
.hero-badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}.hero-badge{border:1px solid var(--line);background:color-mix(in srgb,var(--surface) 80%,transparent);padding:8px 11px;border-radius:999px;font-size:11px;color:var(--muted)}
.stat,.feature,.workflow>div,.tip,.result,.settings-card{transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}
.stat:hover,.feature:hover,.workflow>div:hover,.tip:hover{transform:translateY(-4px);border-color:var(--accent);box-shadow:0 20px 45px rgba(0,0,0,.14)}
.feature{position:relative;overflow:hidden}.feature .icon{width:46px;height:46px;border-radius:14px;display:grid;place-items:center;background:var(--surface2);border:1px solid var(--line);transition:transform .25s ease}.feature:hover .icon{transform:rotate(-5deg) scale(1.07)}
.section-head h2:after{content:"";display:block;width:34px;height:3px;border-radius:3px;background:var(--accent);margin-top:10px}
.workflow b{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;background:var(--surface2);border:1px solid var(--line)}
.stButton button{transition:transform .18s ease,filter .18s ease!important}.stButton button:hover{transform:translateY(-2px)!important;filter:brightness(1.06)}
[data-testid="stFileUploaderDropzone"]{transition:border-color .2s ease,background .2s ease,transform .2s ease}[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent)!important;background:var(--surface2)!important;transform:translateY(-2px)}
.scanner{position:relative;overflow:hidden}.scanner:after{content:"";position:absolute;left:8%;right:8%;height:2px;top:12%;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:.7;animation:scanline 3.2s ease-in-out infinite;pointer-events:none}@keyframes scanline{0%,100%{top:12%;opacity:.2}50%{top:86%;opacity:.8}}
.result{position:relative;overflow:hidden}.result:before{content:"AI CONSENSUS";position:absolute;right:22px;top:18px;font-size:9px;letter-spacing:.14em;color:var(--muted)}
[data-testid="stSidebar"]{box-shadow:18px 0 50px rgba(0,0,0,.08)}
@media(max-width:850px){.hero-card{min-height:260px}.hero-card:after{font-size:150px}.hero-badges{max-width:75%}}
</style>
"""

SHAPE_STYLE = r"""
<style>
/* === ROCOLE SHAPE + INTERACTION UPGRADE (presentation only) === */
.block-container{max-width:1480px!important;padding-top:1.35rem!important}

/* organic hero */
.hero-card{
  position:relative!important; isolation:isolate; overflow:hidden!important;
  border-radius:34px 34px 90px 34px!important;
  padding:48px 50px!important;
  border:1px solid var(--line)!important;
}
.hero-card::before{
  content:""!important; position:absolute!important; z-index:-1!important;
  width:440px!important;height:440px!important;right:-120px!important;top:-190px!important;
  border-radius:44% 56% 68% 32% / 46% 37% 63% 54%!important;
  background:linear-gradient(145deg,color-mix(in srgb,var(--accent) 24%,transparent),transparent)!important;
  animation:blobMorph 9s ease-in-out infinite alternate!important;
}
.hero-card::after{
  content:""!important; position:absolute!important; z-index:-1!important;
  width:190px!important;height:190px!important;right:17%!important;bottom:-90px!important;
  border:1px solid color-mix(in srgb,var(--accent) 35%,transparent)!important;
  border-radius:62% 38% 48% 52% / 43% 57% 43% 57%!important;
  opacity:.55!important; animation:floatLeaf 6s ease-in-out infinite!important;
}
@keyframes blobMorph{
  0%{border-radius:44% 56% 68% 32% / 46% 37% 63% 54%;transform:rotate(0deg)}
  100%{border-radius:63% 37% 41% 59% / 58% 62% 38% 42%;transform:rotate(10deg)}
}
@keyframes floatLeaf{0%,100%{transform:translateY(0) rotate(-7deg)}50%{transform:translateY(-14px) rotate(5deg)}}

/* decorative orbit dots */
.hero-badges{position:relative;z-index:2}
.hero-badge{transition:.22s ease!important}
.hero-badge:hover{transform:translateY(-3px) scale(1.02);border-color:var(--accent)!important}

/* varied shapes instead of identical boxes */
.stats{gap:13px!important}
.stat:nth-child(1){border-radius:26px 10px 26px 10px!important}
.stat:nth-child(2){border-radius:10px 26px 10px 26px!important}
.stat:nth-child(3){border-radius:26px 26px 10px 10px!important}
.stat:nth-child(4){border-radius:10px 10px 26px 26px!important}
.stat strong{transition:transform .25s ease!important;display:inline-block}
.stat:hover strong{transform:scale(1.08)}

.feature-grid{gap:18px!important}
.feature{
  min-height:210px!important;padding:24px!important;
  border-radius:28px 12px 28px 12px!important;
  transform-origin:center bottom;
}
.feature:nth-child(2){border-radius:12px 28px 12px 28px!important}
.feature:nth-child(3){border-radius:28px 28px 12px 28px!important}
.feature:hover{transform:translateY(-7px) rotate(-.35deg)!important}
.feature:nth-child(2):hover{transform:translateY(-7px) rotate(.35deg)!important}
.feature::before{
 content:"";position:absolute;width:95px;height:95px;right:-28px;top:-30px;
 border-radius:58% 42% 70% 30% / 40% 58% 42% 60%;
 border:1px solid color-mix(in srgb,var(--accent) 22%,transparent);
 transition:transform .45s ease;
}
.feature:hover::before{transform:rotate(28deg) scale(1.15)}
.feature .icon{border-radius:18px 8px 18px 8px!important}

/* workflow becomes a connected path */
.workflow{position:relative!important;gap:14px!important}
.workflow::before{
 content:"";position:absolute;left:10%;right:10%;top:50%;height:1px;
 background:linear-gradient(90deg,transparent,var(--line),var(--accent),var(--line),transparent);
 z-index:0;opacity:.8
}
.workflow>div{position:relative!important;z-index:1!important;border-radius:24px 10px 24px 10px!important}
.workflow>div:nth-child(2){border-radius:10px 24px 10px 24px!important}
.workflow>div:hover b{transform:rotate(8deg) scale(1.08)}
.workflow b{transition:.22s ease}

/* upload / scan workspace */
[data-testid="stFileUploaderDropzone"]{
 border-radius:30px 12px 30px 12px!important;
 background-image:radial-gradient(circle at 90% 15%,color-mix(in srgb,var(--accent) 9%,transparent),transparent 30%)!important;
}
[data-testid="stFileUploaderDropzone"]:hover{transform:translateY(-3px) scale(1.003)!important}
.scanner{border-radius:30px 12px 30px 12px!important}

/* buttons feel tactile */
.stButton>button{border-radius:14px!important;min-height:44px!important}
.stButton>button:active{transform:translateY(1px) scale(.985)!important}

/* result emphasis */
.result{border-radius:30px 12px 30px 12px!important}
.result:hover{border-color:color-mix(in srgb,var(--accent) 65%,var(--line))!important}
.result::after{
 content:"";position:absolute;width:150px;height:150px;right:-65px;bottom:-75px;
 border-radius:42% 58% 65% 35% / 55% 42% 58% 45%;
 background:color-mix(in srgb,var(--accent) 7%,transparent);
 pointer-events:none;
}

/* tips/settings/history get character without clutter */
.tip{border-radius:22px 8px 22px 8px!important}
.tip:nth-of-type(even){border-radius:8px 22px 8px 22px!important}
.settings-card{border-radius:28px 12px 28px 12px!important}

/* sidebar active/hover motion */
[data-testid="stSidebar"] [role="radiogroup"] label{
 border-radius:14px 7px 14px 7px!important;
 transition:transform .2s ease,background .2s ease!important
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{transform:translateX(5px)!important}

/* focus and accessibility */
button:focus-visible,input:focus-visible,textarea:focus-visible,[tabindex]:focus-visible{
 outline:2px solid var(--accent)!important;outline-offset:2px!important
}
@media(prefers-reduced-motion:reduce){
 *,*::before,*::after{animation:none!important;transition:none!important}
}
@media(max-width:900px){
 .hero-card{padding:32px 26px!important;border-radius:26px 26px 55px 26px!important}
 .workflow::before{display:none}
 .feature{min-height:auto!important}
}
</style>
"""

COOL_STYLE = r"""
<style>
/* ROCOLE COOL PRODUCT LAYER — UI ONLY */
:root{--ease:cubic-bezier(.2,.8,.2,1)}
.block-container{max-width:1500px!important}

/* subtle ambient grid/noise-like depth */
.stApp:before{
 content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;
 background-image:linear-gradient(color-mix(in srgb,var(--line) 18%,transparent) 1px,transparent 1px),
 linear-gradient(90deg,color-mix(in srgb,var(--line) 18%,transparent) 1px,transparent 1px);
 background-size:42px 42px;mask-image:linear-gradient(to bottom,rgba(0,0,0,.28),transparent 55%);
}

/* hero gets product-style visual depth */
.hero-card{box-shadow:0 28px 80px rgba(0,0,0,.12)!important}
.hero-card h1{max-width:820px;letter-spacing:-.045em!important}
.hero-card p{max-width:720px}
.hero-card .eyebrow:before{
 content:"";display:inline-block;width:7px;height:7px;border-radius:50%;
 background:var(--accent);margin-right:8px;box-shadow:0 0 0 5px color-mix(in srgb,var(--accent) 13%,transparent);
 animation:livePulse 2s ease-in-out infinite
}
@keyframes livePulse{50%{box-shadow:0 0 0 9px transparent}}

/* glassy compact metrics */
.stat{backdrop-filter:blur(12px);position:relative;overflow:hidden}
.stat:after{content:"";position:absolute;inset:0;background:linear-gradient(120deg,transparent 25%,color-mix(in srgb,var(--accent) 8%,transparent),transparent 72%);transform:translateX(-120%);transition:transform .7s var(--ease)}
.stat:hover:after{transform:translateX(120%)}

/* interactive feature cards */
.feature{cursor:default}
.feature h3{transition:transform .25s var(--ease)}
.feature:hover h3{transform:translateX(4px)}
.feature p{transition:opacity .25s ease}.feature:hover p{opacity:.9}

/* native controls feel less Streamlit */
div[data-baseweb="select"]>div,
div[data-baseweb="input"]>div,
.stTextArea textarea{
 border-radius:14px!important;border-color:var(--line)!important;
 transition:border-color .2s ease,box-shadow .2s ease!important
}
div[data-baseweb="select"]>div:focus-within,
div[data-baseweb="input"]>div:focus-within,
.stTextArea textarea:focus{
 border-color:var(--accent)!important;
 box-shadow:0 0 0 4px color-mix(in srgb,var(--accent) 10%,transparent)!important
}

/* segmented radios */
div[role="radiogroup"]{gap:8px!important}
div[role="radiogroup"] label{
 border:1px solid var(--line);border-radius:14px!important;padding:9px 13px!important;
 transition:.2s var(--ease)
}
div[role="radiogroup"] label:hover{border-color:var(--accent);transform:translateY(-2px)}

/* uploader as interactive analysis zone */
[data-testid="stFileUploaderDropzone"]{
 min-height:160px!important;position:relative!important;overflow:hidden!important;
 border-width:1px!important
}
[data-testid="stFileUploaderDropzone"]:before{
 content:"DROP COFFEE LEAF";position:absolute;right:18px;top:14px;
 font-size:9px;letter-spacing:.16em;color:var(--muted);opacity:.65
}
[data-testid="stFileUploaderDropzone"]:after{
 content:"";position:absolute;width:140px;height:140px;right:-65px;bottom:-75px;
 border:1px solid color-mix(in srgb,var(--accent) 28%,transparent);
 border-radius:62% 38% 70% 30% / 44% 58% 42% 56%;pointer-events:none;
 animation:uploaderLeaf 7s ease-in-out infinite alternate
}
@keyframes uploaderLeaf{to{transform:rotate(20deg) translate(-8px,-6px)}}

/* progress */
[data-testid="stProgress"]>div>div>div{border-radius:999px!important}
[data-testid="stProgress"]{animation:progressIn .35s ease both}
@keyframes progressIn{from{opacity:0;transform:scaleX(.96)}}

/* result becomes focal panel */
.result{
 box-shadow:0 24px 70px rgba(0,0,0,.13)!important;
 animation:resultPop .45s var(--ease) both
}
@keyframes resultPop{from{opacity:0;transform:translateY(12px) scale(.985)}to{opacity:1;transform:none}}
.result strong{letter-spacing:-.025em}

/* status dots / chips */
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent);margin-right:7px;box-shadow:0 0 0 5px color-mix(in srgb,var(--accent) 11%,transparent);animation:livePulse 2s infinite}

/* chat polish */
[data-testid="stChatMessage"]{
 border:1px solid var(--line)!important;border-radius:20px 8px 20px 20px!important;
 background:color-mix(in srgb,var(--surface) 94%,transparent)!important;
 animation:chatIn .28s var(--ease) both
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){border-radius:8px 20px 20px 20px!important}
@keyframes chatIn{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
[data-testid="stChatInput"]{border-radius:18px!important}

/* archive/history hover */
[data-testid="stExpander"]{
 border:1px solid var(--line)!important;border-radius:18px!important;
 overflow:hidden;transition:transform .2s var(--ease),border-color .2s ease
}
[data-testid="stExpander"]:hover{transform:translateY(-2px);border-color:color-mix(in srgb,var(--accent) 55%,var(--line))!important}

/* sidebar brand feel */
[data-testid="stSidebar"]{border-right:1px solid var(--line)}
[data-testid="stSidebar"] hr{opacity:.45}

/* custom scrollbar */
*{scrollbar-width:thin;scrollbar-color:color-mix(in srgb,var(--accent) 50%,var(--line)) transparent}

/* mobile */
@media(max-width:760px){
 .block-container{padding-left:1rem!important;padding-right:1rem!important}
 .hero-card{padding:28px 22px!important}
 .hero-card h1{font-size:clamp(2rem,11vw,3.2rem)!important}
 .stats{grid-template-columns:repeat(2,1fr)!important}
}
@media(prefers-reduced-motion:reduce){.hero-card .eyebrow:before,.status-dot,[data-testid="stFileUploaderDropzone"]:after{animation:none!important}}
</style>
"""



# ========================= FINAL COOL DESIGN OVERRIDE — UI ONLY =========================
FINAL_COOL_STYLE = r"""
<style>
/* Premium botanical editorial redesign. Presentation only. */
:root{--bg:#08110d;--surface:#0d1812;--surface2:#132219;--line:rgba(225,239,221,.13);--text:#f4f0df;--muted:#9cac9f;--accent:#b9ef78;--accent2:#75c98a;--shadow:0 30px 80px rgba(0,0,0,.28)}
.stApp{background:
 radial-gradient(circle at 82% 5%,rgba(117,201,138,.13),transparent 28rem),
 radial-gradient(circle at 3% 82%,rgba(185,239,120,.07),transparent 25rem),#08110d!important}
.block-container{max-width:1440px!important;padding:2rem clamp(1.1rem,4vw,4.5rem) 6rem!important}
[data-testid="stSidebar"]{background:rgba(8,17,13,.92)!important;backdrop-filter:blur(24px);border-right:1px solid var(--line)!important}
.brand{font-size:28px!important;letter-spacing:-1.2px!important}.brand-sub{letter-spacing:.08em;text-transform:uppercase;font-size:9px!important}
.eyebrow{display:inline-flex;align-items:center;gap:9px;font-size:10px!important;letter-spacing:.19em!important;color:#b9ef78!important}
.eyebrow:before{content:"";width:22px;height:1px;background:#b9ef78}
.title{font-family:Georgia,"Times New Roman",serif!important;font-weight:500!important;font-size:clamp(48px,7vw,96px)!important;line-height:.92!important;letter-spacing:-.055em!important;max-width:950px}
.hero-card{min-height:520px!important;padding:clamp(38px,6vw,82px)!important;border-radius:42px 42px 130px 42px!important;background:linear-gradient(135deg,rgba(18,35,25,.94),rgba(9,20,14,.96))!important;border:1px solid rgba(185,239,120,.16)!important;box-shadow:0 45px 120px rgba(0,0,0,.3)!important}
.hero-card:before{content:""!important;position:absolute;right:-7%;top:-35%;width:580px!important;height:580px!important;border-radius:38% 62% 66% 34% / 47% 37% 63% 53%!important;background:linear-gradient(145deg,rgba(185,239,120,.22),rgba(117,201,138,.03))!important;filter:blur(.2px);animation:organic 10s ease-in-out infinite alternate!important}
.hero-card:after{content:"✦"!important;right:8%!important;top:auto!important;bottom:8%!important;font-size:210px!important;color:#b9ef78!important;opacity:.08!important;transform:none!important}
@keyframes organic{to{transform:rotate(13deg) scale(1.08);border-radius:61% 39% 43% 57% / 58% 64% 36% 42%}}
.hero-card h1{font-family:Georgia,"Times New Roman",serif!important;font-weight:500!important;font-size:clamp(56px,7.2vw,105px)!important;line-height:.88!important;letter-spacing:-.065em!important;max-width:900px;margin-top:18px!important}
.hero-card p{font-size:17px!important;line-height:1.75!important;max-width:590px!important;color:#aebdb0!important}
.hero-badges{margin-top:30px!important;gap:10px!important}.hero-badge{background:rgba(255,255,255,.035)!important;border:1px solid rgba(255,255,255,.11)!important;border-radius:999px!important;padding:9px 14px!important;color:#c2cdc4!important}
.stats{gap:10px!important;margin-top:12px!important}.stat{background:transparent!important;border-width:0 0 1px!important;border-radius:0!important;padding:26px 8px 22px!important;box-shadow:none!important}.stat strong{font-family:Georgia,serif;font-size:42px!important;font-weight:500!important;color:#f4f0df}.stat small{font-size:9px!important;letter-spacing:.15em!important}.stat:hover{transform:translateY(-4px)!important;border-color:#b9ef78!important}
.section-head{margin-top:70px!important}.section-head h2{font-family:Georgia,serif!important;font-size:clamp(30px,4vw,48px)!important;font-weight:500!important}.section-head h2:after{width:56px!important;height:1px!important}
.feature-grid{grid-template-columns:1.15fr .85fr 1fr!important;gap:16px!important}.feature{min-height:250px!important;background:linear-gradient(145deg,rgba(19,34,25,.9),rgba(11,24,17,.9))!important;border:1px solid var(--line)!important;border-radius:32px!important;padding:30px!important}.feature:nth-child(1){border-radius:32px 90px 32px 32px!important}.feature:nth-child(2){transform:translateY(30px);border-radius:70px 32px 32px 32px!important}.feature:nth-child(3){border-radius:32px 32px 90px 32px!important}.feature:hover{transform:translateY(-8px)!important;border-color:rgba(185,239,120,.48)!important}.feature:nth-child(2):hover{transform:translateY(22px)!important}.feature .icon{width:54px!important;height:54px!important;border-radius:50%!important;background:rgba(185,239,120,.08)!important;border:1px solid rgba(185,239,120,.18)!important}.feature h3{font-family:Georgia,serif!important;font-size:22px!important;font-weight:500!important;margin-top:26px!important}.feature p{font-size:13px!important;line-height:1.75!important}
.workflow{margin-top:34px!important;background:transparent!important;border:0!important;gap:12px!important;overflow:visible!important}.workflow:before{display:none!important}.workflow>div{background:rgba(255,255,255,.025)!important;border:1px solid var(--line)!important;border-radius:22px!important;padding:26px!important}.workflow b{border-radius:999px!important;width:auto!important;height:auto!important;display:inline-block!important;padding:7px 10px!important;font-size:9px!important}.workflow strong{font-family:Georgia,serif;font-size:18px!important;margin-top:18px!important}
.stButton button,.stDownloadButton button{border-radius:999px!important;padding-left:22px!important;padding-right:22px!important;min-height:48px!important;font-weight:650!important;letter-spacing:.01em!important;background:rgba(255,255,255,.035)!important}.stButton button[kind="primary"]{background:#b9ef78!important;color:#0a160e!important;border-color:#b9ef78!important}.stButton button:hover,.stDownloadButton button:hover{border-color:#b9ef78!important;box-shadow:0 10px 35px rgba(185,239,120,.10)!important}
[data-testid="stFileUploaderDropzone"]{min-height:220px!important;border-radius:32px!important;background:linear-gradient(145deg,rgba(18,34,24,.72),rgba(10,22,15,.72))!important;border:1px dashed rgba(185,239,120,.34)!important}.scanner{border-radius:32px!important;padding:9px!important;background:#0c1811!important}.scanner img{border-radius:25px!important}
.pipeline{grid-template-columns:repeat(4,1fr)!important}.pipeline>div{border-radius:18px!important;background:rgba(255,255,255,.025)!important;text-align:center!important}
.result{border-radius:36px!important;padding:40px!important;background:linear-gradient(145deg,rgba(21,39,28,.96),rgba(10,22,15,.96))!important;border:1px solid rgba(185,239,120,.22)!important}.result h2{font-family:Georgia,serif!important;font-weight:500!important;font-size:clamp(40px,6vw,72px)!important}.result .score{font-family:Georgia,serif!important;font-size:64px!important;color:#b9ef78!important}
.tip,.settings-card,.record,[data-testid="stExpander"],[data-testid="stChatMessage"]{background:rgba(16,29,21,.78)!important;border:1px solid var(--line)!important;box-shadow:none!important}.tip,.settings-card,.record{border-radius:24px!important}
[data-testid="stChatMessage"]{border-radius:24px 24px 24px 7px!important}.stChatInputContainer,[data-testid="stChatInput"]{border-radius:999px!important}
div[role="radiogroup"] label{border-radius:999px!important;background:rgba(255,255,255,.025)!important}
hr{border-color:var(--line)!important}
@media(max-width:900px){.hero-card{min-height:430px!important;border-radius:30px 30px 70px 30px!important}.feature-grid{grid-template-columns:1fr!important}.feature:nth-child(2){transform:none!important}.pipeline{grid-template-columns:1fr 1fr!important}.stats{grid-template-columns:1fr 1fr!important}}
</style>
"""

# ========================= SESSION STATE =========================
def initialize_state():
    defaults = {"lab_page": PAGES[0], "path_overrides": {}, "mapping_overrides": {},
                "model_diagnostics": [], "analysis_result": None, "lab_archive": [],
                "ai_turns": [], "ai_display": [], "provider_models": [], "ai_diagnostic": {},
                "specimen": None, "specimen_hash": None, "scan_task": None,
                "scan_counter": 0, "pending_question": None, "upload_revision": 0,
                "ui_theme": "Dark", "ui_language": "English", "scan_images": {}, "history_page": 0}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def go(page):
    st.session_state.lab_page = page
    st.session_state.nav_page = page

def markup(value):
    st.markdown(value, unsafe_allow_html=True)

def page_heading(kicker, title):
    markup(f'<div class="eyebrow">{esc(kicker)}</div><div class="title">{title}</div>')

def render_pipeline(states):
    cells = []
    for name, state in states.items():
        style = "err" if state == "ERROR" else "busy" if state in ("LOADING", "INFERENCE") else ""
        cells.append(f'<div>{esc(name)}<strong class="{style}">{esc(state)}</strong></div>')
    markup('<div class="pipeline">' + ''.join(cells) + '</div>')

def render_result(result, key):
    markup(f'<div class="result"><div class="eyebrow">CONSENSUS / {len(result["predictions"]):02d} OF 04 MODELS</div>'
           f'<h2>{esc(result["label"].upper())}</h2><div class="score">{result["confidence"]:.1f}%</div></div>')
    st.caption(f'MODEL AGREEMENT {result["votes"]}/{len(result["predictions"])} · mean probability ensemble')
    st.progress(result["agreement"] / 100)
    if result["failed_models"]:
        st.warning("Partial ensemble. Failed: " + ", ".join(result["failed_models"]))
    if result["votes"] != len(result["predictions"]):
        st.warning("Models disagree. Compare the individual outputs below.")
    for p in result["predictions"]:
        st.write(f'**{p["model"]}** · {p["label"]} · {p["confidence"]:.1f}%')
        if p["class_source"].startswith("UNVERIFIED"):
            st.warning(f'{p["model"]}: label order is inherited from the original app and needs training verification. Ask the operator to provide the verified training classes.json.')
    with st.expander("FARMER MODE", expanded=True):
        st.write(f'The models predicted **{result["label"]}**. This is not a confirmed agricultural diagnosis.')
        st.write("Inspect several leaves and both leaf surfaces, note whether symptoms affect old or new growth, "
                 "and check watering and drainage. For a nutrient prediction, confirm with soil or leaf testing "
                 "before changing fertilizer. Coffee AI can explain the predicted class in your language.")
    with st.expander("CLASS PROBABILITIES"):
        st.table([{"Class": c, "Mean probability (%)": round(p * 100, 2)}
                  for c, p in zip(result["classes"], result["probabilities"])])
    st.write(guidance(result["label"]))
    indicators = uncertainty(result)
    st.caption(f"Probability margin: {indicators['top_two_margin']:.3f} · normalized entropy: {indicators['normalized_entropy']:.3f}. Descriptive indicators only; probabilities are not calibrated risk estimates.")
    st.caption("Severity: unavailable without a validated severity model. Grad-CAM: unavailable pending trained-checkpoint validation.")
    image_data = st.session_state.scan_images.get(result.get("record_id", str(result.get("scan_id"))))
    if image_data:
        st.image(image_data, caption="Original scan image", width=320)
    st.download_button("DOWNLOAD ANALYSIS / PDF", pdf_report(result, image_data), "rocole_scan.pdf", "application/pdf", key=key+"_pdf")
    report = json.dumps(result, indent=2)
    st.download_button("DOWNLOAD ANALYSIS / JSON", report, "rocole_scan.json", "application/json", key=key)

UI_TEXT = {
"English":{"home":"Home","analyze":"Analyze Leaf","ai":"Coffee AI","history":"History","settings":"Settings","coffee_intel":"COFFEE LEAF INTELLIGENCE","hero":"Understand your coffee leaves.","hero_sub":"Analyze Robusta coffee leaves with four deep-learning models for disease and nutrient-deficiency detection.","ask_ai":"Ask Coffee AI","models":"AI Models","disease":"Disease Classes","nutrient":"Nutrient Classes","scans":"Session Scans","home_note":"Choose Disease or Nutrient analysis explicitly, upload a leaf image, and review the four-model consensus.","appearance":"Appearance","language":"Website language","feedback":"Suggest an improvement","feedback_help":"Your suggestion can be sent directly to the RoCoLe owner by email.","name":"Name (optional)","email":"Your email (optional)","suggestion":"Your suggestion","send":"Send suggestion"},
"ಕನ್ನಡ":{"home":"ಮುಖಪುಟ","analyze":"ಎಲೆ ವಿಶ್ಲೇಷಣೆ","ai":"ಕಾಫಿ AI","history":"ಇತಿಹಾಸ","settings":"ಸೆಟ್ಟಿಂಗ್ಸ್","coffee_intel":"ಕಾಫಿ ಎಲೆ ಬುದ್ಧಿಮತ್ತೆ","hero":"ನಿಮ್ಮ ಕಾಫಿ ಎಲೆಗಳನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ.","hero_sub":"ನಾಲ್ಕು ಡೀಪ್ ಲರ್ನಿಂಗ್ ಮಾದರಿಗಳ ಮೂಲಕ ರೋಬಸ್ಟಾ ಕಾಫಿ ಎಲೆಯ ರೋಗ ಮತ್ತು ಪೋಷಕಾಂಶ ಕೊರತೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ.","ask_ai":"ಕಾಫಿ AI ಕೇಳಿ","models":"AI ಮಾದರಿಗಳು","disease":"ರೋಗ ವರ್ಗಗಳು","nutrient":"ಪೋಷಕಾಂಶ ವರ್ಗಗಳು","scans":"ಸ್ಕ್ಯಾನ್‌ಗಳು","home_note":"ರೋಗ ಅಥವಾ ಪೋಷಕಾಂಶ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ, ಎಲೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಮಾದರಿಗಳ ಫಲಿತಾಂಶ ನೋಡಿ.","appearance":"ರೂಪ","language":"ವೆಬ್‌ಸೈಟ್ ಭಾಷೆ","feedback":"ಸುಧಾರಣೆ ಸೂಚಿಸಿ","feedback_help":"ನಿಮ್ಮ ಸಲಹೆಯನ್ನು RoCoLe ಮಾಲೀಕರಿಗೆ ಇಮೇಲ್ ಮೂಲಕ ಕಳುಹಿಸಬಹುದು.","name":"ಹೆಸರು (ಐಚ್ಛಿಕ)","email":"ನಿಮ್ಮ ಇಮೇಲ್ (ಐಚ್ಛಿಕ)","suggestion":"ನಿಮ್ಮ ಸಲಹೆ","send":"ಸಲಹೆ ಕಳುಹಿಸಿ"},
"हिंदी":{"home":"होम","analyze":"पत्ती विश्लेषण","ai":"कॉफी AI","history":"इतिहास","settings":"सेटिंग्स","coffee_intel":"कॉफी लीफ इंटेलिजेंस","hero":"अपनी कॉफी की पत्तियों को समझें।","hero_sub":"चार डीप-लर्निंग मॉडल से रोबस्टा कॉफी की पत्तियों में रोग और पोषक तत्वों की कमी का विश्लेषण करें।","ask_ai":"कॉफी AI से पूछें","models":"AI मॉडल","disease":"रोग वर्ग","nutrient":"पोषक वर्ग","scans":"स्कैन","home_note":"रोग या पोषक विश्लेषण चुनें, पत्ती की तस्वीर अपलोड करें और चार मॉडलों की सहमति देखें।","appearance":"दिखावट","language":"वेबसाइट भाषा","feedback":"सुधार सुझाएँ","feedback_help":"आपका सुझाव ईमेल द्वारा RoCoLe के मालिक तक भेजा जा सकता है।","name":"नाम (वैकल्पिक)","email":"आपका ईमेल (वैकल्पिक)","suggestion":"आपका सुझाव","send":"सुझाव भेजें"},
"தமிழ்":{"home":"முகப்பு","analyze":"இலை பகுப்பாய்வு","ai":"காபி AI","history":"வரலாறு","settings":"அமைப்புகள்","coffee_intel":"காபி இலை நுண்ணறிவு","hero":"உங்கள் காபி இலைகளைப் புரிந்துகொள்ளுங்கள்.","hero_sub":"நான்கு டீப்-லெர்னிங் மாதிரிகள் மூலம் ரோபஸ்டா காபி இலை நோய் மற்றும் ஊட்டச்சத்து குறைபாட்டை பகுப்பாய்வு செய்யுங்கள்.","ask_ai":"காபி AI-யிடம் கேளுங்கள்","models":"AI மாதிரிகள்","disease":"நோய் வகைகள்","nutrient":"ஊட்டச்சத்து வகைகள்","scans":"ஸ்கேன்கள்","home_note":"நோய் அல்லது ஊட்டச்சத்து பகுப்பாய்வைத் தேர்ந்தெடுத்து இலை படத்தை பதிவேற்றி முடிவுகளைப் பாருங்கள்.","appearance":"தோற்றம்","language":"இணையதள மொழி","feedback":"மேம்பாட்டை பரிந்துரைக்கவும்","feedback_help":"உங்கள் பரிந்துரையை RoCoLe உரிமையாளருக்கு மின்னஞ்சலில் அனுப்பலாம்.","name":"பெயர் (விருப்பம்)","email":"உங்கள் மின்னஞ்சல் (விருப்பம்)","suggestion":"உங்கள் பரிந்துரை","send":"பரிந்துரை அனுப்பு"},
"മലയാളം":{"home":"ഹോം","analyze":"ഇല വിശകലനം","ai":"കോഫി AI","history":"ചരിത്രം","settings":"ക്രമീകരണങ്ങൾ","coffee_intel":"കോഫി ലീഫ് ഇന്റലിജൻസ്","hero":"നിങ്ങളുടെ കോഫി ഇലകളെ മനസ്സിലാക്കൂ.","hero_sub":"നാല് ഡീപ് ലേണിംഗ് മോഡലുകൾ ഉപയോഗിച്ച് റോബസ്റ്റ കോഫി ഇലകളിലെ രോഗവും പോഷകക്കുറവും വിശകലനം ചെയ്യുക.","ask_ai":"കോഫി AI-യോട് ചോദിക്കുക","models":"AI മോഡലുകൾ","disease":"രോഗ വിഭാഗങ്ങൾ","nutrient":"പോഷക വിഭാഗങ്ങൾ","scans":"സ്കാനുകൾ","home_note":"രോഗമോ പോഷക വിശകലനമോ തിരഞ്ഞെടുക്കുക, ഇല ചിത്രം അപ്‌ലോഡ് ചെയ്ത് മോഡൽ ഫലം പരിശോധിക്കുക.","appearance":"രൂപം","language":"വെബ്സൈറ്റ് ഭാഷ","feedback":"മെച്ചപ്പെടുത്തൽ നിർദ്ദേശിക്കുക","feedback_help":"നിങ്ങളുടെ നിർദ്ദേശം RoCoLe ഉടമയ്ക്ക് ഇമെയിൽ വഴി അയയ്ക്കാം.","name":"പേര് (ഐച്ഛികം)","email":"നിങ്ങളുടെ ഇമെയിൽ (ഐച്ഛികം)","suggestion":"നിങ്ങളുടെ നിർദ്ദേശം","send":"നിർദ്ദേശം അയയ്ക്കുക"}
}

def submit_feedback(name, sender, suggestion):
    """Attempt storage and configured email independently; expose safe status only."""
    status = {"storage": "not_configured", "email": "not_configured"}
    st.session_state.feedback_status = status
    if not suggestion.strip():
        return False
    record_id = str(uuid.uuid4())
    db_path = setting("FEEDBACK_DB_PATH")
    if db_path and setting("FEEDBACK_DB_PERSISTENT", "false").lower() == "true":
        try:
            path = Path(db_path).expanduser()
            if not path.is_absolute() or not path.parent.is_dir():
                raise ValueError("Invalid directory")
            with closing(sqlite3.connect(str(path), timeout=10)) as conn:
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("CREATE TABLE IF NOT EXISTS feedback (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, name TEXT, email TEXT, message TEXT NOT NULL)")
                conn.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?)", (record_id, datetime.now(timezone.utc).isoformat(), name[:200], sender[:254], suggestion[:3000]))
                conn.commit()
            status["storage"] = "saved"
        except (OSError, sqlite3.Error, ValueError):
            status["storage"] = "storage_error"
    host, user, password, target = (setting(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "FEEDBACK_TO_EMAIL"))
    if any((host, user, password, target)):
        status["email"] = "configuration_error"
        if all((host, user, password, target)):
            try:
                port = int(setting("SMTP_PORT", "587"))
                if port not in (465, 587):
                    raise ValueError("Use TLS mail ports")
                msg = EmailMessage()
                msg["Subject"] = "RoCoLe improvement suggestion"
                msg["From"], msg["To"] = user, target
                msg.set_content(f"Feedback ID: {record_id}\nName: {name[:200]}\nReply email: {sender[:254]}\n\n{suggestion[:3000]}")
                context = ssl.create_default_context()
                smtp = smtplib.SMTP_SSL(host, port, timeout=15, context=context) if port == 465 else smtplib.SMTP(host, port, timeout=15)
                with smtp:
                    if port == 587:
                        smtp.starttls(context=context)
                    smtp.login(user, password)
                    status["email"] = "recipient_refused" if smtp.send_message(msg) else "accepted"
            except smtplib.SMTPAuthenticationError:
                status["email"] = "authentication_error"
            except smtplib.SMTPRecipientsRefused:
                status["email"] = "recipient_refused"
            except ssl.SSLError:
                status["email"] = "tls_error"
            except (socket.timeout, OSError, smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected):
                status["email"] = "connection_error"
            except (ValueError, smtplib.SMTPNotSupportedError):
                status["email"] = "configuration_error"
            except smtplib.SMTPException:
                status["email"] = "server_error"
    return status["storage"] == "saved" or status["email"] == "accepted"


# ========================= ROBUSTA MARKET =========================
@st.cache_data(ttl=600, show_spinner=False, max_entries=8)
def fetch_robusta_market_data(symbol="RM*0"):
    """Barchart OnDemand getQuote; only validated Robusta futures, no fallback prices.

    API key stays in the server-side POST body, never a URL or cached result.
    The provider must license ICE Robusta data for this application's display use.
    """
    try:
        key = setting("MARKET_API_KEY")
        if not key:
            logger.warning("Robusta market unavailable: API credential not configured")
            return None
        if not re.fullmatch(r"RM(?:\*[01]|[FHKNUX]\d{2})", symbol):
            raise ValueError("Unsupported Robusta symbol")
        with httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=False) as client:
            response = client.post("https://ondemand.websol.barchart.com/getQuote.json",
                                   data={"apikey": key, "symbols": symbol})
        if response.status_code != 200:
            logger.warning("Robusta provider HTTP status=%s", response.status_code)
            return None
        payload = response.json()
        if payload.get("status", {}).get("code") != 200:
            logger.warning("Robusta provider rejected quote request")
            return None
        rows = payload.get("results")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("Expected one quote")
        quote = rows[0]
        name, actual = str(quote.get("name", "")), str(quote.get("symbol", ""))
        # Never accidentally display Arabica (KC) or an unidentified contract.
        if "robusta" not in name.lower() or not re.fullmatch(r"RM[FHKNUX]\d{2}", actual):
            raise ValueError("Unexpected benchmark")
        price, change, percent = (float(quote[k]) for k in ("lastPrice", "netChange", "percentChange"))
        if not all(math.isfinite(v) for v in (price, change, percent)) or price <= 0:
            raise ValueError("Invalid quote values")
        trade_time = datetime.fromisoformat(str(quote["tradeTimestamp"]).replace("Z", "+00:00"))
        if trade_time.tzinfo is None:
            raise ValueError("Missing timestamp timezone")
        if (trade_time - datetime.now(timezone.utc)).total_seconds() > 300:
            raise ValueError("Future trade timestamp")
        mode = str(quote.get("mode", "")).upper()
        # Even an R feed is a cached snapshot here, never a streaming live trade.
        timing = {"I": "Delayed snapshot", "D": "End-of-day snapshot",
                  "R": "Latest available snapshot (real-time source feed)"}.get(mode, "Latest available snapshot")
        return {"price": price, "change": change, "percent": percent,
                "timestamp": trade_time.isoformat(), "symbol": actual, "name": name,
                "currency": "USD", "unit": "metric tonne", "timing": timing,
                "retrieved_at": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        logger.warning("Robusta fetch failed: type=%s", type(exc).__name__)
        return None


@st.fragment(run_every="10m")
def render_market_card():
    # Fragment refreshes don't rerun analysis or load models.
    with st.container(border=True):
        markup('<div class="eyebrow">ROBUSTA MARKET</div>')
        symbol = setting("MARKET_SYMBOL", "RM*0")
        if st.button("Refresh price", key="refresh_robusta_price"):
            fetch_robusta_market_data.clear(symbol)
        quote = fetch_robusta_market_data(symbol)
        if quote is None:
            st.info("Market data is temporarily unavailable.")
            return
        st.metric("Latest available price", f"${quote['price']:,.0f}",
                  f"{quote['change']:+,.0f} ({quote['percent']:+.2f}%)",
                  delta_color="off")
        st.caption("USD / metric tonne · Change versus previous close")
        st.write(f"Benchmark: {quote['name']} ({quote['symbol']}) · ICE Futures Europe")
        st.caption(f"Last market update: {quote['timestamp']} · {quote['timing']}")
        st.caption(f"Fetched: {quote['retrieved_at']} · Refreshes every 10 minutes while this page is open.")
        st.caption("Market open/closed status is not supplied by this endpoint. When trading is closed, "
                   "the latest available trade remains displayed with its original timestamp.")
        st.caption("Source: [Barchart OnDemand](https://www.barchart.com/ondemand) · "
                   "[ICE Robusta benchmark](https://www.ice.com/products/37089079) · "
                   "Futures benchmark, not a local farm-gate price.")


def settings_page():
    t=UI_TEXT[st.session_state.ui_language]
    page_heading(t["settings"].upper(), t["settings"])
    st.subheader(t["appearance"])
    st.radio(t["appearance"], ["Dark","Light"], horizontal=True, key="ui_theme", label_visibility="collapsed")
    st.subheader(t["language"])
    st.selectbox(t["language"], list(UI_TEXT), key="ui_language", label_visibility="collapsed")
    t=UI_TEXT[st.session_state.ui_language]
    st.divider(); st.subheader(t["feedback"]); st.caption("Share a suggestion or tell us what could work better.")
    with st.form("feedback_form", clear_on_submit=False):
        name=st.text_input(t["name"]); sender=st.text_input(t["email"]); suggestion=st.text_area(t["suggestion"], height=150, max_chars=3000)
        submitted=st.form_submit_button("Submit Feedback", type="primary")
    if submitted:
        if not suggestion.strip(): st.warning("Please enter a suggestion.")
        else:
            if submit_feedback(name.strip(), sender.strip(), suggestion.strip()):
                status = st.session_state.feedback_status
                st.success("Feedback saved." if status["storage"] == "saved" else "Mail server accepted your feedback; inbox delivery is not verified.")
            else:
                st.warning("Feedback could not be submitted.")
            status = st.session_state.feedback_status
            explanations = {"not_configured": "not configured", "saved": "saved to configured storage", "accepted": "accepted by mail server (inbox delivery unverified)", "authentication_error": "mail authentication failed", "recipient_refused": "mail recipient rejected", "connection_error": "mail server connection failed", "configuration_error": "mail configuration incomplete or unsupported", "tls_error": "secure mail connection failed", "server_error": "mail server rejected the operation", "storage_error": "storage write failed"}
            st.caption("Storage: " + explanations[status["storage"]] + ". Email: " + explanations[status["email"]] + ".")

# ========================= HOME =========================
def home_page():
    t = UI_TEXT[st.session_state.ui_language]
    markup(f"""
    <div class="hero-card">
      <div class="eyebrow">{t["coffee_intel"]} · ROBUSTA LEAF INTELLIGENCE</div>
      <h1>{t["hero"]}</h1>
      <p>{t["hero_sub"]}</p>
      <div class="hero-badges">
        <span class="hero-badge">◉ 4-model ensemble</span>
        <span class="hero-badge">🌿 Disease + nutrient analysis</span>
        <span class="hero-badge">✦ Multilingual Coffee AI</span>
      </div>
    </div>
    """)
    render_market_card()
    c1,c2,c3=st.columns([1,1,1.35])
    c1.button(t["analyze"], type="primary", on_click=go, args=(PAGES[1],), use_container_width=True)
    c2.button(t["ask_ai"], on_click=go, args=(PAGES[2],), use_container_width=True)
    with c3:
        st.caption("Choose a task → upload a clear leaf → review the four-model consensus.")
    markup('<div class="stats">'+''.join(f'<div class="stat"><strong>{a}</strong><small>{b}</small></div>' for a,b in [("04",t["models"]),("03",t["disease"]),("09",t["nutrient"]),(str(len(st.session_state.lab_archive)),t["scans"])])+'</div>')
    markup("""<div class="section-head"><h2>Explore RoCoLe</h2><span>Everything you need, without the clutter</span></div>
    <div class="feature-grid">
      <div class="feature"><div class="icon">◉</div><h3>Disease analysis</h3><p>Analyze Healthy, Rust and Red Spider Mite with the four-model ensemble.</p></div>
      <div class="feature"><div class="icon">🌱</div><h3>Nutrient analysis</h3><p>Check nine nutrient classes while keeping disease and deficiency tasks separate.</p></div>
      <div class="feature"><div class="icon">✦</div><h3>Coffee AI</h3><p>Continue from your scan or ask coffee questions naturally in your preferred language.</p></div>
    </div>""")
    left,right=st.columns([1.35,1],gap="large")
    with left:
        markup("""<div class="section-head"><h2>How analysis works</h2><span>Three simple steps</span></div>
        <div class="workflow">
          <div><b>01</b><strong>Choose</strong><span>Disease or Nutrient.</span></div>
          <div><b>02</b><strong>Upload</strong><span>Add a clear coffee leaf.</span></div>
          <div><b>03</b><strong>Compare</strong><span>Review ensemble + model outputs.</span></div>
        </div>""")
    with right:
        markup("""<div class="section-head"><h2>Photo tips</h2><span>For a cleaner input</span></div>
        <div class="tip"><strong>🌿 Keep the leaf clear</strong><p>Use natural light, keep the leaf in focus and avoid heavy blur or very dark images.</p></div>
        <div class="tip"><strong>☕ Need help?</strong><p>Coffee AI can explain coffee concepts and discuss your latest model result in simple language.</p></div>""")
# ========================= NEURAL SCAN =========================
def scan_page():
    page_heading("NEURAL SCAN / LEAF INTELLIGENCE", "READ THE <em>LEAF.</em>")
    choice = st.radio("Choose the analysis task", ["DISEASE SCAN", "NUTRIENT SCAN"], index=None,
                      horizontal=True, key="task_widget")
    task = {"DISEASE SCAN": "Disease", "NUTRIENT SCAN": "Nutrient"}.get(choice)
    if task != st.session_state.scan_task:
        st.session_state.scan_task = task
        st.session_state.analysis_result = None
    if task:
        st.caption(" · ".join(CLASSES[task]))
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        markup('<div class="eyebrow">LEAF SCANNER / DROP SPECIMEN</div>')
        source = st.radio("Image source", ["Upload", "Camera"], horizontal=True, key="image_source")
        if source == "Camera":
            upload = st.camera_input("Take a leaf photo", key=f'camera_{st.session_state.upload_revision}')
        else:
            upload = st.file_uploader("Drop specimen / browse", type=["jpg", "jpeg", "png", "webp"], key=f'leaf_upload_{st.session_state.upload_revision}')
        if upload:
            raw = upload.getvalue()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != st.session_state.specimen_hash:
                st.session_state.analysis_result = None
                try:
                    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                    st.session_state.specimen = image
                    st.session_state.specimen_hash = digest
                except Exception:
                    st.session_state.specimen = None
                    st.session_state.specimen_hash = None
                    st.error("Image could not be decoded. Upload a valid JPG, PNG or WEBP.")
        image = st.session_state.specimen
        if image:
            buffer = io.BytesIO()
            preview = image.copy()
            preview.thumbnail((900, 700))
            preview.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            markup(f'<div class="scanner"><img alt="Uploaded leaf specimen" src="data:image/png;base64,{encoded}"></div>')
            st.caption("SPECIMEN ACQUIRED / IMAGE READY / 224 × 224 MODEL INPUT")
            for warning in quality_messages(image):
                st.warning(warning)
            st.caption("Checks: dimensions and mean brightness only; no leaf-content validation.")
            if st.button("REMOVE SPECIMEN"):
                st.session_state.specimen = None
                st.session_state.specimen_hash = None
                st.session_state.analysis_result = None
                st.session_state.upload_revision += 1
                st.rerun()
    with right:
        markup('<div class="eyebrow">MODEL PIPELINE / LIVE EXECUTION</div>')
        states = {m: "WAITING" for m in MODEL_ARCHS}
        pipeline = st.empty()
        with pipeline.container():
            render_pipeline(states)
        st.caption("All four models are attempted. Incompatible checkpoints remain visible as errors.")
        if st.button("RUN NEURAL SCAN", type="primary", disabled=not (task and image), use_container_width=True):
            def update(name, status):
                states[name] = status
                with pipeline.container():
                    render_pipeline(states)
            st.session_state.analysis_result = None
            ready = True
            missing = []
            for model_name in MODEL_ARCHS:
                try:
                    resolve_checkpoint(task, model_name)
                except FileNotFoundError:
                    missing.append(model_name)
            if missing:
                from checkpoint_download import download_checkpoints
                try:
                    with st.spinner("Fetching missing trained checkpoints from your Google Drive folder..."):
                        download_checkpoints(tasks=(task.lower(),))
                    checkpoint_candidates.clear()
                except Exception as exc:
                    ready = False
                    st.error("Could not fetch the trained checkpoints from Drive. Please retry the scan. Check folder sharing and the expected filenames.")
                    st.caption(safe_detail(f"{type(exc).__name__}: {exc}"))
            result = run_scan(image, task, st.session_state.specimen_hash, update) if ready else None
            st.session_state.analysis_result = result
            if result:
                st.session_state.scan_counter += 1
                result["scan_id"] = st.session_state.scan_counter
                st.session_state.lab_archive.insert(0, result)
                record_id = str(uuid.uuid4())
                result["record_id"] = record_id
                image_data = png_bytes(image)
                st.session_state.scan_images[record_id] = image_data
                if current_owner() and setting("ROCOLE_DB_PATH"):
                    try:
                        Store(setting("ROCOLE_DB_PATH")).save_scan(current_owner(), result, image_data)
                        st.success("Scan and image saved to your account.")
                    except (OSError, sqlite3.Error, ValueError):
                        st.warning("Account storage failed. This scan remains in the current session.")
            else:
                st.error("No model completed inference. The exact failure for each model is shown below.")
            for diagnostic in st.session_state.model_diagnostics:
                if diagnostic["state"] == "ERROR":
                    with st.expander(diagnostic["model"] + " / ERROR", expanded=True):
                        st.code(diagnostic["error"], language=None)
    if st.session_state.analysis_result:
        render_result(st.session_state.analysis_result, "scan_report")
        st.button("EXPLAIN WITH COFFEE AI ↗", on_click=go, args=(PAGES[2],))

# ========================= COFFEE AI PAGE =========================
def chat_page():
    markup("""<style>
    [data-testid="stChatInput"]{background:var(--surface)!important;border:0!important;border-radius:28px!important}
    [data-testid="stChatInput"]>div{background:var(--surface)!important;color:var(--text)!important;border:1px solid var(--line)!important;border-radius:28px!important;box-shadow:none!important}
    [data-testid="stChatInput"]>div:focus-within{border-color:var(--accent)!important;box-shadow:0 0 0 1px var(--accent)!important}
    [data-testid="stChatInput"] textarea,[data-testid="stChatInput"] textarea:focus{background:transparent!important;color:var(--text)!important;caret-color:var(--text)!important;border:0!important;border-radius:0!important;outline:none!important;box-shadow:none!important}
    [data-testid="stChatInput"] textarea::placeholder{color:var(--muted)!important;opacity:1!important}
    [data-testid="stChatInput"] button{background:var(--surface2)!important;color:var(--text)!important;border:0!important}
    [data-testid="stChatInput"] button:hover:not(:disabled){background:var(--accent)!important;color:var(--bg)!important}
    [data-testid="stChatInput"] button:disabled{color:var(--muted)!important;opacity:.6!important}
    </style>""")
    page_heading("ROCOLE // COFFEE AI", "ASK<br><em>ROCOLE.</em>")
    st.write("Coffee knowledge without the language barrier.")
    diag = st.session_state.ai_diagnostic
    status = "OFFLINE" if not setting("GEMINI_API_KEY") else diag.get("status", "NOT TESTED")
    markup(f'<div class="strip">INTELLIGENCE / {esc(status)} · AUTO / KANGLISH READY</div>')
    with st.expander("CONVERSATION OPTIONS"):
        language = st.selectbox("Response language", ["Auto: match language and script", "English",
            "Kannada script", "Kanglish (Romanized Kannada)", "Hindi", "Romanized Hindi", "Tulu", "Romanized Tulu", "Tamil", "Malayalam"], key="language")
        if st.button("CLEAR CONVERSATION"):
            st.session_state.ai_turns = []
            st.session_state.ai_display = []
            st.session_state.pending_question = None
            st.rerun()
    result = st.session_state.analysis_result
    st.caption(f'ATTACHED SCAN / {result["label"]} / {result["task"]}' if result else "NO SCAN ATTACHED / GENERAL COFFEE QUESTIONS AVAILABLE")
    if not setting("GEMINI_API_KEY"):
        st.info("Coffee AI is temporarily unavailable. Please try again later.")
    prompt_from_chip = None
    if not st.session_state.ai_display:
        chips = [("WHAT IS ROBUSTA?", "What is Robusta coffee?"), ("EXPLAIN MY SCAN", "Explain my latest leaf scan."),
                 ("NITROGEN DEFICIENCY", "Explain nitrogen deficiency in coffee."), ("COFFEE LEAF CARE", "How do I care for coffee leaves?")]
        for col, (label, question) in zip(st.columns(4), chips):
            if col.button(label, use_container_width=True, disabled=not setting("GEMINI_API_KEY")):
                prompt_from_chip = question
    for msg in st.session_state.ai_display:
        with st.chat_message(msg["role"]):
            if msg.get("error"):
                st.warning(msg["text"])
            else:
                st.markdown(msg["text"])
    if "pending_voice_draft" in st.session_state:
        st.session_state.coffee_composer = st.session_state.pop("pending_voice_draft")
    retry = st.session_state.pending_question and st.button("RETRY LAST QUESTION")
    st.caption("Use the microphone in the message box to record. Submitting a recording sends it to Gemini for transcription; review the text here before sending your question.")
    submitted = st.chat_input("ASK ROCOLE ANYTHING ABOUT COFFEE...", key="coffee_composer", accept_audio=True,
                              audio_sample_rate=16000, max_upload_size=8,
                              disabled=not setting("GEMINI_API_KEY"), max_chars=2000)
    typed = None
    if submitted:
        recording = getattr(submitted, "audio", None)
        text = submitted if isinstance(submitted, str) else submitted.text
        if recording:
            from voice import transcribe
            try:
                with st.spinner("Transcribing your recording..."):
                    transcript = transcribe(recording.getvalue(), recording.type, setting("GEMINI_API_KEY"), setting("GEMINI_TRANSCRIPTION_MODEL", setting("GEMINI_MODEL", "gemini-3.1-flash-lite")))
                st.session_state.pending_voice_draft = "\n".join(part for part in (text.strip(), transcript) if part)[:2000]
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc))
        else:
            typed = text.strip()
    question = typed or prompt_from_chip or (st.session_state.pending_question if retry else None)
    if question:
        st.session_state.pending_question = question
        if not retry:
            st.session_state.ai_display.append({"role": "user", "text": question})
            with st.chat_message("user"):
                st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("CONTACTING COFFEE AI..."):
                answer, error, truncated = ask_ai(question, language)
            if error:
                st.warning(error)
            else:
                st.markdown(answer)
                if truncated:
                    st.caption("Answer reached its length limit. Ask 'continue' for more.")
        st.session_state.ai_display.append({"role": "assistant", "text": error or answer, "error": bool(error)})
        st.session_state.ai_display = st.session_state.ai_display[-40:]
        if not error:
            st.session_state.pending_question = None
        st.rerun()

# ========================= ARCHIVE =========================
def current_owner():
    if not st.user.get("is_logged_in", False):
        return None
    issuer, subject = st.user.get("iss"), st.user.get("sub")
    return account_id(issuer, subject) if issuer and subject else None


def archive_page():
    page_heading("SCAN HISTORY", "RESEARCH <em>RECORDS.</em>")
    owner = current_owner()
    store = None
    records = [(r.get("record_id", str(r["scan_id"])), r) for r in st.session_state.lab_archive]
    if owner and setting("ROCOLE_DB_PATH"):
        try:
            store = Store(setting("ROCOLE_DB_PATH"))
            page = st.number_input("History page", min_value=1, value=1, step=1)
            records = store.scans(owner, offset=(page-1)*20)
            st.caption("Account records and images. Retention depends on the operator's storage and backups.")
        except (OSError, sqlite3.Error, ValueError):
            st.warning("Account history is unavailable; showing this session's scans.")
            store = None
    else:
        st.caption("Records and images are kept in this session only.")
    if not records:
        st.info("No saved scans on this page.")
    for identifier, result in records:
        if store:
            image_data = store.image(owner, identifier)
            if image_data:
                st.session_state.scan_images[result["record_id"]] = image_data
        with st.expander(f"{result['time']} / {result['task']} / {result['label']}"):
            render_result(result, "report_"+identifier)
            if st.button("ATTACH THIS RESULT TO AI", key="attach_"+identifier):
                st.session_state.analysis_result = result
                go(PAGES[2])
                st.rerun()
            if store and st.checkbox("Confirm deletion", key="confirm_"+identifier):
                if st.button("Delete saved scan and image", key="delete_"+identifier):
                    store.delete(owner, identifier)
                    st.session_state.scan_images.pop(result["record_id"], None)
                    st.session_state.lab_archive = [r for r in st.session_state.lab_archive if r.get("record_id") != result["record_id"]]
                    st.rerun()
    if st.button("CLEAR SESSION ARCHIVE"):
        st.session_state.lab_archive = []
        st.session_state.scan_images = {}
        st.rerun()


# ========================= SYSTEM =========================
# ========================= NAVIGATION =========================
def main():
    st.set_page_config(page_title="RoCoLe — Coffee Leaf Intelligence", page_icon="🌿", layout="wide", initial_sidebar_state="auto")
    initialize_state()
    markup(STYLE)
    markup(INTERACTIVE_STYLE)
    markup(SHAPE_STYLE)
    markup(COOL_STYLE)
    markup(FINAL_COOL_STYLE)
    theme_attr = "light" if st.session_state.ui_theme == "Light" else "dark"
    markup(f'<script>document.documentElement.setAttribute("data-theme","{theme_attr}");</script>')
    # CSS variables are also overridden directly because Streamlit may isolate injected script execution.
    if st.session_state.ui_theme == "Light":
        markup('<style>:root{--bg:#f5f3e9;--surface:#fffdf7;--surface2:#edf3e8;--line:#d9e2d5;--text:#17241c;--muted:#68776d;--accent:#3f7e4d;--accent2:#6cae6f;--shadow:0 16px 40px rgba(38,63,45,.10)}</style>')
    if st.session_state.ui_theme == "Light":
        markup("""<style>
        .stApp,[data-testid="stSidebar"],.hero-card,.feature,.result,.scanner,.tip,.settings-card,.record,[data-testid="stExpander"],[data-testid="stChatMessage"],[data-testid="stFileUploaderDropzone"]{background:#fffdf7!important;color:#17241c!important}
        .stApp p,.stApp h1,.stApp h2,.stApp h3,.stat strong,.hero-card p,.hero-badge,.result .score{color:#17241c!important}
        .hero-badge,.workflow>div,.pipeline>div{background:#edf3e8!important;color:#17241c!important}
        .stButton button,.stDownloadButton button,div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,.stTextArea textarea{background:#fffdf7!important;color:#17241c!important}
        </style>""")
    t=UI_TEXT[st.session_state.ui_language]
    labels=[t["home"],t["analyze"],t["ai"],t["history"],t["settings"]]
    with st.sidebar:
        markup('<div class="brand">RoCoLe</div><div class="brand-sub">Coffee Leaf Intelligence</div>')
        if st.session_state.get("nav_page") not in PAGES:
            st.session_state.nav_page = st.session_state.lab_page if st.session_state.lab_page in PAGES else PAGES[0]
        chosen = st.radio("Navigation", PAGES, format_func=lambda value: labels[PAGES.index(value)], key="nav_page", label_visibility="collapsed")
        st.session_state.lab_page = chosen
        markup('<div class="strip">ROBUSTA COFFEE · MULTI-MODEL AI</div>')
    {PAGES[0]:home_page,PAGES[1]:scan_page,PAGES[2]:chat_page,PAGES[3]:archive_page,PAGES[4]:settings_page}[st.session_state.lab_page]()
    markup('<div class="strip">RoCoLe · Coffee Leaf Intelligence · Verify AI predictions before treatment</div>')

if __name__ == "__main__":
    main()
