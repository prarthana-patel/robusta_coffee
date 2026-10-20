"""Application services independent of classifier and Coffee AI logic."""
import hashlib
from contextlib import contextmanager
import io
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


def account_id(issuer, subject):
    if not issuer or not subject:
        raise ValueError('Verified issuer and subject are required')
    return hashlib.sha256(json.dumps([issuer, subject]).encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            raise ValueError('ROCOLE_DB_PATH must be absolute')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS profiles(owner TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS scans(id TEXT PRIMARY KEY, owner TEXT NOT NULL, created TEXT NOT NULL, result TEXT NOT NULL, image BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS scan_owner ON scans(owner, created);''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    def profile(self, owner):
        with self.connect() as db:
            row = db.execute('SELECT data FROM profiles WHERE owner=?', (owner,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_profile(self, owner, data):
        with self.connect() as db:
            db.execute('INSERT INTO profiles VALUES (?,?) ON CONFLICT(owner) DO UPDATE SET data=excluded.data', (owner, json.dumps(data)))

    def save_scan(self, owner, result, image):
        identifier = str(uuid.uuid4())
        with self.connect() as db:
            db.execute('INSERT INTO scans VALUES (?,?,?,?,?)', (identifier, owner, datetime.now(timezone.utc).isoformat(), json.dumps(result, allow_nan=False), image))
        return identifier

    def scans(self, owner, limit=20, offset=0):
        with self.connect() as db:
            rows = db.execute('SELECT id,result FROM scans WHERE owner=? ORDER BY created DESC LIMIT ? OFFSET ?', (owner, limit, offset)).fetchall()
        return [(identifier, json.loads(result)) for identifier, result in rows]

    def image(self, owner, identifier):
        with self.connect() as db:
            row = db.execute('SELECT image FROM scans WHERE owner=? AND id=?', (owner, identifier)).fetchone()
        return row[0] if row else None

    def delete(self, owner, identifier):
        with self.connect() as db:
            db.execute('DELETE FROM scans WHERE owner=? AND id=?', (owner, identifier))


def png_bytes(image):
    out = io.BytesIO()
    image.convert('RGB').save(out, format='PNG')
    return out.getvalue()


def guidance(label):
    if label == 'Healthy':
        return 'Continue regular monitoring, balanced nutrition and good drainage. A healthy prediction does not rule out disease.'
    if label == 'Rust':
        return 'Inspect leaf undersides for orange powdery lesions. Improve airflow through appropriate pruning and ask a local extension officer to confirm rust before choosing any control product.'
    if label == 'Red Spider Mite':
        return 'Inspect leaf undersides with magnification for mites, stippling and webbing. Reduce dust and avoid water stress. Ask an extension officer about integrated pest management and protecting beneficial predators.'
    nutrient = label.split(' (')[0]
    observations = {
        'Boron': 'Check young leaves and growing tips for distortion; deficiency and toxicity can both damage growth.',
        'Calcium': 'Check new growth and root tips; review moisture supply and root health.',
        'Iron': 'Look for interveinal yellowing on young leaves and review soil pH.',
        'Magnesium': 'Look for interveinal yellowing on older leaves and review potassium balance.',
        'Manganese': 'Inspect young leaves for interveinal yellowing or speckling; excess manganese can also cause injury.',
        'Nitrogen': 'Look for generalized paling beginning on older leaves and assess overall plant vigor.',
        'Phosphorus': 'Record slow growth and unusually dark or discolored older foliage; check soil availability.',
        'Potassium': 'Inspect older leaves for marginal yellowing or scorching and review drought stress.',
    }
    return observations.get(nutrient, '') + f' Possible {nutrient.lower()} deficiency: record whether old or young leaves are affected, inspect roots and drainage, and obtain soil pH and leaf-tissue tests. Nutrient interactions and root problems can look similar; confirm before changing fertilizer.'


def uncertainty(result):
    p = result['probabilities']
    if not p or any(not math.isfinite(v) or v < 0 for v in p) or abs(sum(p)-1) > .001:
        raise ValueError('Invalid class probabilities')
    ranked = sorted(p, reverse=True)
    return {'normalized_entropy': -sum(v*math.log(v) for v in p if v)/math.log(len(p)) if len(p)>1 else 0,
            'top_two_margin': ranked[0]-ranked[1] if len(p)>1 else ranked[0]}


def pdf_report(result, image=None):
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    out = io.BytesIO()
    styles = getSampleStyleSheet()
    story = []
    def line(value, style='BodyText'):
        story.extend([Paragraph(escape(str(value)), styles[style]), Spacer(1,8)])
    line('RoCoLe leaf analysis', 'Title')
    for key in ['time','task','label','confidence','method']:
        line(f'{key}: {result.get(key, "Unavailable")}')
    line('Model prediction only; not a confirmed agricultural diagnosis. Probabilities are not calibrated clinical or agricultural risk estimates.')
    if image:
        from PIL import Image as PILImage
        with PILImage.open(io.BytesIO(image)) as im:
            width,height=im.size
        scale=min(360/width,240/height)
        story.append(Image(io.BytesIO(image),width=width*scale,height=height*scale))
    line(guidance(result['label']))
    for prediction in result['predictions']:
        line(f"{prediction['model']}: {prediction['label']} ({prediction['confidence']:.1f}%). Class mapping: {prediction['class_source']}")
    line('Failed models: '+(', '.join(result['failed_models']) or 'None'))
    line('Severity: unavailable. Grad-CAM: unavailable pending checkpoint-specific validation.')
    for label, probability in zip(result['classes'],result['probabilities']):
        line(f'{label}: {probability*100:.2f}%')
    SimpleDocTemplate(out).build(story)
    return out.getvalue()
