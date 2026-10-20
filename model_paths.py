"""Portable checkpoint locations, without changing checkpoint filenames or weights."""
import os
from pathlib import Path


def model_directory(task, project_root=None):
    if task not in ('nutrient', 'disease'):
        raise ValueError('Unknown model task')
    override = os.environ.get(f'ROCOLE_{task.upper()}_DIR', '').strip()
    if override:
        return str(Path(override).expanduser())
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent
    portable = root / 'models' / task
    windows = Path('D:/rocole_github_deploy/models') / task
    legacy = Path('/content/drive/MyDrive') / ('MaxViT_Coffee_Project' if task == 'nutrient' else 'RoCoLe_Phase2_Project')
    candidates = [portable]
    if os.name == 'nt':
        candidates.append(windows)
    candidates.append(legacy)
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(portable)
