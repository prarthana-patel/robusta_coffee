"""Fetch only explicitly named, user-provided research checkpoints over verified TLS."""
import hashlib
import os
from pathlib import Path, PurePosixPath
from filelock import FileLock
from model_paths import model_directory

FOLDER_URL = 'https://drive.google.com/drive/folders/1qEo5uP8AdOmzwGaF9X32BM70E0Pjqdq8'
EXPECTED = {
    'disease': ('maxvit_tiny_tf_224_best.pth', 'swin_tiny_patch4_window7_224_best.pth', 'convnextv2_tiny_best.pth', 'mambaout_tiny_best.pth'),
    'nutrient': ('maxvit_tiny_recovered.pth', 'swin_best.pth', 'convnextv2_tiny_final_checkpoint.pth', 'mambaout_tiny_final_checkpoint.pth'),
}


# Explicitly supplied nutrient checkpoint links (ordered as in the user's screenshot).
DIRECT_IDS = {
    'convnextv2_tiny_final_checkpoint.pth': '108wUvxRKaZPFiuTMF0t4ApmUYHHFUqND',
    'mambaout_tiny_final_checkpoint.pth': '1w145BMdHndL8ZWBVUtPYJiMHO1xLSPB4',
    'maxvit_tiny_recovered.pth': '1KDcXJu7-3RAhB5DV1E3Ajngi9zDCZ_Oi',
    'swin_best.pth': '1U4CJzeSHZEi1Pp8BEobBhmKMXUBtsG3m',
}


def select_files(files, tasks):
    selected=[]
    for task in tasks:
        for filename in EXPECTED[task]:
            matches=[f for f in files if PurePosixPath(f.path.replace('\\','/')).name==filename]
            if len(matches)!=1:
                raise ValueError(f'{task}/{filename}: expected one matching file in Drive, found {len(matches)}. Check folder sharing and filenames.')
            selected.append((task,filename,matches[0]))
    return selected


def download_checkpoints(tasks=('disease','nutrient'), progress=None):
    import gdown
    tasks=tuple(tasks)
    if not tasks or any(task not in EXPECTED for task in tasks):
        raise ValueError('Choose disease and/or nutrient checkpoints.')
    lock_path=Path(__file__).resolve().parent/'models'/'.download.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(lock_path),timeout=1):
        pending=[task for task in tasks if any(not (Path(model_directory(task))/name).is_file() for name in EXPECTED[task])]
        if not pending:
            return []
        # Nutrient checkpoints use direct, user-supplied file IDs. Disease keeps
        # the original folder discovery until its direct IDs are provided.
        folder_tasks=[task for task in pending if task != 'nutrient']
        files=[]
        if folder_tasks:
            files=gdown.download_folder(url=FOLDER_URL,quiet=True,skip_download=True,verify=True,use_cookies=False,timeout=(10,60))
            if not files:
                raise ValueError('Drive returned no disease files. Check folder sharing.')
        selected=select_files(files,folder_tasks) if folder_tasks else []
        selected.extend(('nutrient',name,DIRECT_IDS[name]) for name in EXPECTED['nutrient'] if 'nutrient' in pending)
        downloaded=[]
        for task,filename,remote in selected:
            destination=Path(model_directory(task))/filename
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.is_file():
                continue
            if progress:
                progress(f'Downloading {task}/{filename}…')
            temporary=destination.with_suffix(destination.suffix+'.download')
            try:
                result=gdown.download(id=remote if isinstance(remote,str) else remote.id,output=str(temporary),quiet=True,verify=True,use_cookies=False,timeout=(10,60))
                if not result or not temporary.is_file() or temporary.stat().st_size==0:
                    raise ValueError(f'Download failed for {filename}.')
                with temporary.open('rb') as stream:
                    header=stream.read(512).lower().lstrip()
                if header.startswith((b'<!doctype html',b'<html')):
                    raise ValueError(f'Drive returned an HTML page instead of {filename}.')
                # Record the acquired artifact hash; not proof against an original local copy.
                digest=hashlib.sha256()
                with temporary.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):
                        digest.update(chunk)
                os.replace(temporary,destination)
                destination.with_suffix(destination.suffix+'.sha256').write_text(digest.hexdigest()+'\n')
                downloaded.append(str(destination))
            finally:
                temporary.unlink(missing_ok=True)
        # Class metadata is optional but never guessed; download only unambiguous sidecars.
        for task,filename,remote in selected:
            if isinstance(remote,str):
                continue  # Direct IDs have no known metadata sidecar folder.
            remote_parent=PurePosixPath(remote.path.replace('\\','/')).parent
            names=(Path(filename).with_suffix('.classes.json').name,'classes.json')
            for name in names:
                matches=[f for f in files if PurePosixPath(f.path.replace('\\','/'))==remote_parent/name]
                destination=Path(model_directory(task))/name
                if len(matches)==1 and not destination.exists():
                    temporary=destination.with_suffix(destination.suffix+'.download')
                    try:
                        result=gdown.download(id=matches[0].id,output=str(temporary),quiet=True,verify=True,use_cookies=False,timeout=(10,60))
                        if result:
                            import json
                            json.loads(temporary.read_text())
                            os.replace(temporary,destination)
                    finally:
                        temporary.unlink(missing_ok=True)
        return downloaded


if __name__=='__main__':
    paths=download_checkpoints(progress=print)
    print(f'Downloaded {len(paths)} checkpoints. Existing files preserved. Run the app to check real model compatibility.')
