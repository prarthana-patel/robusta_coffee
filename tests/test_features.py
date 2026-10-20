import ast
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from features import Store, account_id, png_bytes, pdf_report, uncertainty

ROOT=Path(__file__).resolve().parents[1]

def result():
    return dict(scan_id=1,record_id='r',label='Rust',task='Disease',time='2026-10-20T00:00:00+00:00',confidence=70,votes=1,agreement=100,probabilities=[.2,.1,.7],classes=['Healthy','Red Spider Mite','Rust'], predictions=[dict(model='Test',label='Rust',confidence=70,class_source='verified fixture')],failed_models=[],method='fixture')

def test_owner_isolation_and_restart(tmp_path):
    path=tmp_path/'db.sqlite'
    store=Store(path)
    alice=account_id('issuer','a'); bob=account_id('issuer','b')
    store.save_profile(alice, {'name':'Alice'})
    image=png_bytes(Image.new('RGB',(40,80),'green'))
    record=store.save_scan(alice,result(),image)
    reopened=Store(path)
    assert reopened.profile(bob)=={}
    assert reopened.scans(bob)==[] and reopened.image(bob,record) is None
    reopened.delete(bob,record)
    assert reopened.scans(alice)[0][0]==record
    assert reopened.image(alice,record)==image
    reopened.delete(alice,record)
    assert reopened.scans(alice)==[]

def test_pdf_and_uncertainty():
    data=pdf_report(result(),png_bytes(Image.new('RGB',(100,50))))
    assert data.startswith(b'%PDF-') and b'%%EOF' in data
    assert uncertainty(result())['top_two_margin']==pytest.approx(.5)
    bad=result();bad['probabilities']=[float('nan')]
    with pytest.raises(ValueError): uncertainty(bad)

def test_navigation_and_theme(monkeypatch):
    monkeypatch.setenv("MARKET_API_KEY", "")
    a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    for page in ['Analyze Leaf','Coffee AI','History','Settings','Account','SYSTEM','Home','Settings']:
        a.sidebar.radio(key='nav_page').set_value(page).run()
        assert not a.exception, [e.message for e in a.exception]
        assert a.session_state['lab_page']==page
    a.radio(key='ui_theme').set_value('Light').run()
    assert not a.exception and a.session_state['lab_page']=='Settings'
    assert a.session_state['ui_theme']=='Light'

def test_original_model_and_ai_unchanged():
    original=Path('/workspace/attachments/d40de163-491e-488b-b8ea-4370beeb7661/app.py')
    if not original.exists(): pytest.skip('Original upload unavailable')
    old=ast.parse(original.read_text());new=ast.parse((ROOT/'app.py').read_text())
    protected=['ml_runtime','checkpoint_candidates','resolve_checkpoint','extract_state_dict','align_state_dict','metadata_classes','canonical_class','parse_classes','load_model_checked','run_scan','build_contents','ask_ai','ai_exchange']
    for name in protected:
        before=next(n for n in old.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        after=next(n for n in new.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        assert ast.dump(before)==ast.dump(after),name
    for name in ['SYSTEM_PROMPT','MODEL_ARCHS','CLASSES','DISEASE_CLASSES','NUTRIENT_CLASSES','ROOTS','EXACT_CHECKPOINTS']:
        def assignment(tree):
            return next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert ast.dump(assignment(old))==ast.dump(assignment(new)),name

def test_feedback_both_backends_and_safe_failures(tmp_path,monkeypatch):
    import app
    monkeypatch.setenv('FEEDBACK_DB_PATH',str(tmp_path/'feedback.sqlite'))
    monkeypatch.setenv('FEEDBACK_DB_PERSISTENT','true')
    for name,value in {'SMTP_HOST':'mail.example.test','SMTP_USER':'test','SMTP_PASSWORD':'secret','FEEDBACK_TO_EMAIL':'owner@example.test'}.items(): monkeypatch.setenv(name,value)
    smtp=MagicMock();smtp.__enter__.return_value=smtp;smtp.send_message.return_value={}
    with patch.object(app.smtplib,'SMTP',return_value=smtp):
        assert app.submit_feedback('a','b','hello')
        smtp.starttls.assert_called_once();smtp.send_message.assert_called_once()
        assert app.st.session_state.feedback_status=={'storage':'saved','email':'accepted'}
    with patch.object(app.smtplib,'SMTP',side_effect=app.smtplib.SMTPAuthenticationError(535,b'secret')):
        assert app.submit_feedback('a','b','hello')
        assert app.st.session_state.feedback_status['email']=='authentication_error'

def test_voice_transcription_mock_and_failure():
    from voice import transcribe
    response=MagicMock(status_code=200)
    response.json.return_value={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'What is coffee?'}]}}]}
    with patch('voice.httpx.post',return_value=response) as post:
        assert transcribe(b'fixture','audio/wav','not-real','gemini-test')=='What is coffee?'
        assert post.call_args.kwargs['follow_redirects'] is False
    with pytest.raises(ValueError): transcribe(b'a','audio/wav','','gemini-test')
    response.status_code=403
    with patch('voice.httpx.post',return_value=response),pytest.raises(ValueError,match='rejected'):
        transcribe(b'a','audio/wav','not-real','gemini-test')

def test_gradcam_synthetic_only():
    import torch
    from explain import spatial_gradcam
    model=torch.nn.Sequential(torch.nn.Conv2d(3,2,1,bias=False),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
    with torch.no_grad(): model[0].weight.fill_(1)
    before=[p.detach().clone() for p in model.parameters()]
    x=torch.arange(3*8*8,dtype=torch.float32).reshape(1,3,8,8)/192
    heat=spatial_gradcam(model,model[0],x,0)
    assert heat.shape==(8,8) and torch.isfinite(heat).all()
    assert float(heat.max())==pytest.approx(1)
    assert model.training and len(model[0]._forward_hooks)==0
    assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
    with pytest.raises(ValueError): spatial_gradcam(model,model[0],x,99)
    assert len(model[0]._forward_hooks)==0

def test_history_image_reports_and_camera(monkeypatch):
    monkeypatch.setenv('MARKET_API_KEY','')
    a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    a.session_state['lab_archive']=[result()]
    a.session_state['scan_images']={'r':png_bytes(Image.new('RGB',(30,30)))}
    a.sidebar.radio(key='nav_page').set_value('History').run()
    assert not a.exception, [e.message for e in a.exception]
    assert len(a.get('download_button'))==2
    a.sidebar.radio(key='nav_page').set_value('Analyze Leaf').run()
    a.radio(key='image_source').set_value('Camera').run()
    assert not a.exception and len(a.get('camera_input'))==1

def test_portable_model_paths(tmp_path,monkeypatch):
    from model_paths import model_directory
    for task in ('nutrient','disease'):
        monkeypatch.delenv('ROCOLE_'+task.upper()+'_DIR',raising=False)
        folder=tmp_path/'models'/task
        folder.mkdir(parents=True)
        assert model_directory(task,tmp_path)==str(folder)
        override=tmp_path/'mounted'/task
        monkeypatch.setenv('ROCOLE_'+task.upper()+'_DIR',str(override))
        assert model_directory(task,tmp_path)==str(override)
    with pytest.raises(ValueError): model_directory('other',tmp_path)


def test_exact_checkpoint_in_portable_root(tmp_path,monkeypatch):
    import app
    folder=tmp_path/'models'/'nutrient'
    folder.mkdir(parents=True)
    checkpoint=folder/'swin_best.pth'
    checkpoint.write_bytes(b'path-resolution-fixture-not-model-weights')
    monkeypatch.setitem(app.ROOTS,'Nutrient',[str(folder/'checkpoints'),str(folder)])
    app.st.session_state.path_overrides={}
    assert app.resolve_checkpoint('Nutrient','Swin Tiny')==str(checkpoint)
