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
    for page in ['Analyze Leaf','Coffee AI','History','Settings','Home','Settings']:
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
    protected=['ml_runtime','checkpoint_candidates','resolve_checkpoint','extract_state_dict','align_state_dict','metadata_classes','canonical_class','parse_classes','load_model_checked','run_scan','build_contents','ai_exchange']
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
    with patch('voice.httpx.post',return_value=response),pytest.raises(ValueError,match='denied'):
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

def test_removed_pages_and_inline_microphone(monkeypatch):
    monkeypatch.setenv('MARKET_API_KEY','')
    monkeypatch.setenv('GEMINI_API_KEY','test-not-a-real-key')
    a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert a.sidebar.radio(key='nav_page').options==['Home','Analyze Leaf','Coffee AI','History','Settings']
    a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30)
    a.session_state['nav_page']='SYSTEM'
    a.session_state['lab_page']='SYSTEM'
    a.run()
    assert not a.exception and a.session_state['lab_page']=='Home'
    a.sidebar.radio(key='nav_page').set_value('Coffee AI').run()
    assert not a.exception
    assert a.chat_input[0].proto.accept_audio
    assert len(a.get('audio_input'))==0
    assert all(e.label!='VOICE QUESTION' for e in a.expander)


def test_inline_recording_becomes_review_draft(monkeypatch):
    from streamlit.elements.widgets.chat import ChatInputValue
    monkeypatch.setenv('MARKET_API_KEY','')
    monkeypatch.setenv('GEMINI_API_KEY','test-not-a-real-key')
    audio=MagicMock(type='audio/wav')
    audio.getvalue.return_value=b'synthetic-audio'
    submitted=ChatInputValue(text='',audio=audio,_include_audio=True)
    a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    with patch('streamlit.chat_input',side_effect=[submitted,None]),patch('voice.transcribe',return_value='What is Robusta?') as transcribe:
        a.sidebar.radio(key='nav_page').set_value('Coffee AI').run()
        assert not a.exception, [e.message for e in a.exception]
        transcribe.assert_called_once()
        assert a.session_state['coffee_composer']=='What is Robusta?'
        assert a.session_state['ai_display']==[]

def test_drive_selection_rejects_missing_and_ambiguous_files():
    from types import SimpleNamespace
    from checkpoint_download import select_files, EXPECTED
    files=[SimpleNamespace(path='nutrient/'+name,id=str(i)) for i,name in enumerate(EXPECTED['nutrient'])]
    assert len(select_files(files,['nutrient']))==4
    with pytest.raises(ValueError,match='found 0'):select_files(files[:-1],['nutrient'])
    with pytest.raises(ValueError,match='found 2'):select_files(files+[files[0]],['nutrient'])


def test_drive_download_atomic_and_repeatable(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import checkpoint_download as cd
    import gdown
    monkeypatch.setattr(cd,'model_directory',lambda task:str(tmp_path/task))
    files=[SimpleNamespace(path='nutrient/'+name,id=str(i)) for i,name in enumerate(cd.EXPECTED['nutrient'])]
    monkeypatch.setattr(gdown,'download_folder',lambda **kwargs: files)
    def download(**kwargs):
        assert kwargs['verify'] is True
        Path(kwargs['output']).write_bytes(b'fixture-not-model-weights')
        return kwargs['output']
    monkeypatch.setattr(gdown,'download',download)
    assert len(cd.download_checkpoints(['nutrient']))==4
    assert cd.download_checkpoints(['nutrient'])==[]
    assert len(list((tmp_path/'nutrient').glob('*.pth.sha256')))==4
    assert not list(tmp_path.rglob('*.download'))

def test_ai_error_messages_are_specific_and_safe():
    import httpx
    from app import check_http, AIError
    for code, category in [(401,'AUTHENTICATION'),(403,'PERMISSION'),(404,'MODEL_UNAVAILABLE'),(429,'QUOTA'),(503,'PROVIDER')]:
        with pytest.raises(AIError) as caught:
            check_http(httpx.Response(code,json={'error':{'message':'private-provider-detail'}}))
        assert caught.value.category==category
        assert 'private-provider-detail' not in str(caught.value)
    with pytest.raises(AIError) as caught:
        check_http(httpx.Response(400,json={'error':{'details':[{'reason':'API_KEY_INVALID'}]}}))
    assert caught.value.category=='AUTHENTICATION'
    assert 'invalid or expired' in str(caught.value)


def test_ai_fallback_and_no_quota_retry(monkeypatch):
    import app
    from unittest.mock import AsyncMock
    monkeypatch.setenv('GEMINI_API_KEY','test-key')
    monkeypatch.setenv('GEMINI_MODEL','gemini-3.1-flash-lite')
    monkeypatch.delenv('GEMINI_BACKUP_MODEL',raising=False)
    app.initialize_state()
    app.st.session_state.ai_turns=[]
    app.st.session_state.analysis_result=None
    app.st.session_state.preferred_model=''
    exchange=AsyncMock(side_effect=[app.AIError('PROVIDER','temporary',503),('Coffee answer',{},'models/gemini-3.5-flash-lite',[],False)])
    with patch.object(app,'ai_exchange',exchange):
        answer,error,_=app.ask_ai('What is coffee?','English')
    assert answer=='Coffee answer' and error is None
    assert exchange.await_count==2
    assert [call.args[2] for call in exchange.await_args_list]==['gemini-3.1-flash-lite','gemini-3.5-flash-lite']
    assert app.st.session_state.ai_diagnostic['status']=='ONLINE'
    exchange=AsyncMock(side_effect=app.AIError('QUOTA','Quota reached',429))
    with patch.object(app,'ai_exchange',exchange):
        answer,error,_=app.ask_ai('What is coffee?','English')
    assert answer is None and error=='Quota reached' and exchange.await_count==1

def test_voice_fallback_uses_canonical_audio_payload():
    import httpx
    from voice import transcribe
    unavailable=httpx.Response(400,json={'error':{'status':'INVALID_ARGUMENT'}})
    success=httpx.Response(200,json={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'What is coffee?'}]}}]})
    with patch('voice.httpx.post',side_effect=[unavailable,success]) as post:
        assert transcribe(b'fixture','audio/wav','fake','gemini-primary','gemini-backup')=='What is coffee?'
        assert post.call_count==2
        assert '/gemini-backup:' in post.call_args.args[0]
        data=post.call_args.kwargs['json']['contents'][0]['parts'][1]['inlineData']
        assert data['mimeType']=='audio/wav'


def test_voice_quota_auth_and_invalid_response_no_retry():
    import httpx
    from voice import transcribe
    for code,payload,message in [(429,{},'quota'),(403,{},'denied'),(400,{'error':{'details':[{'reason':'API_KEY_INVALID'}]}},'authentication')]:
        with patch('voice.httpx.post',return_value=httpx.Response(code,json=payload)) as post:
            with pytest.raises(ValueError,match=message):transcribe(b'fixture','audio/wav','fake','gemini-primary','gemini-backup')
            assert post.call_count==1
    with patch('voice.httpx.post',return_value=httpx.Response(200,content=b'not-json')):
        with pytest.raises(ValueError,match='invalid response'):transcribe(b'fixture','audio/wav','fake','gemini-primary')
