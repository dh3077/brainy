"""
brainy — 4-step ML training studio
Teaches students how ML works by training image (VAE) or text (CharLSTM) models.
Port 5008.
"""
import os
# Must be set before torch/transformers import their native BLAS/OMP backends
os.environ.setdefault('TOKENIZERS_PARALLELISM',      'false')
os.environ.setdefault('OMP_NUM_THREADS',             '1')
os.environ.setdefault('MKL_NUM_THREADS',             '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS',        '1')
# Allow MPS ops not yet in PyTorch to silently fall back to CPU (required for DistilGPT-2 on Apple Silicon)
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import json
import re
import shutil
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context

from software.ai.image_trainer        import ImageTrainer
from software.ai.text_trainer         import TextTrainer
from software.ai.finetune_trainer     import FinetuneTrainer
from software.ai.smart_prompt_trainer import SmartPromptTrainer
from software.ai.classifier_trainer  import ClassifierTrainer

_OLLAMA_URL   = os.environ.get('OLLAMA_URL',   'http://localhost:11434')
_OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llava')

app = Flask(
    __name__,
    template_folder='software/ui/templates',
    static_folder='assets',
)
app.secret_key = 'brainy-secret-2024'

_MODELS_DIR = Path('data/models')
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_sessions: dict[str, dict] = {}


def _sid() -> str:
    if 'id' not in session:
        session['id'] = str(uuid.uuid4())
    return session['id']


def _state() -> dict:
    sid = _sid()
    if sid not in _sessions:
        _sessions[sid] = {'mode': None, 'training_mode': 'scratch', 'input_mode': 'image', 'trainer': None}
    return _sessions[sid]


# ── Model library helpers ───────────────────────────────────────────────────────

def _model_dir(model_id: str) -> Path:
    return _MODELS_DIR / model_id


def _read_meta(model_id: str) -> dict | None:
    p = _model_dir(model_id) / 'metadata.json'
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _write_meta(model_id: str, meta: dict) -> None:
    _model_dir(model_id).mkdir(parents=True, exist_ok=True)
    (_model_dir(model_id) / 'metadata.json').write_text(json.dumps(meta, indent=2))


def _list_models() -> list:
    entries = []
    if _MODELS_DIR.exists():
        for d in _MODELS_DIR.iterdir():
            if d.is_dir():
                meta = _read_meta(d.name)
                if meta:
                    entries.append(meta)
    entries.sort(key=lambda e: e.get('created_at', ''), reverse=True)
    return entries


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get('/')
def index():
    return render_template('trainer.html')


def _make_trainer(mode: str, training_mode: str, input_mode: str = 'image'):
    if mode == 'image':
        return ImageTrainer()
    if mode == 'classifier':
        return ClassifierTrainer(input_mode=input_mode, training_mode=training_mode)
    if training_mode == 'finetune':
        return FinetuneTrainer()
    if training_mode == 'smart_prompt':
        return SmartPromptTrainer()
    return TextTrainer()


@app.post('/api/mode')
def set_mode():
    data          = request.get_json(force=True) or {}
    mode          = data.get('mode', 'image')
    training_mode = data.get('training_mode', 'scratch')
    input_mode    = data.get('input_mode', 'image')
    if mode not in ('image', 'text', 'classifier'):
        return jsonify({'ok': False, 'error': 'Invalid mode.'})
    if training_mode not in ('scratch', 'finetune', 'smart_prompt'):
        training_mode = 'scratch'
    if input_mode not in ('image', 'text', 'audio'):
        input_mode = 'image'
    state = _state()
    if (state['mode'] != mode or state.get('training_mode') != training_mode
            or state.get('input_mode') != input_mode):
        old_trainer    = state.get('trainer')
        old_input_mode = state.get('input_mode')
        state['mode']          = mode
        state['training_mode'] = training_mode
        state['input_mode']    = input_mode
        new_trainer = _make_trainer(mode, training_mode, input_mode)
        # When only the training_mode changes within classifier, preserve labels + examples
        if (mode == 'classifier' and isinstance(old_trainer, ClassifierTrainer)
                and old_input_mode == input_mode):
            new_trainer.labels    = old_trainer.labels[:]
            new_trainer._examples = {k: list(v) for k, v in old_trainer._examples.items()}
        state['trainer'] = new_trainer
    return jsonify({'ok': True, 'mode': mode, 'training_mode': training_mode,
                    'input_mode': input_mode})


@app.post('/api/upload')
def upload():
    data  = request.get_json(force=True) or {}
    state = _state()
    if state['trainer'] is None:
        return jsonify({'ok': False, 'error': 'No mode set.'})
    try:
        if state['mode'] == 'image':
            src = (data.get('image') or '').strip()
            if not src:
                return jsonify({'ok': False, 'error': 'No image data.'})
            state['trainer'].add_image(src)
            idx   = state['trainer'].count() - 1
            thumb = state['trainer'].thumbnail_b64(idx)
            return jsonify({'ok': True, 'count': state['trainer'].count(), 'thumb': thumb})
        else:
            text = (data.get('text') or '').strip()
            if not text:
                return jsonify({'ok': False, 'error': 'No text.'})
            state['trainer'].add_text(text)
            return jsonify({'ok': True, 'count': state['trainer'].count()})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.post('/api/clear')
def clear():
    state = _state()
    if state['trainer']:
        state['trainer'].clear()
    return jsonify({'ok': True})


@app.get('/api/info')
def info():
    state = _state()
    if not state['trainer']:
        return jsonify({'ok': True, 'mode': None, 'trained': False, 'count': 0})
    return jsonify({'ok': True, 'mode': state['mode'], **state['trainer'].get_info()})


@app.get('/api/train-stream')
def train_stream():
    state = _state()
    if not state['trainer']:
        def _err():
            yield 'event: train_error\ndata: {"message": "No trainer — set a mode first."}\n\n'
        return Response(_err(), mimetype='text/event-stream')

    epochs = int(request.args.get('epochs', 60))
    lr     = float(request.args.get('lr', 0.001))

    if state['mode'] == 'image':
        batch_size     = int(request.args.get('batch_size', 8))
        latent_dim     = int(request.args.get('latent_dim', 128))
        augment_factor = int(request.args.get('augment_factor', 50))
        preview_every  = int(request.args.get('preview_every', 5))
        gen = state['trainer'].train(
            epochs=epochs, lr=lr, batch_size=batch_size,
            latent_dim=latent_dim, augment_factor=augment_factor,
            preview_every=preview_every,
        )
    elif isinstance(state['trainer'], ClassifierTrainer):
        batch_size = int(request.args.get('batch_size', 8))
        gen = state['trainer'].train(epochs=epochs, lr=lr, batch_size=batch_size)
    elif isinstance(state['trainer'], FinetuneTrainer):
        batch_size = int(request.args.get('batch_size', 4))
        max_length = int(request.args.get('max_length', 64))
        gen = state['trainer'].train(
            epochs=epochs, lr=lr,
            batch_size=batch_size, max_length=max_length,
        )
    else:
        hidden_size = int(request.args.get('hidden_size', 40))
        seq_len     = int(request.args.get('seq_len', 50))
        gen = state['trainer'].train(
            epochs=epochs, lr=lr,
            hidden_size=hidden_size, seq_len=seq_len,
        )

    def _stream():
        for event in gen:
            phase = event.get('phase', 'message')
            if phase == 'error':
                phase = 'train_error'
            yield f'event: {phase}\ndata: {json.dumps(event)}\n\n'
        yield 'event: stream_end\ndata: {}\n\n'

    return Response(
        stream_with_context(_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection':        'keep-alive',
        },
    )


@app.post('/api/generate')
def generate():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not state['trainer']:
        return jsonify({'ok': False, 'error': 'No trainer.'})
    try:
        if state['mode'] == 'image':
            if data.get('type') == 'interpolate':
                steps = int(data.get('steps', 10))
                strip = state['trainer'].interpolate(steps=steps)
                return jsonify({'ok': True, 'strip': strip})
            else:
                n    = int(data.get('n', 16))
                grid = state['trainer'].generate(n=n)
                return jsonify({'ok': True, 'grid': grid})
        else:
            prompt      = data.get('prompt', '')
            length      = int(data.get('length', 200))
            temperature = float(data.get('temperature', 1.0))
            text        = state['trainer'].generate(prompt=prompt, length=length, temperature=temperature)
            result      = {'ok': True, 'text': text}
            # Smart Prompt trainers attach retrieval context to each generation
            extra = getattr(state['trainer'], '_last_generate_meta', None)
            if extra:
                result.update(extra)
            return jsonify(result)
    except RuntimeError as e:
        return jsonify({'ok': False, 'error': str(e)})


# ── Model library ──────────────────────────────────────────────────────────────

_TYPE_MAP  = {'image': 'image_generator', 'text': 'text_generator', 'classifier': 'classifier'}
_MODE_MAP  = {'image_generator': 'image', 'text_generator': 'text', 'classifier': 'classifier'}
_EMOJI_MAP = {
    'image_generator': '🎨',
    'text_generator':  '✍️',
    'audio_generator': '🎵',
    'classifier':      '🏷️',
}


@app.post('/api/save')
def save_model():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not state['trainer'] or not state['trainer'].trained:
        return jsonify({'ok': False, 'error': 'No trained model to save.'})

    name      = (data.get('name') or 'Untitled').strip()[:40]
    model_type = _TYPE_MAP.get(state['mode'], state['mode'])
    emoji      = (data.get('emoji') or '').strip() or _EMOJI_MAP.get(model_type, '🤖')

    model_id      = str(uuid.uuid4())
    mdir          = _model_dir(model_id)
    mdir.mkdir(parents=True, exist_ok=True)
    weights_sub   = getattr(state['trainer'], 'WEIGHTS_SUBPATH', 'weights.pt')
    wpath         = mdir / weights_sub
    training_mode = state.get('training_mode', 'scratch')

    try:
        state['trainer'].save(wpath)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

    info = state['trainer'].get_info()
    meta = {
        'id':            model_id,
        'name':          name,
        'emoji':         emoji,
        'model_type':    model_type,
        'training_mode': training_mode,
        'input_mode':    state.get('input_mode', 'image'),
        'created_at':    time.strftime('%Y-%m-%dT%H:%M:%S'),
        'training_params': {
            'epochs':         data.get('epochs'),
            'lr':             data.get('lr'),
            'batch_size':     data.get('batch_size'),
            'latent_dim':     data.get('latent_dim'),
            'image_size':     128 if state['mode'] == 'image' else None,
            'augment_factor': data.get('augment_factor'),
        },
        'training_stats': {
            'final_loss':       info.get('loss'),
            'final_accuracy':   info.get('final_accuracy'),
            'duration_seconds': data.get('duration_seconds'),
            'dataset_size':     info.get('count'),
            'n_params':         info.get('n_params'),
        },
        'weights_file': weights_sub,
    }
    if state['mode'] == 'classifier':
        meta['labels'] = info.get('labels', [])
    _write_meta(model_id, meta)
    return jsonify({'ok': True, 'entry': meta})


@app.get('/api/smart-status')
def smart_status():
    """Check whether sentence-transformers and Ollama/LLaVA are available."""
    try:
        import sentence_transformers as _st
        st_ok = True
    except ImportError:
        st_ok = False

    import urllib.request, urllib.error
    ollama_ok = False
    llava_ok  = False
    try:
        with urllib.request.urlopen(f'{_OLLAMA_URL}/api/tags', timeout=3) as resp:
            data        = json.loads(resp.read())
            ollama_ok   = True
            model_names = [m.get('name', '').split(':')[0] for m in data.get('models', [])]
            llava_ok    = _OLLAMA_MODEL.split(':')[0] in model_names
    except Exception:
        pass

    return jsonify({
        'ok':                   True,
        'sentence_transformers': st_ok,
        'ollama':                ollama_ok,
        'llava':                 llava_ok,
        'ollama_url':            _OLLAMA_URL,
        'ollama_model':          _OLLAMA_MODEL,
        'available':             st_ok and ollama_ok and llava_ok,
    })


@app.get('/api/library')
def get_library():
    return jsonify({'ok': True, 'entries': _list_models()})


@app.post('/api/load')
def load_model():
    data     = request.get_json(force=True) or {}
    model_id = data.get('id', '')
    meta     = _read_meta(model_id)
    if not meta:
        return jsonify({'ok': False, 'error': 'Model not found.'})

    weights_sub = meta.get('weights_file', 'weights.pt')
    wpath = _model_dir(model_id) / weights_sub
    if not wpath.exists():
        return jsonify({'ok': False, 'error': 'Model weights file missing from disk.'})

    mode          = _MODE_MAP.get(meta.get('model_type', ''), 'image')
    training_mode = meta.get('training_mode', 'scratch')
    input_mode    = meta.get('input_mode', 'image')
    s = _state()
    s['mode']          = mode
    s['training_mode'] = training_mode
    s['input_mode']    = input_mode
    s['trainer']       = _make_trainer(mode, training_mode, input_mode)

    try:
        s['trainer'].load(wpath)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

    return jsonify({'ok': True, 'mode': mode, 'training_mode': training_mode,
                    'input_mode': input_mode, 'entry': meta})


@app.post('/api/rename')
def rename_model():
    data     = request.get_json(force=True) or {}
    model_id = data.get('id', '')
    meta     = _read_meta(model_id)
    if not meta:
        return jsonify({'ok': False, 'error': 'Model not found.'})

    new_name  = (data.get('name')  or '').strip()[:40]
    new_emoji = (data.get('emoji') or '').strip()
    if new_name:
        meta['name'] = new_name
    if new_emoji:
        meta['emoji'] = new_emoji
    _write_meta(model_id, meta)
    return jsonify({'ok': True, 'entry': meta})


@app.post('/api/delete')
def delete_model():
    data     = request.get_json(force=True) or {}
    model_id = data.get('id', '')
    mdir     = _model_dir(model_id)
    if not mdir.exists():
        return jsonify({'ok': False, 'error': 'Not found.'})
    shutil.rmtree(mdir)
    return jsonify({'ok': True})


# ── Classifier routes ──────────────────────────────────────────────────────────

@app.post('/api/classifier/labels')
def classifier_labels():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not isinstance(state.get('trainer'), ClassifierTrainer):
        return jsonify({'ok': False, 'error': 'No classifier active.'})
    labels = data.get('labels', [])
    if not labels or not isinstance(labels, list):
        return jsonify({'ok': False, 'error': 'labels must be a non-empty list.'})
    state['trainer'].set_labels([str(l).strip() for l in labels if str(l).strip()])
    return jsonify({'ok': True, 'labels': state['trainer']._labels})


@app.post('/api/classifier/example')
def classifier_example():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not isinstance(state.get('trainer'), ClassifierTrainer):
        return jsonify({'ok': False, 'error': 'No classifier active.'})
    label = (data.get('label') or '').strip()
    raw   = (data.get('data')  or '').strip()
    if not label or not raw:
        return jsonify({'ok': False, 'error': 'label and data are required.'})
    action = data.get('action', 'add')
    if action == 'remove':
        idx = int(data.get('idx', 0))
        state['trainer'].remove_example(label, idx)
        return jsonify({'ok': True, 'counts': state['trainer'].count_per_label()})
    try:
        state['trainer'].add_example(label, raw)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})
    return jsonify({'ok': True, 'counts': state['trainer'].count_per_label()})


@app.post('/api/classifier/rename-label')
def rename_label():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not isinstance(state.get('trainer'), ClassifierTrainer):
        return jsonify({'ok': False, 'error': 'No classifier active.'})
    old_label = (data.get('old_label') or '').strip()
    new_label = (data.get('new_label') or '').strip()
    if not old_label or not new_label:
        return jsonify({'ok': False, 'error': 'old_label and new_label required.'})
    trainer = state['trainer']
    if old_label not in trainer.labels:
        return jsonify({'ok': False, 'error': f'Label "{old_label}" not found.'})
    if new_label in trainer.labels:
        return jsonify({'ok': False, 'error': f'Label "{new_label}" already exists.'})
    idx = trainer.labels.index(old_label)
    trainer.labels[idx] = new_label
    trainer._examples[new_label] = trainer._examples.pop(old_label, [])
    return jsonify({'ok': True, 'labels': trainer.labels})


@app.post('/api/predict')
def predict():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not isinstance(state.get('trainer'), ClassifierTrainer):
        return jsonify({'ok': False, 'error': 'No classifier active.'})
    raw = (data.get('data') or '').strip()
    if not raw:
        return jsonify({'ok': False, 'error': 'data is required.'})
    try:
        result = state['trainer'].predict(raw)
        return jsonify({'ok': True, **result})
    except RuntimeError as e:
        return jsonify({'ok': False, 'error': str(e)})


# ── Bot designs ────────────────────────────────────────────────────────────────

_BOTS_DIR = Path('data/bots')
_BOTS_DIR.mkdir(parents=True, exist_ok=True)


@app.get('/api/bots')
def list_bots():
    entries = []
    if _BOTS_DIR.exists():
        for f in sorted(_BOTS_DIR.glob('*.json'),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                entries.append(json.loads(f.read_text()))
            except Exception:
                pass
    return jsonify({'ok': True, 'bots': entries})


@app.post('/api/bots/save')
def save_bot():
    data   = request.get_json(force=True) or {}
    bot_id = data.get('id') or str(uuid.uuid4())
    _BOTS_DIR.mkdir(parents=True, exist_ok=True)
    p   = _BOTS_DIR / f'{bot_id}.json'
    bot = {**data, 'id': bot_id, 'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    p.write_text(json.dumps(bot, indent=2))
    return jsonify({'ok': True, 'bot': bot})


@app.post('/api/bots/delete')
def delete_bot():
    data   = request.get_json(force=True) or {}
    bot_id = data.get('id', '')
    p      = _BOTS_DIR / f'{bot_id}.json'
    if p.exists():
        p.unlink()
    return jsonify({'ok': True})


if __name__ == '__main__':
    print('✓  brainy → http://localhost:5008')
    app.run(debug=False, port=5008, threaded=True, use_reloader=False)
