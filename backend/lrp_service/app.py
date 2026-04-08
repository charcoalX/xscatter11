from __future__ import print_function
import os, threading
from flask import Flask, request, jsonify
from lrp_engine import LRPEngine

app = Flask(__name__)

CKPT_PATH = os.environ.get('CKPT_PATH', '/model/model.ckpt-200')
engine    = LRPEngine(ckpt_path=CKPT_PATH, num_classes=17)
lock      = threading.Lock()


@app.route('/heatmap', methods=['POST'])
def heatmap():
    data      = request.get_json(force=True)
    image_path = data.get('image_path', '')
    class_idx  = data.get('class_idx', None)

    if not image_path or not os.path.exists(image_path):
        return jsonify({'status': 'error',
                        'message': 'image not found: ' + image_path}), 404
    try:
        class_idx = int(class_idx)
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'message': 'invalid class_idx'}), 400
    if class_idx < 0 or class_idx >= 17:
        return jsonify({'status': 'error', 'message': 'class_idx out of range'}), 400

    with lock:
        heatmap_b64, pred_prob = engine.compute(image_path, class_idx)

    return jsonify({
        'status':      'ok',
        'heatmap_b64': heatmap_b64,
        'pred_prob':   pred_prob,
        'class_idx':   class_idx,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
