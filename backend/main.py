#!/usr/bin/env python3

print('Important Hints: ******************************************************')
print('Please make sure that you have installed the followings: ')
print('1. PostgreSQL Database. The default port number is 5434.')
print('2. Localhost server')
print('3. Install python libraries such as Flask, psycopg2')
print('***********************************************************************')

# Load .env file if present (local development)
import os
try:
    with open(os.path.join(os.path.dirname(__file__), '.env')) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())
except FileNotFoundError:
    pass

# start import modules
from modules import *
from query import *
# from info import *
from flask import send_from_directory

app = Flask(__name__)
app.debug = True

# Read connection information from config.json
with open('config.json') as config_file:
    conn_info = json.load(config_file)

# Define database connection (environment variables override config.json)
import os as _os
host     = _os.environ.get('DB_HOST',     conn_info['host'])
dbname   = _os.environ.get('DB_NAME',     conn_info['dbname'])
user     = _os.environ.get('DB_USER',     conn_info['user'])
port     = _os.environ.get('DB_PORT',     conn_info['port'])
password = _os.environ.get('DB_PASSWORD', conn_info['password'])

# Create connection string
conn_str =  "host='" + host + "' dbname='" + dbname + "' user='" + user + "' password='" + password + "' port='" + port + "'"

# try connecting to postgresql database
try:
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    print('Connected to postgresql database ...')
    print('')
except:
    print('Please check the following! then run this code again: ')
    print('1. You have installed the PostgreSQL database.')
    print('2. The connection information in the config.json file are correct.')
    sys.exit()

@app.route('/')
def index():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'data')
    feature_files = {}
    for f in os.listdir(data_dir):
        filepath = os.path.join(data_dir, f)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as fh:
                feature_files[f] = fh.read()
    return render_template('index.html', feature_files=feature_files)

@app.route('/QueryAll', methods = ['POST'])
def Route_query_all():
    """ Query all data """
    obj = request.json
    json = query_all(params = obj, conn = conn, cursor = cursor)
    return jsonify(json)

@app.route('/QueryAll_Count', methods = ['POST'])
def Route_query_all_count():
    """ Query all data """
    obj = request.json
    json = query_all(params = obj, conn = conn, cursor = cursor)
    return jsonify(json)

@app.route('/GetCluster', methods = ['POST'])
def Route_get_cluster():
    obj = request.json 
    json = query_clustering(params = obj)
    return jsonify(json)

@app.route('/GetClusterDBSCAN', methods = ['POST'])
def Route_get_cluster_dbscan():
    obj = request.json
    json = query_clustering_DBSCAN(params = obj)
 
    return jsonify(json)

@app.route('/GetMutualInfo', methods = ['POST'])
def Route_get_metual_info():
    obj = request.json
    json = query_get_mutual_info(params = obj, conn = conn, cursor = cursor)
    return jsonify(json)


@app.route('/GetCountInfo', methods = ['POST'])
def Route_get_count_info():
    obj = request.json
    json = query_get_count_info(params = obj, conn = conn, cursor = cursor)
    return jsonify(json)

@app.route('/GetTsne', methods = ['POST'])
def Route_get_tsne():
    obj = request.json
    json = query_get_tsne(params = obj, conn = conn, cursor = cursor)
    return jsonify(json)


LRP_SERVICE_URL = _os.environ.get('LRP_SERVICE_URL', 'http://localhost:5001')

def _start_lrp_service():
    import subprocess, time
    container_name = 'xscatter-lrp-service'

    # Check if already running
    try:
        check = subprocess.Popen(
            ['docker', 'ps', '--filter', 'name=' + container_name, '--format', '{{.Names}}'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print('[LRP] docker not found, skipping auto-start (running inside container?)')
        return
    out, _ = check.communicate()
    if container_name in out.decode():
        print('[LRP] Service already running')
        return

    # Remove any stopped container with the same name
    subprocess.Popen(['docker', 'rm', container_name],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

    project_dir = _os.path.dirname(_os.path.abspath(__file__))
    model_path  = _os.path.join(project_dir, 'lrp_service', 'model').replace('\\', '/')
    images_path = _os.path.join(project_dir, 'static', 'images').replace('\\', '/')

    proc = subprocess.Popen([
        'docker', 'run', '-d',
        '--name', container_name,
        '-p', '5001:5001',
        '-v', model_path  + ':/model:ro',
        '-v', images_path + ':/images:ro',
        '-e', 'CKPT_PATH=/model/model.ckpt-200',
        'xscatter-lrp'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        print('[LRP] Failed to start container:', err.decode().strip())
        return

    print('[LRP] Container started, loading model (~40s)...')
    for _ in range(30):
        time.sleep(3)
        try:
            r = http_requests.get('http://localhost:5001/health', timeout=2)
            if r.json().get('status') == 'ok':
                print('[LRP] Service ready')
                return
        except Exception:
            pass
    print('[LRP] Warning: service did not become healthy in time')

_start_lrp_service()

@app.route('/GetLRPHeatmap', methods=['POST'])
def Route_get_lrp_heatmap():
    obj       = request.json
    image_id  = str(obj.get('image_id', ''))
    class_idx = int(obj.get('class_idx', 0))
    data_type = obj.get('data_type', 'synthetic')

    if data_type == 'synthetic':
        rel_path = '/images/vis_filtered_thumbnails/{0}.jpg'.format(image_id)
    elif data_type == 'experimental':
        rel_path = '/images/exp_filtered_thumbnails/{0}.png'.format(image_id)
    elif data_type == 'cifar10':
        rel_path = '/images/cifar10_images/{0}.png'.format(image_id)
    else:
        return jsonify({'status': 'error', 'message': 'unknown data_type'}), 400

    try:
        resp = http_requests.post(
            LRP_SERVICE_URL + '/heatmap',
            json={'image_path': rel_path, 'class_idx': class_idx},
            timeout=90
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except http_requests.exceptions.ConnectionError:
        return jsonify({'status': 'error', 'message': 'LRP service unavailable'}), 503
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


_ASSISTANT_SYSTEM_PROMPT = """You are a helpful assistant embedded in XScatter, an interactive visual analysis tool for X-ray scattering images and CIFAR-10 datasets using a ResNet-50 deep learning model.

Key features of XScatter:
1. Scatter Plot (main view): Shows T-SNE or PCA embeddings of images. Each dot = one image. Colors = class labels. Click a dot to load it into the Detailed Images panel.
2. Selection modes: "Single" (click one dot) or "Multiple" (lasso-draw a region). Selected images appear in the Detailed Images panel at the bottom.
3. Detailed Images panel (bottom): Each selected image card shows:
   - The raw image thumbnail
   - PRD (Prediction Probability) grid: colored boxes showing the model's predicted probability for each of the 17 attributes
   - ACT (Actual Label) grid: ground-truth labels
   - Clicking a PRD box triggers an LRP heatmap overlay on the raw image showing which pixels drove that prediction
   - An opacity slider controls heatmap transparency
4. LRP Heatmap: Gradient×Input relevance map rendered as a jet-colormap overlay. Warm colors = high relevance pixels.
5. Vis Panel (middle): Gallery / Statistics / Clustering views for lasso-selected image groups. Use lasso selection on the scatter plot to create groups; each group gets its own panel.
6. Attributes Selection (right filter panel): Filter which images appear in the scatter plot by toggling attribute values on/off.
7. Group Selection: Select and compare groups of images side by side.
8. Layer Compare: Compare scatter plots across different ResNet layers (1, 3, or 6 layers).
9. Model Architecture: Visualizes the ResNet-50 architecture with hover tooltips showing layer shapes.
10. Attribute Study: Pairwise attribute analysis — mutual information, correlation, conditional entropy heatmaps.
11. Data Types: Only Synthetic X-ray (17 attributes) is currently available. Experimental X-ray and CIFAR-10 are not available. If asked about these, inform the user they are not supported.
12. Embedding Methods: Only T-SNE is currently available. PCA is not available. If asked, inform the user.
13. Background slider: Controls opacity of the background image in the scatter plot.

Answer questions about how to use this tool, what features mean, and how to interpret visualizations. Be concise and practical. If unsure, say so."""

@app.route('/AskAssistant', methods=['POST'])
def Route_ask_assistant():
    api_key = _os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        return jsonify({'status': 'error', 'message': 'ANTHROPIC_API_KEY not set'}), 500

    obj = request.json or {}
    messages = obj.get('messages', [])
    if not messages:
        return jsonify({'status': 'error', 'message': 'no messages'}), 400

    try:
        resp = http_requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            json={
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 1024,
                'system': _ASSISTANT_SYSTEM_PROMPT,
                'messages': messages
            },
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return jsonify({'status': 'ok', 'reply': data['content'][0]['text']})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Serve React frontend (production build)
DIST_DIR = os.path.join(os.path.dirname(__file__), 'dist')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path and os.path.exists(os.path.join(DIST_DIR, path)):
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, 'index.html')


if __name__ == '__main__':
    print('Running local server at: http://127.0.0.1:8085/')
    # app.run(host = "0.0.0.0", port = 8888)
    app.run(host = "127.0.0.1", port = 8085)