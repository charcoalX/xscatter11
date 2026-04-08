# xscatter11

A React + D3 interactive visualization platform for analyzing X-ray scattering images using a ResNet-50 deep learning model. Provides scatter plot exploration, clustering analysis, attribute co-occurrence statistics, and pairwise information-theoretic metrics.

This project is a frontend reimplementation of [xscatter10](https://github.com/charcoalX/xscatter10), using React 19 + D3 v7 instead of vanilla JS, with the same Flask backend.

---

## Features

- **Scatter Plot** — 2D embedding of image features (t-SNE / layer activations). Supports lasso multi-select, single-click select, zoom/pan, and flower-plot mode.
- **Compare Mode** — Side-by-side view of 1, 3, or 6 feature layers.
- **Selection Groups** — Lasso or click to create named groups with color coding; drag to reorder.
- **Gallery Tab** — Browse selected images with true/predicted label display.
- **Statistics Tab** — Parallel coordinates showing attribute distributions per selection.
- **Clustering Tab** — K-Means and DBSCAN clustering with silhouette / Davies-Bouldin quality scores.
- **Attribute Filter** — Toggle individual attributes; select all / deselect all checkbox.
- **Attribute Study Panel**
  - *Pairwise Attribute Information* — D3 heatmap of mutual information, correlation, or conditional entropy between attributes, with optional cluster reordering.
  - *Coexisting Attributes* — Sortable co-occurrence count table.
- **Model Architecture Panel** — Annotated diagram of the ResNet-50 feature extraction pipeline.
- **AI Assistant** — Chat panel powered by Claude (Anthropic API).
- **Detailed Image View** — Bottom panel showing full-resolution images for clicked selections.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, D3 v7, Zustand, Vite |
| Backend | Python 3, Flask |
| Database | PostgreSQL |
| ML Model | ResNet-50 (Tensorflow) |
| AI Assistant| Anthropic Claude API |

---

## Prerequisites

- Node.js 18+
- Python 3.9+
- PostgreSQL (default port 5434)
- The `static/` image dataset folder (not included in repo due to size)

---

## Setup & Running

### 1. Clone the repo

```bash
git clone https://github.com/charcoalX/xscatter11.git
cd xscatter11
```

### 2. Configure the backend

The repo includes two template files: `config.example.json` (database connection) and `.env.example` (API key). Copy each one and fill in your own values:

```bash
cd backend

# config.example.json is already in the repo — copy it and fill in your PostgreSQL credentials
cp config.example.json config.json
```

`config.json` should look like:
```json
{
    "host": "localhost",
    "port": "5434",
    "dbname": "your_db_name",
    "user": "your_db_user",
    "password": "your_db_password"
}
```

```bash
# .env.example is already in the repo — copy it and fill in your Anthropic API key
cp .env.example .env
```

`.env` should look like:
```
ANTHROPIC_API_KEY=sk-ant-...
```

# Install Python dependencies
pip install -r requirements.txt

# Place the static image folder here (copy from your data source)
# backend/static/images/vis_filtered_thumbnails/   ← X-ray thumbnails
```

### 3. Start the Flask backend

```bash
# From the backend/ directory
python main.py
# Runs on http://127.0.0.1:8085
```

### 4. Start the Vite frontend

```bash
# From the project root (xscatter11/)
npm install
npm run dev
# Runs on http://localhost:5173
```

Open `http://localhost:5173` in your browser. The frontend automatically proxies all API calls to the Flask backend at port 8085.

---

## Project Structure

```
xscatter11/
  backend/                  Flask backend (copied from xscatter10)
    main.py                 Flask app entry point
    query.py                Database query logic
    utils.py / modules.py   Helper utilities
    requirements.txt        Python dependencies
    config.example.json     DB connection template
    .env.example            API key template
    lrp_service/            LRP heatmap microservice
  src/
    api/index.js            Axios wrappers for all Flask endpoints
    store/useStore.js       Zustand global state
    components/
      ScatterPlot.jsx       D3 scatter plot
      VisPanelTabs.jsx      Gallery / Statistics / Clustering tabs
      ModelArchPanel.jsx    ResNet-50 architecture diagram
    styles/                 Component CSS
    App.jsx                 Root layout and all panel wiring
  public/
    17tags_meta.txt         X-ray attribute names (17 classes)
  vite.config.js            Proxy config (API + /static → port 8085)
```

---

## API Endpoints (Flask)

| Endpoint | Purpose |
|---|---|
| `POST /QueryAll` | Fetch embedding data |
| `POST /GetCluster` | K-Means clustering |
| `POST /GetClusterDBSCAN` | DBSCAN clustering |
| `POST /GetMutualInfo` | Pairwise attribute metrics |
| `POST /GetCountInfo` | Attribute co-occurrence counts |
| `POST /GetTsne` | On-demand t-SNE |
| `POST /GetLRPHeatmap` | LRP heatmap overlay |
| `POST /AskAssistant` | AI assistant chat |
