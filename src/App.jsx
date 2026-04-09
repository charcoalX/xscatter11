import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import * as d3 from 'd3'
import { queryAll, getCountInfo, getMutualInfo } from './api'
import { useStore } from './store/useStore'
import ScatterPlot from './components/ScatterPlot'
import { GalleryTab, StatisticsTab, ClusteringTab } from './components/VisPanelTabs'
import ModelArchPanel from './components/ModelArchPanel'

import './styles/style.css'
import './styles/scatterplot.css'
import './styles/vis.css'
import './styles/selection.css'
import './styles/gallery.css'
import './styles/clustering.css'
import './styles/count.css'

const ERROR_METHOD_OPTIONS = ['Cosine', 'Euclidean', 'Manhattan']

const FEATURE_FILES = {
  '17tags_meta.txt': '17tags_meta.txt',
}

const PLOT_OPTIONS = [
  { value: 'act',    label: 'Actual Label (ACT)' },
  { value: 'layer1', label: 'SCALE1' },
  { value: 'layer3', label: 'SCALE3' },
  { value: 'layer5', label: 'SCALE5' },
  { value: 'fea',    label: 'Feature-2048 (FEA)' },
  { value: 'prd',    label: 'Prediction Prob (PRD)' },
]

function dataTypeToApiKey(label) {
  if (label === 'Synthetic')    return 'synthetic'
  if (label === 'Experimental') return 'experimental'
  return 'cifar10'
}

function imageSrc(dataType, id) {
  if (dataType === 'cifar10') return `/static/images/cifar10_images/${id}.png`
  return `/static/images/vis_filtered_thumbnails/${id}.jpg`
}

// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [loading, setLoading]             = useState(true)
  const [loadError, setLoadError]         = useState(null)
  const [dataTypeLabel, setDataTypeLabel] = useState('Synthetic')
  const [errorMethod, setErrorMethod]     = useState('Cosine')
  const [featureFile, setFeatureFile]     = useState('17tags_meta.txt')
  const [selectMode, setSelectMode]       = useState('single')
  const [visTabs, setVisTabs]             = useState({})
  const [comparePanelOpen, setComparePanelOpen] = useState(false)
  // panel visibility — mirrors original toggle buttons
  const [filterOpen, setFilterOpen]       = useState(true)   // Attributes Selection
  const [visOpen, setVisOpen]             = useState(true)   // Group Selection

  // Attribute Study — count table
  const [countData, setCountData]         = useState(null)
  const [countLoading, setCountLoading]   = useState(false)
  const [countNum, setCountNum]           = useState(2)

  // Attribute Study — pairwise matrix
  const [matrixData, setMatrixData]       = useState(null)
  const [matrixLoading, setMatrixLoading] = useState(false)
  const [matrixMetric, setMatrixMetric]   = useState('correlation')
  const [matrixClustered, setMatrixClustered] = useState(false)

  const {
    setOptions, loadData, setLabels,
    dots, distanceOfErrors, labelColors, filterLabels, attrSelectAll,
    selectedDots, selectImageIds,
    plotType, setPlotType,
    compareMode, setCompareMode,
    dotMode, setDotMode,
    bgOpacity, setBgOpacity,
    hoveredDot, setHoveredDot,
    selectDot, addLassoSelection, clearSelections, removeSelection, reorderSelections, removeImageId,
    selectAllLabels, deselectAllLabels,
    attrStudyOpen, toggleAttrStudy,
    modelArchOpen, toggleModelArch, closeModelArch,
    aiAssistantOpen, toggleAIAssistant, closeAIAssistant,
  } = useStore()

  // ── load feature file labels ──────────────────────────────────────────────
  useEffect(() => {
    fetch(`/${featureFile}`)
      .then(r => r.text())
      .then(text => {
        const labels = text.split('\n').map(s => s.trim()).filter(Boolean)
        setLabels(labels)
      })
      .catch(console.error)
  }, [featureFile])

  // ── load scatter data ─────────────────────────────────────────────────────
  useEffect(() => {
    setLoading(true)
    setLoadError(null)
    const params = {
      'Data type':         dataTypeToApiKey(dataTypeLabel),
      'Embedding method':  'tsne',
      'Distance of error': errorMethod.toLowerCase(),
    }
    setOptions(params)
    queryAll(params)
      .then(data => { loadData(data); setLoading(false) })
      .catch(e  => { console.error(e); setLoadError(e.message); setLoading(false) })
  }, [dataTypeLabel, errorMethod])

  // ── load count data (lazy, when Attribute Study panel opens) ────────────────
  useEffect(() => {
    if (!attrStudyOpen || countData || countLoading) return
    setCountLoading(true)
    const params = {
      'Data type':         dataTypeToApiKey(dataTypeLabel),
      'Embedding method':  'tsne',
      'Distance of error': errorMethod.toLowerCase(),
    }
    getCountInfo(params)
      .then(data => { setCountData(data); setCountLoading(false) })
      .catch(e   => { console.error(e);   setCountLoading(false) })
  }, [attrStudyOpen])

  // Reset count data when data type changes so it's re-fetched
  useEffect(() => { setCountData(null) }, [dataTypeLabel, errorMethod])

  // ── load matrix data (lazy, when Attribute Study panel opens) ────────────────
  useEffect(() => {
    if (!attrStudyOpen || matrixData || matrixLoading) return
    setMatrixLoading(true)
    const params = {
      'Data type':         dataTypeToApiKey(dataTypeLabel),
      'Embedding method':  'tsne',
      'Distance of error': errorMethod.toLowerCase(),
    }
    getMutualInfo(params)
      .then(data => { setMatrixData(data); setMatrixLoading(false) })
      .catch(e   => { console.error(e);    setMatrixLoading(false) })
  }, [attrStudyOpen])

  // Reset matrix data when data type changes
  useEffect(() => { setMatrixData(null) }, [dataTypeLabel, errorMethod])

  const dataType      = dataTypeToApiKey(dataTypeLabel)
  const is6Layer      = compareMode === '6layers'
  const is3Layer      = compareMode === '3layers'
  const isCompare     = is6Layer || is3Layer
  const isCompareOpen = compareMode !== 'none'
  const activePlotLabel = PLOT_OPTIONS.find(o => o.value === plotType)?.label ?? 'Actual Label (ACT)'

  // ── panel widths (mirrors original toggle logic) ─────────────────────────
  const filterWidth  = filterOpen ? 15 : 0
  const visWidth     = visOpen    ? (isCompareOpen ? 40 : 50) : 0
  const scatterWidth = 100 - filterWidth - visWidth

  // ── toggle handlers ───────────────────────────────────────────────────────
  const setVisTab = (selectId, tab) =>
    setVisTabs(prev => ({ ...prev, [selectId]: tab }))

  // Clean up tab state when selections are removed
  useEffect(() => {
    const activeIds = new Set(selectedDots.map(s => s.selectId))
    setVisTabs(prev => {
      const next = {}
      for (const [k, v] of Object.entries(prev))
        if (activeIds.has(Number(k))) next[k] = v
      return next
    })
  }, [selectedDots])

  // shared ScatterPlot props
  const spCommon = {
    dots, distanceOfErrors, selectedDots, selectImageIds, hoveredDot, labelColors, filterLabels, attrSelectAll,
    dotMode, bgOpacity, selectMode,
    onDotClick:  selectDot,
    onDotHover:  setHoveredDot,
    onLassoEnd:  addLassoSelection,
  }

  return (
    <div style={{ height: '100%' }}>

      {/* ══════════════════════════════════════════════════════ NAVBAR */}
      <div id="navbar">
        <div id="navbar-title">Visual Analysis of X-Ray Scattering Images</div>

        <div id="navbar-options">
          <span className="navbar-item">
            <label htmlFor="feature-file-select">Feature Names:</label>
            <select
              id="feature-file-select" className="navbar-selects"
              value={featureFile}
              onChange={e => setFeatureFile(e.target.value)}
            >
              {Object.keys(FEATURE_FILES).map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </span>

          <span className="navbar-item">
            <label>Data types:</label>
            <select
              id="data-type-option" className="navbar-selects"
              value={dataTypeLabel}
              onChange={e => setDataTypeLabel(e.target.value)}
            >
              <option>Synthetic</option>
              <option disabled title="Not available">Experimental</option>
            </select>
          </span>

          <span className="navbar-item">
            <label>Error methods:</label>
            <select
              id="error-method-option" className="navbar-selects"
              value={errorMethod}
              onChange={e => setErrorMethod(e.target.value)}
            >
              {ERROR_METHOD_OPTIONS.map(o => <option key={o}>{o}</option>)}
            </select>
          </span>

          <span className="navbar-item">
            <label>Embedding methods:</label>
            <select id="embedding-method-option" className="navbar-selects">
              <option>T-SNE</option>
              <option disabled title="Not available">PCA</option>
            </select>
          </span>

          <button className="navbar-buttons" onClick={() => { toggleAttrStudy(); setFilterOpen(true) }}>Open Attribute Study</button>
          <button className="navbar-buttons" onClick={toggleModelArch}>Model Architecture</button>
          <button className="navbar-buttons" onClick={toggleAIAssistant}>AI Assistant</button>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════ MAIN */}
      <div id="main-container">

        {/* ── TOP CONTAINER ── */}
        <div id="top-container">

          {/* ════ SCATTERPLOT CONTAINER */}
          <div id="scatterplot-container" style={{ width: `${scatterWidth}%` }}>

            <button
              id="single-select-btn"
              style={{ background: selectMode === 'single' ? '#525252' : '#ffffff',
                       color:      selectMode === 'single' ? '#ffffff' : '#252525' }}
              onClick={() => setSelectMode('single')}
            >
              ○ Single
            </button>
            <button
              id="multiple-select-btn"
              style={{ background: selectMode === 'multiple' ? '#525252' : '#ffffff',
                       color:      selectMode === 'multiple' ? '#ffffff' : '#252525' }}
              onClick={() => setSelectMode('multiple')}
            >
              ⊞ Multiple
            </button>

            {/* plot-options: hidden in compare mode, like original */}
            {!isCompare && (
              <select
                id="plot-options" className="navbar-selects"
                value={plotType}
                onChange={e => setPlotType(e.target.value)}
              >
                {PLOT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            )}

            {/* dot color buttons */}
            <button
              id="default-dot-btn"
              className={`dot-color-btn${dotMode === 'default' ? ' selected' : ''}`}
              onClick={() => setDotMode('default')}
            >Default</button>
            <button
              id="prediction-dot-btn"
              className={`dot-color-btn${dotMode === 'prediction' ? ' selected' : ''}`}
              onClick={() => setDotMode('prediction')}
            >Prediction</button>
            <button
              id="flower-dot-btn"
              className={`dot-color-btn${dotMode === 'flower' ? ' selected' : ''}`}
              onClick={() => setDotMode('flower')}
            >FlowerPlot</button>

            <div id="background-opacity-container">
              <label id="background-opacity-label">Background</label>
              <input
                type="range" id="background-opacity-slider"
                min="0" max="1" step="0.05" value={bgOpacity}
                onChange={e => setBgOpacity(+e.target.value)}
              />
            </div>

            {/* hover image preview */}
            <div id="scatterplot-image" style={{ opacity: hoveredDot ? 1 : 0 }}>
              {hoveredDot && (
                <img src={imageSrc(dataType, hoveredDot.id)} alt=""
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              )}
            </div>
            <div id="scatterplot-imageid" style={{ opacity: hoveredDot ? 1 : 0 }}>
              {hoveredDot ? `Image ID: ${hoveredDot.id}` : ''}
            </div>

            {/* ── scatter panels ── */}
            {(() => {
              const w1 = '100%'
              const w3 = '33.2%'
              const w6 = '16.6%'

              const actW   = isCompare ? (is6Layer ? w6 : w3) : w1
              const layerW = is6Layer ? w6 : '0%'
              const feaW   = isCompare ? (is6Layer ? w6 : w3) : '0%'
              const prdW   = isCompare ? (is6Layer ? w6 : w3) : '0%'

              const sp = (type) => !loading && !loadError
                ? <ScatterPlot {...spCommon} type={type} />
                : loading
                  ? <div style={{ padding: 10, fontSize: 12, color: '#888' }}>Loading…</div>
                  : <div style={{ padding: 10, fontSize: 11, color: 'red' }}>Error: {loadError}</div>

              return <>
                {/* ACT */}
                <div id="act-container" style={{ width: actW }}>
                  <div className="container-title">
                    {isCompare ? 'Actual Label (ACT)' : activePlotLabel}
                  </div>
                  <div id="act-content" className="container-content">
                    {sp(isCompare ? 'act' : plotType)}
                  </div>
                </div>

                {/* SCALE1 */}
                <div id="layer2-container" className={is6Layer ? 'visible' : ''} style={{ width: layerW }}>
                  <div className="container-title">SCALE1</div>
                  <div id="layer2-content" className="container-content">
                    {is6Layer && sp('layer1')}
                  </div>
                </div>

                {/* SCALE3 */}
                <div id="layer3-container" className={is6Layer ? 'visible' : ''} style={{ width: layerW }}>
                  <div className="container-title">SCALE3</div>
                  <div id="layer3-content" className="container-content">
                    {is6Layer && sp('layer3')}
                  </div>
                </div>

                {/* SCALE5 */}
                <div id="layer4-container" className={is6Layer ? 'visible' : ''} style={{ width: layerW }}>
                  <div className="container-title">SCALE5</div>
                  <div id="layer4-content" className="container-content">
                    {is6Layer && sp('layer5')}
                  </div>
                </div>

                {/* FEA */}
                <div id="fea-container" style={{ width: feaW }}>
                  <div className="container-title">Feature-2048 (FEA)</div>
                  <div id="fea-content" className="container-content">
                    {isCompare && sp('fea')}
                  </div>
                </div>

                {/* PRD */}
                <div id="prd-container" style={{ width: prdW }}>
                  <div className="container-title">Prediction Prob (PRD)</div>
                  <div id="prd-content" className="container-content">
                    {isCompare && sp('prd')}
                  </div>
                </div>
              </>
            })()}

            {/* compare dropdown menu */}
            {comparePanelOpen && (
              <div id="compare-menu" style={{ display: 'block' }}>
                <button onClick={() => { setCompareMode('none');    setComparePanelOpen(false) }}>1 Layer</button>
                <button onClick={() => { setCompareMode('3layers'); setComparePanelOpen(false) }}>3 Layers</button>
                <button onClick={() => { setCompareMode('6layers'); setComparePanelOpen(false) }}>6 Layers</button>
              </div>
            )}

            <button
              id="compare-toggle-btn"
              className={isCompareOpen ? 'open' : ''}
              onClick={() => setComparePanelOpen(p => !p)}
            >Layer Compare</button>

          </div>{/* /scatterplot-container */}

          {/* ════ VIS CONTAINER — scrollable, one panel per selection */}
          <div id="vis-container" style={{ width: `${visWidth}%`, overflowY: 'auto' }}>
            {Array.from({ length: Math.max(2, selectedDots.length) }, (_, i) => {
              const sel = selectedDots[i] ?? null
              const key = sel?.selectId ?? `empty-${i}`
              return (
                <VisPanel
                  key={key}
                  panelId={i + 1}
                  tab={visTabs[sel?.selectId] ?? 'gallery'}
                  onTab={tab => sel && setVisTab(sel.selectId, tab)}
                  selection={sel}
                  dataType={dataType}
                />
              )
            })}
          </div>

          {/* ════ FILTER CONTAINER */}
          <div id="filter-container" style={{ width: `${filterWidth}%` }}>
            <div id="attribute-container">
              <div className="container-title">
                Attributes
                {filterOpen && (
                  <input
                    type="checkbox"
                    title={attrSelectAll ? 'Deselect all' : 'Select all'}
                    checked={attrSelectAll}
                    onChange={() => attrSelectAll ? deselectAllLabels() : selectAllLabels()}
                    style={{ position: 'absolute', right: 4, top: '50%', transform: 'translateY(-50%)', cursor: 'pointer', margin: 0 }}
                  />
                )}
              </div>
              <button
                id="selection-toggle-btn"
                className={filterOpen ? 'open' : ''}
                onClick={() => setFilterOpen(prev => !prev)}
              >Attributes Selection</button>
              <button
                id="group-selection-toggle-btn"
                className={visOpen ? 'open' : ''}
                onClick={() => setVisOpen(prev => !prev)}
              >Group Selection</button>
              <div id="attribute-content" className="container-content">
                <AttributeList />
              </div>
            </div>

            <div id="attribute-vis-content"><RelationsPanel /></div>

            <div id="selection-container">
              <div className="container-title">Drag/Drop Selections <button id="clear-selections-btn" onClick={clearSelections} title="Clear all">&#x2715;</button></div>
              <div id="selection-content" className="container-content">
                {selectedDots.map((sel, i) => (
                  <SelectionRow key={sel.selectId} selection={sel} index={i}
                    onRemove={() => removeSelection(sel.selectId)} />
                ))}
              </div>
            </div>
          </div>

        </div>{/* /top-container */}

        {/* ════ MODEL ARCH PANEL */}
        <div id="model-arch-panel" className={modelArchOpen ? 'open' : ''}>
          <div id="model-arch-title" className="container-title">
            Model Architecture (ResNet-50)
            <button id="model-arch-close-btn" onClick={closeModelArch} title="Close">&#x2715;</button>
          </div>
          <div id="model-arch-content">
            {modelArchOpen && <ModelArchPanel />}
          </div>
        </div>

        <div id="arch-tooltip" />

        {/* ════ AI ASSISTANT */}
        <AIAssistantPanel open={aiAssistantOpen} onClose={closeAIAssistant} />

        {/* ════ BOTTOM CONTAINER */}
        <div id="bottom-container">
          <div id="image-container">
            <div className="container-title rotate">Detailed Images <button id="clear-images-btn" onClick={() => useStore.getState().clearSelectImageIds()} title="Clear all">&#x2715;</button></div>
            <div id="image-content" className="container-content rotate">
              {selectImageIds.map(id => (
                <DetailedImage key={id} id={id} dataType={dataType} />
              ))}
            </div>
          </div>
        </div>

        {/* ════ ATTRIBUTE VIS CONTAINER */}
        <div id="attribute-vis-container" style={{ width: attrStudyOpen ? '50%' : '0%' }}>

          {/* ── count table (top 40%) ── */}
          <div id="attribute-matrix2-content">
            <div className="container-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 6px' }}>
              <span>Coexisting Attributes Statistics</span>
              <select
                id="attribute-matrix2-option"
                value={countNum}
                onChange={e => setCountNum(Number(e.target.value))}
              >
                {[1,2,3,4,5].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <div className="attr-panel-body">
              {countLoading && <div style={{ padding: 8, fontSize: 11, color: '#888' }}>Loading…</div>}
              {countData && <CountPanel countData={countData} countNum={countNum} />}
            </div>
          </div>

          {/* ── heatmap (bottom 60%) ── */}
          <div id="attribute-matrix-content">
            <div className="container-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 6px' }}>
              <span>Pairwise Attributes Information</span>
              <span style={{ display: 'flex', gap: 4 }}>
                <select
                  id="attribute-matrix-option"
                  value={matrixMetric}
                  onChange={e => setMatrixMetric(e.target.value)}
                >
                  <option value="MI">Mutual Info</option>
                  <option value="correlation">Correlation</option>
                  <option value="conditional_entropy_truelabel">Cond. Entropy TrueLabel</option>
                  <option value="conditional_entropy_prediction">Cond. Entropy Prediction</option>
                </select>
                <button
                  id="matrix-cluster-btn"
                  className={matrixClustered ? 'selected' : ''}
                  onClick={() => setMatrixClustered(c => !c)}
                >Cluster</button>
              </span>
            </div>
            <div className="attr-panel-body">
              {matrixLoading && <div style={{ padding: 8, fontSize: 11, color: '#888' }}>Loading…</div>}
              {matrixData && <MatrixPanel matrixData={matrixData} matrixMetric={matrixMetric} matrixClustered={matrixClustered} />}
            </div>
          </div>

        </div>

      </div>{/* /main-container */}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function VisPanel({ panelId, tab, onTab, selection, dataType }) {
  const tabs = ['gallery', 'statistics', 'clustering']
  return (
    <div className="vis-panel">
      <div className="container-title">
        {selection
          ? <>
              <span style={{
                display: 'inline-block', width: 10, height: 10,
                background: selection.color, marginRight: 4, verticalAlign: 'middle',
              }} />
              <strong>Selection:</strong> {selection.selectId}&nbsp;
              <strong>Total:</strong> {selection.ids.length} images
            </>
          : `Selection ${panelId}: 0`
        }
      </div>
      <div className="tab">
        {tabs.map(t => (
          <button
            key={t}
            className={`tab-buttons${tab === t ? ' active' : ''}`}
            onClick={() => onTab(t)}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      <div className="vis-panel-content container-content">
        {tab === 'gallery'    && <GalleryTab    selection={selection} dataType={dataType} />}
        {tab === 'statistics' && <StatisticsTab selection={selection} />}
        {tab === 'clustering' && <ClusteringTab selection={selection} dataType={dataType} />}
      </div>
    </div>
  )
}

function DetailedImage({ id, dataType }) {
  const { dots, labels, labelColors, setHoveredDot, removeImageId } = useStore()
  const dot = dots.find(d => d.id === id)
  const heatmapRef  = useRef(null)
  const tipRef      = useRef(null)
  const imgRef      = useRef(null)
  const overlayRef  = useRef(null)

  // LRP state: null | { src, label, opacity, loading, error }
  const [lrp, setLrp] = useState(null)

  // Request LRP heatmap from backend when a PRD rect is clicked
  const requestLRP = useCallback(async (attrIdx) => {
    const labelName = labels[attrIdx] ?? `class ${attrIdx}`
    setLrp(prev => ({ ...(prev ?? {}), loading: true, error: null, label: `computing ${labelName}…` }))
    try {
      const { getLRPHeatmap } = await import('./api')
      const result = await getLRPHeatmap({ image_id: id, class_idx: attrIdx, data_type: dataType })
      if (result.status === 'ok') {
        const pVal = dot?.predProb?.[attrIdx] != null
          ? dot.predProb[attrIdx].toFixed(2)
          : (result.pred_prob ?? 0).toFixed(2)
        const src   = `data:image/png;base64,${result.heatmap_b64}`
        const label = `LRP: ${labelName} (p=${pVal})`
        setLrp({ src, label, opacity: 0.7, loading: false, error: null })
      } else {
        setLrp(prev => ({ ...(prev ?? {}), loading: false, error: result.message ?? 'error' }))
      }
    } catch {
      setLrp(prev => ({ ...(prev ?? {}), loading: false, error: 'LRP unavailable' }))
    }
  }, [id, dataType, dot, labels])

  // Draw PRD / ACT heatmap — mirrors original formula: rh = floor(heatmapH/2/rows) - 10
  const drawHeatmap = useCallback((imgEl) => {
    const el = heatmapRef.current
    if (!el || !dot?.predProb?.length) return
    d3.select(el).selectAll('*').remove()

    const W        = el.offsetWidth || 120
    const cols     = 6
    const rows     = Math.ceil(dot.predProb.length / cols)
    const distance = 3

    const resolvedImg = imgEl || imgRef.current
    const containerH  = el.closest('#image-content')?.offsetHeight ?? 200
    const imgMaxH     = Math.floor(containerH * 0.5)
    const imgH        = resolvedImg ? Math.min(resolvedImg.offsetHeight, imgMaxH) : imgMaxH
    const overhead    = 50
    const heatmapH    = Math.max(50, containerH - imgH - overhead)

    const rh = Math.max(3, Math.floor(heatmapH / 2 / rows) - 10)
    const rw = Math.floor(W / cols) - distance
    const H  = heatmapH

    const svg = d3.select(el).append('svg').attr('width', W).attr('height', H)
    const tip = d3.select(tipRef.current)

    const px = i => (rw + distance) * (i % cols)
    const py = (i, yBase) => (rh + distance) * Math.floor(i / cols) + yBase

    const showTip = (event, val, i) => {
      tip.style('opacity', 1)
        .html(`<strong>${labels[i] ?? i}</strong><br/>${typeof val === 'number' ? val.toFixed(3) : val}`)
        .style('left', (event.clientX + 14) + 'px')
        .style('top',  (event.clientY - 14) + 'px')
    }
    const hideTip = () => tip.style('opacity', 0)

    // ── PRD (top half) — clickable ──
    const prdBase = 15
    svg.append('text').attr('x', 0).attr('y', 10).attr('font-size', 10).attr('fill', '#444').text('PRD')
    svg.selectAll('.prd-bg').data(dot.predProb).join('rect')
      .attr('x', (_, i) => px(i)).attr('y', (_, i) => py(i, prdBase))
      .attr('width', rw).attr('height', rh).attr('fill', '#f0f0f0')
    svg.selectAll('.prd-fill').data(dot.predProb).join('rect')
      .attr('x', (_, i) => px(i)).attr('y', (_, i) => py(i, prdBase))
      .attr('width', rw).attr('height', rh)
      .attr('fill', (_, i) => labelColors[i] ?? '#252525')
      .attr('opacity', d => dataType === 'synthetic' ? (d > 0.5 ? 1 : 0) : d)
      .attr('cursor', 'pointer')
      .on('mouseover', (event, d) => { const i = dot.predProb.indexOf(d); showTip(event, d, i) })
      .on('mousemove', (event, d) => { const i = dot.predProb.indexOf(d); showTip(event, d, i) })
      .on('mouseout', hideTip)
      .on('click', (event, d) => {
        event.stopPropagation()
        const i = dot.predProb.indexOf(d)
        requestLRP(i)
      })

    // ── ACT (bottom half) ──
    const actBase = H / 2 + 10
    svg.append('text').attr('x', 0).attr('y', H / 2 + 1).attr('font-size', 10).attr('fill', '#444').text('ACT')
    svg.selectAll('.act-fill').data(dot.trueLabel ?? []).join('rect')
      .attr('x', (_, i) => px(i)).attr('y', (_, i) => py(i, actBase))
      .attr('width', rw).attr('height', rh)
      .attr('fill', (d, i) => d > 0.5 ? (labelColors[i] ?? '#252525') : '#f0f0f0')
      .attr('cursor', 'pointer')
      .on('mouseover', (event, d) => { const i = (dot.trueLabel ?? []).indexOf(d); showTip(event, d, i) })
      .on('mousemove', (event, d) => { const i = (dot.trueLabel ?? []).indexOf(d); showTip(event, d, i) })
      .on('mouseout', hideTip)
  }, [dot, labels, labelColors, dataType, requestLRP])

  useEffect(() => { drawHeatmap() }, [drawHeatmap])

  // Sync overlay opacity when slider changes
  const handleOpacity = (e) => {
    const opacity = parseFloat(e.target.value)
    setLrp(prev => prev ? { ...prev, opacity } : prev)
  }

  return (
    <div
      style={{
        display: 'inline-block', verticalAlign: 'top', background: '#fff',
        marginRight: 8, padding: '6px 6px 4px', position: 'relative',
        fontSize: 11, width: 120, flexShrink: 0,
      }}
      onMouseEnter={() => dot && setHoveredDot(dot)}
      onMouseLeave={() => setHoveredDot(null)}
    >
      {/* header */}
      <div style={{ marginBottom: 2, paddingRight: 36, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        Image ID: {id}
      </div>

      {/* LRP label — always reserves space so layout is stable */}
      <div style={{ fontSize: 9, color: lrp?.error ? '#c00' : '#444', height: 14, overflow: 'hidden',
                    textOverflow: 'ellipsis', whiteSpace: 'nowrap', lineHeight: '14px', marginBottom: 1 }}>
        {lrp?.loading ? 'computing…' : (lrp?.error ?? lrp?.label ?? '')}
      </div>

      {/* opacity slider — only shown when overlay is ready */}
      {lrp?.src && !lrp.loading && (
        <input type="range" min={0} max={1} step={0.05} value={lrp.opacity}
          onChange={handleOpacity}
          className="slider-thin"
          style={{ position: 'absolute', top: 5, right: 18, width: 34, zIndex: 10 }}
        />
      )}

      {/* close button */}
      <button
        style={{ position: 'absolute', top: 2, right: 2, border: 'none',
                 background: 'transparent', cursor: 'pointer', fontSize: 11, lineHeight: 1 }}
        onClick={() => removeImageId(id)}
      >&#x2715;</button>

      {/* image + LRP overlay */}
      <div style={{ position: 'relative', width: '100%' }}>
        <img
          ref={imgRef}
          src={imageSrc(dataType, id)}
          alt={`id ${id}`}
          style={{ width: '100%', height: 'auto', display: 'block',
                   imageRendering: 'pixelated', maxHeight: '50%', objectFit: 'contain' }}
          onLoad={e => drawHeatmap(e.target)}
        />
        {lrp?.src && (
          <img
            ref={overlayRef}
            src={lrp.src}
            alt="LRP overlay"
            style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
                     objectFit: 'contain', opacity: lrp.opacity, pointerEvents: 'none' }}
          />
        )}
      </div>

      {/* PRD / ACT heatmap */}
      <div ref={heatmapRef} style={{ width: '100%' }} />

      {/* tooltip */}
      <div ref={tipRef} className="heatmap-tooltip" style={{
        position: 'fixed', opacity: 0, pointerEvents: 'none', zIndex: 9999,
        transition: 'opacity 0.1s',
      }} />
    </div>
  )
}

// ── Count table helpers ───────────────────────────────────────────────────────

function processCountRows(countData, num) {
  // Backend returns { "1": {attr_0:…}, "2": {attr_0_1:…}, … }
  const subData = countData[num] ?? countData[String(num)]
  if (!subData) return []
  const rows = []
  for (const [key, val] of Object.entries(subData)) {
    const ids = key.split('_').slice(1).map(Number)  // strip leading 'attr'
    // num=1: imageIDs is [[…]] (numpy.where tuple); num>=2: flat […]
    const number = ids.length === 1
      ? (val.imageIDs?.[0]?.length ?? 0)
      : (val.imageIDs?.length ?? 0)
    rows.push({
      Attributes: ids,
      Number:     number,
      CorNum:     val.imageIDs_correctPred?.length ?? 0,
    })
  }
  return rows
}

function CountPanel({ countData, countNum }) {
  const { labels, labelColors, selectAllLabels, toggleFilterLabel } = useStore()
  const [sort, setSort] = useState({ col: 'Number', dir: 'desc' })

  const rows = useMemo(() => {
    if (!countData) return []
    const result = processCountRows(countData, countNum)
    const dir = sort.dir === 'desc' ? -1 : 1
    result.sort((a, b) => dir * (b[sort.col] - a[sort.col]))
    return result
  }, [countData, countNum, sort])

  const cycleSort = (col) => {
    setSort(prev =>
      prev.col === col
        ? { col, dir: prev.dir === 'desc' ? 'asc' : 'desc' }
        : { col, dir: 'desc' }
    )
  }

  const handleAttrClick = (ids) => {
    selectAllLabels()
    ids.forEach(i => toggleFilterLabel(i))
  }

  const thClass = (col) =>
    sort.col === col ? (sort.dir === 'desc' ? 'aes' : 'des') : 'header'

  return (
    <table className="scroll">
      <thead>
        <tr>
          <th className="header">Attributes</th>
          <th className={thClass('Number')}  onClick={() => cycleSort('Number')}>Number</th>
          <th className={thClass('CorNum')}  onClick={() => cycleSort('CorNum')}>CorNum</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            <td style={{ cursor: 'pointer' }} onClick={() => handleAttrClick(row.Attributes)}>
              {row.Attributes.map(idx => (
                <span key={idx} style={{ marginRight: 4 }}>
                  <span style={{ display: 'inline-block', width: 10, height: 10, background: labelColors[idx] ?? '#252525', verticalAlign: 'middle' }} />
                  {' '}{labels[idx] ?? idx}
                </span>
              ))}
            </td>
            <td>{row.Number}</td>
            <td>{row.CorNum}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// MatrixPanel — Pairwise attribute information heatmap
// ─────────────────────────────────────────────────────────────────────────────

function setNum(num) {
  const res = num.toFixed(2)
  if (res[0] === '-' && res[1] === '0') return res.replace('-0', '-')
  if (res[0] === '0' && res[1] === '.') return res.replace('0.', '.')
  return res
}

function clusterAttributes(mutualInfo, N) {
  const D = []
  for (let i = 0; i < N; i++) {
    D[i] = []
    for (let j = 0; j < N; j++) {
      if (i === j) { D[i][j] = 0; continue }
      let val = mutualInfo[i + '-' + j]
      if (val === undefined) val = mutualInfo[j + '-' + i]
      D[i][j] = (val === undefined || val >= 1000000) ? 1 : 1 - Math.abs(val)
    }
  }
  const tree = d3.range(N).map(i => ({ leaves: [i] }))
  const active = d3.range(N)
  const CD = D.map(row => row.slice())
  while (active.length > 1) {
    let minD = Infinity, ai = -1, bi = -1
    for (let p = 0; p < active.length - 1; p++)
      for (let q = p + 1; q < active.length; q++)
        if (CD[active[p]][active[q]] < minD) { minD = CD[active[p]][active[q]]; ai = p; bi = q }
    const a = active[ai], b = active[bi]
    const lA = tree[a].leaves.length, lB = tree[b].leaves.length
    const newIdx = tree.length
    tree.push({ leaves: tree[a].leaves.concat(tree[b].leaves) })
    CD.push([])
    for (let k = 0; k < active.length; k++) {
      const c = active[k]
      if (c === a || c === b) continue
      const dv = (CD[a][c] * lA + CD[b][c] * lB) / (lA + lB)
      CD[newIdx][c] = dv; CD[c][newIdx] = dv
    }
    CD[newIdx][newIdx] = 0
    active.splice(bi, 1); active.splice(ai, 1); active.push(newIdx)
  }
  return tree[active[0]].leaves
}

function MatrixPanel({ matrixData, matrixMetric, matrixClustered }) {
  const { labels, selectAllLabels, toggleFilterLabel } = useStore()
  const svgRef       = useRef()
  const containerRef = useRef()
  const [size, setSize] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setSize({ w: Math.floor(width), h: Math.floor(height) })
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  useEffect(() => {
    const el = svgRef.current
    if (!el || !matrixData || !labels.length || !size.w || !size.h) return
    const svg = d3.select(el)
    svg.selectAll('*').remove()

    // Resolve metric key and mode
    const isCondEntropy = matrixMetric.startsWith('conditional_entropy')
    const metricKey  = isCondEntropy ? 'conditional_entropy' : matrixMetric
    const mutualInfo = matrixData[metricKey]
    if (!mutualInfo) return

    const kvPred    = mutualInfo.predProb    ?? {}
    const kvTrue    = mutualInfo.trueLabel   ?? {}
    const kvBetween = mutualInfo.between     ?? {}

    // Compute value ranges (exclude sentinel 1000000)
    const validValues = arr => Object.values(arr).filter(v => v < 1000000)
    const rangeOf = arr => { const v = validValues(arr); return [d3.min(v) ?? 0, d3.max(v) ?? 1] }
    const [minPred, maxPred]     = rangeOf(kvPred)
    const [minTrue, maxTrue]     = rangeOf(kvTrue)
    const [minBetw, maxBetw]     = rangeOf(kvBetween)

    const scalePred = d3.scaleLinear().domain([minPred, maxPred]).range([minPred, maxPred])
    const scaleTrue = d3.scaleLinear().domain([minTrue, maxTrue]).range([minTrue, maxTrue])
    const scaleBetw = d3.scaleLinear().domain([minBetw, maxBetw]).range([minBetw, maxBetw])
    // For drawMatrix (single triangle) we normalize to [0,1]
    const scaleNorm = d3.scaleLinear().domain([minTrue, maxTrue]).range([0, 1])
    const scaleNormP= d3.scaleLinear().domain([minPred, maxPred]).range([0, 1])

    const N = labels.length
    const margin = { left: 240, top: 110, right: 50, bottom: 30 }
    const width  = size.w || 500
    const height = size.h || 400
    const distance = 0
    const minOpa = isCondEntropy ? 0.2 : 0.1

    const cellSize = Math.min(17, (width - margin.left - margin.right) / N, (height - margin.top - margin.bottom) / N)
    const rw = cellSize + 6
    const rh = cellSize

    // Attribute ordering
    const order = matrixClustered
      ? clusterAttributes(kvTrue, N)
      : d3.range(N)
    const pos = new Array(N)
    order.forEach((origIdx, visPos) => { pos[origIdx] = visPos })

    svg.attr('width', width).attr('height', height)

    // ── click handler ────────────────────────────────────────────
    function onCellClick(i, j) {
      selectAllLabels()
      toggleFilterLabel(parseInt(i))
      if (i !== j) toggleFilterLabel(parseInt(j))
    }

    // ── hover/out helpers (highlight row/col labels) ──────────────
    function onCellOver(i, j) {
      d3.select(`#mx-col-${j}`).attr('fill', 'red').style('font-weight', 'bold').style('font-size', '14px')
      d3.select(`#mx-row-${i}`).attr('fill', 'red').style('font-weight', 'bold').style('font-size', '14px')
      d3.select(`#mx-col-${i}`).attr('fill', 'blue').style('font-weight', 'bold').style('font-size', '14px')
      d3.select(`#mx-row-${j}`).attr('fill', 'blue').style('font-weight', 'bold').style('font-size', '14px')
      d3.select(`#mx-txt-${i}-${j}`).style('stroke', 'blue').style('font-size', '10px').style('stroke-width', '0.8px')
      d3.select(`#mx-txt-${j}-${i}`).style('stroke', 'red').style('font-size', '10px').style('stroke-width', '0.8px')
    }
    function onCellOut(i, j) {
      d3.select(`#mx-col-${j}`).attr('fill', '#000').style('font-weight', 'normal').style('font-size', '12px')
      d3.select(`#mx-row-${i}`).attr('fill', '#000').style('font-weight', 'normal').style('font-size', '12px')
      d3.select(`#mx-col-${i}`).attr('fill', '#000').style('font-weight', 'normal').style('font-size', '12px')
      d3.select(`#mx-row-${j}`).attr('fill', '#000').style('font-weight', 'normal').style('font-size', '12px')
      d3.select(`#mx-txt-${i}-${j}`).style('stroke', '#000').style('font-size', '9px').style('stroke-width', '0.1px')
      d3.select(`#mx-txt-${j}-${i}`).style('stroke', '#000').style('font-size', '9px').style('stroke-width', '0.1px')
    }

    // ── draw axis labels ─────────────────────────────────────────
    const orderedLabels = order.map(i => labels[i])
    svg.selectAll('.mx-col-label').data(orderedLabels).enter().append('text')
      .attr('id', (d, i) => `mx-col-${i}`)
      .attr('class', 'mx-col-label')
      .attr('dy', '0.5em').attr('dx', '0.5em')
      .attr('text-anchor', 'right').style('font-size', '12px')
      .attr('transform', (d, i) => `translate(${margin.left + i * (rw + distance)},${margin.top - 10}) rotate(-45)`)
      .text(d => d)

    svg.selectAll('.mx-row-label').data(orderedLabels).enter().append('text')
      .attr('id', (d, i) => `mx-row-${i}`)
      .attr('class', 'mx-row-label')
      .attr('dy', '1em').attr('text-anchor', 'end').style('font-size', '12px')
      .attr('transform', (d, i) => `translate(${margin.left - 10},${margin.top + i * (rh + distance)})`)
      .text(d => d)

    if (isCondEntropy) {
      // ── drawMatrix: single triangle ───────────────────────────
      const labelSuffix = matrixMetric.split('_')[2]  // 'truelabel' or 'prediction'
      const useTrue = labelSuffix === 'truelabel'
      const kvSingle = useTrue ? kvTrue : kvPred
      const scaleSingle = useTrue ? scaleNorm : scaleNormP
      const color = useTrue ? '#a6d96a' : '#92c5de'
      const markerText = useTrue ? 'true-label' : 'prediction'

      const entries = Object.entries(kvSingle)
      const txys = [], tdata = [], tijs = []

      svg.selectAll('.mx-rect-single').data(entries).enter().append('rect')
        .attr('class', 'mx-rect-single')
        .attr('id', (d) => { const [k] = d; const [ii, jj] = k.split('-'); return `mx-rect-${ii}-${jj}` })
        .attr('width', rw).attr('height', rh)
        .attr('fill', color)
        .attr('stroke', 'black').attr('stroke-width', '1px')
        .attr('opacity', (d, idx) => {
          const [k, v] = entries[idx]
          const [ii, jj] = k.split('-')
          const res = scaleSingle(v)
          const opa = v >= 1000000 ? minOpa : Math.max(minOpa, Math.abs(res))
          tdata.push(res); tijs.push(`${pos[+ii]}_${pos[+jj]}`)
          txys.push(`${pos[+ii] * (rw + distance) + margin.left},${pos[+jj] * (rh + distance) + margin.top}`)
          return opa
        })
        .attr('transform', (d) => {
          const [k] = d; const [ii, jj] = k.split('-')
          return `translate(${pos[+ii] * (rw + distance) + margin.left},${pos[+jj] * (rh + distance) + margin.top})`
        })
        .style('cursor', 'pointer')
        .on('click', (event, d) => { const [k] = d; const [ii, jj] = k.split('-'); onCellClick(ii, jj) })
        .on('mouseover', function(event, d) { const [k] = d; const [ii, jj] = k.split('-'); onCellOver(pos[+ii], pos[+jj]) })
        .on('mouseout',  function(event, d) { const [k] = d; const [ii, jj] = k.split('-'); onCellOut(pos[+ii], pos[+jj]) })

      svg.selectAll('.mx-txt-single').data(tdata).enter().append('text')
        .attr('id', (d, i) => `mx-txt-${tijs[i].replace('_','-').split('-')[0]}-${tijs[i].replace('_','-').split('-')[1]}`)
        .attr('class', 'mx-txt-single number_text')
        .attr('dy', '1.3em').attr('dx', '0.5em')
        .attr('text-anchor', 'start').style('font-size', '9px').style('cursor', 'pointer')
        .attr('transform', (d, i) => `translate(${txys[i]})`)
        .text(d => d > 100000 ? '.00' : setNum(d))
        .on('mouseover', function(event, d) {
          const id = d3.select(this).attr('id').replace('mx-txt-','').split('-')
          onCellOver(+id[0], +id[1])
        })
        .on('mouseout', function(event, d) {
          const id = d3.select(this).attr('id').replace('mx-txt-','').split('-')
          onCellOut(+id[0], +id[1])
        })
        .on('click', function(event, d) {
          const id = d3.select(this).attr('id').replace('mx-txt-','').split('-')
          onCellClick(id[0], id[1])
        })

      // markers
      svg.append('text').attr('class','mutual_marker').attr('text-anchor','end')
        .style('font-size','12px').style('font-weight','bold').attr('fill','#f4a582')
        .attr('transform',`translate(${margin.left - 22},${margin.top - 10})`)
        .text('uncertain attribute')
      svg.append('text').attr('class','mutual_marker').attr('text-anchor','start')
        .style('font-size','12px').style('font-weight','bold').attr('fill','#f4a582')
        .attr('transform',`translate(${margin.left - 10},${margin.top - 24}) rotate(-90)`)
        .text('condition')
      svg.append('text').attr('class','mutual_marker').attr('text-anchor','start')
        .style('font-size','12px').style('font-weight','bold').attr('fill', color)
        .attr('transform',`translate(${margin.left / 2 + 125},${margin.top + N * rh + 15})`)
        .text(markerText)
      svg.append('text').attr('class','mutual_marker').attr('text-anchor','end')
        .style('font-size','12px').style('font-weight','bold').attr('fill', color)
        .attr('transform',`translate(${margin.left + N * rw + 14},110) rotate(270)`)
        .text(markerText)

    } else {
      // ── drawMatrixMerged: upper=trueLabel (green), lower=predProb (blue), diag=between (orange) ──
      const trueEntries  = Object.entries(kvTrue)
      const predEntries  = Object.entries(kvPred)
      const diagEntries  = Object.entries(kvBetween)

      function drawTriangle(entries, scale, color, cls, isUpper, isDiag) {
        const txys = [], tdata = [], tijs = []
        svg.selectAll(`.${cls}`).data(entries).enter().append('rect')
          .attr('class', cls)
          .attr('id', (d) => { const [k] = d; const [ii,jj] = k.split('-'); return `mx-rect-${ii}-${jj}` })
          .attr('width', rw).attr('height', rh)
          .attr('fill', color)
          .attr('stroke', isDiag ? color : 'black').attr('stroke-width', '1px')
          .attr('opacity', (d, idx) => {
            const [k, v] = entries[idx]
            const [ii, jj] = k.split('-')
            const vi = pos[+ii], vj = pos[+jj]
            const inRegion = isDiag ? (vi === vj) : isUpper ? (vi > vj) : (vi < vj)
            if (!inRegion) return 0
            const res = scale(v)
            const opa = v >= 1000000 ? minOpa : Math.max(minOpa, res < 0 ? Math.abs(res) : res)
            tdata.push(isDiag ? scale(v) : res)
            tijs.push(`${vi}_${vj}`)
            txys.push(`${vi * (rw + distance) + margin.left},${vj * (rh + distance) + margin.top}`)
            return opa
          })
          .attr('transform', (d) => {
            const [k] = d; const [ii, jj] = k.split('-')
            const vi = pos[+ii], vj = pos[+jj]
            return `translate(${vi * (rw + distance) + margin.left},${vj * (rh + distance) + margin.top})`
          })
          .style('cursor', 'pointer')
          .on('click', (event, d) => { const [k]=d; const [ii,jj]=k.split('-'); onCellClick(ii, jj) })
          .on('mouseover', function(event, d) { const [k]=d; const [ii,jj]=k.split('-'); onCellOver(pos[+ii], pos[+jj]) })
          .on('mouseout',  function(event, d) { const [k]=d; const [ii,jj]=k.split('-'); onCellOut(pos[+ii], pos[+jj]) })

        // value text
        svg.selectAll(`.${cls}-txt`).data(tdata).enter().append('text')
          .attr('id', (d, i) => `mx-txt-${tijs[i].split('_')[0]}-${tijs[i].split('_')[1]}`)
          .attr('class', `${cls}-txt number_text`)
          .attr('dy','1.3em').attr('dx','0.5em')
          .attr('text-anchor','start').style('font-size','9px').style('cursor','pointer')
          .attr('transform', (d, i) => `translate(${txys[i]})`)
          .text(d => d > 10000 ? '.00' : setNum(d))
          .on('mouseover', function() {
            const id = d3.select(this).attr('id').replace('mx-txt-','').split('-')
            onCellOver(+id[0], +id[1])
          })
          .on('mouseout', function() {
            const id = d3.select(this).attr('id').replace('mx-txt-','').split('-')
            onCellOut(+id[0], +id[1])
          })
          .on('click', function() {
            const id = d3.select(this).attr('id').replace('mx-txt-','').split('-')
            onCellClick(id[0], id[1])
          })
      }

      drawTriangle(trueEntries, scaleTrue, '#a6d96a', 'mx-rect-true', true,  false)
      drawTriangle(predEntries, scalePred,  '#92c5de', 'mx-rect-pred', false, false)
      drawTriangle(diagEntries, scaleBetw, '#f4a582', 'mx-rect-diag', false, true)

      // ── markers ─────────────────────────────────────────────────
      svg.append('text').attr('class','mutual_marker').attr('text-anchor','end')
        .style('font-size','12px').style('font-weight','bold').attr('fill','#f4a582')
        .attr('transform',`translate(${margin.left - 24},${margin.top - 10})`).text('prediction')
      svg.append('text').attr('class','mutual_marker').attr('text-anchor','start')
        .style('font-size','12px').style('font-weight','bold').attr('fill','#f4a582')
        .attr('transform',`translate(${margin.left - 10},${margin.top - 24}) rotate(-90)`).text('true-label')

      svg.append('line').attr('x1', margin.left - 100).attr('y1', margin.top - 9)
        .attr('x2', margin.left - 9).attr('y2', margin.top - 9).attr('stroke','#f4a582').attr('stroke-width',1)
      svg.append('line').attr('x1', margin.left - 9).attr('y1', margin.top - 84)
        .attr('x2', margin.left - 9).attr('y2', margin.top - 9).attr('stroke','#f4a582').attr('stroke-width',1)

      svg.append('text').attr('class','mutual_marker').attr('text-anchor','start')
        .style('font-size','12px').style('font-weight','bold').attr('fill','#92c5de')
        .attr('transform',`translate(${margin.left / 2 + 120},${margin.top + N * rh + 15})`).text('prediction')
      svg.append('text').attr('class','mutual_marker').attr('text-anchor','end')
        .style('font-size','12px').style('font-weight','bold').attr('fill','#a6d96a')
        .attr('transform',`translate(${margin.left + N * rw + 14},110) rotate(-90)`).text('true-label ')

      svg.append('line').attr('x1', margin.left / 2 + 120).attr('y1', margin.top + N * rh + 6)
        .attr('x2', margin.left / 2 + 245).attr('y2', margin.top + N * rh + 6).attr('stroke','#92c5de').attr('stroke-width',1)
      svg.append('line').attr('x1', margin.left + N * rw + 6).attr('y1', 110)
        .attr('x2', margin.left + N * rw + 6).attr('y2', 230).attr('stroke','#a6d96a').attr('stroke-width',1)

      // boundaries
      svg.append('line').attr('x1', margin.left / 2 + 120).attr('y1', margin.top + N * rh)
        .attr('x2', margin.left + N * rw).attr('y2', margin.top + N * rh).attr('stroke','#92c5de').attr('stroke-width',1)
      svg.append('line').attr('x1', margin.left / 2 + 120).attr('y1', margin.top)
        .attr('x2', margin.left + N * rw).attr('y2', margin.top).attr('stroke','#a6d96a').attr('stroke-width',1)
      svg.append('line').attr('x1', margin.left / 2 + 120).attr('y1', 110)
        .attr('x2', margin.left / 2 + 120).attr('y2', 110 + N * rh).attr('stroke','#92c5de').attr('stroke-width',1)
      svg.append('line').attr('x1', margin.left + N * rw).attr('y1', 110)
        .attr('x2', margin.left + N * rw).attr('y2', 110 + N * rh).attr('stroke','#a6d96a').attr('stroke-width',1)
    }

  }, [matrixData, matrixMetric, matrixClustered, labels, size])

  if (!matrixData) return null

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', overflow: 'hidden' }}>
      <svg ref={svgRef} />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// RelationsPanel — XOR-gate circle matrix + bar chart
// Renders when filterLabels has at least one attribute selected
// ─────────────────────────────────────────────────────────────────────────────
function RelationsPanel() {
  const { dots, filterLabels, labels } = useStore()
  const svgRef       = useRef()
  const containerRef = useRef()

  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    const svg = d3.select(el)
    svg.selectAll('*').remove()

    if (!filterLabels.length || !dots.length) return

    // Filter dots where ALL selected attrs have trueLabel === 1
    const filtered = dots.filter(dot => {
      if (!dot.trueLabel || !dot.predProb) return false
      return filterLabels.every(idx => Number(dot.trueLabel[idx]) === 1)
    })

    if (!filtered.length) {
      svg.attr('width', 200).attr('height', 30)
      svg.append('text').attr('x', 6).attr('y', 18)
        .style('font-size', '11px').attr('fill', '#888')
        .text('No co-occurring images found')
      return
    }

    // Compute XOR gates: gate string length = filterLabels.length
    const gates = {}
    for (const dot of filtered) {
      const gate = filterLabels.map(idx => Number(dot.predProb[idx]) > 0.5 ? '1' : '0').join('')
      if (!gates[gate]) gates[gate] = { count: 0, imageIds: [] }
      gates[gate].count++
      gates[gate].imageIds.push(dot.id)
    }

    const tableKeys     = Object.keys(gates).sort()
    const tableValues   = tableKeys.map(k => gates[k].count)
    const tableRows_len = tableKeys.length
    const tableCols_len = filterLabels.length

    const container  = containerRef.current
    const totalWidth = container.clientWidth  || 400
    const totalHeight= container.clientHeight || 300

    const margin      = { left: 10, top: 4, right: 20, bottom: 4 }
    const width       = totalWidth  - margin.left - margin.right
    const height_text = 70
    const distance    = 3
    const radius      = Math.min(6, Math.max(3, (width / tableCols_len / 2) - 1))

    // Bar chart start position (right of last column of circles)
    const lastCircleX = (tableCols_len - 1) * (radius + distance) * 2 + radius + margin.left
    const xBarStart   = lastCircleX + radius + distance * 4
    const xBarMaxWidth= Math.max(50, width - xBarStart - 40)

    const linearScale = d3.scaleLinear()
      .domain([0, d3.max(tableValues)])
      .range([0, 1])

    svg
      .attr('width',  totalWidth)
      .attr('height', totalHeight)

    // Flattened circle items (row-major: rows=gates, cols=attrs)
    const flatten_items = []
    for (let i = 0; i < tableRows_len; i++)
      for (let j = 0; j < tableCols_len; j++)
        flatten_items.push(tableKeys[i][j])

    // ── circles ──────────────────────────────────────────────────
    svg.selectAll('.xor-circle')
      .data(flatten_items)
      .enter().append('circle')
      .attr('class', 'xor-circle')
      .attr('r', radius)
      .attr('fill',         d => d === '0' ? '#ffffff' : '#4d4d4d')
      .attr('stroke',       '#000')
      .attr('stroke-width', '1px')
      .attr('cx', (d, k) => (k % tableCols_len) * (radius + distance) * 2 + radius + margin.left)
      .attr('cy', (d, k) => Math.floor(k / tableCols_len) * (radius + distance) * 2 + radius + height_text + distance + margin.top)

    // ── attribute name labels (rotated -45°) ─────────────────────
    const sortedAttrs = [...filterLabels].sort((a, b) => a - b)
    svg.selectAll('.xor-attr-label')
      .data(sortedAttrs)
      .enter().append('text')
      .attr('class', 'xor-attr-label')
      .attr('dy', '.3em')
      .style('font-size', '11px')
      .attr('transform', (d, i) => {
        const x = i * (radius + distance) * 2 + margin.left
        const y = height_text - distance + margin.top
        return `translate(${x},${y}) rotate(-45)`
      })
      .text(d => labels[d] ?? String(d))

    // ── bar chart column label ────────────────────────────────────
    svg.append('text')
      .attr('dy', '.3em')
      .style('font-size', '11px')
      .attr('transform', () => {
        const x = xBarStart + distance
        const y = height_text - distance + margin.top
        return `translate(${x},${y}) rotate(-45)`
      })
      .text('count')

    // ── bars and count labels ─────────────────────────────────────
    const barWidths = tableValues.map(v => linearScale(v) * xBarMaxWidth)

    svg.selectAll('.xor-bar')
      .data(tableValues)
      .enter().append('rect')
      .attr('class', 'xor-bar')
      .attr('x',      xBarStart)
      .attr('y',      (d, i) => i * (radius + distance) * 2 + height_text + distance + margin.top)
      .attr('width',  (d, i) => barWidths[i])
      .attr('height', radius * 2)
      .attr('fill',   '#b2182b')
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('fill', '#f4a582')
        const i = tableValues.indexOf(d)
        const imageIds = gates[tableKeys[i]].imageIds
        const types = ['act', 'layer1', 'layer3', 'layer5', 'fea', 'prd']
        for (const id of imageIds) {
          for (const type of types) {
            d3.select(`#scatterdot-${type}-${id}`)
              .append('circle')
              .attr('class', 'scatterdot-hover')
              .attr('r', 0)
              .attr('fill', 'none')
              .attr('stroke', 'red')
              .attr('stroke-opacity', 0)
              .attr('stroke-width', '2px')
              .transition().duration(200)
                .attr('r', 11).attr('stroke-opacity', 1)
              .transition().duration(150)
                .attr('r', 7).attr('stroke-opacity', 0.8)
          }
        }
      })
      .on('mouseout', function() {
        d3.select(this).attr('fill', '#b2182b')
        d3.selectAll('.scatterdot-hover').remove()
      })
      .on('click', function(event, d) {
        const i = tableValues.indexOf(d)
        const imageIds = gates[tableKeys[i]].imageIds
        useStore.getState().addLassoSelection(imageIds)
      })

    svg.selectAll('.xor-count')
      .data(tableValues)
      .enter().append('text')
      .attr('class', 'xor-count')
      .attr('dy', '.3em')
      .style('font-size', '11px')
      .attr('fill', 'black')
      .attr('x', (d, i) => xBarStart + barWidths[i] + distance)
      .attr('y', (d, i) => i * (radius + distance) * 2 + height_text + radius + distance + margin.top)
      .text((d, i) => gates[tableKeys[i]].count)

  }, [filterLabels, dots, labels])

  if (!filterLabels.length) {
    return (
      <div style={{ padding: 8, fontSize: 11, color: '#888' }}>
        Click an attribute row above to view co-occurrence patterns
      </div>
    )
  }

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', overflow: 'auto' }}>
      <svg ref={svgRef} />
    </div>
  )
}

function AttributeList() {
  const { labels, labelColors, filterLabels, attrSelectAll, toggleFilterLabel } = useStore()

  if (!labels.length) return (
    <div style={{ padding: 4, fontSize: 11, color: '#888' }}>Loading attributes…</div>
  )

  return (
    <>
      {labels.map((label, i) => {
        const isSelected = attrSelectAll || filterLabels.includes(i)
        return (
          <div
            key={i}
            id={`attribute-btn-${i}`}
            className={`attribute-btn${isSelected ? ' selected' : ''}`}
            onClick={() => toggleFilterLabel(i)}
          >
            <span style={{
              display: 'inline-block', width: 10, height: 10,
              background: labelColors[i] ?? '#252525',
              marginRight: 4, verticalAlign: 'middle',
            }} />
            {label}
          </div>
        )
      })}
    </>
  )
}

let _dragSrcIdx = null   // index within the list — for reordering

function SelectionRow({ selection, index, onRemove }) {
  const { reorderSelections, updateSelectionColor } = useStore()
  const rowRef   = useRef(null)
  const enterCnt = useRef(0)

  const handleDragStart = (e) => {
    _dragSrcIdx = index
    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.setData('text/plain', String(index))
  }
  const handleDragEnter = (e) => {
    e.preventDefault()
    enterCnt.current++
    if (rowRef.current) rowRef.current.style.outline = '2px dashed #525252'
  }
  const handleDragOver = (e) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }
  const handleDragLeave = () => {
    enterCnt.current--
    if (enterCnt.current === 0 && rowRef.current)
      rowRef.current.style.outline = ''
  }
  const handleDrop = (e) => {
    e.preventDefault()
    enterCnt.current = 0
    if (rowRef.current) rowRef.current.style.outline = ''
    if (_dragSrcIdx !== null && _dragSrcIdx !== index)
      reorderSelections(_dragSrcIdx, index)
    _dragSrcIdx = null
  }

  return (
    <div
      ref={rowRef}
      className="selection-header"
      draggable={true}
      onDragStart={handleDragStart}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{ cursor: 'grab' }}
    >
      <label style={{ lineHeight: 0, marginRight: 4, cursor: 'pointer' }}>
        <span
          className="colorpicker-button"
          style={{ background: selection.color, display: 'inline-block', pointerEvents: 'none' }}
        />
        <input
          type="color"
          value={selection.color}
          onChange={e => updateSelectionColor(selection.selectId, e.target.value)}
          style={{ opacity: 0, width: 0, height: 0, padding: 0, border: 0, position: 'absolute' }}
        />
      </label>
      <span style={{ flex: 1, fontSize: 11 }}>
        Selection {selection.selectId} ({selection.ids.length})
      </span>
      <button
        className="selection-remove-button"
        onClick={onRemove}
        onDragStart={e => e.stopPropagation()}
      >&#x2715;</button>
    </div>
  )
}

// Split assistant text into lines, ensuring numbered/bulleted items start on new lines
function formatAIMessage(text) {
  if (!text) return null
  // Insert newline before numbered list items (e.g. "1. ", "2. ") and bullets ("- ", "• ")
  const normalized = text
    .replace(/([^\n])(\n?)(\s*)(\d+\.\s)/g, (_, pre, nl, sp, num) => nl ? `${pre}\n${num}` : `${pre}\n${num}`)
    .replace(/([^\n])(\n?)(\s*)([-•]\s)/g,  (_, pre, nl, sp, bul) => nl ? `${pre}\n${bul}` : `${pre}\n${bul}`)
  const lines = normalized.split('\n')
  return lines.map((line, i) => (
    <span key={i}>{line}{i < lines.length - 1 && <br />}</span>
  ))
}

function AIAssistantPanel({ open, onClose }) {
  const [messages, setMessages] = useState([])
  const [input, setInput]       = useState('')
  const [sending, setSending]   = useState(false)

  const send = async () => {
    if (!input.trim() || sending) return
    const userMsg = { role: 'user', content: input.trim() }
    const next = [...messages, userMsg]
    setMessages(next)
    setInput('')
    setSending(true)
    try {
      const { askAssistant } = await import('./api')
      const data = await askAssistant(next)
      setMessages([...next, { role: 'assistant', content: data.reply ?? String(data) }])
    } catch {
      setMessages([...next, { role: 'assistant', content: '(error)', isError: true }])
    }
    setSending(false)
  }

  return (
    <div id="ai-assistant-panel" className={open ? 'open' : ''}>
      <div id="ai-assistant-header">
        <span>AI Assistant</span>
        <button id="ai-assistant-close" onClick={onClose} title="Close">&#x2715;</button>
      </div>
      <div id="ai-assistant-messages">
        {messages.map((m, i) => (
          <div key={i} className={
            `ai-msg ${m.role === 'user' ? 'ai-msg-user' : 'ai-msg-assistant'}` +
            (m.isError ? ' ai-msg-error' : '')
          }>
            {formatAIMessage(m.content)}
          </div>
        ))}
      </div>
      <div id="ai-assistant-input-row">
        <input
          id="ai-assistant-input" type="text"
          placeholder="Ask about this tool…" autoComplete="off"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
        />
        <button id="ai-assistant-send" onClick={send} disabled={sending}>
          {sending ? '…' : 'Send'}
        </button>
      </div>
    </div>
  )
}
