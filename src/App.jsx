import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import * as d3 from 'd3'
import { queryAll, getCountInfo } from './api'
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
  'cifar10.txt': 'cifar10.txt',
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

  const {
    setOptions, loadData, setLabels,
    dots, distanceOfErrors, labelColors, filterLabels,
    selectedDots, selectImageIds,
    plotType, setPlotType,
    compareMode, setCompareMode,
    dotMode, setDotMode,
    bgOpacity, setBgOpacity,
    hoveredDot, setHoveredDot,
    selectDot, addLassoSelection, clearSelections, removeSelection, reorderSelections, removeImageId,
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
    dots, distanceOfErrors, selectedDots, selectImageIds, hoveredDot, labelColors, filterLabels,
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

        <div id="navbar-file">
          <label htmlFor="feature-file-select">Feature Names: </label>
          <select
            id="feature-file-select" className="navbar-selects"
            value={featureFile}
            onChange={e => setFeatureFile(e.target.value)}
          >
            {Object.keys(FEATURE_FILES).map(f => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>

        <div id="navbar-options">
          <label>Data types: </label>
          <select
            id="data-type-option" className="navbar-selects"
            value={dataTypeLabel}
            onChange={e => setDataTypeLabel(e.target.value)}
          >
            <option>Synthetic</option>
            <option>Experimental</option>
            <option disabled title="Not available">Cifar10</option>
          </select>

          <label>Error methods: </label>
          <select
            id="error-method-option" className="navbar-selects"
            value={errorMethod}
            onChange={e => setErrorMethod(e.target.value)}
          >
            {ERROR_METHOD_OPTIONS.map(o => <option key={o}>{o}</option>)}
          </select>

          <label>Embedding methods: </label>
          <select id="embedding-method-option" className="navbar-selects">
            <option>T-SNE</option>
            <option disabled title="Not available">PCA</option>
          </select>

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
              <div className="container-title">Attributes</div>
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
            {countLoading && <div style={{ padding: 8, fontSize: 11, color: '#888' }}>Loading…</div>}
            {countData && (
              <CountPanel
                countData={countData}
                countNum={countNum}
              />
            )}
          </div>

          {/* ── heatmap (bottom 60%) — TODO ── */}
          <div id="attribute-matrix-content"></div>

          <div id="matrix-option-row">
            <select id="attribute-matrix-option">
              <option value="MI">Mutual Info</option>
              <option value="correlation">Correlation</option>
              <option value="conditional_entropy_truelabel">Cond. Entropy Truelabel</option>
              <option value="conditional_entropy_prediction">Cond. Entropy Prediction</option>
            </select>
            <button id="matrix-cluster-btn">Cluster</button>
          </div>

          <select
            id="attribute-matrix2-option"
            value={countNum}
            onChange={e => setCountNum(Number(e.target.value))}
          >
            {[1,2,3,4,5].map(n => <option key={n} value={n}>{n}</option>)}
          </select>

          <div id="attribute-matrix-title">Coexisting Attributes Statistics</div>
          <div id="attribute-matrix2-title">Pairwise Attributes Information</div>
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
    const overhead    = 26
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
                  <span style={{ color: labelColors[idx] ?? '#252525', fontSize: 16 }}>■</span>
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
      .on('mouseover', function() { d3.select(this).attr('fill', '#f4a582') })
      .on('mouseout',  function() { d3.select(this).attr('fill', '#b2182b') })
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
  const { labels, labelColors, filterLabels, selectAllLabels, toggleFilterLabel } = useStore()

  if (!labels.length) return (
    <div style={{ padding: 4, fontSize: 11, color: '#888' }}>Loading attributes…</div>
  )

  const allSelected = filterLabels.length === 0

  return (
    <>
      <div
        id="attribute-selectall-btn"
        className={`attribute-btn${allSelected ? ' selected' : ''}`}
        onClick={selectAllLabels}
      >
        <span style={{
          display: 'inline-block', width: 10, height: 10,
          background: '#000', marginRight: 4, verticalAlign: 'middle',
        }} />
        Select All
      </div>

      {labels.map((label, i) => {
        const isSelected = allSelected || filterLabels.includes(i)
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
            {m.content}
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
