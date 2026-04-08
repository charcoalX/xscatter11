import { useEffect, useRef, useState, useCallback } from 'react'
import * as d3 from 'd3'
import { useStore } from '../store/useStore'
import { getCluster, getClusterDBSCAN } from '../api'

// ── helpers ───────────────────────────────────────────────────────────────────

function imageSrc(dataType, id) {
  if (dataType === 'cifar10') return `/static/images/cifar10_images/${id}.png`
  return `/static/images/vis_filtered_thumbnails/${id}.jpg`
}

// Group selected dot ids by count of trueLabel === 1 (mirrors vis.getGalleryData)
function getGalleryData(selectionIds, dots) {
  const data = {}
  for (const dot of dots) {
    if (!selectionIds.includes(dot.id)) continue
    const count = (dot.trueLabel ?? []).filter(v => v === 1).length
    if (data[count] == null) data[count] = []
    data[count].push(dot.id)
  }
  return data
}

// Extract cluster vectors from dots (mirrors vis.getClusterVectors)
function getClusterVectors(selectionIds, dots, clusterBy) {
  const vectors = []
  for (const dot of dots) {
    if (!selectionIds.includes(dot.id)) continue
    let x, y
    if (clusterBy === 'act') { x = dot.xTrueLabel; y = dot.yTrueLabel }
    else if (clusterBy === 'fea') { x = dot.xFeature; y = dot.yFeature }
    else { x = dot.xPredict; y = dot.yPredict }
    if (x != null && y != null) vectors.push([x, y])
  }
  return vectors
}

// ── GalleryTab ─────────────────────────────────────────────────────────────────

export function GalleryTab({ selection, dataType }) {
  const { dots, addLassoSelection, selectDot } = useStore()

  if (!selection) return <div style={{ padding: 8, fontSize: 11, color: '#888' }}>No selection</div>

  const grouped = getGalleryData(selection.ids, dots)
  const groups = Object.keys(grouped).sort((a, b) => +a - +b)

  if (!groups.length)
    return <div style={{ padding: 8, fontSize: 11, color: '#888' }}>No images</div>

  return (
    <div>
      {groups.map(count => (
        <div key={count} className="gallery-label-container">
          {/* label header — click to add new selection */}
          <div
            style={{
              width: 100, height: 60, border: '2px solid #d9d9d9',
              margin: 2, float: 'left', lineHeight: '60px',
              textAlign: 'center', cursor: 'pointer', fontSize: 11,
            }}
            onClick={() => addLassoSelection(grouped[count].map(Number))}
          >
            #Attributes: {count}
          </div>

          {grouped[count].map(id => (
            <span key={id} style={{ display: 'inline-block' }}>
              <img
                src={imageSrc(dataType, id)}
                alt=""
                style={{
                  width: 60, height: 60,
                  border: '2px solid #fff', margin: 2,
                  cursor: 'pointer', imageRendering: 'pixelated',
                }}
                onClick={() => selectDot({ id: Number(id) })}
                onMouseEnter={e => e.currentTarget.style.border = '2px solid rgb(43,20,217)'}
                onMouseLeave={e => e.currentTarget.style.border = '2px solid #fff'}
              />
              <span style={{ display: 'block', fontSize: 9, textAlign: 'center' }}>{id}</span>
            </span>
          ))}
        </div>
      ))}
    </div>
  )
}

// ── StatisticsTab ──────────────────────────────────────────────────────────────

function computeStats(selectionIds, dots, labels) {
  // TP/TN/FP/FN per label
  const top = labels.map(label => ({ Label: label, TP: 0, TN: 0, FP: 0, FN: 0 }))

  for (const dot of dots) {
    if (!selectionIds.includes(dot.id)) continue
    const probs  = dot.predProb  ?? []
    const truths = dot.trueLabel ?? []
    for (let i = 0; i < probs.length; i++) {
      const pred  = probs[i]  > 0.5 ? 1 : 0
      const truth = truths[i] ?? 0
      if (truth === 1 && pred === 1) top[i].TP++
      else if (truth === 0 && pred === 1) top[i].FP++
      else if (truth === 1 && pred === 0) top[i].FN++
      else top[i].TN++
    }
  }

  // filter to labels with any activity
  const filtered = top.filter(d => d.TP + d.FP + d.FN > 0)
  filtered.sort((a, b) => (b.TP + b.FP + b.FN) - (a.TP + a.FP + a.FN))

  // compute precision/recall/accuracy/F1
  const bottom = filtered.map(d => {
    const p = d.TP + d.FN
    const n = d.FP + d.TN
    let accuracy  = (d.TP + d.TN) / (p + n)
    let precision = d.TP / (d.TP + d.FP) || 0
    let recall    = d.TP / (d.TP + d.FN) || 0
    let f1        = (precision + recall > 0)
      ? 2 * precision * recall / (precision + recall) : 0

    accuracy  = Math.round(accuracy  * 100) / 100 || 0
    precision = Math.round(precision * 100) / 100
    recall    = Math.round(recall    * 100) / 100
    f1        = Math.round(f1        * 100) / 100

    return { Label: d.Label, Accuracy: accuracy, F1: f1, Precision: precision, Recall: recall }
  })

  return { top: filtered, bottom }
}

function ParallelCoord({ data, dimensions, labelColors, labels }) {
  const ref = useRef(null)

  useEffect(() => {
    if (!data.length) return
    const el = ref.current
    if (!el) return
    d3.select(el).selectAll('svg').remove()

    const margin = { top: 20, right: 10, bottom: 20, left: 10 }
    const width  = el.clientWidth  - margin.left - margin.right
    const height = el.clientHeight - margin.top  - margin.bottom
    if (width <= 0 || height <= 0) return

    const x = d3.scaleBand().rangeRound([0, width]).padding(1).domain(dimensions)
    const y = {}
    dimensions.forEach(d => {
      const vals = data.map(p => p[d])
      const allNum = vals.every(v => !isNaN(parseFloat(v)))
      if (allNum) {
        y[d] = d3.scaleLinear()
          .domain(d3.extent(data, p => +p[d]))
          .range([height, 0])
      } else {
        const unique = [...new Set(vals)]
        y[d] = d3.scalePoint().domain(unique).range([height, 0])
      }
    })

    const svg = d3.select(el).append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const line = d3.line()
    const path = dd => line(dimensions.map(p => [x(p), y[p](dd[p])]))

    svg.append('g').attr('class', 'background')
      .selectAll('path').data(data).join('path')
      .attr('d', path).attr('fill', 'none')
      .attr('stroke', '#ddd').attr('stroke-opacity', 0.4)

    svg.append('g').attr('class', 'foreground')
      .selectAll('path').data(data).join('path')
      .attr('d', path).attr('fill', 'none')
      .attr('stroke', dd => {
        const pos = labels.indexOf(dd.Label)
        return labelColors[pos] ?? '#888'
      })
      .attr('stroke-opacity', 0.7).attr('stroke-width', '1px')

    const axes = svg.selectAll('.dimension').data(dimensions).join('g')
      .attr('class', 'dimension')
      .attr('transform', d => `translate(${x(d)})`)

    axes.append('g').attr('class', 'axis')
      .each(function(d) {
        d3.select(this).call(d3.axisLeft(y[d]).ticks(4))
      })
      .append('text')
        .attr('fill', '#000').style('text-anchor', 'middle')
        .attr('y', -9).style('font-size', '10px')
        .text(d => d)
  }, [data, dimensions, labelColors, labels])

  return <div ref={ref} style={{ width: '100%', height: '50%', overflow: 'hidden' }} />
}

export function StatisticsTab({ selection }) {
  const { dots, labels, labelColors } = useStore()

  if (!selection) return <div style={{ padding: 8, fontSize: 11, color: '#888' }}>No selection</div>
  if (!labels.length) return <div style={{ padding: 8, fontSize: 11, color: '#888' }}>Load feature file first</div>

  const { top, bottom } = computeStats(selection.ids, dots, labels)
  if (!top.length) return <div style={{ padding: 8, fontSize: 11, color: '#888' }}>No data</div>

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ParallelCoord data={top}    dimensions={['Label','TP','TN','FP','FN']}               labelColors={labelColors} labels={labels} />
      <ParallelCoord data={bottom} dimensions={['Label','Precision','Recall','Accuracy','F1']} labelColors={labelColors} labels={labels} />
    </div>
  )
}

// ── ClusteringTab ──────────────────────────────────────────────────────────────

const CLUSTER_BY_OPTIONS = [
  { label: 'ACT tSNE', value: 'act' },
  { label: 'FEA tSNE', value: 'fea' },
  { label: 'PRD tSNE', value: 'prd' },
]

function FlowerPlot({ dot, labelColors, filterLabels, attrSelectAll = true, size = 50 }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current || !dot) return
    d3.select(ref.current).selectAll('*').remove()

    const svg = d3.select(ref.current)
    const r = size / 2
    const innerR = 0.3 * r
    const cx = r, cy = r
    const g = svg.append('g').attr('transform', `translate(${cx},${cy})`)

    const probs  = dot.predProb  ?? []
    const truths = dot.trueLabel ?? []
    const activeLabels = attrSelectAll
      ? labelColors.map((_, i) => i)
      : filterLabels

    const pie = d3.pie().sort(null).value(() => 1)
    const arc = d3.arc()
      .innerRadius(innerR)
      .outerRadius(dd => (r - innerR) * parseFloat(dd.data) + innerR)
    const outlineArc = d3.arc().innerRadius(innerR).outerRadius(r)

    g.selectAll('.solidArc')
      .data(pie(probs)).join('path')
      .attr('class', 'solidArc')
      .attr('fill', (_, i) => activeLabels.includes(i) ? (labelColors[i] ?? 'none') : 'none')
      .attr('fill-opacity', 0.5).attr('d', arc)
      .attr('stroke', (dd, i) => {
        if (!activeLabels.includes(i)) return 'none'
        if (truths[i] === 1) return labelColors[i] ?? '#000'
        return dd.data > 0.5 ? '#000' : 'none'
      })
      .attr('stroke-width', '1px')

    g.selectAll('.outlineArc')
      .data(pie(truths)).join('path')
      .attr('class', 'outlineArc').attr('fill', 'none')
      .attr('stroke', (dd, i) => dd.data === 1 ? (labelColors[i] ?? 'none') : 'none')
      .attr('d', outlineArc)

    // center circle with id
    g.append('circle').attr('r', 11).attr('fill', '#000')
    g.append('circle').attr('r', 10).attr('fill', '#fff')
    g.append('text').attr('dy', '.3em').style('text-anchor', 'middle')
      .style('fill', '#000').style('font-size', '10px').text(dot.id)
  }, [dot, labelColors, filterLabels, attrSelectAll, size])

  return <svg ref={ref} width={size} height={size} style={{ cursor: 'pointer' }} />
}

export function ClusteringTab({ selection, dataType }) {
  const { dots, labelColors, filterLabels, attrSelectAll, selectDot } = useStore()
  const [clusterBy,     setClusterBy]     = useState('act')
  const [clusterWith,   setClusterWith]   = useState('kmean')
  const [clusterNum,    setClusterNum]    = useState(3)
  const [eps,           setEps]           = useState(0.01)
  const [minSample,     setMinSample]     = useState(3)
  const [result,        setResult]        = useState(null)
  const [loading,       setLoading]       = useState(false)

  const selectionIds = selection?.ids ?? []

  const runClustering = useCallback(() => {
    if (!selectionIds.length) return
    const selDots = dots.filter(d => selectionIds.includes(d.id))
    const vectors = getClusterVectors(selectionIds, selDots, clusterBy)
    if (!vectors.length) return

    setLoading(true)
    const req = clusterWith === 'kmean'
      ? getCluster({ vectors, clusterNum })
      : getClusterDBSCAN({ vectors, eps, min_samples: minSample })

    req.then(data => {
      // attach dot refs for display
      const clusterResult = data.cluster
      const numCluster = data.num_cluster ?? clusterNum
      const dotsInOrder = selDots.filter(d => {
        if (clusterBy === 'act') return d.xTrueLabel != null && d.yTrueLabel != null
        if (clusterBy === 'fea') return d.xFeature != null && d.yFeature != null
        return d.xPredict != null && d.yPredict != null
      })
      setResult({ clusterResult, numCluster, dotsInOrder,
        silh: data.Silh_score, davies: data.davies_score,
        outlier: data.outlier ?? false })
      setLoading(false)
    }).catch(e => { console.error(e); setLoading(false) })
  }, [selectionIds, dots, clusterBy, clusterWith, clusterNum, eps, minSample])

  // Run on selection or controls change
  useEffect(() => { if (selectionIds.length) runClustering() }, [selectionIds.join(','), clusterBy, clusterWith, clusterNum, eps, minSample])

  if (!selection) return <div style={{ padding: 8, fontSize: 11, color: '#888' }}>No selection</div>

  const clusterColors = d3.schemeCategory10

  return (
    <div style={{ position: 'relative', overflow: 'auto', height: '100%' }}>
      {/* Controls */}
      <div className="cluster-controls" style={{ padding: '4px 8px', fontSize: 11 }}>
        <div style={{ marginBottom: 4 }}>Cluster By:{' '}
          <select className="clusterby-selection" value={clusterBy} onChange={e => setClusterBy(e.target.value)}>
            {CLUSTER_BY_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </div>
        <div style={{ marginBottom: 4 }}>Cluster Methods:{' '}
          <select className="clusterby-selection" value={clusterWith} onChange={e => setClusterWith(e.target.value)}>
            <option value="kmean">K-Means</option>
            <option value="dbscan">DBSCAN</option>
          </select>
        </div>

        {clusterWith === 'kmean' && (
          <div style={{ marginBottom: 4 }}>
            Clusters:{' '}
            <input type="range" className="cluster-slider" min={1} max={5} step={1}
              value={clusterNum} onChange={e => setClusterNum(+e.target.value)} />
            {' '}{clusterNum}
          </div>
        )}

        {clusterWith === 'dbscan' && (
          <>
            <div style={{ marginBottom: 4 }}>EPS:{' '}
              <input type="range" className="cluster-slider" min={0} max={1} step={0.01}
                value={eps} onChange={e => setEps(+e.target.value)} />
              {' '}{eps}
            </div>
            <div style={{ marginBottom: 4 }}>MinPts:{' '}
              <input type="range" className="cluster-slider" min={3} max={10} step={1}
                value={minSample} onChange={e => setMinSample(+e.target.value)} />
              {' '}{minSample}
            </div>
          </>
        )}
      </div>

      {loading && <div style={{ padding: 8, fontSize: 11, color: '#888' }}>Clustering…</div>}

      {result && !loading && (
        <>
          {/* Legends */}
          <div id="cluster-legends" className="cluster-legends">
            {Array.from({ length: result.numCluster }, (_, i) => (
              <div key={i} className="cluster-legends-btn">
                <span style={{
                  display: 'inline-block', width: 12, height: 12,
                  background: clusterColors[i], opacity: 0.5, marginRight: 4
                }} />
                {i === result.numCluster - 1 && result.outlier ? 'Outliers' : `Cluster ${i + 1}`}
              </div>
            ))}
          </div>

          {/* Cluster rows */}
          {Array.from({ length: result.numCluster }, (_, ci) => (
            <div key={ci} style={{
              width: 'calc(100% - 210px)', marginLeft: 200,
              height: 'auto', marginTop: 5,
              border: `2px solid ${clusterColors[ci]}`,
              overflowY: 'auto', overflowX: 'hidden',
              display: 'flex', flexWrap: 'wrap',
            }}>
              {result.dotsInOrder.map((dot, j) =>
                result.clusterResult[j] === ci ? (
                  <div key={dot.id}
                    style={{ width: 50, height: 50, marginRight: 2, float: 'left' }}
                    onClick={() => selectDot(dot)}
                  >
                    <FlowerPlot dot={dot} labelColors={labelColors} filterLabels={filterLabels} attrSelectAll={attrSelectAll} size={50} />
                  </div>
                ) : null
              )}
            </div>
          ))}

          {/* Quality scores */}
          <div className="cluster-coefficient" style={{ padding: 8, fontSize: 11, marginTop: 12 }}>
            <div style={{ marginBottom: 8 }}><strong>Cluster Quality Scores</strong></div>
            <div style={{ marginBottom: 6 }}>
              <strong>Silhouette Score:</strong> {result.silh?.toFixed(4) ?? 'N/A'}
              {' '}
              <span className="cluster-hint-icon" title="The best value is 1&#10;The worst value is -1">?</span>
            </div>
            <div style={{ marginBottom: 0 }}>
              <strong>Davies Bouldin Score:</strong> {result.davies?.toFixed(4) ?? 'N/A'}
              {' '}
              <span className="cluster-hint-icon" title="The minimum score is zero.&#10;Lower is better.">?</span>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
