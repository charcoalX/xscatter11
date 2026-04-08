import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

const CIRCLE_W = 40

const COORD_MAP = {
  act:    d => ({ x: d.xTrueLabel, y: d.yTrueLabel }),
  fea:    d => ({ x: d.xFeature,   y: d.yFeature   }),
  prd:    d => ({ x: d.xPredict,   y: d.yPredict   }),
  layer1: d => ({ x: d.xLayer1,    y: d.yLayer1    }),
  layer3: d => ({ x: d.xLayer3,    y: d.yLayer3    }),
  layer5: d => ({ x: d.xLayer5,    y: d.yLayer5    }),
}

// Ray-casting point-in-polygon
function pointInPolygon(point, polygon) {
  const [px, py] = point
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]
    const intersect = (yi > py) !== (yj > py) &&
      px < ((xj - xi) * (py - yi)) / (yj - yi) + xi
    if (intersect) inside = !inside
  }
  return inside
}

export default function ScatterPlot({
  dots = [],
  type = 'act',
  distanceOfErrors = [],
  selectedDots = [],
  selectImageIds = [],
  hoveredDot = null,
  labelColors = [],
  filterLabels = [],
  attrSelectAll = true,
  dotMode = 'default',
  bgOpacity = 1,
  selectMode = 'single',
  onDotClick,
  onDotHover,
  onLassoEnd,
}) {
  const containerRef = useRef(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setSize({ w: width, h: height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const container = containerRef.current
    if (!container || dots.length === 0) return
    const width  = size.w || container.clientWidth
    const height = size.h || container.clientHeight
    if (width === 0 || height === 0) return

    d3.select(container).selectAll('svg').remove()

    const getCoords = COORD_MAP[type] ?? COORD_MAP.act
    const validDots = dots.filter(d => {
      const c = getCoords(d)
      return c.x != null && c.y != null
    })
    if (!validDots.length) return

    // pixel coords — same formula as original
    const toX = v => v * (width  - CIRCLE_W * 2) + CIRCLE_W
    const toY = v => v * (height - 20 - CIRCLE_W * 2) + CIRCLE_W

    // opacity
    const errMin = d3.min(distanceOfErrors) ?? 0
    const errMax = d3.max(distanceOfErrors) ?? 1
    const opacityScale = d3.scaleLinear().domain([errMin, errMax]).range([0.3, 1])

    // selected id → color
    const selColorMap = new Map()
    for (const sel of selectedDots)
      for (const id of sel.ids) selColorMap.set(id, sel.color)

    // active label indices
    const activeLabels = attrSelectAll
      ? labelColors.map((_, i) => i)
      : filterLabels

    // dot fill color
    const getDotColor = dot => {
      if (dotMode !== 'prediction') return '#252525'
      if (!labelColors.length || !dot.predProb?.length) return '#252525'
      const maxIdx = activeLabels.reduce((best, i) =>
        dot.predProb[i] > (dot.predProb[best] ?? -1) ? i : best, activeLabels[0] ?? 0)
      return labelColors[maxIdx] ?? '#252525'
    }

    // ── SVG ────────────────────────────────────────────────────────
    const svg = d3.select(container).append('svg')
      .attr('width', width).attr('height', height)
      .style('display', 'block')

    const zoomRect = svg.append('rect')
      .attr('class', 'zoom-rect')
      .attr('width', width).attr('height', height)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')

    const g = svg.append('g').attr('class', `scatter-layer-${type}`)

    // ── dots ───────────────────────────────────────────────────────
    const groups = g.selectAll('g.dot')
      .data(validDots, d => d.id)
      .join('g')
        .attr('class', `dot scatterdot-${type}`)
        .attr('id', d => `scatterdot-${type}-${d.id}`)
        .attr('transform', d => {
          const { x, y } = getCoords(d)
          return `translate(${toX(x)},${toY(y)})`
        })
        .attr('cursor', 'pointer')

    // base circle
    groups.append('circle')
      .attr('class', `scattercircle-${type}`)
      .attr('cx', CIRCLE_W / 2).attr('cy', CIRCLE_W / 2)
      .attr('transform', `translate(-${CIRCLE_W/2},-${CIRCLE_W/2})`)
      .attr('r', 5)
      .attr('fill', d => getDotColor(d))
      .attr('fill-opacity', d => opacityScale(distanceOfErrors[d.id] ?? errMax))
      .attr('opacity', bgOpacity)

    // selected highlight + label
    groups.filter(d => selColorMap.has(d.id))
      .each(function(d) {
        const grp = d3.select(this)
        grp.append('circle')
          .attr('class', `scatterselectedcircle-${type}`)
          .attr('cx', CIRCLE_W / 2).attr('cy', CIRCLE_W / 2)
          .attr('transform', `translate(-${CIRCLE_W/2},-${CIRCLE_W/2})`)
          .attr('r', 9)
          .attr('fill', selColorMap.get(d.id))
          .attr('fill-opacity', opacityScale(distanceOfErrors[d.id] ?? errMax))
        grp.append('text')
          .attr('class', `scatterselectedtext-${type}`)
          .attr('dx', '1em').attr('dy', '-0.7em')
          .style('text-anchor', 'middle')
          .style('fill', '#000').style('font-size', '12px')
          .style('pointer-events', 'none')
          .text(d.id)
      })

    // clicked-dot red ring highlight
    const clickedSet = new Set(selectImageIds)
    groups.filter(d => clickedSet.has(d.id))
      .append('circle')
        .attr('class', `red-ring-clicked scatterclickedcircle-${type}`)
        .attr('cx', CIRCLE_W / 2).attr('cy', CIRCLE_W / 2)
        .attr('transform', `translate(-${CIRCLE_W/2},-${CIRCLE_W/2})`)
        .attr('r', 7)
        .attr('fill', 'none')
        .attr('stroke', 'red')
        .attr('stroke-width', 2)
        .attr('pointer-events', 'none')

    // flower plot
    if (dotMode === 'flower') {
      groups.each(function(dot) {
        if (!dot.predProb?.length) return
        const grp = d3.select(this)
        const r = CIRCLE_W / 2
        const innerR = 0.3 * r
        const pie = d3.pie().sort(null).value(() => 1)
        const arc = d3.arc()
          .innerRadius(innerR)
          .outerRadius(dd => (r - innerR) * parseFloat(dd.data) + innerR)
        const outlineArc = d3.arc().innerRadius(innerR).outerRadius(r)

        grp.selectAll('.solidArc')
          .data(pie(dot.predProb)).join('path')
          .attr('class', 'solidArc')
          .attr('fill', (_, i) => activeLabels.includes(i) ? (labelColors[i] ?? 'none') : 'none')
          .attr('fill-opacity', 0.5).attr('d', arc)
          .attr('stroke', (dd, i) => {
            if (!activeLabels.includes(i)) return 'none'
            if (dot.trueLabel?.[i] === 1) return labelColors[i] ?? '#000'
            return dd.data > 0.5 ? '#000' : 'none'
          })
          .attr('stroke-width', '1px')

        grp.selectAll('.outlineArc')
          .data(pie(dot.trueLabel ?? [])).join('path')
          .attr('class', 'outlineArc')
          .attr('fill', 'none')
          .attr('stroke', (dd, i) =>
            dd.data === 1 ? (labelColors[i] ?? 'none') : 'none')
          .attr('d', outlineArc)
      })
    }

    // ── dot events (click + hover) — shared by both modes ──────────
    groups
      .on('click', (event, d) => { event.stopPropagation(); onDotClick?.(d) })
      .on('mouseenter', (_, d) => onDotHover?.(d))
      .on('mouseleave', ()     => onDotHover?.(null))

    // ── zoom state — shared (used by lasso for coord conversion) ───
    const zoomState = { k: 1, x: 0, y: 0 }
    const zoom = d3.zoom()
      .scaleExtent([1, 20])
      .on('zoom', event => {
        g.attr('transform', event.transform)
        zoomState.k = event.transform.k
        zoomState.x = event.transform.x
        zoomState.y = event.transform.y
      })

    // ── SINGLE MODE: zoom on rect ───────────────────────────────────
    if (selectMode === 'single') {
      zoomRect.call(zoom)
    }

    // ── MULTIPLE MODE: lasso ────────────────────────────────────────
    if (selectMode === 'multiple') {
      // Wheel-only zoom still works in lasso mode (like original)
      svg.call(
        d3.zoom()
          .scaleExtent([1, 20])
          .filter(e => e.type === 'wheel')
          .on('zoom', event => {
            g.attr('transform', event.transform)
            zoomState.k = event.transform.k
            zoomState.x = event.transform.x
            zoomState.y = event.transform.y
          })
      )

      const lassoG = svg.append('g').attr('class', 'lasso')
      let lassoPoints = []
      let dragging = false

      svg.on('mousedown.lasso', function(event) {
        // Only start lasso on SVG background, not on dots
        if (event.target !== zoomRect.node() && !svg.node().isSameNode(event.target)) {
          // clicked on a dot — let dot click handler run
          return
        }
        event.preventDefault()
        dragging = true
        const [x, y] = d3.pointer(event, svg.node())
        lassoPoints = [[x, y]]

        lassoG.selectAll('*').remove()
        lassoG.append('circle')
          .attr('class', 'origin')
          .attr('cx', x).attr('cy', y).attr('r', 4)
        lassoG.append('path').attr('class', 'drawn').attr('fill', '#525252').attr('fill-opacity', 0.05)
        lassoG.append('path').attr('class', 'loop_close')
      })

      svg.on('mousemove.lasso', function(event) {
        if (!dragging) return
        const [x, y] = d3.pointer(event, svg.node())
        lassoPoints.push([x, y])

        const pathStr = 'M' + lassoPoints.map(p => p.join(',')).join('L') + 'Z'
        lassoG.select('.drawn').attr('d', pathStr)

        const first = lassoPoints[0]
        const last  = lassoPoints[lassoPoints.length - 1]
        lassoG.select('.loop_close')
          .attr('d', `M${last[0]},${last[1]}L${first[0]},${first[1]}`)
          .attr('stroke', '#525252').attr('stroke-width', 2)
          .attr('stroke-dasharray', '4,4').attr('fill', 'none')
      })

      svg.on('mouseup.lasso', function() {
        if (!dragging || lassoPoints.length < 3) {
          dragging = false
          lassoPoints = []
          lassoG.selectAll('*').remove()
          return
        }
        dragging = false

        // Convert each dot from g-space → SVG space using current zoom transform
        const toSVG = (gx, gy) => [
          gx * zoomState.k + zoomState.x,
          gy * zoomState.k + zoomState.y,
        ]

        const selectedIds = []
        groups.each(function(d) {
          const { x, y } = getCoords(d)
          const [svgX, svgY] = toSVG(toX(x), toY(y))
          if (pointInPolygon([svgX, svgY], lassoPoints)) {
            selectedIds.push(d.id)
          }
        })

        lassoG.selectAll('*').remove()
        lassoPoints = []

        if (selectedIds.length > 0) onLassoEnd?.(selectedIds)
      })

      // Cancel lasso if mouse leaves SVG
      svg.on('mouseleave.lasso', function() {
        if (!dragging) return
        dragging = false
        lassoG.selectAll('*').remove()
        lassoPoints = []
      })
    }

    return () => d3.select(container).selectAll('svg').remove()
  }, [dots, type, distanceOfErrors, selectedDots, selectImageIds, labelColors, filterLabels,
      attrSelectAll, dotMode, bgOpacity, selectMode, size])

  // Hover pulse — lightweight, no full redraw
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    // remove pulse from all rings in this panel
    d3.select(container).selectAll('.red-ring-clicked')
      .classed('red-ring-pulse', false)
    // add pulse to the hovered dot if it's a clicked dot
    if (hoveredDot && selectImageIds.includes(hoveredDot.id)) {
      d3.select(container)
        .select(`#scatterdot-${type}-${hoveredDot.id}`)
        .select('.red-ring-clicked')
        .classed('red-ring-pulse', true)
    }
  }, [hoveredDot, selectImageIds, type])

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
}
