import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

const NODES = [
  {
    id: 'input', label: 'Input', sub1: '256×256×1', sub2: '→ 224×224', shape: null, flatSize: null,
    color: { fill: '#F1EFE8', stroke: '#888780', text: '#3a3a38', sub: '#5F5E5A' },
    hasLabel: true,
  },
  {
    id: 'scale1', label: 'scale1', sub1: 'conv1  7×7 s2', sub2: '+ BN + ReLU',
    shape: '112×112×64', flatSize: '802 816',
    color: { fill: '#E1F5EE', stroke: '#0F6E56', text: '#085041', sub: '#1a9068' },
  },
  {
    id: 'scale2', label: 'scale2', sub1: 'MaxPool s2', sub2: '3×Bottleneck',
    shape: '56×56×256', flatSize: '802 816',
    color: { fill: '#EEEDFE', stroke: '#534AB7', text: '#3C3489', sub: '#6b63c8' },
  },
  {
    id: 'scale3', label: 'scale3', sub1: '4×Bottleneck s2', sub2: '128 → 512 ch',
    shape: '28×28×512', flatSize: '401 408',
    color: { fill: '#E6F1FB', stroke: '#185FA5', text: '#0C447C', sub: '#2c7bc5' },
  },
  {
    id: 'scale4', label: 'scale4', sub1: '6×Bottleneck s2', sub2: '256 → 1024 ch',
    shape: '14×14×1024', flatSize: '200 704',
    color: { fill: '#FAEEDA', stroke: '#854F0B', text: '#633806', sub: '#a86510' },
  },
  {
    id: 'scale5', label: 'scale5', sub1: '3×Bottleneck s2', sub2: '512 → 2048 ch',
    shape: '7×7×2048', flatSize: '100 352',
    color: { fill: '#FAECE7', stroke: '#993C1D', text: '#712B13', sub: '#c04e26' },
  },
  {
    id: 'gap', label: 'GAP', sub1: 'GlobalAvgPool', sub2: '7×7 → 1×1',
    shape: '2048', flatSize: '2 048',
    color: { fill: '#EAF3DE', stroke: '#3B6D11', text: '#27500A', sub: '#4e8f17' },
  },
  {
    id: 'pred', label: 'Prediction', sub1: 'FC  2048→17', sub2: '+ Sigmoid',
    shape: '17 classes', flatSize: '17',
    color: { fill: '#FBEAF0', stroke: '#993556', text: '#72243E', sub: '#c24268' },
  },
]

export default function ModelArchPanel() {
  const svgRef = useRef(null)

  useEffect(() => {
    const container = svgRef.current
    if (!container) return

    const W = 1160, H = 150
    const NODE_W = 115, NODE_H = 78
    const GAP_X  = 14
    const Y_MID  = 16
    const totalW = NODES.length * NODE_W + (NODES.length - 1) * GAP_X
    const xStart = (W - totalW) / 2

    const nodes = NODES.map((d, i) => ({
      ...d,
      cx: xStart + i * (NODE_W + GAP_X) + NODE_W / 2,
      x:  xStart + i * (NODE_W + GAP_X),
      y:  Y_MID,
    }))

    d3.select(container).selectAll('*').remove()

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')

    // arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arch-arrow-react')
      .attr('viewBox', '0 0 10 10').attr('refX', 8).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto-start-reverse')
      .append('path').attr('d', 'M2 1L8 5L2 9')
        .attr('fill', 'none').attr('stroke', '#aaaaaa')
        .attr('stroke-width', 1.5).attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round')

    svg.append('text').attr('class', 'arch-section-header')
      .attr('x', W / 2).attr('y', 11)
      .text('ResNet-50 feature extraction pipeline')

    // arrows
    for (let i = 0; i < nodes.length - 1; i++) {
      const a = nodes[i], b = nodes[i + 1]
      svg.append('line').attr('class', 'arch-link')
        .attr('x1', a.x + NODE_W + 2).attr('y1', a.y + NODE_H / 2)
        .attr('x2', b.x - 2).attr('y2', b.y + NODE_H / 2)
        .attr('stroke', b.color.stroke)
        .attr('marker-end', 'url(#arch-arrow-react)')
    }

    // node groups
    const nodeG = svg.selectAll('.node-group').data(nodes).join('g')
      .attr('class', 'node-group')
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .style('cursor', 'default')

    nodeG.append('rect').attr('class', 'node-rect')
      .attr('width', NODE_W).attr('height', NODE_H).attr('rx', 6)
      .attr('fill', d => d.color.fill).attr('stroke', d => d.color.stroke).attr('stroke-width', 1.2)

    nodeG.append('rect').attr('width', NODE_W).attr('height', 3).attr('rx', 2)
      .attr('fill', d => d.color.stroke).attr('opacity', 0.85)

    nodeG.append('text').attr('class', 'node-title')
      .attr('x', NODE_W / 2).attr('y', 17)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', d => d.color.text).text(d => d.label)

    nodeG.append('text').attr('class', 'node-sub')
      .attr('x', NODE_W / 2).attr('y', 30)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', d => d.color.sub).text(d => d.sub1)

    nodeG.append('text').attr('class', 'node-sub')
      .attr('x', NODE_W / 2).attr('y', 41)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', d => d.color.sub).text(d => d.sub2)

    nodeG.filter(d => d.shape).append('line')
      .attr('x1', 6).attr('x2', NODE_W - 6).attr('y1', 52).attr('y2', 52)
      .attr('stroke', d => d.color.stroke).attr('stroke-width', 0.6).attr('opacity', 0.4)

    nodeG.filter(d => d.shape).append('text').attr('class', 'node-shape')
      .attr('x', NODE_W / 2).attr('y', 62)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', d => d.color.text).text(d => d.shape)

    // true-label badge on input node
    nodeG.filter(d => d.hasLabel).append('rect')
      .attr('x', 5).attr('y', NODE_H - 18).attr('width', NODE_W - 10).attr('height', 13).attr('rx', 2)
      .attr('fill', '#3B6D11')

    nodeG.filter(d => d.hasLabel).append('text')
      .attr('x', NODE_W / 2).attr('y', NODE_H - 11)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', '#EAF3DE').attr('font-size', '7px').attr('font-weight', '600')
      .text('true label  (1, 17)')

    // flat-size labels
    svg.selectAll('.flat-label').data(nodes.filter(d => d.flatSize)).join('text')
      .attr('class', 'flat-label')
      .attr('x', d => d.cx).attr('y', Y_MID + NODE_H + 10)
      .text(d => d.flatSize)

    // true-label path
    const inp = nodes[0], pred = nodes[nodes.length - 1]
    const pathY = Y_MID + NODE_H + 22
    svg.append('path').attr('class', 'true-label-path')
      .attr('d', `M${inp.cx} ${inp.y + NODE_H} L${inp.cx} ${pathY} L${pred.cx} ${pathY} L${pred.cx} ${pred.y + NODE_H}`)
      .attr('marker-end', 'url(#arch-arrow-react)')

    svg.append('text').attr('class', 'bce-label')
      .attr('x', (inp.cx + pred.cx) / 2).attr('y', pathY + 11)
      .text('BCE loss  ·  sigmoid_cross_entropy_with_logits')

  }, [])

  return (
    <div ref={svgRef} style={{ width: '100%', height: '100%' }} />
  )
}
