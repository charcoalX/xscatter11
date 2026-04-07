import { create } from 'zustand'
import * as d3 from 'd3'

// ── helpers ──────────────────────────────────────────────────────────────────

export function getAttributeColors(n) {
  if (n === 0) return []
  return Array.from({ length: n }, (_, i) =>
    d3.interpolatePlasma(n <= 1 ? 0.5 : i / (n - 1))
  )
}

export function prepareScatterplot(rawData) {
  const dots = []
  const errors = []
  let index = 0
  for (const img of Object.values(rawData)) {
    dots.push({
      id: index,
      error:       img['Distance of error'],
      xTrueLabel:  img['truelabel17-x'],
      yTrueLabel:  img['truelabel17-y'],
      xFeature:    img['feature2048-x'],
      yFeature:    img['feature2048-y'],
      xPredict:    img['prediction17-x'],
      yPredict:    img['prediction17-y'],
      xLayer1: img['s1-x'], yLayer1: img['s1-y'],
      xLayer2: img['s2-x'], yLayer2: img['s2-y'],
      xLayer3: img['s3-x'], yLayer3: img['s3-y'],
      xLayer4: img['s4-x'], yLayer4: img['s4-y'],
      xLayer5: img['s5-x'], yLayer5: img['s5-y'],
      predProb:  img['predProb'],   // array
      trueLabel: img['trueLabel'],  // array
    })
    errors.push(img['Distance of error'])
    index++
  }
  return { dots, errors }
}

// ── store ─────────────────────────────────────────────────────────────────────

export const useStore = create((set, get) => ({
  // Query options — mirrors main.embedding.options
  options: {
    'Data type': 'synthetic',
    'Embedding method': 'TSNE',
    'Distance of error': 'Cosine',
  },

  // Raw + processed data
  rawData: null,
  dots: [],
  distanceOfErrors: [],
  labelColors: [],
  numLabels: 0,
  labels: [],        // attribute names loaded from feature file

  // Attribute filter: [] = all selected; [0,2,5] = only those indices
  filterLabels: [],

  // Dot display mode: 'default' | 'prediction' | 'flower'
  dotMode: 'default',

  // Which plot type is shown: 'act' | 'fea' | 'prd' | 'layer1' | 'layer3' | 'layer5'
  plotType: 'act',

  // Compare mode: 'none' | '3layers' | '6layers'
  compareMode: 'none',

  // Background opacity (0–1)
  bgOpacity: 1,

  // Selections
  selectedDots: [],   // [{ selectId, color, ids, lock }]
  selectImageIds: [], // flat list of selected IDs (for Detailed Images)
  selectionCount: 0,

  // Hover
  hoveredDot: null,

  // Panel visibility
  attrStudyOpen: false,
  modelArchOpen: false,
  aiAssistantOpen: false,

  // ── actions ──────────────────────────────────────────────────────────────

  setOptions: (opts) => set({ options: opts }),

  loadData: (rawData) => {
    const { dots, errors } = prepareScatterplot(rawData)
    const numLabels = dots[0]?.predProb?.length ?? 0
    set({
      rawData,
      dots,
      distanceOfErrors: errors,
      numLabels,
      labelColors: getAttributeColors(numLabels),
    })
  },

  setLabels: (labels) => {
    const labelColors = getAttributeColors(labels.length)
    set({ labels, labelColors })
  },

  // Select All: clear filterLabels
  selectAllLabels: () => set({ filterLabels: [] }),

  // Toggle one label index in/out of filterLabels
  toggleFilterLabel: (idx) => {
    const { filterLabels } = get()
    const pos = filterLabels.indexOf(idx)
    if (pos !== -1) {
      set({ filterLabels: filterLabels.filter((_, i) => i !== pos) })
    } else {
      set({ filterLabels: [...filterLabels, idx].sort((a, b) => a - b) })
    }
  },

  setPlotType: (plotType) => set({ plotType }),
  setCompareMode: (compareMode) => set({ compareMode }),
  setDotMode: (dotMode) => set({ dotMode }),
  setBgOpacity: (bgOpacity) => set({ bgOpacity }),

  // Single-dot click: only adds image to bottom detail panel, no group created
  selectDot: (dot) => {
    const { selectImageIds } = get()
    if (selectImageIds.includes(dot.id)) return
    set({ selectImageIds: [...selectImageIds, dot.id] })
  },

  // Lasso multi-select
  addLassoSelection: (ids) => {
    if (!ids.length) return
    const { selectedDots, selectionCount } = get()
    const color = '#' + Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, '0')
    set({
      selectionCount: selectionCount + 1,
      selectedDots: [
        ...selectedDots,
        { selectId: selectionCount + 1, color, ids, lock: false },
      ],
    })
  },

  removeSelection: (selectId) => {
    const { selectedDots } = get()
    set({ selectedDots: selectedDots.filter(s => s.selectId !== selectId) })
  },

  // Move selection at fromIndex to toIndex (for drag-to-reorder)
  reorderSelections: (fromIndex, toIndex) => {
    const { selectedDots } = get()
    if (fromIndex === toIndex) return
    const next = [...selectedDots]
    const [moved] = next.splice(fromIndex, 1)
    next.splice(toIndex, 0, moved)
    set({ selectedDots: next })
  },

  updateSelectionColor: (selectId, color) => {
    const { selectedDots } = get()
    set({
      selectedDots: selectedDots.map(s =>
        s.selectId === selectId ? { ...s, color } : s
      ),
    })
  },

  clearSelections: () => set({ selectedDots: [], selectImageIds: [], selectionCount: 0 }),
  clearSelectImageIds: () => set({ selectImageIds: [] }),
  removeImageId: (id) => set(s => ({ selectImageIds: s.selectImageIds.filter(x => x !== id) })),

  setHoveredDot: (dot) => set({ hoveredDot: dot }),

  toggleAttrStudy:    () => set(s => ({ attrStudyOpen:    !s.attrStudyOpen })),
  toggleModelArch:    () => set(s => ({ modelArchOpen:    !s.modelArchOpen })),
  toggleAIAssistant:  () => set(s => ({ aiAssistantOpen:  !s.aiAssistantOpen })),
  closeModelArch:     () => set({ modelArchOpen: false }),
  closeAIAssistant:   () => set({ aiAssistantOpen: false }),
}))
