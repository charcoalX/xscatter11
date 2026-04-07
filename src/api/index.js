import axios from 'axios'

const api = axios.create({ baseURL: '/' })

export const queryAll = (params) =>
  api.post('/QueryAll', params).then(r => r.data)

export const getCluster = (params) =>
  api.post('/GetCluster', params).then(r => r.data)

export const getClusterDBSCAN = (params) =>
  api.post('/GetClusterDBSCAN', params).then(r => r.data)

export const getMutualInfo = (params) =>
  api.post('/GetMutualInfo', params).then(r => r.data)

export const getTsne = (params) =>
  api.post('/GetTsne', params).then(r => r.data)

export const getCountInfo = (params) =>
  api.post('/GetCountInfo', params).then(r => r.data)

export const getLRPHeatmap = (params) =>
  api.post('/GetLRPHeatmap', params).then(r => r.data)

export const askAssistant = (messages) =>
  api.post('/AskAssistant', { messages }).then(r => r.data)
