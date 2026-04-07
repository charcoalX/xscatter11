import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const FLASK_URL = 'http://127.0.0.1:8085'

const flaskEndpoints = [
  '/QueryAll',
  '/GetCluster',
  '/GetClusterDBSCAN',
  '/GetMutualInfo',
  '/GetTsne',
  '/GetCountInfo',
  '/GetLRPHeatmap',
  '/AskAssistant',
]

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // static files (images served by Flask)
      '/static': { target: FLASK_URL, changeOrigin: true },
      // API endpoints
      ...Object.fromEntries(
        flaskEndpoints.map(ep => [ep, { target: FLASK_URL, changeOrigin: true }])
      ),
    },
  },
})
