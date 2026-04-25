import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    dedupe: ['react', 'react-dom'],
  },
  optimizeDeps: {
    include: ['react', 'react-dom', '@humeai/voice-react', '@xyflow/react'],
  },
  server: {
    proxy: {
      // Proxy backend API calls during dev so the frontend can call /swarms/*
      // and /api/* without dealing with CORS or hardcoded ports.
      '/swarms': { target: 'http://localhost:8000', changeOrigin: true },
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/healthz': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
