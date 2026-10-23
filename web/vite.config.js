import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api is proxied to the Python server. In production the API URL comes from VITE_API_URL
// (frontend on Vercel, API elsewhere) or is same-origin (single-container deploy).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
