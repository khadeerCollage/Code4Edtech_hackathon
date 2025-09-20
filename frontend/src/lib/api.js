import axios from 'axios';

// Base URL: try env var first (Vite exposes import.meta.env), else relative to dev server using proxy.
const baseURL = import.meta.env.VITE_API_BASE || '';

export const api = axios.create({
  baseURL,
  headers: { 'Content-Type': 'application/json' }
});
