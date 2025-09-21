// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export const apiUrl = (endpoint) => `${API_BASE_URL}${endpoint}`;

export default {
  apiUrl,
  API_BASE_URL
};