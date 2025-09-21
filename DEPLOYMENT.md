# OMR Evaluation System - Deployment Guide

## Quick Deploy to Vercel

### Prerequisites
1. GitHub account
2. Vercel account (connect to GitHub)

### Deployment Steps

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Deploy on Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect the configuration

3. **Environment Variables:**
   After deployment, add these environment variables in Vercel dashboard:
   ```
   VITE_API_URL=https://your-app-name.vercel.app
   DATABASE_URL=your_postgresql_connection_string
   JWT_SECRET_KEY=your_secret_key
   ```

4. **Database Setup:**
   - Use a cloud PostgreSQL service (like Supabase, PlanetScale, or Railway)
   - Update the `DATABASE_URL` environment variable

### Local Development
```bash
# Frontend
cd frontend
npm install
npm run dev

# Backend
cd backend
pip install -r requirements.txt
python app.py
```

### Project Structure
- `frontend/` - React.js with Vite
- `backend/` - Flask API
- `vercel.json` - Deployment configuration
- `requirements.txt` - Python dependencies

### Features
- OMR Sheet Processing
- Real-time Results Dashboard
- Authentication System
- File Upload & Management
- Export to CSV