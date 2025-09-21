# OMR Evaluation System - Deployment Guide

## Quick Deploy to Vercel

### 🔧 **FIXED Deployment Issues:**
✅ Removed `cryptography==41.0.8` causing build errors
✅ Added missing `PyJWT==2.8.0` dependency
✅ Switched to `opencv-python-headless` for cloud compatibility
✅ Optimized dependency versions for Vercel

### Prerequisites
1. GitHub account
2. Vercel account (connect to GitHub)

### Deployment Steps

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Fixed dependencies for Vercel deployment"
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
   JWT_SECRET_KEY=1-Ww4g2RFRQEO4MhBR8U1yyKBbkElkTiIIHSRHpEJKg
   ```

4. **Database Setup:**
   - Use a cloud PostgreSQL service (like Supabase, Railway, or Neon)
   - Update the `DATABASE_URL` environment variable

### 📦 **Optimized Dependencies:**
- **Core:** Flask, Flask-CORS, Flask-SQLAlchemy, PyJWT
- **Database:** psycopg2-binary, SQLAlchemy
- **Image Processing:** opencv-python-headless, Pillow, numpy
- **Data:** pandas (for OMR Excel processing)

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
- `backend/` - Flask API with optimized dependencies
- `vercel.json` - Deployment configuration
- `backend/requirements.txt` - Fixed Python dependencies
- `runtime.txt` - Python 3.11 specification

### Features
- ✅ OMR Sheet Processing
- ✅ Real-time Results Dashboard  
- ✅ Authentication System
- ✅ File Upload & Management
- ✅ Export to CSV
- ✅ Live Terminal Output
- ✅ Enhanced Scoring System

**🚀 Ready for deployment!** The cryptography dependency error is now fixed.