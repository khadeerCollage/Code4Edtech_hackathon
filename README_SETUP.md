# OMR Evaluation System - Complete Setup Guide

## 🚀 System Overview

This is a complete OMR (Optical Mark Recognition) evaluation system with:
- **Backend**: Flask API with PostgreSQL database
- **Frontend**: React.js with Vite
- **Core**: Advanced OMR processing using OpenCV and machine learning

## 📋 Prerequisites

### Software Requirements
1. **Python 3.8+** with pip
2. **Node.js 16+** with npm/bun
3. **PostgreSQL 12+** (recommended) or SQLite for development
4. **Git** for version control

### Python Packages (install via pip)
```bash
pip install flask flask-cors flask-sqlalchemy python-dotenv
pip install opencv-python numpy pandas openpyxl scikit-learn
pip install psycopg2-binary  # For PostgreSQL
pip install pillow werkzeug
```

### Node.js Packages (for frontend)
```bash
cd frontend
npm install  # or bun install
```

## 🗄️ Database Setup

### Option 1: PostgreSQL (Recommended for Production)

1. **Install PostgreSQL**
   - Windows: Download from https://www.postgresql.org/download/
   - macOS: `brew install postgresql`
   - Linux: `sudo apt-get install postgresql postgresql-contrib`

2. **Create Database and User**
   ```sql
   -- Connect to PostgreSQL as superuser
   psql -U postgres
   
   -- Create database
   CREATE DATABASE omr_evaluation_db;
   
   -- Create user (optional)
   CREATE USER omr_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE omr_evaluation_db TO omr_user;
   
   -- Exit psql
   \q
   ```

3. **Initialize Schema**
   ```bash
   cd backend
   psql -U postgres -d omr_evaluation_db -f database_schema.sql
   ```

### Option 2: SQLite (Quick Development Setup)
- No setup needed - SQLite file will be created automatically
- Good for testing and development

## ⚙️ Configuration Setup

### 1. Backend Configuration
```bash
cd backend
cp .env.example .env
```

Edit `.env` with your settings:
```env
# For PostgreSQL
DATABASE_URL=postgresql://omr_user:your_password@localhost:5432/omr_evaluation_db

# For SQLite (development)
# DATABASE_URL=sqlite:///omr_evaluation.db

SECRET_KEY=your-super-secret-key-here
FLASK_ENV=development
CORS_ORIGINS=http://localhost:5173
```

### 2. Frontend Configuration
The React frontend is pre-configured to work with the Flask backend.

## 🚀 Running the System

### Step 1: Start Backend Server
```bash
cd backend
python app.py
```

Backend will start on: `http://localhost:5000`

### Step 2: Start Frontend Development Server
```bash
cd frontend
npm run dev  # or bun dev
```

Frontend will start on: `http://localhost:5173`

## 🧪 Testing the System

### 1. Test Backend API
Visit: `http://localhost:5000/` - Should show "OMR Evaluation Backend is running!"

### 2. Test Database Connection
The backend will automatically:
- Create database tables if they don't exist
- Display connection status in console
- Create a default test user

### 3. Test Frontend
Visit: `http://localhost:5173` - Should show the OMR system interface

### 4. Test Complete Workflow
1. **Register/Login**: Create account or use test credentials
2. **Upload OMR Sheets**: Go to Upload page, select images and answer key Excel file
3. **View Results**: Check Dashboard for processing status
4. **Review Batches**: Click on completed batches to see detailed results

## 📁 File Structure

```
backend/
├── app.py                    # Main Flask application
├── test_green_circle_omr_new.py  # Core OMR processing
├── database_schema.sql       # PostgreSQL schema
├── database_utils.py         # Database utilities
├── .env.example             # Environment template
├── uploads/                 # Uploaded images and answer keys
└── results/                 # Processing results and debug files

frontend/
├── src/
│   ├── pages/              # React pages (Dashboard, Upload, Results)
│   ├── components/         # Reusable components
│   └── App.jsx            # Main React app
├── package.json
└── vite.config.js

dataset/
└── data/
    ├── Set_A/             # Sample OMR images Set A
    ├── Set_B/             # Sample OMR images Set B
    └── Key (Set A and B).xlsx  # Answer keys
```

## 🔗 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login

### OMR Processing
- `GET /api/batches` - Get all processing batches
- `POST /api/upload` - Upload OMR images and answer key
- `GET /api/batches/<batch_id>` - Get batch details
- `GET /api/batches/<batch_id>/results` - Get processing results
- `GET /api/batches/<batch_id>/export` - Export results as CSV

### Dashboard
- `GET /api/dashboard/metrics` - Get dashboard statistics

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check PostgreSQL is running: `pg_ctl status`
   - Verify credentials in `.env` file
   - Test connection: `psql -U omr_user -d omr_evaluation_db`

2. **Python Import Errors**
   - Install missing packages: `pip install -r requirements.txt`
   - Activate virtual environment if using one

3. **Frontend Build Errors**
   - Clear node modules: `rm -rf node_modules && npm install`
   - Check Node.js version: `node --version`

4. **File Upload Issues**
   - Check upload folder permissions
   - Verify file size limits (50MB default)
   - Ensure supported file formats (JPG, PNG, XLSX)

### Debug Mode
- Backend: Set `FLASK_DEBUG=True` in `.env`
- Frontend: Development server includes hot reload
- Check browser console for frontend errors
- Check terminal output for backend errors

## 📊 Database Schema Overview

### Tables Created
1. **users** - User authentication and management
2. **omr_batches** - Processing batch information
3. **omr_sheets** - Individual OMR sheet results
4. **subject_scores** - Detailed subject-wise scoring

### Views Available
- **batch_statistics** - Aggregated batch performance data
- **detailed_results** - Combined batch and sheet information

## 🎯 Features Implemented

### Backend Features
- ✅ Complete Flask REST API
- ✅ PostgreSQL/SQLite database support
- ✅ File upload handling (images + Excel)
- ✅ OMR processing integration
- ✅ Subject-wise score calculation
- ✅ Batch processing management
- ✅ CSV export functionality
- ✅ Error handling and logging

### Frontend Features
- ✅ React.js with modern UI
- ✅ User authentication pages
- ✅ File upload with drag-and-drop
- ✅ Dashboard with metrics
- ✅ Results table with filtering
- ✅ Responsive design
- ✅ Real-time processing status

### OMR Core Features
- ✅ Advanced circle detection
- ✅ Set A/B automatic detection
- ✅ Excel answer key integration
- ✅ Subject-wise breakdown
- ✅ Multiple marking detection
- ✅ Debug visualization
- ✅ Comprehensive JSON reporting

## 🚀 Next Steps

1. **Production Deployment**
   - Use Gunicorn/uWSGI for Flask
   - Build React app for production
   - Set up reverse proxy (nginx)
   - Configure SSL certificates

2. **Advanced Features**
   - Real-time processing updates via WebSocket
   - User role management (admin/teacher/student)
   - Email notifications
   - Advanced analytics and reporting
   - Bulk operations and batch management

3. **Performance Optimization**
   - Background task processing (Celery)
   - Database query optimization
   - Image processing caching
   - CDN for static files

## 💡 Default Test Credentials

For testing purposes, the system creates default users:
- **Email**: `test@example.com`
- **Password**: `password123`

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Check database connection and schema
4. Review backend/frontend console logs
5. Ensure all environment variables are set correctly