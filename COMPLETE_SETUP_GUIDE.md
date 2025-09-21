# 🚀 COMPLETE OMR EVALUATION SYSTEM - SETUP GUIDE
**Full-Stack Integration: PostgreSQL + Flask Backend + React Frontend**

## 📋 PREREQUISITES

### Required Software:
- **Python 3.8+** (Download from python.org)
- **Node.js 16+** (Download from nodejs.org) 
- **PostgreSQL 13+** (Download from postgresql.org)
- **Git** (Download from git-scm.com)

### System Requirements:
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 5GB free space
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+

---

## 🗄️ PHASE 1: DATABASE SETUP

### Step 1.1: Install and Start PostgreSQL
```bash
# Windows (using chocolatey)
choco install postgresql

# macOS (using homebrew)  
brew install postgresql
brew services start postgresql

# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### Step 1.2: Create Database and User
```bash
# Access PostgreSQL as superuser
psql -U postgres

# Or on some systems:
sudo -u postgres psql
```

### Step 1.3: Execute the Complete Schema
```sql
-- In psql, run our complete schema:
\i C:/Users/USER/Desktop/Code4Edtech_hackathon/Code4Edtech_hackathon/backend/frontend_based_complete_schema.sql

-- OR copy-paste the entire content of the file
```

### Step 1.4: Verify Database Setup
```sql
-- Check if everything was created:
\c omr_evaluation_db
\dt  -- List all tables
SELECT COUNT(*) FROM users;     -- Should show test users
SELECT COUNT(*) FROM omr_batches;  -- Should show test batches
\q   -- Exit psql
```

---

## 🐍 PHASE 2: BACKEND SETUP

### Step 2.1: Create Python Virtual Environment
```bash
# Navigate to project directory
cd "C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon"

# Create virtual environment (if not exists)
python -m venv hackathon_env

# Activate virtual environment
# Windows:
hackathon_env\Scripts\activate
# macOS/Linux:
source hackathon_env/bin/activate
```

### Step 2.2: Install Python Dependencies
```bash
# Navigate to backend folder
cd backend

# Install all requirements
pip install -r requirements.txt

# Verify installation
pip list | grep Flask
pip list | grep psycopg2
```

### Step 2.3: Set Environment Variables
Create `.env` file in backend folder:
```env
# Database Configuration
DATABASE_URL=postgresql://omr_user:omr_password123@localhost:5432/omr_evaluation_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=omr_evaluation_db
DB_USER=omr_user
DB_PASSWORD=omr_password123

# Flask Configuration
FLASK_APP=complete_frontend_integrated_app.py
FLASK_ENV=development
SECRET_KEY=your-super-secret-key-change-in-production
JWT_SECRET_KEY=jwt-secret-change-in-production

# File Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=104857600

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000
```

### Step 2.4: Test Backend
```bash
# Run the Flask application
python complete_frontend_integrated_app.py

# Expected output:
# ==================================================
# OMR EVALUATION SYSTEM - BACKEND SERVER
# ==================================================
# Frontend can access API at: http://127.0.0.1:5000
# CORS enabled for: http://localhost:3000, http://localhost:5173
# ==================================================
# * Running on http://127.0.0.1:5000
```

### Step 2.5: Verify API Endpoints
Open new terminal and test:
```bash
# Test basic connectivity
curl http://127.0.0.1:5000/api/auth/login

# Expected: {"error": "Email/Username and password are required"}
```

---

## ⚛️ PHASE 3: FRONTEND SETUP

### Step 3.1: Navigate to Frontend
```bash
# Open new terminal, navigate to frontend
cd "C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\frontend"
```

### Step 3.2: Install Frontend Dependencies
```bash
# Install all packages
npm install
# OR
yarn install

# Verify installation
npm list react
npm list vite
```

### Step 3.3: Create Frontend Environment File
Create `.env` file in frontend folder:
```env
# API Configuration
VITE_API_BASE_URL=http://127.0.0.1:5000/api
VITE_FILE_BASE_URL=http://127.0.0.1:5000/api/files

# Development Configuration
VITE_NODE_ENV=development
```

### Step 3.4: Update API Integration (Critical!)
Follow the **FRONTEND_INTEGRATION_GUIDE.md** to update:

1. **Login.jsx** - Replace mock login with real API
2. **Register.jsx** - Replace mock registration with real API  
3. **Dashboard.jsx** - Replace mock data with real API calls
4. **Upload.jsx** - Replace mock upload with real file upload
5. **Results.jsx** - Replace mock results with real API data
6. **Batches.jsx** - Replace mock batches with real API data

### Step 3.5: Start Frontend Development Server
```bash
# Start the development server
npm run dev
# OR
yarn dev

# Expected output:
# Local:   http://localhost:5173/
# Network: use --host to expose
```

---

## 🧪 PHASE 4: TESTING & VERIFICATION

### Step 4.1: Test User Authentication
1. **Open**: http://localhost:5173
2. **Login with test credentials**:
   - Email: `test@example.com`
   - Password: `password123`
3. **Verify**: Successfully redirected to dashboard

### Step 4.2: Test Registration
1. **Go to**: Register page
2. **Fill form with new details**:
   - Username: `newuser`
   - Email: `newuser@example.com`
   - Password: `newpassword123`
   - Full Name: `New Test User`
3. **Verify**: User created and logged in automatically

### Step 4.3: Test Dashboard
1. **Check**: Dashboard loads with real statistics
2. **Verify**: Recent batches show test data
3. **Confirm**: All metrics display correctly

### Step 4.4: Test File Upload
1. **Navigate**: To Upload page
2. **Select**: Sample OMR images from `dataset/data/Set_A/`
3. **Select**: Answer key Excel file from `dataset/data/Key (Set A and B).xlsx`
4. **Fill form**:
   - Exam Name: `Test Mathematics Exam`
   - Answer Key Version: `A`
   - School: `Test School`
   - Grade: `Grade 12`
   - Subject: `Mathematics`
5. **Upload**: Click upload and verify processing starts

### Step 4.5: Test Results View
1. **After upload**: Navigate to results
2. **Verify**: Student results display
3. **Test**: Subject filtering (Math, Physics, etc.)
4. **Test**: Export functionality

---

## 🔧 TROUBLESHOOTING

### Common Issues & Solutions:

#### 1. Database Connection Errors
```bash
# Error: psycopg2.OperationalError
# Solution: Check PostgreSQL service is running
sudo systemctl status postgresql  # Linux
brew services list | grep postgresql  # macOS
```

#### 2. Python Module Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'xyz'
# Solution: Ensure virtual environment is activated
source hackathon_env/bin/activate  # macOS/Linux
hackathon_env\Scripts\activate     # Windows

pip install -r requirements.txt
```

#### 3. CORS Errors in Frontend
```javascript
// Error: CORS policy blocked
// Solution: Verify backend CORS settings
// In complete_frontend_integrated_app.py, ensure:
CORS(app, origins=["http://localhost:3000", "http://localhost:5173"])
```

#### 4. File Upload Errors
```bash
# Error: 413 Payload Too Large
# Solution: Check MAX_CONTENT_LENGTH in app config
# Set in .env file:
MAX_CONTENT_LENGTH=104857600  # 100MB
```

#### 5. JWT Token Issues
```javascript
// Error: Invalid token
// Solution: Check token storage
console.log(localStorage.getItem('auth_token'));

// Clear and login again if needed
localStorage.removeItem('auth_token');
localStorage.removeItem('user_data');
```

---

## 🚦 VERIFICATION CHECKLIST

### ✅ Backend Verification:
- [ ] PostgreSQL service running
- [ ] Database `omr_evaluation_db` exists
- [ ] Tables created successfully  
- [ ] Test data inserted
- [ ] Flask app starts without errors
- [ ] API endpoints respond correctly
- [ ] File uploads work
- [ ] Authentication works

### ✅ Frontend Verification:
- [ ] React development server starts
- [ ] Login page loads
- [ ] Login with test credentials works
- [ ] Dashboard shows real data
- [ ] Upload page accepts files
- [ ] Results page displays data
- [ ] Navigation works between pages
- [ ] API calls return real data

### ✅ Integration Verification:
- [ ] Frontend ↔ Backend communication
- [ ] Authentication flow complete
- [ ] File upload → Processing → Results
- [ ] Database updates from frontend actions
- [ ] Error handling works properly
- [ ] Export functionality works

---

## 🎯 FINAL TESTING FLOW

### Complete End-to-End Test:
1. **Start Services**:
   ```bash
   # Terminal 1: Backend
   cd backend
   python complete_frontend_integrated_app.py
   
   # Terminal 2: Frontend  
   cd frontend
   npm run dev
   ```

2. **Test Authentication**:
   - Visit http://localhost:5173
   - Login with `test@example.com` / `password123`
   - Verify dashboard loads with statistics

3. **Test Upload Flow**:
   - Go to Upload page
   - Select OMR images from `dataset/data/Set_A/`
   - Select answer key from `dataset/data/Key (Set A and B).xlsx`
   - Fill exam details and upload
   - Verify batch appears in Batches page

4. **Test Results Flow**:
   - Navigate to Results from uploaded batch
   - Verify student results display
   - Test subject filtering
   - Export results as CSV

5. **Test Management**:
   - View all batches in Batches page
   - Filter by status, subject, school
   - Check batch statistics match dashboard

---

## 📞 SUPPORT & RESOURCES

### Documentation:
- **Backend API**: Check `complete_frontend_integrated_app.py` comments
- **Database Schema**: Review `frontend_based_complete_schema.sql`
- **Frontend Integration**: Follow `FRONTEND_INTEGRATION_GUIDE.md`

### Test Data Available:
- **Users**: 3 test users with different roles
- **Batches**: 4 sample batches with different statuses
- **Sheets**: 5 test OMR sheets with mock results
- **Images**: Sample OMR sheets in `dataset/data/Set_A/`

### Default Credentials:
```
Admin: admin@omr.com / password123
Teacher: test@example.com / password123  
Teacher: teacher@school.edu / password123
```

### Port Configuration:
- **Backend**: http://127.0.0.1:5000
- **Frontend**: http://localhost:5173
- **Database**: localhost:5432

---

## 🎉 SUCCESS!

If all steps completed successfully, you now have a **fully functional OMR Evaluation System** with:

- ✅ **Authentication System** (Login/Register/JWT)
- ✅ **File Upload & Processing** (Images + Excel)
- ✅ **Real-time Dashboard** (Statistics & Metrics)
- ✅ **Results Management** (View/Filter/Export)
- ✅ **Batch Management** (Create/Track/Monitor)
- ✅ **Database Integration** (PostgreSQL with full schema)
- ✅ **Frontend Integration** (React components with real data)

**Your OMR system is ready for the hackathon! 🚀**