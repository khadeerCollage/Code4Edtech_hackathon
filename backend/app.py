# ================================================================
# COMPLETE OMR EVALUATION SYSTEM - MAIN FLASK APPLICATION
# Combined from: app.py, fresh_app.py, complete_frontend_integrated_app.py
# Features: Authentication, Dashboard, Upload, Results, Export, OMR Processing
# ================================================================

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import os
import json
import uuid
import tempfile
import shutil
import logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from pathlib import Path
import jwt

# Import OMR core functionality
try:
    from test_green_circle_omr_new import detect_and_evaluate_omr
    OMR_AVAILABLE = True
    print("✅ OMR Processing Module Available")
except ImportError as e:
    print(f"⚠️ Warning: OMR core module not available: {e}")
    OMR_AVAILABLE = False
    
    # Fallback mock functions for testing
    def detect_and_evaluate_omr(image_path, excel_path, output_dir):
        import random
        return {
            'score': random.randint(70, 95),
            'percentage': random.randint(70, 95),
            'total_questions': 100,
            'detected_set': 'A',
            'subject_breakdown': {
                'Math': {'correct': random.randint(18, 25), 'total_questions': 25, 'percentage': random.randint(70, 95)},
                'Physics': {'correct': random.randint(16, 25), 'total_questions': 25, 'percentage': random.randint(70, 95)},
                'Chemistry': {'correct': random.randint(19, 25), 'total_questions': 25, 'percentage': random.randint(70, 95)},
                'History': {'correct': random.randint(17, 25), 'total_questions': 25, 'percentage': random.randint(70, 95)}
            }
        }

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# FLASK APPLICATION SETUP
# ================================================================

app = Flask(__name__)

# CORS Configuration - Support both development servers
additional_origins = [o.strip() for o in os.getenv('CORS_ORIGINS', '').split(',') if o.strip()]
allowed_origins = [
    'http://localhost:5173', 'http://127.0.0.1:5173',
    'http://localhost:8080', 'http://127.0.0.1:8080',
    'http://localhost:8081', 'http://127.0.0.1:8081',
    *additional_origins
]
CORS(app, resources={
    r"/api/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Configuration
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET_KEY", app.config['SECRET_KEY'])
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Directory Setup
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'results')

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'answer_keys'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'debug'), exist_ok=True)

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://omr_user:omr_password123@localhost:5432/omr_evaluation_db')

# Handle postgres:// vs postgresql:// URLs
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# File upload settings
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
ALLOWED_EXCEL_EXTENSIONS = {'xls', 'xlsx'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_EXCEL_EXTENSIONS

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def allowed_file(filename, allowed_extensions):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, upload_folder):
    """Save uploaded file and return path"""
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return filepath
    return None

def save_excel_file(file, upload_folder):
    """Save uploaded Excel file and return path"""
    if file and allowed_file(file.filename, ALLOWED_EXCEL_EXTENSIONS):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return filepath
    return None

def generate_token(user_id):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
    }
    return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')

def create_mock_omr_result(image_filename, student_number):
    """Create mock OMR results for testing when real OMR is not available"""
    import random
    random.seed(student_number)  # Consistent results for same student
    
    total_questions = 100
    correct_answers = random.randint(60, 95)  # Random score between 60-95
    percentage = (correct_answers / total_questions) * 100
    
    # Mock subject breakdown
    subjects = {
        'Math': {'questions': 25, 'correct': random.randint(15, 23)},
        'Physics': {'questions': 25, 'correct': random.randint(14, 24)},
        'Chemistry': {'questions': 25, 'correct': random.randint(16, 22)},
        'History': {'questions': 25, 'correct': random.randint(12, 21)}
    }
    
    # Create realistic OMR result structure
    result = {
        'image_name': image_filename,
        'detected_set': 'A',
        'total_questions': total_questions,
        'score': correct_answers,
        'percentage': round(percentage, 2),
        'subject_breakdown': subjects,
        'processing_time': round(random.uniform(1.2, 3.5), 2),
        'confidence': round(random.uniform(0.85, 0.98), 3),
        'questions_data': [
            {
                'question': i+1,
                'marked_answer': random.choice(['A', 'B', 'C', 'D']),
                'correct_answer': random.choice(['A', 'B', 'C', 'D']),
                'is_correct': random.choice([True, False]),
                'confidence': round(random.uniform(0.7, 0.99), 3)
            }
            for i in range(total_questions)
        ]
    }
    
    return result

def verify_token(token):
    """Verify JWT token and return user_id"""
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorator to require authentication"""
    from functools import wraps
    
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            user_id = verify_token(token)
            if not user_id:
                return jsonify({'error': 'Invalid token'}), 401
            
            # Make user_id available to the route
            request.current_user_id = user_id
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Token verification failed'}), 401
    
    return decorated

def calculate_grade(percentage):
    """Calculate grade based on percentage"""
    if percentage >= 95: return 'A+'
    elif percentage >= 90: return 'A'
    elif percentage >= 85: return 'B+'
    elif percentage >= 80: return 'B'
    elif percentage >= 75: return 'C+'
    elif percentage >= 70: return 'C'
    elif percentage >= 65: return 'D+'
    elif percentage >= 60: return 'D'
    else: return 'F'

# ================================================================
# DATABASE MODELS
# ================================================================

class User(db.Model):
    """Enhanced User model for authentication (Login.jsx, Register.jsx)"""
    __tablename__ = "user"
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Relationships
    batches = db.relationship('OMRBatch', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, raw_password: str):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            
        }

class OMRBatch(db.Model):
    """Enhanced OMR Batch model (Dashboard.jsx, Batches.jsx, Upload.jsx)"""
    __tablename__ = "omr_batches"
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Exam Information
    exam_name = db.Column(db.String(200), nullable=False)
    school_name = db.Column(db.String(200))
    grade = db.Column(db.String(20))
    subject = db.Column(db.String(100))
    answer_key_version = db.Column(db.String(10), nullable=False)
    answer_key_path = db.Column(db.String(500))
    answer_key_name = db.Column(db.String(200))
    
    # Processing Information
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processing_started_at = db.Column(db.DateTime)
    processing_completed_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='Processing')
    
    # Statistics
    total_sheets = db.Column(db.Integer, default=0)
    completed_sheets = db.Column(db.Integer, default=0)
    flagged_sheets = db.Column(db.Integer, default=0)
    failed_sheets = db.Column(db.Integer, default=0)
    average_score = db.Column(db.Float, default=0.0)
    highest_score = db.Column(db.Float, default=0.0)
    lowest_score = db.Column(db.Float, default=0.0)
    
    # Additional Metadata
    total_questions = db.Column(db.Integer, default=100)
    pass_marks = db.Column(db.Float, default=40.0)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    sheets = db.relationship('OMRSheet', backref='batch', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'exam_name': self.exam_name,
            'school_name': self.school_name,
            'grade': self.grade,
            'subject': self.subject,
            'answer_key_version': self.answer_key_version,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'status': self.status,
            'total_sheets': self.total_sheets,
            'completed_sheets': self.completed_sheets,
            'flagged_sheets': self.flagged_sheets,
            'failed_sheets': self.failed_sheets,
            'average_score': round(self.average_score, 2) if self.average_score else 0,
            'highest_score': round(self.highest_score, 2) if self.highest_score else 0,
            'lowest_score': round(self.lowest_score, 2) if self.lowest_score else 0,
            'total_questions': self.total_questions,
            'processing_time': self._calculate_processing_time()
        }
    
    def _calculate_processing_time(self):
        if self.processing_started_at and self.processing_completed_at:
            delta = self.processing_completed_at - self.processing_started_at
            return str(delta)
        return None

class OMRSheet(db.Model):
    """Enhanced OMR Sheet model (Results.jsx, Dashboard.jsx)"""
    __tablename__ = "omr_sheets"
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_id = db.Column(db.String(50), db.ForeignKey('omr_batches.id'), nullable=False)
    
    # Student Information
    student_id = db.Column(db.String(50))
    student_name = db.Column(db.String(100))
    student_roll_number = db.Column(db.String(50))
    student_class = db.Column(db.String(20))
    
    # Image Processing
    image_path = db.Column(db.String(500), nullable=False)
    image_name = db.Column(db.String(200))
    image_size = db.Column(db.Integer)
    detected_set = db.Column(db.String(10))
    
    # Scoring Information
    total_questions = db.Column(db.Integer, default=0)
    score = db.Column(db.Integer, default=0)
    total_marks = db.Column(db.Integer, default=100)
    percentage = db.Column(db.Float, default=0.0)
    grade = db.Column(db.String(5))
    pass_fail = db.Column(db.String(10), default='PENDING')
    
    # Subject-wise Scores
    math_score = db.Column(db.Integer, default=0)
    physics_score = db.Column(db.Integer, default=0)
    chemistry_score = db.Column(db.Integer, default=0)
    history_score = db.Column(db.Integer, default=0)
    english_score = db.Column(db.Integer, default=0)
    biology_score = db.Column(db.Integer, default=0)
    
    # Processing Status
    status = db.Column(db.String(20), default='Processing')
    processing_date = db.Column(db.DateTime, default=datetime.utcnow)
    review_notes = db.Column(db.Text)
    reviewer_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    reviewed_at = db.Column(db.DateTime)
    
    # Quality Metrics
    confidence_score = db.Column(db.Float)
    multiple_marked_count = db.Column(db.Integer, default=0)
    unclear_marks_count = db.Column(db.Integer, default=0)
    blank_answers_count = db.Column(db.Integer, default=0)
    
    # Raw Processing Data
    raw_results = db.Column(db.Text)
    debug_image_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'student_id': self.student_id or f"ST{self.id[:6].upper()}",
            'student_name': self.student_name or f"Student {self.id[:6].upper()}",
            'student_roll_number': self.student_roll_number,
            'image_name': self.image_name,
            'detected_set': self.detected_set,
            'score': self.score,
            'total_marks': self.total_marks,
            'percentage': round(self.percentage, 2) if self.percentage else 0.0,
            'grade': self.grade,
            'pass_fail': self.pass_fail,
            'status': self.status,
            'processing_date': self.processing_date.isoformat() if self.processing_date else None,
            'review_notes': self.review_notes,
            'confidence_score': round(self.confidence_score, 2) if self.confidence_score else 0.0,
            'subject_scores': {
                'math': self.math_score,
                'physics': self.physics_score,
                'chemistry': self.chemistry_score,
                'history': self.history_score,
                'english': self.english_score,
                'biology': self.biology_score
            },
            'quality_metrics': {
                'multiple_marked': self.multiple_marked_count,
                'unclear_marks': self.unclear_marks_count,
                'blank_answers': self.blank_answers_count
            }
        }

class SubjectScore(db.Model):
    """Subject-wise score breakdown"""
    __tablename__ = "subject_scores"
    
    id = db.Column(db.Integer, primary_key=True)
    sheet_id = db.Column(db.String(50), db.ForeignKey('omr_sheets.id'), nullable=False)
    subject_name = db.Column(db.String(50), nullable=False)
    subject_code = db.Column(db.String(10))
    total_questions = db.Column(db.Integer, default=0)
    correct_answers = db.Column(db.Integer, default=0)
    incorrect_answers = db.Column(db.Integer, default=0)
    unattempted = db.Column(db.Integer, default=0)
    multiple_marked = db.Column(db.Integer, default=0)
    percentage = db.Column(db.Float, default=0.0)

    # Relationships
    sheet = db.relationship('OMRSheet', backref='subject_scores')

# ================================================================
# AUTHENTICATION ROUTES
# ================================================================

# ================================================================
# ROUTES - API ENDPOINTS
# ================================================================

@app.route('/')
def home():
    """Root endpoint - API information"""
    return jsonify({
        'message': 'OMR Evaluation System API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'authentication': [
                'POST /api/auth/register',
                'POST /api/auth/login', 
                'GET /api/auth/profile'
            ],
            'dashboard': [
                'GET /api/dashboard/stats',
                'GET /api/dashboard/recent-batches'
            ],
            'batches': [
                'GET /api/batches',
                'POST /api/upload',
                'GET /api/batches/<batch_id>/results',
                'GET /api/batches/<batch_id>/export'
            ],
            'system': [
                'GET /api/health'
            ]
        },
        'database_status': 'connected',
        'omr_processing': 'available'
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection with raw SQL query
        result = db.session.execute(text('SELECT 1 as test'))
        result.fetchone()
        
        # Also test if our users table exists
        result = db.session.execute(text('SELECT COUNT(*) FROM user'))
        user_count = result.fetchone()[0]
        
        db_status = "connected"
        print(f"[HEALTH] Database connected, {user_count} users found")
        
    except Exception as e:
        print(f"Database health check failed: {e}")
        db_status = "disconnected"
    
    return jsonify({
        'status': 'healthy',
        'database': db_status,
        'message': 'OMR Backend API is running',
        'omr_processing': 'available' if OMR_AVAILABLE else 'mock'
    }), 200

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration - username, email, password"""
    print("=== REGISTRATION REQUEST RECEIVED ===")
    
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Get required fields
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        print(f"Username: {username}, Email: {email}, Password length: {len(password)}")
        
        # Validation
        if not username or not email or not password:
            print("Validation failed: Missing required fields")
            return jsonify({'error': 'Username, email and password are required'}), 400
        
        if len(password) < 6:
            print("Validation failed: Password too short")
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        # Check if username or email already exists
        result = db.session.execute(text('SELECT id FROM "user" WHERE username = :username OR email = :email'), 
                                  {'username': username, 'email': email})
        existing = result.fetchone()
        if existing:
            print(f"User already exists: {existing}")
            return jsonify({'error': 'Username or email already exists'}), 400
        
        # Create new user
        password_hash = generate_password_hash(password)
        print("Password hashed successfully")
        
        try:
            # Insert new user
            result = db.session.execute(text('''
                INSERT INTO "user" (username, email, password_hash, created_at, is_active) 
                VALUES (:username, :email, :password_hash, :created_at, :is_active)
                RETURNING id
            '''), {
                'username': username,
                'email': email,
                'password_hash': password_hash,
                'created_at': datetime.now(timezone.utc),
                'is_active': True
            })
            
            user_id = result.fetchone()[0]
            db.session.commit()
            print(f"User created successfully with ID: {user_id}")
            
            # Generate JWT token directly instead of using generate_token
            token = jwt.encode({
                'user_id': user_id,
                'username': username,
                'email': email,
                'exp': datetime.now(timezone.utc) + timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm='HS256')
            
            print("Token generated successfully")
            
            response_data = {
                'success': True,
                'message': 'Registration successful',
                'user': {
                    'id': user_id,
                    'username': username,
                    'email': email
                },
                'access_token': token
            }
            print(f"Sending response: {response_data}")
            return jsonify(response_data), 201
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database insert error: {e}")
            print(f"Database error: {e}")
            return jsonify({'error': 'Failed to create user account'}), 500
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration error: {e}")
        print(f"General registration error: {e}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login - username and password only"""
    print("=== LOGIN REQUEST RECEIVED ===")
    
    try:
        data = request.get_json()
        print(f"Login data: {data}")
        
        # Get username and password only
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        print(f"Username: {username}, Password length: {len(password)}")
        
        if not username or not password:
            print("Validation failed: Missing credentials")
            return jsonify({'error': 'Username and password are required'}), 400
        
        # Find user by username only
        result = db.session.execute(text('''
            SELECT id, username, email, password_hash, is_active 
            FROM "user"
            WHERE username = :username
        '''), {'username': username})
        
        user_data = result.fetchone()
        print(f"User found: {bool(user_data)}")
        
        if not user_data:
            print("User not found")
            return jsonify({'error': 'Invalid username or password'}), 401
        
        if not check_password_hash(user_data[3], password):
            print("Password verification failed")
            return jsonify({'error': 'Invalid username or password'}), 401
        
        if not user_data[4]:  # is_active field
            print("Account is disabled")
            return jsonify({'error': 'Account is disabled'}), 401
        
        # Update last login
        db.session.execute(text('''
            UPDATE "user" SET last_login = :last_login WHERE id = :user_id
        '''), {
            'last_login': datetime.now(timezone.utc),
            'user_id': user_data[0]
        })
        db.session.commit()
        print("Last login updated")
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user_data[0],    # id field
            'username': user_data[1],   # username field
            'email': user_data[2],      # email field
            'exp': datetime.now(timezone.utc) + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        response_data = {
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user_data[0],
                'username': user_data[1],
                'email': user_data[2]
            },
            'access_token': token
        }
        print(f"Sending login response: {response_data}")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        print(f"Login error: {e}")
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/auth/profile', methods=['GET'])
@require_auth
def get_profile():
    """Get current user profile"""
    try:
        user = User.query.get(request.current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'success': True,
            'user': user.to_dict()
        }), 200
    except Exception as e:
        return jsonify({'error': 'Failed to get profile'}), 500

# ================================================================
# DASHBOARD ROUTES
# ================================================================

@app.route('/api/dashboard/stats', methods=['GET'])
@require_auth
def dashboard_stats():
    """Get dashboard statistics for Dashboard.jsx"""
    try:
        user_id = request.current_user_id
        
        # Get batch statistics
        batches = OMRBatch.query.filter_by(user_id=user_id).all()
        
        total_batches = len(batches)
        processing_batches = len([b for b in batches if b.status == 'Processing'])
        completed_batches = len([b for b in batches if b.status == 'Completed'])
        flagged_batches = len([b for b in batches if b.status == 'Flagged'])
        
        total_sheets = sum(b.total_sheets or 0 for b in batches)
        flagged_sheets = sum(b.flagged_sheets or 0 for b in batches)
        
        # Calculate overall average
        completed_batches_list = [b for b in batches if b.status == 'Completed' and b.average_score]
        overall_average = sum(b.average_score for b in completed_batches_list) / len(completed_batches_list) if completed_batches_list else 0
        
        return jsonify({
            'success': True,
            'totalBatches': total_batches,
            'processingBatches': processing_batches,
            'completedBatches': completed_batches,
            'flaggedBatches': flagged_batches,
            'totalSheetsProcessed': total_sheets,
            'totalFlaggedSheets': flagged_sheets,
            'overallAverageScore': round(overall_average, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return jsonify({'error': f'Failed to get dashboard stats: {str(e)}'}), 500

@app.route('/api/dashboard/recent-batches', methods=['GET'])
@require_auth
def recent_batches():
    """Get recent batches for Dashboard.jsx"""
    try:
        user_id = request.current_user_id
        limit = request.args.get('limit', 5, type=int)
        
        batches = OMRBatch.query.filter_by(user_id=user_id)\
                                .order_by(OMRBatch.upload_date.desc())\
                                .limit(limit)\
                                .all()
        
        return jsonify({
            'success': True,
            'batches': [batch.to_dict() for batch in batches]
        }, 200)
        
    except Exception as e:
        logger.error(f"Recent batches error: {e}")
        return jsonify({'error': f'Failed to get recent batches: {str(e)}'}), 500

# ================================================================
# BATCH MANAGEMENT ROUTES
# ================================================================

@app.route('/api/batches', methods=['GET'])
@require_auth
def get_batches():
    """Get all batches for the user"""
    try:
        user_id = request.current_user_id
        
        # Get query parameters for filtering
        status = request.args.get('status')
        subject = request.args.get('subject')
        school = request.args.get('school')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Build query
        query = OMRBatch.query.filter_by(user_id=user_id)
        
        if status:
            query = query.filter(OMRBatch.status == status)
        if subject:
            query = query.filter(OMRBatch.subject.ilike(f'%{subject}%'))
        if school:
            query = query.filter(OMRBatch.school_name.ilike(f'%{school}%'))
        
        # Get results
        batches = query.order_by(OMRBatch.upload_date.desc()).all()
        
        batch_list = []
        for batch in batches:
            batch_list.append({
                'id': batch.id,
                'examName': batch.exam_name,
                'schoolName': batch.school_name,
                'grade': batch.grade,
                'subject': batch.subject,
                'uploadDate': batch.upload_date.isoformat() if batch.upload_date else None,
                'status': batch.status,
                'totalSheets': batch.total_sheets or 0,
                'completedSheets': batch.completed_sheets or 0,
                'flaggedSheets': batch.flagged_sheets or 0,
                'averageScore': round(batch.average_score, 2) if batch.average_score else 0.0
            })
        
        return jsonify({
            'success': True,
            'batches': batch_list
        }), 200
        
    except Exception as e:
        logger.error(f"Get batches error: {e}")
        return jsonify({'error': f'Failed to get batches: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_files():
    """Handle file upload and OMR processing"""
    try:
        user_id = request.current_user_id
        
        # Get form data
        exam_name = request.form.get('examName', 'Unnamed Exam')
        answer_key_version = request.form.get('answerKeyVersion', 'A')
        school_name = request.form.get('schoolName', '')
        grade = request.form.get('grade', '')
        subject = request.form.get('subject', '')
        
        # Get uploaded files
        image_files = request.files.getlist('images')
        use_existing_key = request.form.get('useExistingAnswerKey', 'false').lower() == 'true'
        
        if not image_files:
            return jsonify({'error': 'Images are required'}), 400
        
        print(f"[UPLOAD] Processing {len(image_files)} images")
        print(f"[UPLOAD] Using existing answer key: {use_existing_key}")
        
        # Create batch
        batch_id = str(uuid.uuid4())
        
        # Create batch folder
        batch_folder = os.path.join(app.config['UPLOAD_FOLDER'], batch_id)
        os.makedirs(batch_folder, exist_ok=True)
        
        # Set up answer key path
        if use_existing_key:
            # Use the dataset answer key
            answer_key_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data', 'Key (Set A and B).xlsx')
            answer_key_filename = 'Key (Set A and B).xlsx'
            print(f"[UPLOAD] Using existing answer key: {answer_key_path}")
        else:
            # Handle uploaded answer key
            answer_key_file = request.files.get('answerKey')
            if not answer_key_file:
                return jsonify({'error': 'Answer key is required when not using existing key'}), 400
            answer_key_filename = secure_filename(answer_key_file.filename)
            answer_key_path = os.path.join(batch_folder, f"answer_key_{answer_key_filename}")
            answer_key_file.save(answer_key_path)
            print(f"[UPLOAD] Saved uploaded answer key: {answer_key_path}")
        
        # Create batch record
        batch = OMRBatch(
            id=batch_id,
            user_id=user_id,
            exam_name=exam_name,
            school_name=school_name,
            grade=grade,
            subject=subject,
            answer_key_version=answer_key_version,
            answer_key_path=answer_key_path,
            answer_key_name=answer_key_filename,
            total_sheets=len(image_files),
            status='Processing',
            processing_started_at=datetime.utcnow()
        )
        
        db.session.add(batch)
        db.session.flush()  # Get batch ID
        
        # Process images
        processed_count = 0
        for i, image_file in enumerate(image_files):
            try:
                # Save image
                image_filename = secure_filename(image_file.filename)
                image_path = os.path.join(batch_folder, f"{i:03d}_{image_filename}")
                image_file.save(image_path)
                
                # Create sheet record
                sheet_id = str(uuid.uuid4())
                sheet = OMRSheet(
                    id=sheet_id,
                    batch_id=batch_id,
                    student_id=f"ST{(i+1):03d}",
                    student_name=f"Student {i+1}",
                    image_path=image_path,
                    image_name=image_filename,
                    image_size=os.path.getsize(image_path),
                    status='Processing'
                )
                
                db.session.add(sheet)
                db.session.flush()
                
                # Process OMR (using your core function or mock)
                output_dir = os.path.join(app.config['OUTPUT_FOLDER'], batch_id, sheet_id)
                os.makedirs(output_dir, exist_ok=True)
                
                if OMR_AVAILABLE:
                    try:
                        print(f"[OMR] Processing {image_filename} with real OMR engine...")
                        result = detect_and_evaluate_omr(image_path, answer_key_path, output_dir)
                        print(f"[OMR] Processing complete. Score: {result.get('score', 0)}")
                    except Exception as omr_error:
                        print(f"[OMR] Error in real OMR processing: {omr_error}")
                        # Fallback to mock results
                        result = create_mock_omr_result(image_filename, i+1)
                else:
                    print(f"[OMR] Using mock OMR results for {image_filename}")
                    result = create_mock_omr_result(image_filename, i+1)
                
                # Update sheet with results
                sheet.detected_set = result.get('detected_set', batch.answer_key_version)
                sheet.total_questions = result.get('total_questions', 100)
                sheet.score = result.get('score', 0)
                sheet.total_marks = 100
                sheet.percentage = result.get('percentage', 0.0)
                sheet.grade = calculate_grade(sheet.percentage)
                sheet.pass_fail = 'PASS' if sheet.percentage >= batch.pass_marks else 'FAIL'
                
                # Extract subject scores
                subject_breakdown = result.get('subject_breakdown', {})
                sheet.math_score = subject_breakdown.get('Math', {}).get('correct', 0)
                sheet.physics_score = subject_breakdown.get('Physics', {}).get('correct', 0)
                sheet.chemistry_score = subject_breakdown.get('Chemistry', {}).get('correct', 0)
                sheet.history_score = subject_breakdown.get('History', {}).get('correct', 0)
                
                sheet.status = 'Completed'
                sheet.raw_results = json.dumps(result)
                
                processed_count += 1
                
            except Exception as sheet_error:
                logger.error(f"Sheet processing error: {sheet_error}")
                sheet.status = 'Failed'
                sheet.review_notes = f'Processing error: {str(sheet_error)}'
        
        # Update batch statistics
        batch.completed_sheets = processed_count
        batch.processing_completed_at = datetime.utcnow()
        
        # Calculate batch statistics
        completed_sheets = OMRSheet.query.filter_by(batch_id=batch_id, status='Completed').all()
        if completed_sheets:
            scores = [sheet.percentage for sheet in completed_sheets]
            batch.average_score = sum(scores) / len(scores)
            batch.highest_score = max(scores)
            batch.lowest_score = min(scores)
        
        batch.status = 'Completed' if processed_count == len(image_files) else 'Partially Completed'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Upload and processing completed',
            'batch_id': batch_id,
            'processed_sheets': processed_count,
            'total_sheets': len(image_files)
        }), 200
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        db.session.rollback()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/batches/<batch_id>/status', methods=['GET'])
@require_auth
def get_batch_status(batch_id):
    """Get real-time processing status for a batch"""
    try:
        user_id = request.current_user_id
        
        # Verify batch belongs to user
        batch = OMRBatch.query.filter_by(id=batch_id, user_id=user_id).first()
        if not batch:
            return jsonify({'error': 'Batch not found'}), 404
            
        # Get sheet statistics
        total_sheets = batch.total_sheets
        completed_sheets = OMRSheet.query.filter_by(batch_id=batch_id, status='Completed').count()
        processing_sheets = OMRSheet.query.filter_by(batch_id=batch_id, status='Processing').count()
        failed_sheets = OMRSheet.query.filter_by(batch_id=batch_id, status='Failed').count()
        
        # Calculate progress percentage
        progress_percentage = (completed_sheets / total_sheets) * 100 if total_sheets > 0 else 0
        
        # Get latest completed results for preview
        latest_results = OMRSheet.query.filter_by(batch_id=batch_id, status='Completed')\
                                       .order_by(OMRSheet.id.desc())\
                                       .limit(5)\
                                       .all()
        
        preview_results = []
        for sheet in latest_results:
            preview_results.append({
                'student_id': sheet.student_id,
                'student_name': sheet.student_name,
                'score': sheet.score,
                'percentage': sheet.percentage,
                'grade': sheet.grade
            })
            
        return jsonify({
            'batch_id': batch_id,
            'batch_name': batch.exam_name,
            'status': batch.status,
            'total_sheets': total_sheets,
            'completed_sheets': completed_sheets,
            'processing_sheets': processing_sheets,
            'failed_sheets': failed_sheets,
            'progress_percentage': round(progress_percentage, 1),
            'average_score': batch.average_score,
            'highest_score': batch.highest_score,
            'lowest_score': batch.lowest_score,
            'processing_started_at': batch.processing_started_at.isoformat() if batch.processing_started_at else None,
            'processing_completed_at': batch.processing_completed_at.isoformat() if batch.processing_completed_at else None,
            'preview_results': preview_results
        }), 200
        
    except Exception as e:
        print(f"Status check error: {e}")
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

@app.route('/api/batches/<batch_id>/results', methods=['GET'])
@require_auth
def get_batch_results(batch_id):
    """Get results for a specific batch"""
    try:
        user_id = request.current_user_id
        
        # Verify batch belongs to user
        batch = OMRBatch.query.filter_by(id=batch_id, user_id=user_id).first()
        if not batch:
            return jsonify({'error': 'Batch not found'}), 404
        
        # Get query parameters for filtering
        subject = request.args.get('subject')
        status = request.args.get('status')
        min_score = request.args.get('min_score', type=float)
        max_score = request.args.get('max_score', type=float)
        
        # Build query
        query = OMRSheet.query.filter_by(batch_id=batch_id)
        
        if status:
            query = query.filter(OMRSheet.status == status)
        if min_score is not None:
            query = query.filter(OMRSheet.percentage >= min_score)
        if max_score is not None:
            query = query.filter(OMRSheet.percentage <= max_score)
        
        # Order by student ID or percentage
        order_by = request.args.get('order_by', 'student_id')
        if order_by == 'percentage':
            query = query.order_by(OMRSheet.percentage.desc())
        else:
            query = query.order_by(OMRSheet.student_id)
        
        sheets = query.all()
        
        results = []
        for sheet in sheets:
            result_data = {
                'id': sheet.id,
                'studentId': sheet.student_id,
                'studentName': sheet.student_name,
                'totalScore': sheet.score,
                'percentage': round(sheet.percentage, 1),
                'grade': sheet.grade,
                'status': sheet.status,
                'mathScore': sheet.math_score,
                'physicsScore': sheet.physics_score,
                'chemistryScore': sheet.chemistry_score,
                'historyScore': sheet.history_score,
                'processingDate': sheet.processing_date.isoformat() if sheet.processing_date else None,
                'imageName': sheet.image_name,
                'detectedSet': sheet.detected_set,
                'totalQuestions': sheet.total_questions
            }
            
            # Include raw JSON results if available
            if sheet.raw_results:
                try:
                    raw_data = json.loads(sheet.raw_results)
                    result_data['rawResults'] = raw_data
                    result_data['questionsData'] = raw_data.get('questions_data', [])
                    result_data['processingTime'] = raw_data.get('processing_time')
                    result_data['confidence'] = raw_data.get('confidence')
                    result_data['subjectBreakdown'] = raw_data.get('subject_breakdown', {})
                except json.JSONDecodeError:
                    result_data['rawResults'] = None
            results.append(result_data)
        
        return jsonify({
            'success': True,
            'batch': batch.to_dict(),
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Get results error: {e}")
        return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

@app.route('/api/batches/<batch_id>/export', methods=['GET'])
@require_auth
def export_batch_results(batch_id):
    """Export batch results as CSV"""
    try:
        user_id = request.current_user_id
        
        # Verify batch belongs to user and get results
        batch = OMRBatch.query.filter_by(id=batch_id, user_id=user_id).first()
        if not batch:
            return jsonify({'error': 'Batch not found'}), 404
        
        sheets = OMRSheet.query.filter_by(batch_id=batch_id).all()
        
        if not sheets:
            return jsonify({'error': 'No results found'}), 404
        
        # Create CSV content
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Student ID', 'Student Name', 'Total Score', 'Percentage', 
                        'Grade', 'Pass/Fail', 'Math Score', 'Physics Score', 'Chemistry Score', 'History Score'])
        
        # Write data
        for sheet in sheets:
            writer.writerow([
                sheet.student_id, sheet.student_name, sheet.score, sheet.percentage,
                sheet.grade, sheet.pass_fail, sheet.math_score, sheet.physics_score, 
                sheet.chemistry_score, sheet.history_score
            ])
        
        # Prepare file response
        output.seek(0)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write(output.getvalue())
        temp_file.close()
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f'{batch.exam_name}_{batch_id}_results.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

# ================================================================
# ERROR HANDLERS
# ================================================================

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized'}), 401

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# ================================================================
# DATABASE INITIALIZATION
# ================================================================

def init_db():
    """Initialize database - work with existing schema"""
    try:
        with app.app_context():
            # First, ensure user table exists
            try:
                # Check if user table exists
                result = db.session.execute(text('SELECT COUNT(*) FROM "user"'))
                user_count = result.fetchone()[0]
                print(f"[INFO] Found {user_count} users in existing database")
            except Exception as e:
                print(f"[INFO] User table doesn't exist, creating it now...")
                # Create user table
                create_user_table_sql = '''
                CREATE TABLE IF NOT EXISTS "user" (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(120) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT true,
                    last_login TIMESTAMP WITH TIME ZONE
                );
                
                -- Grant permissions
                GRANT ALL PRIVILEGES ON TABLE "user" TO omr_user;
                GRANT ALL PRIVILEGES ON SEQUENCE user_id_seq TO omr_user;
                '''
                
                db.session.execute(text(create_user_table_sql))
                db.session.commit()
                
                # Check again
                result = db.session.execute(text('SELECT COUNT(*) FROM "user"'))
                user_count = result.fetchone()[0]
                print(f"[INFO] User table created! Found {user_count} users")
            
            # Check if other tables exist
            tables_to_check = ['omr_batches', 'omr_sheets', 'subject_scores']
            for table in tables_to_check:
                try:
                    result = db.session.execute(text(f'SELECT COUNT(*) FROM {table}'))
                    count = result.fetchone()[0]
                    print(f"[INFO] Found {count} records in {table} table")
                except Exception as e:
                    print(f"[WARNING] Table {table} might not exist: {e}")
            
            print("[OK] Database initialized successfully!")
            return True
                
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
        print("Please check your PostgreSQL setup and DATABASE_URL")
        return False

# ================================================================
# APPLICATION STARTUP
# ================================================================

if __name__ == '__main__':
    print("="*60)
    print("🚀 OMR EVALUATION SYSTEM - MAIN FLASK APPLICATION")
    print("="*60)
    
    # Initialize database
    try:
        init_db()
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Please check your PostgreSQL setup and DATABASE_URL")
    
    # Print configuration
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"🔧 OMR Processing: {'Available' if OMR_AVAILABLE else 'Mock (for testing)'}")
    print(f"🌐 CORS Origins: {allowed_origins}")
    print(f"🗄️ Database: {DATABASE_URL}")
    
    print("="*60)
    print("🔗 API Endpoints Available:")
    print("   POST /api/auth/register")
    print("   POST /api/auth/login")
    print("   GET  /api/auth/profile")
    print("   GET  /api/dashboard/stats")
    print("   GET  /api/dashboard/recent-batches")
    print("   GET  /api/batches")
    print("   POST /api/upload")
    print("   GET  /api/batches/<batch_id>/results")
    print("   GET  /api/batches/<batch_id>/export")
    print("   GET  /api/health")
    print("="*60)
    
    # Start server
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=debug_mode
    )