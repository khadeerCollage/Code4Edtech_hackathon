"""
Fresh OMR Evaluation System Backend
Complete Flask Application with PostgreSQL Integration
Author: OMR Team
Date: December 2024
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import json
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tempfile
import logging

# Import OMR core functionality
try:
    from test_green_circle_omr_new import detect_and_evaluate_omr
    OMR_AVAILABLE = True
except ImportError:
    print("Warning: OMR core module not available")
    OMR_AVAILABLE = False

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# Flask Application Setup
# ========================================

app = Flask(__name__)

# CORS Configuration
additional_origins = [o.strip() for o in os.getenv('CORS_ORIGINS', '').split(',') if o.strip()]
allowed_origins = ['http://localhost:5173', 'http://127.0.0.1:5173'] + additional_origins
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Configuration
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "dev-secret-key")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'results')

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Database configuration
db_url = os.getenv('DATABASE_URL')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url or 'sqlite:///fresh_omr.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# File upload settings
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
ALLOWED_EXCEL_EXTENSIONS = {'xls', 'xlsx'}

# ========================================
# Database Models
# ========================================

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    
    # Relationships
    batches = db.relationship('OMRBatch', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class OMRBatch(db.Model):
    __tablename__ = "omr_batches"
    id = db.Column(db.String(50), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exam_name = db.Column(db.String(200), nullable=False)
    answer_key_version = db.Column(db.String(10), nullable=False)
    answer_key_path = db.Column(db.String(500))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='Processing')
    total_sheets = db.Column(db.Integer, default=0)
    completed_sheets = db.Column(db.Integer, default=0)
    flagged_sheets = db.Column(db.Integer, default=0)
    average_score = db.Column(db.Float, default=0.0)
    
    # Relationships
    sheets = db.relationship('OMRSheet', backref='batch', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'exam_name': self.exam_name,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'status': self.status,
            'total_sheets': self.total_sheets,
            'completed_sheets': self.completed_sheets,
            'flagged_sheets': self.flagged_sheets,
            'average_score': round(self.average_score, 2) if self.average_score else 0,
            'answer_key_version': self.answer_key_version
        }

class OMRSheet(db.Model):
    __tablename__ = "omr_sheets"
    id = db.Column(db.String(50), primary_key=True)
    batch_id = db.Column(db.String(50), db.ForeignKey('omr_batches.id'), nullable=False)
    student_id = db.Column(db.String(50))
    student_name = db.Column(db.String(100))
    image_path = db.Column(db.String(500), nullable=False)
    detected_set = db.Column(db.String(10))
    total_questions = db.Column(db.Integer, default=0)
    score = db.Column(db.Integer, default=0)
    total_marks = db.Column(db.Integer, default=100)
    percentage = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='Processing')
    processing_date = db.Column(db.DateTime, default=datetime.utcnow)
    review_notes = db.Column(db.Text)
    raw_results = db.Column(db.Text)
    
    # Subject scores
    math_score = db.Column(db.Integer, default=0)
    physics_score = db.Column(db.Integer, default=0)
    chemistry_score = db.Column(db.Integer, default=0)
    history_score = db.Column(db.Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id or f"ST{self.id[:6].upper()}",
            'student_name': self.student_name or f"Student {self.id[:6].upper()}",
            'score': self.score,
            'total_marks': self.total_marks,
            'percentage': round(self.percentage, 2),
            'status': self.status,
            'review_notes': self.review_notes,
            'detected_set': self.detected_set,
            'math': self.math_score,
            'physics': self.physics_score,
            'chemistry': self.chemistry_score,
            'history': self.history_score,
            'processing_date': self.processing_date.isoformat() if self.processing_date else None
        }

class SubjectScore(db.Model):
    __tablename__ = "subject_scores"
    id = db.Column(db.Integer, primary_key=True)
    sheet_id = db.Column(db.String(50), db.ForeignKey('omr_sheets.id'), nullable=False)
    subject_name = db.Column(db.String(50), nullable=False)
    total_questions = db.Column(db.Integer, default=0)
    correct_answers = db.Column(db.Integer, default=0)
    incorrect_answers = db.Column(db.Integer, default=0)
    unattempted = db.Column(db.Integer, default=0)
    multiple_marked = db.Column(db.Integer, default=0)
    percentage = db.Column(db.Float, default=0.0)
    
    def to_dict(self):
        return {
            'subject_name': self.subject_name,
            'total_questions': self.total_questions,
            'correct': self.correct_answers,
            'incorrect': self.incorrect_answers,
            'unattempted': self.unattempted,
            'multiple_marked': self.multiple_marked,
            'percentage': round(self.percentage, 2)
        }

# ========================================
# Utility Functions
# ========================================

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, upload_folder):
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return filepath
    return None

def save_excel_file(file, upload_folder):
    if file and allowed_file(file.filename, ALLOWED_EXCEL_EXTENSIONS):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return filepath
    return None

# ========================================
# API Routes
# ========================================

@app.route('/')
def health_check():
    return jsonify({
        "message": "OMR Evaluation System - Fresh Backend Running!",
        "status": "healthy",
        "version": "2.0.0",
        "omr_available": OMR_AVAILABLE,
        "database": "connected" if db.engine else "disconnected"
    })

# ========================================
# Authentication Routes
# ========================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()
        
        # Validation
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered"}), 409
        
        # Create new user
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        logger.info(f"New user registered: {email}")
        
        return jsonify({
            "message": "Registration successful",
            "user": user.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({"error": "Registration failed"}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        # Find user
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid email or password"}), 401
        
        logger.info(f"User logged in: {email}")
        
        return jsonify({
            "message": "Login successful",
            "user": user.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Login failed"}), 500

# ========================================
# Dashboard Routes
# ========================================

@app.route('/api/dashboard/metrics', methods=['GET'])
def get_dashboard_metrics():
    """Get dashboard overview metrics"""
    try:
        total_batches = OMRBatch.query.count()
        processing_batches = OMRBatch.query.filter_by(status='Processing').count()
        completed_batches = OMRBatch.query.filter_by(status='Completed').count()
        flagged_sheets = db.session.query(db.func.sum(OMRBatch.flagged_sheets)).scalar() or 0
        
        return jsonify({
            'total_batches': total_batches,
            'processing_batches': processing_batches,
            'completed_batches': completed_batches,
            'flagged_sheets': flagged_sheets
        })
        
    except Exception as e:
        logger.error(f"Dashboard metrics error: {e}")
        return jsonify({"error": "Failed to fetch metrics"}), 500

# ========================================
# Batch Management Routes
# ========================================

@app.route('/api/batches', methods=['GET'])
def get_batches():
    """Get all OMR batches"""
    try:
        batches = OMRBatch.query.order_by(OMRBatch.upload_date.desc()).all()
        return jsonify([batch.to_dict() for batch in batches])
    except Exception as e:
        logger.error(f"Get batches error: {e}")
        return jsonify({"error": "Failed to fetch batches"}), 500

@app.route('/api/batches/<batch_id>', methods=['GET'])
def get_batch_details(batch_id):
    """Get specific batch details with sheets"""
    try:
        batch = OMRBatch.query.get_or_404(batch_id)
        sheets = [sheet.to_dict() for sheet in batch.sheets]
        
        result = batch.to_dict()
        result['sheets'] = sheets
        return jsonify(result)
    except Exception as e:
        logger.error(f"Get batch details error: {e}")
        return jsonify({"error": "Failed to fetch batch details"}), 500

@app.route('/api/batches/<batch_id>/results', methods=['GET'])
def get_batch_results(batch_id):
    """Get processed results for a specific batch"""
    try:
        batch = OMRBatch.query.get_or_404(batch_id)
        sheets = [sheet.to_dict() for sheet in batch.sheets]
        
        # Calculate summary statistics
        total_sheets = len(sheets)
        completed_sheets = len([s for s in sheets if s['status'] == 'Completed'])
        flagged_sheets = len([s for s in sheets if s['status'] == 'Flagged'])
        
        if completed_sheets > 0:
            average_score = sum(s['percentage'] for s in sheets if s['status'] == 'Completed') / completed_sheets
        else:
            average_score = 0
        
        return jsonify({
            'batch_info': batch.to_dict(),
            'summary': {
                'total_sheets': total_sheets,
                'completed_sheets': completed_sheets,
                'flagged_sheets': flagged_sheets,
                'average_score': round(average_score, 2)
            },
            'results': sheets
        })
    except Exception as e:
        logger.error(f"Get batch results error: {e}")
        return jsonify({"error": "Failed to fetch batch results"}), 500

# ========================================
# File Upload and Processing Routes
# ========================================

@app.route('/api/upload', methods=['POST'])
def upload_omr_batch():
    """Upload OMR sheets and answer key for processing"""
    if not OMR_AVAILABLE:
        return jsonify({"error": "OMR processing module not available"}), 503
    
    try:
        # Check files
        if 'images' not in request.files or 'answer_key' not in request.files:
            return jsonify({"error": "Both images and answer key files are required"}), 400
        
        images = request.files.getlist('images')
        answer_key = request.files['answer_key']
        exam_name = request.form.get('exam_name', 'Untitled Exam')
        answer_key_version = request.form.get('answer_key_version', 'A')
        
        if not images or not answer_key:
            return jsonify({"error": "No files provided"}), 400
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        batch_folder = os.path.join(app.config['UPLOAD_FOLDER'], batch_id)
        os.makedirs(batch_folder, exist_ok=True)
        
        # Save answer key
        answer_key_path = save_excel_file(answer_key, batch_folder)
        if not answer_key_path:
            return jsonify({"error": "Invalid answer key file format"}), 400
        
        # Create batch record
        batch = OMRBatch(
            id=batch_id,
            user_id=1,  # Default user for now
            exam_name=exam_name,
            answer_key_version=answer_key_version,
            answer_key_path=answer_key_path,
            total_sheets=len(images),
            status='Processing'
        )
        db.session.add(batch)
        
        # Process each image
        processed_sheets = []
        for i, image_file in enumerate(images):
            try:
                image_path = save_uploaded_file(image_file, batch_folder)
                if not image_path:
                    continue
                
                sheet_id = f"{batch_id}_{i:03d}"
                sheet = OMRSheet(
                    id=sheet_id,
                    batch_id=batch_id,
                    image_path=image_path,
                    status='Processing'
                )
                db.session.add(sheet)
                processed_sheets.append(sheet)
                
            except Exception as e:
                logger.error(f"Error processing image {image_file.filename}: {e}")
                continue
        
        db.session.commit()
        
        # Process OMR sheets
        for sheet in processed_sheets:
            try:
                process_omr_sheet(sheet, answer_key_path)
            except Exception as e:
                logger.error(f"Error processing sheet {sheet.id}: {e}")
                sheet.status = 'Failed'
                sheet.review_notes = str(e)
        
        # Update batch status
        completed_count = len([s for s in processed_sheets if s.status == 'Completed'])
        flagged_count = len([s for s in processed_sheets if s.status == 'Flagged'])
        
        batch.completed_sheets = completed_count
        batch.flagged_sheets = flagged_count
        batch.status = 'Completed' if completed_count + flagged_count == len(processed_sheets) else 'Processing'
        
        if completed_count > 0:
            total_percentage = sum(s.percentage for s in processed_sheets if s.status == 'Completed')
            batch.average_score = total_percentage / completed_count
        
        db.session.commit()
        
        return jsonify({
            "message": "Files uploaded and processing completed",
            "batch_id": batch_id,
            "total_images": len(images),
            "processed_sheets": len(processed_sheets),
            "completed": completed_count,
            "flagged": flagged_count
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to process upload"}), 500

def process_omr_sheet(sheet, answer_key_path):
    """Process individual OMR sheet using the core OMR function"""
    try:
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], sheet.batch_id, sheet.id)
        os.makedirs(output_dir, exist_ok=True)
        
        report = detect_and_evaluate_omr(sheet.image_path, answer_key_path, output_dir)
        
        # Update sheet with results
        sheet.detected_set = report.get('detected_set', 'A')
        sheet.total_questions = report.get('total_questions', 0)
        sheet.score = report.get('score', 0)
        sheet.percentage = report.get('percentage', 0.0)
        sheet.raw_results = json.dumps(report)
        
        # Extract subject scores
        subject_breakdown = report.get('subject_breakdown', {})
        sheet.math_score = subject_breakdown.get('Math', {}).get('correct', 0)
        sheet.physics_score = subject_breakdown.get('Physics', {}).get('correct', 0)
        sheet.chemistry_score = subject_breakdown.get('Chemistry', {}).get('correct', 0)
        sheet.history_score = subject_breakdown.get('History', {}).get('correct', 0)
        
        # Check for review flags
        if report.get('overall_summary', {}).get('multiple_marked', 0) > 5:
            sheet.status = 'Flagged'
            sheet.review_notes = f"High number of multiple marked answers: {report.get('overall_summary', {}).get('multiple_marked', 0)}"
        else:
            sheet.status = 'Completed'
        
        # Save detailed subject scores
        for subject_name, subject_data in subject_breakdown.items():
            subject_score = SubjectScore(
                sheet_id=sheet.id,
                subject_name=subject_name,
                total_questions=subject_data.get('total_questions', 0),
                correct_answers=subject_data.get('correct', 0),
                incorrect_answers=subject_data.get('incorrect', 0),
                unattempted=subject_data.get('unattempted', 0),
                multiple_marked=subject_data.get('multiple_marked', 0),
                percentage=subject_data.get('percentage', 0.0)
            )
            db.session.add(subject_score)
        
        db.session.commit()
        logger.info(f"Successfully processed sheet {sheet.id}")
        
    except Exception as e:
        logger.error(f"Error processing sheet {sheet.id}: {e}")
        sheet.status = 'Failed'
        sheet.review_notes = f"Processing error: {str(e)}"
        db.session.commit()

# ========================================
# Error Handlers
# ========================================

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def file_too_large_error(error):
    return jsonify({"error": "File too large. Maximum size is 50MB"}), 413

# ========================================
# Database Initialization
# ========================================

def init_db():
    """Initialize database"""
    try:
        with app.app_context():
            db.create_all()
            
            # Check tables
            inspector = db.inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            required_tables = ['users', 'omr_batches', 'omr_sheets', 'subject_scores']
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
                db.create_all()
                logger.info("All tables created successfully!")
            else:
                logger.info("All required tables exist.")
            
            # Create default test user if none exists
            if not User.query.first():
                test_user = User(email='test@example.com')
                test_user.set_password('password123')
                db.session.add(test_user)
                db.session.commit()
                logger.info("Created default test user: test@example.com / password123")
                
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# ========================================
# Application Startup
# ========================================

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    print("=" * 60)
    print("🚀 FRESH OMR EVALUATION SYSTEM BACKEND")
    print("=" * 60)
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"🗃️  Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"🌐 CORS allowed origins: {allowed_origins}")
    print(f"🔧 OMR Processing: {'Available' if OMR_AVAILABLE else 'Not Available'}")
    print("=" * 60)
    print("🔗 Fresh API Endpoints:")
    print("   POST /api/auth/register    - User registration")
    print("   POST /api/auth/login       - User login")
    print("   GET  /api/dashboard/metrics- Dashboard statistics")
    print("   GET  /api/batches          - Get all batches")
    print("   POST /api/upload           - Upload OMR images and answer key")
    print("   GET  /api/batches/<id>     - Get batch details")
    print("   GET  /api/batches/<id>/results - Get batch results")
    print("=" * 60)
    print("🔐 Test Login Credentials:")
    print("   Email: test@example.com")
    print("   Password: password123")
    print("=" * 60)
    
    # Start the application
    app.run(debug=True, host='0.0.0.0', port=5000)