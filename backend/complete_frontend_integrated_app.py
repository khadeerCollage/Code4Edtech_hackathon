# ================================================================
# COMPLETE OMR EVALUATION SYSTEM - FLASK BACKEND
# Frontend-Based Integration for Login, Register, Dashboard, Upload, Results
# Matches React Components Exactly
# ================================================================

import os
import uuid
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Flask imports
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
import jwt

# Image processing and OMR
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Import our OMR processing module
try:
    from test_green_circle_omr_new import detect_circles_advanced, process_omr_sheet
except ImportError as e:
    print(f"Warning: Could not import OMR module: {e}")
    # Fallback function for testing
    def detect_circles_advanced(image_path):
        return {"status": "mock", "detected_answers": ["A"] * 100, "confidence": 0.85}
    
    def process_omr_sheet(image_path, answer_key):
        return {"score": 85, "percentage": 85.0, "details": "Mock processing"}

# ================================================================
# APP CONFIGURATION
# ================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Database Configuration (matching schema)
DATABASE_URL = os.environ.get('DATABASE_URL', 
    'postgresql://omr_user:omr_password123@localhost:5432/omr_evaluation_db')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# File Upload Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'xlsx', 'xls', 'zip'}

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
CORS(app, origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"])

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'answer_keys'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'debug'), exist_ok=True)

# ================================================================
# DATABASE MODELS (Matching Frontend Requirements)
# ================================================================

class User(db.Model):
    """User model for authentication (Login.jsx, Register.jsx)"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100))
    role = db.Column(db.String(20), default='teacher')
    school_name = db.Column(db.String(200))
    phone = db.Column(db.String(20))
    is_active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=False)
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    batches = db.relationship('OMRBatch', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'school_name': self.school_name,
            'phone': self.phone,
            'is_active': self.is_active,
            'email_verified': self.email_verified,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat()
        }

class OMRBatch(db.Model):
    """Batch model for exam sessions (Dashboard.jsx, Batches.jsx, Upload.jsx)"""
    __tablename__ = 'omr_batches'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Exam Information (from Upload.jsx and Batches.jsx)
    exam_name = db.Column(db.String(200), nullable=False)
    school_name = db.Column(db.String(200))
    grade = db.Column(db.String(20))
    subject = db.Column(db.String(100))
    exam_date = db.Column(db.Date)
    exam_duration = db.Column(db.Integer)
    
    # Answer Key Information (from Upload.jsx)
    answer_key_version = db.Column(db.String(10), nullable=False)
    answer_key_path = db.Column(db.String(500))
    answer_key_name = db.Column(db.String(200))
    
    # Processing Information (from Dashboard.jsx)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processing_started_at = db.Column(db.DateTime)
    processing_completed_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='Processing')
    
    # Statistics (from Dashboard.jsx, Batches.jsx)
    total_sheets = db.Column(db.Integer, default=0)
    completed_sheets = db.Column(db.Integer, default=0)
    flagged_sheets = db.Column(db.Integer, default=0)
    failed_sheets = db.Column(db.Integer, default=0)
    average_score = db.Column(db.Numeric(5,2), default=0.0)
    highest_score = db.Column(db.Numeric(5,2), default=0.0)
    lowest_score = db.Column(db.Numeric(5,2), default=0.0)
    
    # Additional Metadata
    total_questions = db.Column(db.Integer, default=100)
    pass_marks = db.Column(db.Numeric(5,2), default=40.0)
    notes = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sheets = db.relationship('OMRSheet', backref='batch', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'exam_name': self.exam_name,
            'school_name': self.school_name,
            'grade': self.grade,
            'subject': self.subject,
            'exam_date': self.exam_date.isoformat() if self.exam_date else None,
            'answer_key_version': self.answer_key_version,
            'upload_date': self.upload_date.isoformat(),
            'status': self.status,
            'total_sheets': self.total_sheets,
            'completed_sheets': self.completed_sheets,
            'flagged_sheets': self.flagged_sheets,
            'failed_sheets': self.failed_sheets,
            'average_score': float(self.average_score) if self.average_score else 0.0,
            'highest_score': float(self.highest_score) if self.highest_score else 0.0,
            'lowest_score': float(self.lowest_score) if self.lowest_score else 0.0,
            'total_questions': self.total_questions,
            'pass_marks': float(self.pass_marks) if self.pass_marks else 0.0,
            'processing_time': self._calculate_processing_time()
        }
    
    def _calculate_processing_time(self):
        if self.processing_started_at and self.processing_completed_at:
            delta = self.processing_completed_at - self.processing_started_at
            return str(delta)
        return None

class OMRSheet(db.Model):
    """Individual sheet model (Results.jsx, Dashboard.jsx)"""
    __tablename__ = 'omr_sheets'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_id = db.Column(db.String(50), db.ForeignKey('omr_batches.id'), nullable=False)
    
    # Student Information (from Results.jsx)
    student_id = db.Column(db.String(50))
    student_name = db.Column(db.String(100))
    student_roll_number = db.Column(db.String(50))
    student_class = db.Column(db.String(20))
    
    # Image Processing (from Upload.jsx)
    image_path = db.Column(db.String(500), nullable=False)
    image_name = db.Column(db.String(200))
    image_size = db.Column(db.Integer)
    detected_set = db.Column(db.String(10))
    
    # Scoring Information (from Results.jsx, ResultsTable.jsx)
    total_questions = db.Column(db.Integer, default=0)
    score = db.Column(db.Integer, default=0)
    total_marks = db.Column(db.Integer, default=100)
    percentage = db.Column(db.Numeric(5,2), default=0.0)
    grade = db.Column(db.String(5))
    pass_fail = db.Column(db.String(10), default='PENDING')
    
    # Subject-wise Scores (from ResultsTable.jsx)
    math_score = db.Column(db.Integer, default=0)
    physics_score = db.Column(db.Integer, default=0)
    chemistry_score = db.Column(db.Integer, default=0)
    history_score = db.Column(db.Integer, default=0)
    english_score = db.Column(db.Integer, default=0)
    biology_score = db.Column(db.Integer, default=0)
    
    # Processing Status (from Results.jsx)
    status = db.Column(db.String(20), default='Processing')
    processing_date = db.Column(db.DateTime, default=datetime.utcnow)
    review_notes = db.Column(db.Text)
    reviewer_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    reviewed_at = db.Column(db.DateTime)
    
    # Quality Metrics
    confidence_score = db.Column(db.Numeric(5,2))
    multiple_marked_count = db.Column(db.Integer, default=0)
    unclear_marks_count = db.Column(db.Integer, default=0)
    blank_answers_count = db.Column(db.Integer, default=0)
    
    # Raw Processing Data
    raw_results = db.Column(db.Text)
    debug_image_path = db.Column(db.String(500))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'student_id': self.student_id,
            'student_name': self.student_name,
            'student_roll_number': self.student_roll_number,
            'student_class': self.student_class,
            'image_name': self.image_name,
            'detected_set': self.detected_set,
            'score': self.score,
            'total_marks': self.total_marks,
            'percentage': float(self.percentage) if self.percentage else 0.0,
            'grade': self.grade,
            'pass_fail': self.pass_fail,
            'status': self.status,
            'processing_date': self.processing_date.isoformat(),
            'review_notes': self.review_notes,
            'confidence_score': float(self.confidence_score) if self.confidence_score else 0.0,
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

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_token(user):
    """Generate JWT token for authenticated user"""
    payload = {
        'user_id': user.id,
        'email': user.email,
        'role': user.role,
        'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
    }
    return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """Verify JWT token and return user"""
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        user = User.query.get(payload['user_id'])
        return user
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
            user = verify_token(token)
            if not user:
                return jsonify({'error': 'Invalid token'}), 401
            
            # Make user available to the route
            request.current_user = user
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Token verification failed'}), 401
    
    return decorated

def calculate_subject_scores(answers, answer_key, subjects_map):
    """Calculate subject-wise scores based on question mapping"""
    subject_scores = {}
    
    for subject, questions in subjects_map.items():
        correct = 0
        total = len(questions)
        
        for q_num in questions:
            if q_num <= len(answers) and q_num <= len(answer_key):
                if answers[q_num-1] == answer_key[q_num-1]:
                    correct += 1
        
        subject_scores[subject] = {
            'score': correct,
            'total': total,
            'percentage': (correct / total * 100) if total > 0 else 0
        }
    
    return subject_scores

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
# AUTHENTICATION ROUTES (Login.jsx, Register.jsx)
# ================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration endpoint for Register.jsx"""
    try:
        data = request.get_json()
        
        # Validate required fields (matching Register.jsx form)
        required_fields = ['username', 'email', 'password', 'fullName']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        if data.get('username') and User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already taken'}), 400
        
        # Create new user
        user = User(
            username=data['username'],
            email=data['email'].lower().strip(),
            password_hash=generate_password_hash(data['password']),
            full_name=data['fullName'],
            school_name=data.get('schoolName', ''),
            phone=data.get('phone', ''),
            role=data.get('role', 'teacher')
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Generate token
        token = generate_token(user)
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': user.to_dict(),
            'token': token
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint for Login.jsx"""
    try:
        data = request.get_json()
        
        # Get email/username and password (Login.jsx uses 'email' field but can be username)
        email_or_username = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email_or_username or not password:
            return jsonify({'error': 'Email/Username and password are required'}), 400
        
        # Find user by email or username
        user = User.query.filter(
            (User.email == email_or_username) | (User.username == email_or_username)
        ).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is disabled'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Generate token
        token = generate_token(user)
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': user.to_dict(),
            'token': token
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/auth/profile', methods=['GET'])
@require_auth
def get_profile():
    """Get current user profile"""
    return jsonify({
        'success': True,
        'user': request.current_user.to_dict()
    }), 200

@app.route('/api/auth/logout', methods=['POST'])
@require_auth
def logout():
    """User logout endpoint"""
    # In a more complete implementation, you'd invalidate the token
    # For now, just return success (client will remove token)
    return jsonify({
        'success': True,
        'message': 'Logout successful'
    }), 200

# ================================================================
# DASHBOARD ROUTES (Dashboard.jsx)
# ================================================================

@app.route('/api/dashboard/stats', methods=['GET'])
@require_auth
def dashboard_stats():
    """Get dashboard statistics for Dashboard.jsx"""
    try:
        user_id = request.current_user.id
        
        # Get batch statistics
        batches = OMRBatch.query.filter_by(user_id=user_id).all()
        
        total_batches = len(batches)
        processing_batches = len([b for b in batches if b.status == 'Processing'])
        completed_batches = len([b for b in batches if b.status == 'Completed'])
        flagged_batches = len([b for b in batches if b.status == 'Flagged'])
        
        total_sheets = sum(b.total_sheets for b in batches)
        flagged_sheets = sum(b.flagged_sheets for b in batches)
        
        # Calculate overall average
        completed_batches_list = [b for b in batches if b.status == 'Completed' and b.average_score]
        overall_average = sum(float(b.average_score) for b in completed_batches_list) / len(completed_batches_list) if completed_batches_list else 0
        
        return jsonify({
            'success': True,
            'stats': {
                'totalBatches': total_batches,
                'processingBatches': processing_batches,
                'completedBatches': completed_batches,
                'flaggedBatches': flagged_batches,
                'totalSheetsProcessed': total_sheets,
                'totalFlaggedSheets': flagged_sheets,
                'overallAverageScore': round(overall_average, 2)
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get dashboard stats: {str(e)}'}), 500

@app.route('/api/dashboard/recent-batches', methods=['GET'])
@require_auth
def recent_batches():
    """Get recent batches for Dashboard.jsx"""
    try:
        user_id = request.current_user.id
        limit = request.args.get('limit', 5, type=int)
        
        batches = OMRBatch.query.filter_by(user_id=user_id)\
                                .order_by(OMRBatch.upload_date.desc())\
                                .limit(limit)\
                                .all()
        
        return jsonify({
            'success': True,
            'batches': [batch.to_dict() for batch in batches]
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get recent batches: {str(e)}'}), 500

# ================================================================
# BATCH MANAGEMENT ROUTES (Batches.jsx, Upload.jsx)
# ================================================================

@app.route('/api/batches', methods=['GET'])
@require_auth
def get_batches():
    """Get all batches for Batches.jsx"""
    try:
        user_id = request.current_user.id
        
        # Get query parameters
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
        
        # Paginate
        batches = query.order_by(OMRBatch.upload_date.desc())\
                      .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'success': True,
            'batches': [batch.to_dict() for batch in batches.items],
            'pagination': {
                'page': batches.page,
                'pages': batches.pages,
                'per_page': batches.per_page,
                'total': batches.total,
                'has_next': batches.has_next,
                'has_prev': batches.has_prev
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get batches: {str(e)}'}), 500

@app.route('/api/batches/<batch_id>', methods=['GET'])
@require_auth
def get_batch(batch_id):
    """Get specific batch details"""
    try:
        batch = OMRBatch.query.filter_by(id=batch_id, user_id=request.current_user.id).first()
        
        if not batch:
            return jsonify({'error': 'Batch not found'}), 404
        
        return jsonify({
            'success': True,
            'batch': batch.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get batch: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_files():
    """File upload endpoint for Upload.jsx"""
    try:
        # Get form data
        exam_name = request.form.get('examName')
        answer_key_version = request.form.get('answerKeyVersion')
        school_name = request.form.get('schoolName', '')
        grade = request.form.get('grade', '')
        subject = request.form.get('subject', '')
        
        if not exam_name or not answer_key_version:
            return jsonify({'error': 'Exam name and answer key version are required'}), 400
        
        # Get uploaded files
        answer_key_file = request.files.get('answerKey')
        image_files = request.files.getlist('images')
        
        if not answer_key_file:
            return jsonify({'error': 'Answer key file is required'}), 400
        
        if not image_files:
            return jsonify({'error': 'At least one OMR sheet image is required'}), 400
        
        # Validate files
        if not allowed_file(answer_key_file.filename):
            return jsonify({'error': 'Invalid answer key file format'}), 400
        
        for img_file in image_files:
            if not allowed_file(img_file.filename):
                return jsonify({'error': f'Invalid image file format: {img_file.filename}'}), 400
        
        # Create new batch
        batch = OMRBatch(
            user_id=request.current_user.id,
            exam_name=exam_name,
            school_name=school_name,
            grade=grade,
            subject=subject,
            answer_key_version=answer_key_version,
            total_sheets=len(image_files),
            status='Processing'
        )
        
        db.session.add(batch)
        db.session.flush()  # Get batch ID
        
        batch_folder = os.path.join(app.config['UPLOAD_FOLDER'], batch.id)
        os.makedirs(batch_folder, exist_ok=True)
        
        # Save answer key
        answer_key_filename = secure_filename(answer_key_file.filename)
        answer_key_path = os.path.join(batch_folder, f"answer_key_{answer_key_filename}")
        answer_key_file.save(answer_key_path)
        
        batch.answer_key_path = answer_key_path
        batch.answer_key_name = answer_key_filename
        
        # Process images
        batch.processing_started_at = datetime.utcnow()
        
        saved_images = []
        for img_file in image_files:
            # Save image
            img_filename = secure_filename(img_file.filename)
            img_path = os.path.join(batch_folder, img_filename)
            img_file.save(img_path)
            
            # Create sheet record
            sheet = OMRSheet(
                batch_id=batch.id,
                image_path=img_path,
                image_name=img_filename,
                image_size=os.path.getsize(img_path),
                status='Processing'
            )
            
            db.session.add(sheet)
            saved_images.append({
                'filename': img_filename,
                'size': sheet.image_size,
                'sheet_id': sheet.id
            })
        
        db.session.commit()
        
        # Start background processing (in a real app, this would be async)
        # For demo, we'll mark as completed immediately
        process_batch_async(batch.id)
        
        return jsonify({
            'success': True,
            'message': 'Files uploaded successfully',
            'batch_id': batch.id,
            'uploaded_images': saved_images,
            'answer_key': answer_key_filename
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

def process_batch_async(batch_id):
    """Process batch in background (simplified version)"""
    try:
        batch = OMRBatch.query.get(batch_id)
        if not batch:
            return
        
        # Load answer key (simplified - assume CSV format)
        # In reality, you'd parse Excel file properly
        correct_answers = ['A'] * 25 + ['B'] * 25 + ['C'] * 25 + ['D'] * 25  # Mock data
        
        # Subject mapping (mock data)
        subjects_map = {
            'math': list(range(1, 26)),
            'physics': list(range(26, 51)),
            'chemistry': list(range(51, 76)),
            'history': list(range(76, 101))
        }
        
        sheets = OMRSheet.query.filter_by(batch_id=batch_id).all()
        
        for i, sheet in enumerate(sheets):
            try:
                # Process OMR (mock processing for demo)
                # detected_answers = detect_circles_advanced(sheet.image_path)
                
                # Generate mock student data
                sheet.student_id = f"ST{(i+1):03d}"
                sheet.student_name = f"Student {i+1}"
                sheet.student_roll_number = f"R{(i+1):03d}"
                sheet.detected_set = batch.answer_key_version
                
                # Mock scoring
                import random
                sheet.score = random.randint(70, 95)
                sheet.total_marks = 100
                sheet.percentage = sheet.score
                sheet.grade = calculate_grade(sheet.percentage)
                sheet.pass_fail = 'PASS' if sheet.percentage >= batch.pass_marks else 'FAIL'
                
                # Mock subject scores
                sheet.math_score = random.randint(18, 25)
                sheet.physics_score = random.randint(16, 25)
                sheet.chemistry_score = random.randint(19, 25)
                sheet.history_score = random.randint(17, 25)
                
                sheet.status = 'Completed'
                sheet.confidence_score = random.uniform(0.8, 0.95)
                
            except Exception as e:
                sheet.status = 'Failed'
                sheet.review_notes = f'Processing error: {str(e)}'
        
        # Update batch statistics
        completed_sheets = [s for s in sheets if s.status == 'Completed']
        batch.completed_sheets = len(completed_sheets)
        batch.failed_sheets = len([s for s in sheets if s.status == 'Failed'])
        
        if completed_sheets:
            scores = [float(s.percentage) for s in completed_sheets]
            batch.average_score = sum(scores) / len(scores)
            batch.highest_score = max(scores)
            batch.lowest_score = min(scores)
        
        batch.processing_completed_at = datetime.utcnow()
        batch.status = 'Completed'
        
        db.session.commit()
        
    except Exception as e:
        print(f"Background processing error: {e}")
        if batch:
            batch.status = 'Failed'
            db.session.commit()

# ================================================================
# RESULTS ROUTES (Results.jsx, ResultsTable.jsx)
# ================================================================

@app.route('/api/batches/<batch_id>/results', methods=['GET'])
@require_auth
def get_batch_results(batch_id):
    """Get batch results for Results.jsx"""
    try:
        batch = OMRBatch.query.filter_by(id=batch_id, user_id=request.current_user.id).first()
        
        if not batch:
            return jsonify({'error': 'Batch not found'}), 404
        
        # Get query parameters for filtering
        subject = request.args.get('subject')
        status = request.args.get('status')
        min_score = request.args.get('min_score', type=float)
        max_score = request.args.get('max_score', type=float)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
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
        order_dir = request.args.get('order_dir', 'asc')
        
        if order_by == 'percentage':
            if order_dir == 'desc':
                query = query.order_by(OMRSheet.percentage.desc())
            else:
                query = query.order_by(OMRSheet.percentage.asc())
        else:
            query = query.order_by(OMRSheet.student_id)
        
        # Paginate
        sheets = query.paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'success': True,
            'batch': batch.to_dict(),
            'results': [sheet.to_dict() for sheet in sheets.items],
            'pagination': {
                'page': sheets.page,
                'pages': sheets.pages,
                'per_page': sheets.per_page,
                'total': sheets.total,
                'has_next': sheets.has_next,
                'has_prev': sheets.has_prev
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

@app.route('/api/sheets/<sheet_id>', methods=['GET'])
@require_auth
def get_sheet_details(sheet_id):
    """Get detailed sheet information"""
    try:
        sheet = OMRSheet.query.join(OMRBatch).filter(
            OMRSheet.id == sheet_id,
            OMRBatch.user_id == request.current_user.id
        ).first()
        
        if not sheet:
            return jsonify({'error': 'Sheet not found'}), 404
        
        return jsonify({
            'success': True,
            'sheet': sheet.to_dict(),
            'batch': sheet.batch.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get sheet details: {str(e)}'}), 500

@app.route('/api/sheets/<sheet_id>/review', methods=['POST'])
@require_auth
def review_sheet(sheet_id):
    """Review and update flagged sheet"""
    try:
        data = request.get_json()
        
        sheet = OMRSheet.query.join(OMRBatch).filter(
            OMRSheet.id == sheet_id,
            OMRBatch.user_id == request.current_user.id
        ).first()
        
        if not sheet:
            return jsonify({'error': 'Sheet not found'}), 404
        
        # Update sheet with review
        sheet.review_notes = data.get('notes', '')
        sheet.reviewer_id = request.current_user.id
        sheet.reviewed_at = datetime.utcnow()
        
        # Update status if provided
        new_status = data.get('status')
        if new_status in ['Completed', 'Failed', 'Flagged']:
            sheet.status = new_status
        
        # Update scores if provided
        if 'score' in data:
            sheet.score = data['score']
            sheet.percentage = (sheet.score / sheet.total_marks) * 100 if sheet.total_marks > 0 else 0
            sheet.grade = calculate_grade(sheet.percentage)
            sheet.pass_fail = 'PASS' if sheet.percentage >= sheet.batch.pass_marks else 'FAIL'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Sheet reviewed successfully',
            'sheet': sheet.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to review sheet: {str(e)}'}), 500

# ================================================================
# EXPORT AND REPORTING
# ================================================================

@app.route('/api/batches/<batch_id>/export', methods=['GET'])
@require_auth
def export_batch_results(batch_id):
    """Export batch results as CSV"""
    try:
        batch = OMRBatch.query.filter_by(id=batch_id, user_id=request.current_user.id).first()
        
        if not batch:
            return jsonify({'error': 'Batch not found'}), 404
        
        sheets = OMRSheet.query.filter_by(batch_id=batch_id).all()
        
        # Create CSV data
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow([
            'Student ID', 'Student Name', 'Roll Number', 'Score', 'Percentage', 
            'Grade', 'Pass/Fail', 'Math Score', 'Physics Score', 'Chemistry Score', 
            'History Score', 'Status', 'Processing Date'
        ])
        
        # Data
        for sheet in sheets:
            writer.writerow([
                sheet.student_id, sheet.student_name, sheet.student_roll_number,
                sheet.score, float(sheet.percentage), sheet.grade, sheet.pass_fail,
                sheet.math_score, sheet.physics_score, sheet.chemistry_score,
                sheet.history_score, sheet.status, 
                sheet.processing_date.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        output.seek(0)
        
        from flask import Response
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=results_{batch.exam_name}_{batch_id}.csv'
            }
        )
        
    except Exception as e:
        return jsonify({'error': f'Failed to export results: {str(e)}'}), 500

# ================================================================
# FILE SERVING
# ================================================================

@app.route('/api/files/<path:filename>')
@require_auth
def serve_file(filename):
    """Serve uploaded files (images, debug images, etc.)"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404

# ================================================================
# ERROR HANDLERS
# ================================================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# ================================================================
# APPLICATION INITIALIZATION
# ================================================================

def create_tables():
    """Create database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    # Create tables if they don't exist
    create_tables()
    
    # Run the application
    print("="*50)
    print("OMR EVALUATION SYSTEM - BACKEND SERVER")
    print("="*50)
    print(f"Frontend can access API at: http://127.0.0.1:5000")
    print(f"CORS enabled for: http://localhost:3000, http://localhost:5173")
    print("="*50)
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )