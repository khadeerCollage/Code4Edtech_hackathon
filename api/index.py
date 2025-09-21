# Vercel serverless function entry point
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["*"])

# Mock data for demonstration
DEMO_BATCHES = [
    {
        'id': '91d51108-9c46-417a-8153-0686b8702f34',
        'exam_name': 'Code4Edtech Hackathon Demo',
        'school_name': 'Demo School',
        'grade': '10th',
        'subject': 'Mixed Topics',
        'answer_key_version': 'A',
        'upload_date': datetime.now().isoformat(),
        'status': 'Completed',
        'total_sheets': 5,
        'completed_sheets': 5,
        'flagged_sheets': 0,
        'failed_sheets': 0,
        'average_score': 85.2,
        'highest_score': 95.0,
        'lowest_score': 72.0,
        'total_questions': 100
    }
]

DEMO_RESULTS = [
    {
        'id': '1',
        'student_id': 'STU001',
        'student_name': 'John Doe',
        'score': 85,
        'percentage': 85.0,
        'grade': 'B+',
        'total_questions': 100,
        'correct_answers': 85,
        'incorrect_answers': 10,
        'unattempted': 5,
        'status': 'Completed'
    },
    {
        'id': '2', 
        'student_id': 'STU002',
        'student_name': 'Jane Smith',
        'score': 92,
        'percentage': 92.0,
        'grade': 'A-',
        'total_questions': 100,
        'correct_answers': 92,
        'incorrect_answers': 6,
        'unattempted': 2,
        'status': 'Completed'
    }
]

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if path == '':
        return jsonify({
            'message': 'OMR Evaluation System API',
            'status': 'running',
            'version': '1.0.0'
        })
    
    if path.startswith('batches/') and path.endswith('/status'):
        batch_id = path.split('/')[1]
        batch = next((b for b in DEMO_BATCHES if b['id'] == batch_id), None)
        if batch:
            return jsonify(batch)
        return jsonify({'error': 'Batch not found'}), 404
    
    if path.startswith('batches/') and path.endswith('/results'):
        batch_id = path.split('/')[1]
        return jsonify({
            'batch_id': batch_id,
            'sheets': DEMO_RESULTS,
            'total': len(DEMO_RESULTS)
        })
    
    if path == 'dashboard/stats':
        return jsonify({
            'total_batches': 1,
            'total_sheets': 5,
            'completed_sheets': 5,
            'average_score': 85.2
        })
    
    if path == 'dashboard/recent-batches':
        return jsonify({
            'batches': DEMO_BATCHES[:3]
        })
    
    if path == 'batches':
        return jsonify({
            'batches': DEMO_BATCHES
        })
    
    if path == 'upload' and request.method == 'POST':
        return jsonify({
            'message': 'Demo mode - files uploaded successfully',
            'batch_id': '91d51108-9c46-417a-8153-0686b8702f34',
            'status': 'success'
        })
    
    if path == 'auth/login' and request.method == 'POST':
        return jsonify({
            'message': 'Demo login successful',
            'token': 'demo-token-12345',
            'user': {'id': 1, 'email': 'demo@example.com'}
        })
    
    if path == 'auth/register' and request.method == 'POST':
        return jsonify({
            'message': 'Demo registration successful',
            'user': {'id': 1, 'email': 'demo@example.com'}
        })
    
    return jsonify({'error': 'Not found'}), 404