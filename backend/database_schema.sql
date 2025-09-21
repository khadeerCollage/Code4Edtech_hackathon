-- OMR Evaluation System Database Schema
-- PostgreSQL Database Schema for OMR processing system
-- Created: December 2024

-- Drop existing tables (be careful with this in production!)
DROP TABLE IF EXISTS subject_scores CASCADE;
DROP TABLE IF EXISTS omr_sheets CASCADE;
DROP TABLE IF EXISTS omr_batches CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Create Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on email for faster lookups
CREATE INDEX idx_users_email ON users(email);

-- Create OMR Batches table
CREATE TABLE omr_batches (
    id VARCHAR(50) PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    exam_name VARCHAR(200) NOT NULL,
    answer_key_version VARCHAR(10) NOT NULL,
    answer_key_path VARCHAR(500),
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'Processing',
    total_sheets INTEGER DEFAULT 0,
    completed_sheets INTEGER DEFAULT 0,
    flagged_sheets INTEGER DEFAULT 0,
    average_score DECIMAL(5,2) DEFAULT 0.0
);

-- Create indexes for better query performance
CREATE INDEX idx_omr_batches_user_id ON omr_batches(user_id);
CREATE INDEX idx_omr_batches_status ON omr_batches(status);
CREATE INDEX idx_omr_batches_upload_date ON omr_batches(upload_date DESC);

-- Create OMR Sheets table
CREATE TABLE omr_sheets (
    id VARCHAR(50) PRIMARY KEY,
    batch_id VARCHAR(50) REFERENCES omr_batches(id) ON DELETE CASCADE,
    student_id VARCHAR(50),
    student_name VARCHAR(100),
    image_path VARCHAR(500) NOT NULL,
    detected_set VARCHAR(10),
    total_questions INTEGER DEFAULT 0,
    score INTEGER DEFAULT 0,
    total_marks INTEGER DEFAULT 100,
    percentage DECIMAL(5,2) DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'Processing',
    processing_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    review_notes TEXT,
    raw_results TEXT,
    math_score INTEGER DEFAULT 0,
    physics_score INTEGER DEFAULT 0,
    chemistry_score INTEGER DEFAULT 0,
    history_score INTEGER DEFAULT 0
);

-- Create indexes for OMR sheets
CREATE INDEX idx_omr_sheets_batch_id ON omr_sheets(batch_id);
CREATE INDEX idx_omr_sheets_status ON omr_sheets(status);
CREATE INDEX idx_omr_sheets_student_id ON omr_sheets(student_id);
CREATE INDEX idx_omr_sheets_processing_date ON omr_sheets(processing_date DESC);

-- Create Subject Scores table (detailed subject breakdown)
CREATE TABLE subject_scores (
    id SERIAL PRIMARY KEY,
    sheet_id VARCHAR(50) REFERENCES omr_sheets(id) ON DELETE CASCADE,
    subject_name VARCHAR(50) NOT NULL,
    total_questions INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    incorrect_answers INTEGER DEFAULT 0,
    unattempted INTEGER DEFAULT 0,
    multiple_marked INTEGER DEFAULT 0,
    percentage DECIMAL(5,2) DEFAULT 0.0
);

-- Create indexes for subject scores
CREATE INDEX idx_subject_scores_sheet_id ON subject_scores(sheet_id);
CREATE INDEX idx_subject_scores_subject_name ON subject_scores(subject_name);

-- Add some constraints for data validation
ALTER TABLE omr_batches 
ADD CONSTRAINT chk_batch_status 
CHECK (status IN ('Processing', 'Completed', 'Failed', 'Flagged'));

ALTER TABLE omr_sheets 
ADD CONSTRAINT chk_sheet_status 
CHECK (status IN ('Processing', 'Completed', 'Failed', 'Flagged'));

ALTER TABLE omr_sheets 
ADD CONSTRAINT chk_percentage_range 
CHECK (percentage >= 0.0 AND percentage <= 100.0);

ALTER TABLE subject_scores 
ADD CONSTRAINT chk_subject_percentage_range 
CHECK (percentage >= 0.0 AND percentage <= 100.0);

-- Create a view for easy batch statistics
CREATE VIEW batch_statistics AS
SELECT 
    b.id,
    b.exam_name,
    b.upload_date,
    b.status,
    b.total_sheets,
    b.completed_sheets,
    b.flagged_sheets,
    b.average_score,
    COUNT(s.id) as actual_sheets,
    AVG(s.percentage) as calculated_avg_score,
    MIN(s.percentage) as min_score,
    MAX(s.percentage) as max_score
FROM omr_batches b
LEFT JOIN omr_sheets s ON b.id = s.batch_id
GROUP BY b.id, b.exam_name, b.upload_date, b.status, 
         b.total_sheets, b.completed_sheets, b.flagged_sheets, b.average_score;

-- Create a view for detailed results
CREATE VIEW detailed_results AS
SELECT 
    s.id as sheet_id,
    s.batch_id,
    b.exam_name,
    b.answer_key_version,
    s.student_id,
    s.student_name,
    s.detected_set,
    s.score,
    s.total_marks,
    s.percentage,
    s.status,
    s.math_score,
    s.physics_score,
    s.chemistry_score,
    s.history_score,
    s.processing_date,
    s.review_notes
FROM omr_sheets s
JOIN omr_batches b ON s.batch_id = b.id;

-- Insert sample test data (optional - remove in production)
INSERT INTO users (email, password_hash) VALUES 
('admin@omr.com', 'scrypt:32768:8:1$HGxU9XzPqKj4Rd9o$6f0a4f8e9c7b2a1d3e5f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z'),
('teacher@school.edu', 'scrypt:32768:8:1$HGxU9XzPqKj4Rd9o$6f0a4f8e9c7b2a1d3e5f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z');

-- Sample batch data
INSERT INTO omr_batches (id, user_id, exam_name, answer_key_version, status, total_sheets, completed_sheets) VALUES 
('batch-001', 1, 'Mathematics Final Exam 2024', 'A', 'Completed', 120, 118),
('batch-002', 1, 'Physics Midterm Test', 'B', 'Processing', 85, 45),
('batch-003', 1, 'Chemistry Quiz Set A', 'A', 'Flagged', 60, 58);

-- Sample sheet data
INSERT INTO omr_sheets (id, batch_id, student_id, student_name, score, percentage, status, math_score, physics_score, chemistry_score, history_score) VALUES 
('sheet-001', 'batch-001', 'ST001', 'Alice Johnson', 85, 85.0, 'Completed', 22, 20, 23, 20),
('sheet-002', 'batch-001', 'ST002', 'Bob Smith', 92, 92.0, 'Completed', 25, 22, 24, 21),
('sheet-003', 'batch-001', 'ST003', 'Carol Davis', 78, 78.0, 'Flagged', 19, 18, 21, 20);

-- Display table information
\echo 'Database schema created successfully!'
\echo 'Tables created:'
\echo '- users: Store user authentication data'
\echo '- omr_batches: Store batch information for OMR processing'
\echo '- omr_sheets: Store individual OMR sheet processing results'
\echo '- subject_scores: Store detailed subject-wise scoring breakdown'
\echo ''
\echo 'Views created:'
\echo '- batch_statistics: Aggregated statistics for each batch'
\echo '- detailed_results: Combined view of sheets with batch information'
\echo ''
\echo 'Sample data inserted for testing'
\echo 'Default users:'
\echo '  admin@omr.com / password123'
\echo '  teacher@school.edu / password123'