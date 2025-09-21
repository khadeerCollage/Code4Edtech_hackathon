-- ========================================
-- FRESH OMR EVALUATION SYSTEM DATABASE SETUP
-- Complete PostgreSQL Database Setup Script
-- ========================================

-- Step 1: Create Database and User (Run as postgres superuser)
-- Command: psql -U postgres -f this_file.sql

-- Drop existing database if it exists (CAUTION: This will delete all data!)
DROP DATABASE IF EXISTS omr_evaluation_db;
DROP USER IF EXISTS omr_user;

-- Create fresh database
CREATE DATABASE omr_evaluation_db;

-- Create dedicated user for OMR system
CREATE USER omr_user WITH PASSWORD 'omr_password123';

-- Grant all privileges to the user
GRANT ALL PRIVILEGES ON DATABASE omr_evaluation_db TO omr_user;
ALTER USER omr_user CREATEDB;

-- Connect to the new database
\c omr_evaluation_db;

-- Grant schema privileges to the user
GRANT ALL ON SCHEMA public TO omr_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO omr_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO omr_user;

-- ========================================
-- Step 2: Create All Tables (Fresh Schema)
-- ========================================

-- Drop all existing tables (clean slate)
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

-- Create Subject Scores table
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

-- ========================================
-- Step 3: Create Indexes for Performance
-- ========================================

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_omr_batches_user_id ON omr_batches(user_id);
CREATE INDEX idx_omr_batches_status ON omr_batches(status);
CREATE INDEX idx_omr_batches_upload_date ON omr_batches(upload_date DESC);
CREATE INDEX idx_omr_sheets_batch_id ON omr_sheets(batch_id);
CREATE INDEX idx_omr_sheets_status ON omr_sheets(status);
CREATE INDEX idx_omr_sheets_student_id ON omr_sheets(student_id);
CREATE INDEX idx_omr_sheets_processing_date ON omr_sheets(processing_date DESC);
CREATE INDEX idx_subject_scores_sheet_id ON subject_scores(sheet_id);
CREATE INDEX idx_subject_scores_subject_name ON subject_scores(subject_name);

-- ========================================
-- Step 4: Add Data Validation Constraints
-- ========================================

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

-- ========================================
-- Step 5: Create Helpful Views
-- ========================================

-- Batch statistics view
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

-- Detailed results view
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

-- ========================================
-- Step 6: Grant Final Permissions
-- ========================================

-- Grant all permissions to omr_user on new tables and views
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO omr_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO omr_user;
GRANT SELECT ON batch_statistics TO omr_user;
GRANT SELECT ON detailed_results TO omr_user;

-- ========================================
-- Step 7: Insert Test Data (Optional)
-- ========================================

-- Create test user with known password (password123)
INSERT INTO users (email, password_hash) VALUES 
('test@example.com', 'scrypt:32768:8:1$salt$hashedpassword'),
('admin@omr.com', 'scrypt:32768:8:1$salt$hashedpassword');

-- Insert sample batches for testing
INSERT INTO omr_batches (id, user_id, exam_name, answer_key_version, status, total_sheets, completed_sheets) VALUES 
('test-batch-001', 1, 'Sample Mathematics Exam', 'A', 'Completed', 3, 3);

-- Insert sample sheets
INSERT INTO omr_sheets (id, batch_id, student_id, student_name, score, percentage, status, math_score, physics_score, chemistry_score, history_score) VALUES 
('sheet-test-001', 'test-batch-001', 'ST001', 'Test Student 1', 85, 85.0, 'Completed', 22, 20, 23, 20),
('sheet-test-002', 'test-batch-001', 'ST002', 'Test Student 2', 92, 92.0, 'Completed', 25, 22, 24, 21),
('sheet-test-003', 'test-batch-001', 'ST003', 'Test Student 3', 78, 78.0, 'Completed', 19, 18, 21, 20);

-- ========================================
-- Step 8: Verification Queries
-- ========================================

-- Check all tables are created
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check users table
SELECT COUNT(*) as user_count FROM users;

-- Check batches table
SELECT COUNT(*) as batch_count FROM omr_batches;

-- Check sheets table
SELECT COUNT(*) as sheet_count FROM omr_sheets;

-- Display success message
SELECT 'Fresh OMR Evaluation Database Setup Complete!' as status;
SELECT 'Database: omr_evaluation_db' as info;
SELECT 'User: omr_user' as info;
SELECT 'Password: omr_password123' as info;
SELECT 'Connection: postgresql://omr_user:omr_password123@localhost:5432/omr_evaluation_db' as connection_string;