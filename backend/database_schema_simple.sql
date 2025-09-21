-- Simple OMR Evaluation System Database Schema
-- PostgreSQL Database Schema for OMR processing system

-- Drop existing tables if they exist
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

-- Create essential indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_omr_batches_user_id ON omr_batches(user_id);
CREATE INDEX idx_omr_sheets_batch_id ON omr_sheets(batch_id);
CREATE INDEX idx_subject_scores_sheet_id ON subject_scores(sheet_id);

-- Add constraints
ALTER TABLE omr_batches 
ADD CONSTRAINT chk_batch_status 
CHECK (status IN ('Processing', 'Completed', 'Failed', 'Flagged'));

ALTER TABLE omr_sheets 
ADD CONSTRAINT chk_sheet_status 
CHECK (status IN ('Processing', 'Completed', 'Failed', 'Flagged'));

-- Insert test user
INSERT INTO users (email, password_hash) VALUES 
('test@example.com', 'scrypt:32768:8:1$salt$hashedpassword');

SELECT 'Database schema created successfully!' as message;