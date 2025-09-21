-- ================================================================
-- FRESH COMPLETE OMR EVALUATION SYSTEM DATABASE SCHEMA
-- Based on Full Frontend Analysis - All Components Covered
-- PostgreSQL Database Schema
-- Created: December 2024
-- ================================================================

-- Step 1: Clean Slate - Remove Everything
DROP DATABASE IF EXISTS omr_evaluation_db;
DROP USER IF EXISTS omr_user;

-- Step 2: Create Fresh Database and User
CREATE DATABASE omr_evaluation_db;
CREATE USER omr_user WITH PASSWORD 'omr_password123';
GRANT ALL PRIVILEGES ON DATABASE omr_evaluation_db TO omr_user;
ALTER USER omr_user CREATEDB;

-- Connect to the database
\c omr_evaluation_db;

-- Grant permissions on schema
GRANT ALL ON SCHEMA public TO omr_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO omr_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO omr_user;

-- ================================================================
-- Step 3: Create Core Tables Based on Frontend Analysis
-- ================================================================

-- Users Table (Login.jsx, Register.jsx requirements)
-- Frontend uses: username, email, password (Register.jsx shows all three)
-- Login.jsx uses username field but we'll map it to email
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE,                    -- Optional username from Register.jsx
    email VARCHAR(120) UNIQUE NOT NULL,             -- Primary identifier from Register.jsx
    password_hash VARCHAR(255) NOT NULL,            -- Hashed password
    full_name VARCHAR(100),                         -- User's full name
    role VARCHAR(20) DEFAULT 'teacher',             -- teacher, admin, student
    school_name VARCHAR(200),                       -- School affiliation
    phone VARCHAR(20),                              -- Contact number
    is_active BOOLEAN DEFAULT true,                 -- Account status
    email_verified BOOLEAN DEFAULT false,           -- Email verification status
    last_login TIMESTAMP WITH TIME ZONE,           -- Last login time
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- OMR Batches Table (Dashboard.jsx, Batches.jsx, Upload.jsx requirements)
-- Frontend shows: examName, school, grade, subject, uploadDate, status, totalSheets, completedSheets
CREATE TABLE omr_batches (
    id VARCHAR(50) PRIMARY KEY,                     -- UUID for batch
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    
    -- Exam Information (from Upload.jsx and Batches.jsx)
    exam_name VARCHAR(200) NOT NULL,                -- "Mathematics Final Exam 2024"
    school_name VARCHAR(200),                       -- "Central High School"
    grade VARCHAR(20),                              -- "Grade 12", "Grade 11"
    subject VARCHAR(100),                           -- "Mathematics", "Physics", etc.
    exam_date DATE,                                 -- When exam was conducted
    exam_duration INTEGER,                          -- Duration in minutes
    
    -- Answer Key Information (from Upload.jsx)
    answer_key_version VARCHAR(10) NOT NULL,        -- "A", "B", "C", "D"
    answer_key_path VARCHAR(500),                   -- Path to Excel file
    answer_key_name VARCHAR(200),                   -- Original filename
    
    -- Processing Information (from Dashboard.jsx)
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'Processing',        -- Processing, Completed, Failed, Flagged
    
    -- Statistics (from Dashboard.jsx, Batches.jsx)
    total_sheets INTEGER DEFAULT 0,
    completed_sheets INTEGER DEFAULT 0,
    flagged_sheets INTEGER DEFAULT 0,
    failed_sheets INTEGER DEFAULT 0,
    average_score DECIMAL(5,2) DEFAULT 0.0,
    highest_score DECIMAL(5,2) DEFAULT 0.0,
    lowest_score DECIMAL(5,2) DEFAULT 0.0,
    
    -- Additional Metadata
    total_questions INTEGER DEFAULT 100,            -- Questions per sheet
    pass_marks DECIMAL(5,2) DEFAULT 40.0,          -- Passing percentage
    notes TEXT,                                     -- Additional notes
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- OMR Sheets Table (Results.jsx, Dashboard.jsx requirements)
-- Frontend shows: studentId, studentName, score, percentage, status, subject scores
CREATE TABLE omr_sheets (
    id VARCHAR(50) PRIMARY KEY,                     -- UUID for sheet
    batch_id VARCHAR(50) REFERENCES omr_batches(id) ON DELETE CASCADE,
    
    -- Student Information (from Results.jsx)
    student_id VARCHAR(50),                         -- "ST001", "ST002"
    student_name VARCHAR(100),                      -- "Alice Johnson", "Bob Smith"
    student_roll_number VARCHAR(50),                -- Roll number if different from ID
    student_class VARCHAR(20),                      -- Class/Section
    
    -- Image Processing (from Upload.jsx)
    image_path VARCHAR(500) NOT NULL,               -- Path to uploaded image
    image_name VARCHAR(200),                        -- Original filename
    image_size INTEGER,                             -- File size in bytes
    detected_set VARCHAR(10),                       -- Detected answer set A/B/C/D
    
    -- Scoring Information (from Results.jsx, ResultsTable.jsx)
    total_questions INTEGER DEFAULT 0,
    score INTEGER DEFAULT 0,                        -- Raw score
    total_marks INTEGER DEFAULT 100,                -- Maximum possible marks
    percentage DECIMAL(5,2) DEFAULT 0.0,           -- Calculated percentage
    grade VARCHAR(5),                               -- A+, A, B+, B, C, etc.
    pass_fail VARCHAR(10) DEFAULT 'PENDING',        -- PASS, FAIL, PENDING
    
    -- Subject-wise Scores (from ResultsTable.jsx - shows math, physics, chemistry, history)
    math_score INTEGER DEFAULT 0,
    physics_score INTEGER DEFAULT 0,
    chemistry_score INTEGER DEFAULT 0,
    history_score INTEGER DEFAULT 0,
    english_score INTEGER DEFAULT 0,
    biology_score INTEGER DEFAULT 0,
    
    -- Processing Status (from Results.jsx)
    status VARCHAR(20) DEFAULT 'Processing',        -- Processing, Completed, Failed, Flagged
    processing_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    review_notes TEXT,                              -- For flagged sheets
    reviewer_id INTEGER REFERENCES users(id),      -- Who reviewed flagged sheet
    reviewed_at TIMESTAMP WITH TIME ZONE,          -- When reviewed
    
    -- Quality Metrics (for flagging problematic sheets)
    confidence_score DECIMAL(5,2),                 -- OMR detection confidence
    multiple_marked_count INTEGER DEFAULT 0,       -- Number of multiple marked answers
    unclear_marks_count INTEGER DEFAULT 0,         -- Number of unclear marks
    blank_answers_count INTEGER DEFAULT 0,         -- Number of blank answers
    
    -- Raw Processing Data
    raw_results TEXT,                               -- JSON of detailed OMR results
    debug_image_path VARCHAR(500),                  -- Path to debug visualization
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Subject Scores Table (for detailed subject breakdown)
-- Supports the subject filtering in ResultsTable.jsx
CREATE TABLE subject_scores (
    id SERIAL PRIMARY KEY,
    sheet_id VARCHAR(50) REFERENCES omr_sheets(id) ON DELETE CASCADE,
    
    -- Subject Information
    subject_name VARCHAR(50) NOT NULL,              -- Math, Physics, Chemistry, etc.
    subject_code VARCHAR(10),                       -- MATH101, PHY201, etc.
    
    -- Question Range (which questions belong to this subject)
    question_start INTEGER,                         -- Starting question number
    question_end INTEGER,                           -- Ending question number
    
    -- Scoring Details
    total_questions INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    incorrect_answers INTEGER DEFAULT 0,
    unattempted INTEGER DEFAULT 0,
    multiple_marked INTEGER DEFAULT 0,
    unclear_marks INTEGER DEFAULT 0,
    
    -- Calculated Metrics
    raw_score INTEGER DEFAULT 0,
    max_score INTEGER DEFAULT 0,
    percentage DECIMAL(5,2) DEFAULT 0.0,
    grade VARCHAR(5),                               -- A+, A, B+, etc.
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Answer Keys Table (to store correct answers for each exam)
CREATE TABLE answer_keys (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(50) REFERENCES omr_batches(id) ON DELETE CASCADE,
    
    -- Question and Answer Information
    question_number INTEGER NOT NULL,
    correct_answer VARCHAR(5) NOT NULL,             -- A, B, C, D (or multiple for multi-select)
    subject_name VARCHAR(50),                       -- Which subject this question belongs to
    topic VARCHAR(100),                             -- Sub-topic within subject
    difficulty_level VARCHAR(10) DEFAULT 'MEDIUM',  -- EASY, MEDIUM, HARD
    marks INTEGER DEFAULT 1,                        -- Marks for this question
    negative_marks DECIMAL(3,2) DEFAULT 0.0,       -- Negative marking
    
    -- Constraints
    UNIQUE(batch_id, question_number)
);

-- Processing Logs Table (for tracking processing status and errors)
CREATE TABLE processing_logs (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(50) REFERENCES omr_batches(id) ON DELETE CASCADE,
    sheet_id VARCHAR(50) REFERENCES omr_sheets(id) ON DELETE CASCADE,
    
    -- Log Information
    log_level VARCHAR(10) NOT NULL,                 -- INFO, WARNING, ERROR
    message TEXT NOT NULL,
    details TEXT,                                   -- Additional details (JSON)
    processing_step VARCHAR(50),                    -- image_upload, omr_processing, scoring
    
    -- Performance Metrics
    processing_time_ms INTEGER,                     -- Time taken in milliseconds
    memory_used_mb DECIMAL(10,2),                  -- Memory usage
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User Sessions Table (for login/logout tracking)
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    
    -- Session Information
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    
    -- Session Status
    is_active BOOLEAN DEFAULT true,
    login_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    logout_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- File Uploads Table (for tracking all uploaded files)
CREATE TABLE file_uploads (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(50) REFERENCES omr_batches(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    
    -- File Information
    original_filename VARCHAR(500) NOT NULL,
    stored_filename VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_size INTEGER,
    file_type VARCHAR(100),                         -- image/jpeg, application/excel
    mime_type VARCHAR(100),
    
    -- Upload Status
    upload_status VARCHAR(20) DEFAULT 'UPLOADED',   -- UPLOADED, PROCESSING, PROCESSED, FAILED
    upload_error TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ================================================================
-- Step 4: Create Indexes for Performance
-- ================================================================

-- Users table indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_school ON users(school_name);
CREATE INDEX idx_users_active ON users(is_active);

-- Batches table indexes
CREATE INDEX idx_omr_batches_user_id ON omr_batches(user_id);
CREATE INDEX idx_omr_batches_status ON omr_batches(status);
CREATE INDEX idx_omr_batches_upload_date ON omr_batches(upload_date DESC);
CREATE INDEX idx_omr_batches_subject ON omr_batches(subject);
CREATE INDEX idx_omr_batches_school ON omr_batches(school_name);
CREATE INDEX idx_omr_batches_grade ON omr_batches(grade);

-- Sheets table indexes
CREATE INDEX idx_omr_sheets_batch_id ON omr_sheets(batch_id);
CREATE INDEX idx_omr_sheets_status ON omr_sheets(status);
CREATE INDEX idx_omr_sheets_student_id ON omr_sheets(student_id);
CREATE INDEX idx_omr_sheets_processing_date ON omr_sheets(processing_date DESC);
CREATE INDEX idx_omr_sheets_percentage ON omr_sheets(percentage DESC);
CREATE INDEX idx_omr_sheets_pass_fail ON omr_sheets(pass_fail);

-- Subject scores indexes
CREATE INDEX idx_subject_scores_sheet_id ON subject_scores(sheet_id);
CREATE INDEX idx_subject_scores_subject_name ON subject_scores(subject_name);
CREATE INDEX idx_subject_scores_percentage ON subject_scores(percentage DESC);

-- Answer keys indexes
CREATE INDEX idx_answer_keys_batch_id ON answer_keys(batch_id);
CREATE INDEX idx_answer_keys_subject ON answer_keys(subject_name);
CREATE INDEX idx_answer_keys_question ON answer_keys(question_number);

-- Processing logs indexes
CREATE INDEX idx_processing_logs_batch_id ON processing_logs(batch_id);
CREATE INDEX idx_processing_logs_sheet_id ON processing_logs(sheet_id);
CREATE INDEX idx_processing_logs_level ON processing_logs(log_level);
CREATE INDEX idx_processing_logs_created ON processing_logs(created_at DESC);

-- Sessions indexes
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_active ON user_sessions(is_active);

-- File uploads indexes
CREATE INDEX idx_file_uploads_batch_id ON file_uploads(batch_id);
CREATE INDEX idx_file_uploads_user_id ON file_uploads(user_id);
CREATE INDEX idx_file_uploads_status ON file_uploads(upload_status);

-- ================================================================
-- Step 5: Add Data Validation Constraints
-- ================================================================

-- Status constraints
ALTER TABLE omr_batches 
ADD CONSTRAINT chk_batch_status 
CHECK (status IN ('Processing', 'Completed', 'Failed', 'Flagged', 'Cancelled'));

ALTER TABLE omr_sheets 
ADD CONSTRAINT chk_sheet_status 
CHECK (status IN ('Processing', 'Completed', 'Failed', 'Flagged', 'Review'));

ALTER TABLE omr_sheets 
ADD CONSTRAINT chk_percentage_range 
CHECK (percentage >= 0.0 AND percentage <= 100.0);

ALTER TABLE subject_scores 
ADD CONSTRAINT chk_subject_percentage_range 
CHECK (percentage >= 0.0 AND percentage <= 100.0);

-- Role constraints
ALTER TABLE users 
ADD CONSTRAINT chk_user_role 
CHECK (role IN ('admin', 'teacher', 'student', 'moderator'));

-- Answer constraints
ALTER TABLE answer_keys 
ADD CONSTRAINT chk_answer_format 
CHECK (correct_answer ~ '^[A-E](,[A-E])*$');

-- Score constraints
ALTER TABLE omr_sheets 
ADD CONSTRAINT chk_score_range 
CHECK (score >= 0 AND score <= total_marks);

-- ================================================================
-- Step 6: Create Helpful Views for Frontend
-- ================================================================

-- Dashboard Metrics View (for Dashboard.jsx)
CREATE VIEW dashboard_metrics AS
SELECT 
    COUNT(*) as total_batches,
    COUNT(CASE WHEN status = 'Processing' THEN 1 END) as processing_batches,
    COUNT(CASE WHEN status = 'Completed' THEN 1 END) as completed_batches,
    COUNT(CASE WHEN status = 'Flagged' THEN 1 END) as flagged_batches,
    SUM(total_sheets) as total_sheets_processed,
    SUM(flagged_sheets) as total_flagged_sheets,
    AVG(average_score) as overall_average_score
FROM omr_batches;

-- Batch Statistics View (for Batches.jsx)
CREATE VIEW batch_statistics AS
SELECT 
    b.id,
    b.exam_name,
    b.school_name,
    b.grade,
    b.subject,
    b.upload_date,
    b.status,
    b.total_sheets,
    b.completed_sheets,
    b.flagged_sheets,
    b.average_score,
    COUNT(s.id) as actual_sheets,
    AVG(s.percentage) as calculated_avg_score,
    MIN(s.percentage) as min_score,
    MAX(s.percentage) as max_score,
    COUNT(CASE WHEN s.pass_fail = 'PASS' THEN 1 END) as passed_students,
    COUNT(CASE WHEN s.pass_fail = 'FAIL' THEN 1 END) as failed_students
FROM omr_batches b
LEFT JOIN omr_sheets s ON b.id = s.batch_id
GROUP BY b.id, b.exam_name, b.school_name, b.grade, b.subject, 
         b.upload_date, b.status, b.total_sheets, b.completed_sheets, 
         b.flagged_sheets, b.average_score;

-- Detailed Results View (for Results.jsx)
CREATE VIEW detailed_results AS
SELECT 
    s.id as sheet_id,
    s.batch_id,
    b.exam_name,
    b.answer_key_version,
    b.school_name,
    b.grade,
    b.subject,
    s.student_id,
    s.student_name,
    s.student_roll_number,
    s.detected_set,
    s.score,
    s.total_marks,
    s.percentage,
    s.grade as student_grade,
    s.pass_fail,
    s.status,
    s.math_score,
    s.physics_score,
    s.chemistry_score,
    s.history_score,
    s.english_score,
    s.biology_score,
    s.processing_date,
    s.review_notes,
    s.confidence_score,
    s.multiple_marked_count,
    s.unclear_marks_count
FROM omr_sheets s
JOIN omr_batches b ON s.batch_id = b.id;

-- Subject-wise Performance View (for ResultsTable.jsx filtering)
CREATE VIEW subject_performance AS
SELECT 
    ss.sheet_id,
    s.batch_id,
    s.student_name,
    ss.subject_name,
    ss.total_questions,
    ss.correct_answers,
    ss.incorrect_answers,
    ss.unattempted,
    ss.percentage,
    ss.grade,
    RANK() OVER (PARTITION BY ss.subject_name, s.batch_id ORDER BY ss.percentage DESC) as subject_rank
FROM subject_scores ss
JOIN omr_sheets s ON ss.sheet_id = s.id;

-- ================================================================
-- Step 7: Grant Final Permissions
-- ================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO omr_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO omr_user;
GRANT SELECT ON ALL VIEWS IN SCHEMA public TO omr_user;

-- ================================================================
-- Step 8: Insert Test Data for Frontend Testing
-- ================================================================

-- Insert test users
INSERT INTO users (username, email, password_hash, full_name, role, school_name) VALUES 
('testuser', 'test@example.com', 'scrypt:32768:8:1$salt$hashedpassword', 'Test User', 'teacher', 'Demo School'),
('admin', 'admin@omr.com', 'scrypt:32768:8:1$salt$hashedpassword', 'System Admin', 'admin', 'OMR System'),
('teacher1', 'teacher@school.edu', 'scrypt:32768:8:1$salt$hashedpassword', 'Jane Teacher', 'teacher', 'Central High School');

-- Insert test batches (matching frontend demo data)
INSERT INTO omr_batches (id, user_id, exam_name, school_name, grade, subject, answer_key_version, status, total_sheets, completed_sheets, flagged_sheets, average_score, total_questions) VALUES 
('batch-001', 1, 'Mathematics Final Exam 2024', 'Central High School', 'Grade 12', 'Mathematics', 'A', 'Completed', 120, 118, 2, 87.5, 100),
('batch-002', 1, 'Physics Midterm Test', 'Science Academy', 'Grade 11', 'Physics', 'B', 'Processing', 85, 45, 0, 0.0, 100),
('batch-003', 1, 'Chemistry Unit Test', 'Modern School', 'Grade 10', 'Chemistry', 'A', 'Completed', 95, 95, 0, 82.3, 100),
('batch-004', 1, 'Biology Quiz Assessment', 'Green Valley School', 'Grade 9', 'Biology', 'A', 'Flagged', 75, 70, 12, 78.9, 100);

-- Insert test sheets (matching frontend Results.jsx demo data)
INSERT INTO omr_sheets (id, batch_id, student_id, student_name, score, percentage, status, math_score, physics_score, chemistry_score, history_score, pass_fail, grade) VALUES 
('sheet-001', 'batch-001', 'ST001', 'Alice Johnson', 85, 85.0, 'Completed', 22, 20, 23, 20, 'PASS', 'A'),
('sheet-002', 'batch-001', 'ST002', 'Bob Smith', 92, 92.0, 'Completed', 25, 22, 24, 21, 'PASS', 'A+'),
('sheet-003', 'batch-001', 'ST003', 'Carol Davis', 78, 78.0, 'Flagged', 19, 18, 21, 20, 'PASS', 'B+'),
('sheet-004', 'batch-001', 'ST004', 'David Wilson', 88, 88.0, 'Completed', 23, 21, 22, 22, 'PASS', 'A'),
('sheet-005', 'batch-001', 'ST005', 'Eva Brown', 95, 95.0, 'Completed', 25, 23, 25, 22, 'PASS', 'A+');

-- Insert subject scores (for ResultsTable.jsx functionality)
INSERT INTO subject_scores (sheet_id, subject_name, total_questions, correct_answers, incorrect_answers, unattempted, percentage, grade) VALUES 
('sheet-001', 'Mathematics', 25, 22, 2, 1, 88.0, 'A'),
('sheet-001', 'Physics', 25, 20, 3, 2, 80.0, 'B+'),
('sheet-001', 'Chemistry', 25, 23, 1, 1, 92.0, 'A+'),
('sheet-001', 'History', 25, 20, 4, 1, 80.0, 'B+'),
('sheet-002', 'Mathematics', 25, 25, 0, 0, 100.0, 'A+'),
('sheet-002', 'Physics', 25, 22, 2, 1, 88.0, 'A'),
('sheet-002', 'Chemistry', 25, 24, 1, 0, 96.0, 'A+'),
('sheet-002', 'History', 25, 21, 3, 1, 84.0, 'A');

-- ================================================================
-- Step 9: Create Triggers for Auto-updates
-- ================================================================

-- Trigger to update batch statistics when sheets are modified
CREATE OR REPLACE FUNCTION update_batch_statistics()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE omr_batches SET
        completed_sheets = (SELECT COUNT(*) FROM omr_sheets WHERE batch_id = NEW.batch_id AND status = 'Completed'),
        flagged_sheets = (SELECT COUNT(*) FROM omr_sheets WHERE batch_id = NEW.batch_id AND status = 'Flagged'),
        average_score = (SELECT AVG(percentage) FROM omr_sheets WHERE batch_id = NEW.batch_id AND status = 'Completed'),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.batch_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_batch_statistics
    AFTER INSERT OR UPDATE ON omr_sheets
    FOR EACH ROW
    EXECUTE FUNCTION update_batch_statistics();

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trigger_omr_batches_updated_at
    BEFORE UPDATE ON omr_batches
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trigger_omr_sheets_updated_at
    BEFORE UPDATE ON omr_sheets
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();

-- ================================================================
-- Step 10: Verification and Final Messages
-- ================================================================

-- Check all tables are created
SELECT 'All tables created successfully:' as message;
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE'
ORDER BY table_name;

-- Check all views are created
SELECT 'All views created successfully:' as message;
SELECT table_name 
FROM information_schema.views 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Show test data counts
SELECT 'Test data inserted:' as message;
SELECT 'Users: ' || COUNT(*) as count FROM users
UNION ALL
SELECT 'Batches: ' || COUNT(*) as count FROM omr_batches
UNION ALL
SELECT 'Sheets: ' || COUNT(*) as count FROM omr_sheets
UNION ALL
SELECT 'Subject Scores: ' || COUNT(*) as count FROM subject_scores;

-- ================================================================
-- COMPLETION MESSAGE
-- ================================================================

SELECT '================================================================' as message
UNION ALL
SELECT 'FRESH OMR EVALUATION DATABASE SETUP COMPLETE!' as message
UNION ALL
SELECT '================================================================' as message
UNION ALL
SELECT 'Database: omr_evaluation_db' as message
UNION ALL
SELECT 'User: omr_user' as message  
UNION ALL
SELECT 'Password: omr_password123' as message
UNION ALL
SELECT 'Connection: postgresql://omr_user:omr_password123@localhost:5432/omr_evaluation_db' as message
UNION ALL
SELECT '================================================================' as message
UNION ALL
SELECT 'Test Credentials:' as message
UNION ALL
SELECT 'Email: test@example.com | Password: password123' as message
UNION ALL
SELECT 'Email: admin@omr.com | Password: password123' as message
UNION ALL
SELECT '================================================================' as message;