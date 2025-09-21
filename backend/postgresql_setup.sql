-- PostgreSQL Setup Script for OMR System
-- Run this as PostgreSQL superuser (postgres)

-- Connect as superuser: psql -U postgres

-- Create database and user for OMR system
CREATE DATABASE omr_evaluation_db;

-- Create user with password
CREATE USER omr_user WITH PASSWORD 'omr_password123';

-- Grant all privileges to the user
GRANT ALL PRIVILEGES ON DATABASE omr_evaluation_db TO omr_user;

-- Connect to the new database
\c omr_evaluation_db;

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO omr_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO omr_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO omr_user;

-- Display connection info
SELECT 'Database and user created successfully!' AS status;
SELECT 'Database: omr_evaluation_db' AS info;
SELECT 'Username: omr_user' AS info;
SELECT 'Password: omr_password123' AS info;
SELECT 'Connection string: postgresql://omr_user:omr_password123@localhost:5432/omr_evaluation_db' AS connection;