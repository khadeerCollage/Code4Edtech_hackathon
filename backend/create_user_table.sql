-- Create user table (singular) as expected by the backend
-- This is in addition to the users table (plural) from the complete schema

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

-- Insert a test user for initial testing
INSERT INTO "user" (username, email, password_hash, created_at, is_active)
VALUES ('testuser', 'test@example.com', '$2b$12$6k8VR8Mg/ByTd5/LTWFe4.d3/3Ot.bvIBq7gH9wlJGrVN1bfQMQ9y', CURRENT_TIMESTAMP, true)
ON CONFLICT (username) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE "user" TO omr_user;
GRANT ALL PRIVILEGES ON SEQUENCE user_id_seq TO omr_user;

-- Verify table creation
SELECT 'User table created successfully!' as status;
SELECT COUNT(*) as user_count FROM "user";