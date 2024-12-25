-- Create table for teleclasses
CREATE TABLE IF NOT EXISTS teleclasses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    video_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for access links
CREATE TABLE IF NOT EXISTS teleclass_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    access_code VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    expiry_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);