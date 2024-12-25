-- Create a new users table with updated schema
CREATE TABLE users_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    nationality VARCHAR(100),
    phone VARCHAR(20),
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Copy data from old table to new one
INSERT INTO users_new (id, full_name, email, nationality, password_hash, created_at, updated_at)
SELECT id, full_name, email, nationality, password_hash, created_at, updated_at FROM users;

-- Drop old table
DROP TABLE users;

-- Rename new table to original name
ALTER TABLE users_new RENAME TO users;