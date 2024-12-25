-- Add referred_by column to referral_links table
CREATE TABLE referral_links_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    code VARCHAR(255) NOT NULL UNIQUE,
    active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    used_at TIMESTAMP,
    referred_by INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (referred_by) REFERENCES users(id)
);

-- Copy data from old table to new one
INSERT INTO referral_links_new (id, user_id, code, active, created_at, used_at)
SELECT id, user_id, code, active, created_at, used_at FROM referral_links;

-- Drop old table
DROP TABLE referral_links;

-- Rename new table to original name
ALTER TABLE referral_links_new RENAME TO referral_links;