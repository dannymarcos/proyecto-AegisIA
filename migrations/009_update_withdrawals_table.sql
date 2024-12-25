-- Create a new withdrawals table with updated schema
CREATE TABLE withdrawals_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USDT',
    wallet_address TEXT,
    withdrawal_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Copy data from old table to new one
INSERT INTO withdrawals_new (id, user_id, amount, withdrawal_date, status)
SELECT id, user_id, amount, withdrawal_date, status FROM withdrawals;

-- Drop old table
DROP TABLE withdrawals;

-- Rename new table to original name
ALTER TABLE withdrawals_new RENAME TO withdrawals;