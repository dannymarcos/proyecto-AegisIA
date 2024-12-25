CREATE TABLE IF NOT EXISTS investments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    investment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    daily_percentage DECIMAL(5,2),
    monthly_percentage DECIMAL(5,2),
    total_generated DECIMAL(15,2) DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id)
);