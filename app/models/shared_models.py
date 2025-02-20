#proyecto_aegisia_main/app/models/shared_models.py

import logging
from app.models.create_db import db
# from app.Aplicacion import db 

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class User(db.Model):
    """Represents a user in the database."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True)
    nationality = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"

class Strategy(db.Model):
    """Represents a trading strategy in the database."""
    __tablename__ = "strategies"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<Strategy(id={self.id}, name={self.name})>"

class Investment(db.Model):
    """Represents a user's investment in the database."""
    __tablename__ = "investments"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    amount = db.Column(db.Numeric(15,2), nullable=False)
    investment_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    daily_percentage = db.Column(db.Numeric(5,2))
    monthly_percentage = db.Column(db.Numeric(5,2))
    total_generated = db.Column(db.Numeric(15,2), default=0)

    user = db.relationship('User', backref=db.backref('investments', lazy=True))

    def __repr__(self):
        return f"<Investment(id={self.id}, user_id={self.user_id}, amount={self.amount})>"

class Withdrawal(db.Model):
    """Represents a user's withdrawal in the database."""
    __tablename__ = "withdrawals"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    amount = db.Column(db.Numeric(15,2), nullable=False)
    withdrawal_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    status = db.Column(db.String(20), default='pending')

    user = db.relationship('User', backref=db.backref('withdrawals', lazy=True))

    def __repr__(self):
        return f"<Withdrawal(id={self.id}, user_id={self.user_id}, amount={self.amount}, status={self.status})>"

class ReferralLink(db.Model):
    """Represents a referral link in the database."""
    __tablename__ = "referral_links"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    code = db.Column(db.String(255), nullable=False, unique=True)
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    used_at = db.Column(db.DateTime, nullable=True)
    referred_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    user = db.relationship('User', backref=db.backref('referral_links', lazy=True), foreign_keys=[user_id])
    referrer = db.relationship('User', backref=db.backref('referred_links', lazy=True), foreign_keys=[referred_by])

    def __repr__(self):
        return f"<ReferralLink(id={self.id}, user_id={self.user_id}, code={self.code})>"

# End of models

# Removed Referral model to fix mapper initialization error