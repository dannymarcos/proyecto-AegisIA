from shared_resources import app

import os
from kraken_api import time
from flask import current_app, session, jsonify, request, Blueprint, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from shared_models import db, User, ReferralLink
from email.mime.text import MIMEText
import urllib.parse
import secrets

from shared_models import Withdrawal, Investment, User
import smtplib  # For sending emails
import telegram
import urllib.parse
import os
from werkzeug.security import generate_password_hash
from kraken_api import KrakenFuturesAPI
from market_data import cancel_order, get_account_balance, execute_kraken_trade
from trading_logic import evaluate_strategy_performance, start_market_monitor, stop_market_monitor
from sqlalchemy.exc import SQLAlchemyError
from shared_models import Strategy, db, logger
from market_data import fetch_historical_data
import logging
import telegram
import requests
import pandas as pd
import json

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load translations
def load_translations():
    try:
        with open('translations.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading translations: {e}")
        return {}

translations = load_translations()

def get_translated_text(key, language='en'):
    """Get translated text based on current language"""
    try:
        return translations.get(language, {}).get(key, translations['en'].get(key, key))
    except Exception as e:
        logger.error(f"Error getting translation for {key}: {e}")
        return key

routes = Blueprint("routes", __name__)

# Removed manual trading route as per request

@routes.route("/")
def home():
    """Home page route"""
    try:
        # Get current language from session or default to English
        current_language = session.get('language', 'en')
        # Get current user from session
        user_email = session.get('user_email')
        if not user_email:
            # For testing purposes only
            user_email = 'test@example.com'
            session['user_email'] = user_email
        
        user = User.query.filter_by(email=user_email).first()
        if not user:
            # Create test user if it doesn't exist
            user = User(
                full_name='Test User',
                email=user_email,
                nationality='Test Country',
                phone='+1234567890',
                password_hash='test_hash'
            )
            db.session.add(user)
            db.session.commit()
            logger.info("Created test user successfully")
        
        # Get user's investments total
        total_invested = db.session.query(db.func.sum(Investment.amount)).filter_by(user_id=user.id).scalar() or 0
        total_generated = db.session.query(db.func.sum(Investment.total_generated)).filter_by(user_id=user.id).scalar() or 0
        
        # Set default test balances
        balances = {
            'usdt': '1000.00',
            'usd': '1000.00',
            'btc': '0.05'
        }
        
        # Try to get real balances from Kraken API
        try:
            kraken_balance = get_account_balance()
            if not kraken_balance.get('error'):
                balances = {
                    'usdt': kraken_balance.get('USDT', '1000.00'),
                    'usd': kraken_balance.get('ZUSD', '1000.00'),
                    'btc': kraken_balance.get('XXBT', '0.05')
                }
        except Exception as e:
            logger.warning(f"Could not get Kraken balances, using test values: {e}")
        
        logger.info(f"Got user balances successfully: {balances}")
        return render_template("home.html", user=user, balances=balances, current_language=current_language)
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        return render_template("home.html", error=str(e))

@routes.route("/finances")
def finances_route():
    """Finances page route"""
    try:
        # Get test user (or current user in production)
        user = User.query.filter_by(email='test@example.com').first()
        if not user:
            return render_template("finances.html", error="User not found")
        
        # Get user's investments
        investments = Investment.query.filter_by(user_id=user.id).all()
        
        # Calculate totals
        total_invested = sum(investment.amount for investment in investments)
        total_generated = sum(investment.total_generated for investment in investments)
        
        # Get latest investment for percentages
        latest_investment = Investment.query.filter_by(user_id=user.id).order_by(Investment.investment_date.desc()).first()
        daily_percentage = latest_investment.daily_percentage if latest_investment else 0
        monthly_percentage = latest_investment.monthly_percentage if latest_investment else 0
        
        # Get withdrawals
        withdrawals = Withdrawal.query.filter_by(user_id=user.id).all()
        
        return render_template("finances.html",
            total_invested=total_invested,
            total_generated=total_generated,
            daily_percentage=daily_percentage,
            monthly_percentage=monthly_percentage,
            investments=investments,
            withdrawals=withdrawals
        )
    except Exception as e:
        logger.error(f"Error in finances route: {e}")
        return render_template("finances.html", error=str(e))

@routes.route("/request_withdrawal", methods=["POST"])
def request_withdrawal():
    """Handle withdrawal requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        amount = float(data.get('amount', 0))
        currency = data.get('currency', 'USDT')
        wallet_address = data.get('wallet_address')
        
        if amount <= 0:
            return jsonify({"status": "error", "message": "Invalid withdrawal amount"}), 400
        
        if not wallet_address:
            return jsonify({"status": "error", "message": "Wallet address is required"}), 400
        
        # Validate wallet address format based on currency
        if currency == 'USDT' and not wallet_address.startswith('T'):
            return jsonify({"status": "error", "message": "Invalid USDT (TRC20) wallet address"}), 400
        elif currency == 'BTC' and not (wallet_address.startswith('1') or wallet_address.startswith('3') or wallet_address.startswith('bc1')):
            return jsonify({"status": "error", "message": "Invalid BTC wallet address"}), 400
        
        # Get test user (or current user in production)
        user = User.query.filter_by(email='test@example.com').first()
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404
            
        # Calculate total available balance (investments + profits)
        total_invested = db.session.query(db.func.sum(Investment.amount)).filter_by(user_id=user.id).scalar() or 0
        total_generated = db.session.query(db.func.sum(Investment.total_generated)).filter_by(user_id=user.id).scalar() or 0
        total_available = float(total_invested) + float(total_generated)
        
        # Check if withdrawal amount is available
        if amount > total_available:
            return jsonify({
                "status": "error",
                "message": f"Insufficient funds. Available balance: {total_available} {currency}"
            }), 400
        
        # Create withdrawal request
        withdrawal = Withdrawal(
            user_id=user.id,
            amount=amount,
            status='pending',
            currency=currency,
            wallet_address=wallet_address
        )
        db.session.add(withdrawal)
        db.session.commit()
        
        # Redirect to withdrawal confirmation page
        return jsonify({
            "status": "success",
            "message": "Balance verified. Please confirm withdrawal details.",
            "redirect": url_for('routes.confirm_withdrawal', 
                                amount=amount,
                                currency=currency,
                                wallet_address=wallet_address)
        })
    except Exception as e:
        logger.error(f"Error processing withdrawal request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@routes.route("/confirm_withdrawal")
def confirm_withdrawal():
    """Withdrawal confirmation page"""
    try:
        amount = request.args.get('amount')
        currency = request.args.get('currency')
        wallet_address = request.args.get('wallet_address')
        
        if not all([amount, currency, wallet_address]):
            return redirect(url_for('routes.finances_route'))
        
        user = User.query.filter_by(email='test@example.com').first()
        if not user:
            return redirect(url_for('routes.finances_route'))
            
        return render_template('confirm_withdrawal.html',
            amount=amount,
            currency=currency,
            wallet_address=wallet_address,
            user=user
        )
    except Exception as e:
        logger.error(f"Error displaying withdrawal confirmation: {e}")
        return redirect(url_for('routes.finances_route'))

# Email functionality for withdrawal requests
def send_withdrawal_email(user, amount, currency, wallet_address):
    """Send an email notification for a withdrawal request."""
    try:
        msg = MIMEText(f"""
        Withdrawal Request Details:
        
        Full Name: {user.full_name}
        Email: {user.email}
        Phone: {user.phone}
        Amount: {amount} {currency}
        Currency: {currency}
        Wallet Address: {wallet_address}
        """)
        
        msg['Subject'] = 'New Withdrawal Request'
        msg['From'] = 'noreply@aegis-ia.com'
        msg['To'] = 'aegisiaapp@gmail.com'
        
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_username = os.environ.get('SMTP_USERNAME')
        smtp_password = os.environ.get('SMTP_PASSWORD')
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            
        logger.info(f"Withdrawal email sent for {user.email} - Amount: {amount} {currency}, Wallet: {wallet_address}")
    except Exception as e:
        logger.error(f"Error sending withdrawal email: {e}")

# Initialize Telegram bot
bot = telegram.Bot(token='')
TELEGRAM_BOT_USERNAME = 'Mysoportebot'
TELEGRAM_BOT_LINK = f"https://t.me/{TELEGRAM_BOT_USERNAME}?start=welcome"
TELEGRAM_START_COMMAND = '/start'
TELEGRAM_WELCOME_MESSAGE = (
    "🛡️ ¡Bienvenido al Soporte de Aegis-IA!\n\n"
    "Por favor, selecciona una opción:\n\n"
    "1️⃣ Problemas técnicos\n"
    "2️⃣ Problemas con inversiones\n" 
    "3️⃣ Problemas con retiros\n"
    "4️⃣ Otros problemas\n\n"
    "O escribe tu consulta directamente y nuestro equipo te ayudará lo antes posible.\n\n"
    "🔒 Seguridad & Desarrollo\n"
    "⏰ Tiempo de respuesta: 24 horas"
)

@routes.route("/submit_support_request", methods=["POST"])
def submit_support_request():
    """Handle support request submission"""
    try:
        data = request.get_json()
        user = User.query.filter_by(email='test@example.com').first()
        
        if not user:
            return jsonify({"status": "error", "message": "User not logged in"}), 401
        
        # Format support ticket message with user info
        ticket_message = (
            f"{TELEGRAM_WELCOME_MESSAGE}\n\n"
            f"Información del usuario:\n"
            f"Nombre: {user.full_name}\n"
            f"Tipo de problema: {data.get('issue_type', 'No especificado')}\n"
            f"Descripción: {data.get('description', 'Sin descripción')}\n\n"
            f"🔒 Seguridad & Desarrollo\n"
            f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Format support ticket message with Aegis-IA logo and shield
        ticket_message = (
            f"🛡️ AEGIS-IA SUPPORT SYSTEM\n\n"
            f"Welcome to Aegis-IA Support!\n"
            f"Your security and development partner.\n\n"
            f"Please use the following commands:\n"
            f"/name - Enter your name\n"
            f"/issue - Describe your issue\n"
            f"/status - Check ticket status\n\n"
            f"Pre-filled information:\n"
            f"Name: {user.full_name}\n"
            f"Issue Type: {data.get('issue_type', 'Not specified')}\n"
            f"Description: {data.get('description', 'No description provided')}\n\n"
            f"🔒 Security & Development\n"
            f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Return direct bot link
        return jsonify({
            "status": "success",
            "message": "Support ticket submitted successfully",
            "telegram_link": TELEGRAM_BOT_LINK,
            "open_in_new_window": True
        })
    except Exception as e:
        logger.error(f"Error submitting support request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
        
@routes.route("/settings/password", methods=["GET", "POST"])
def settings_password_route():
    """Settings password page route"""
    try:
        if request.method == "POST":
            data = request.get_json()
            user = User.query.filter_by(email='test@example.com').first()
            
            if not user:
                return jsonify({"status": "error", "message": "User not found"}), 404
                
            current_password = data.get('current_password')
            new_password = data.get('new_password')
            
            if not current_password or not new_password:
                return jsonify({"status": "error", "message": "Missing password data"}), 400
                
            if user.password_hash != current_password:  # In production, use proper password verification
                return jsonify({"status": "error", "message": "Current password is incorrect"}), 400
                
            user.password_hash = generate_password_hash(new_password)
            db.session.commit()
            
            return jsonify({
                "status": "success",
                "message": "Password updated successfully"
            })
            
        return render_template("settings/password.html")
        
    except Exception as e:
        logger.error(f"Error in password settings: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@routes.route("/profile", methods=["GET", "POST"])
def profile_route():
    """Profile page route"""
    try:
        # Get or create test user
        user = User.query.filter_by(email='test@example.com').first()
        if not user:
            user = User(
                full_name='Test User',
                email='test@example.com',
                nationality='Test Country',
                phone='+1234567890',
                password_hash='test_hash'
            )
            db.session.add(user)
            db.session.commit()
            logger.info("Created test user successfully")

        if request.method == "POST":
            try:
                data = request.form
                # Update user information
                user.full_name = data.get('full_name', user.full_name)
                user.nationality = data.get('nationality', user.nationality)
                user.phone = data.get('phone', user.phone)
                
                # Handle email change
                new_email = data.get('email')
                if new_email and new_email != user.email:
                    # Check if email is already taken
                    if User.query.filter_by(email=new_email).first():
                        return jsonify({"status": "error", "message": "Email already in use"}), 400
                    user.email = new_email
                
                # Handle password change
                current_password = data.get('current_password')
                new_password = data.get('new_password')
                if current_password and new_password:
                    if user.password_hash == current_password:  # In production, use proper password verification
                        user.password_hash = generate_password_hash(new_password)
                    else:
                        return jsonify({"status": "error", "message": "Current password is incorrect"}), 400
                
                db.session.commit()
                return jsonify({
                    "status": "success",
                    "message": "Profile updated successfully",
                    "user": {
                        "full_name": user.full_name,
                        "email": user.email,
                        "nationality": user.nationality,
                        "phone": user.phone
                    }
                })
            except Exception as e:
                logger.error(f"Error updating profile: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500
        
        return render_template("profile.html", user=user)
        
    except Exception as e:
        logger.error(f"Error in profile route: {e}")
        return render_template("profile.html", error="Error loading profile")

@routes.route("/update_profile", methods=["POST"])
def update_profile():
    """Update user profile"""
    try:
        data = request.get_json()
        # TODO: Update user data in database
        # For now, just return success
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return jsonify({"error": str(e)}), 500

@routes.route("/settings")
def settings_route():
    """Settings page route"""
    return render_template("under_construction.html", page_name="Settings")

@routes.route("/classes")
def classes_route():
    """Classes page route"""
    try:
        # Get test user (admin)
        user = User.query.filter_by(email='test@example.com').first()
        is_admin = bool(user)  # For now, test user is admin
        return render_template("classes.html", is_admin=is_admin)
    except Exception as e:
        logger.error(f"Error in classes route: {e}")
        return render_template("classes.html", error="Error loading classes")

@routes.route("/settings/2fa")
def settings_2fa_route():
    """Settings 2FA page route"""
    return render_template("under_construction.html", page_name="Enable 2FA")

@routes.route("/change_language", methods=["POST"])
def change_language():
    """Change the application language."""
    try:
        data = request.get_json()
        language = data.get('language')
        confirm = data.get('confirm', False)
        
        if not confirm:
            # Return confirmation message in the target language
            confirmation_messages = {
                'en': 'Do you want to change the language to English?',
                'es': '¿Desea cambiar el idioma a Español?',
                'pt': 'Deseja mudar o idioma para Português?',
                'fr': 'Voulez-vous changer la langue en Français?'
            }
            return jsonify({
                "status": "confirm",
                "message": confirmation_messages.get(language, confirmation_messages['en'])
            })
        
        if language not in ['es', 'en', 'pt', 'fr']:
            return jsonify({"error": "Invalid language"}), 400
            
        session['language'] = language
        
        # Load translations for the selected language
        current_translations = translations.get(language, {})
        
        # Update all text elements with translations
        return jsonify({
            "status": "success",
            "translations": current_translations,
            "message": f"Language changed to {language}"
        })
    except Exception as e:
        logger.error(f"Error changing language: {e}")
        return jsonify({"error": str(e)}), 500

@routes.route("/stop_ai_trading", methods=["POST"])
def stop_ai_trading():
    """Stop AI trading"""
    try:
        stop_market_monitor()
        return jsonify({"status": "success", "message": "AI trading stopped"})
    except Exception as e:
        logger.error(f"Error stopping AI trading: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@routes.route("/update_trading_mode", methods=["POST"])
def update_trading_mode():
    """Update trading mode"""
    try:
        data = request.get_json()
        mode = data.get("mode")
        if mode not in ["spot", "futures"]:
            return jsonify({"status": "error", "message": "Invalid trading mode"}), 400
        current_app.config["TRADING_MODE"] = mode
        return jsonify({"status": "success", "mode": mode})
    except Exception as e:
        logger.error(f"Error updating trading mode: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@routes.route("/get_balance")
def get_balance():
    """Get account balance"""
    try:
        trading_mode = current_app.config.get("TRADING_MODE", "spot")
        if trading_mode == "futures":
            futures_client = KrakenFuturesAPI()
            accounts = futures_client.get_accounts()
            if accounts.get("error"):
                return jsonify({"error": accounts["error"]}), 400
            return jsonify({"futures_balance": accounts.get("accounts", [{}])[0]})
        else:
            balance = get_account_balance()
            if balance.get("error"):
                return jsonify({"error": balance["error"]}), 400
            return jsonify({"spot_balance": balance})
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        return jsonify({"error": str(e)}), 500

@routes.route("/get_cryptos")
def get_cryptos():
    """Get available cryptocurrencies"""
    try:
        trading_mode = current_app.config.get("TRADING_MODE", "spot")
        if trading_mode == "futures":
            futures_client = KrakenFuturesAPI()
            tickers = futures_client.get_tickers()
            if tickers.get("error"):
                return jsonify({"error": tickers["error"]}), 400
            cryptos = [{"symbol": t["symbol"], "price": t["last"]} for t in tickers.get("tickers", [])]
        else:
            # For spot trading, get pairs from Kraken API
            try:
                response = requests.get("https://api.kraken.com/0/public/Ticker")
                if response.status_code != 200:
                    logger.error(f"Error fetching tickers: HTTP {response.status_code}")
                    return jsonify({"error": "Failed to fetch tickers"}), 500

                data = response.json()
                if "error" in data and data["error"]:
                    logger.error(f"Kraken API error: {data['error']}")
                    return jsonify({"error": str(data["error"])}), 500

                result = data.get("result", {})
                cryptos = []
                for pair_name, ticker_info in result.items():
                    if "c" in ticker_info:  # "c" contains the last trade closed price
                        price = ticker_info["c"][0]  # First element is the price
                        cryptos.append({
                            "symbol": pair_name,
                            "price": price
                        })

                if not cryptos:
                    logger.error("No crypto data found in response")
                    return jsonify({"error": "No crypto data found"}), 500

                logger.info(f"Successfully fetched {len(cryptos)} cryptocurrencies")
                return jsonify({"cryptos": cryptos})

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error fetching tickers: {e}")
                return jsonify({"error": f"Network error: {str(e)}"}), 500

        return jsonify({"cryptos": cryptos})
    except Exception as e:
        logger.error(f"Error getting cryptocurrencies: {e}")
        return jsonify({"error": str(e)}), 500

@routes.route("/upload_class", methods=["POST"])
def upload_class():
    """Handle class video upload"""
    try:
        # Check if user is admin
        user = User.query.filter_by(email='test@example.com').first()
        if not user:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401

        title = request.form.get('title')
        description = request.form.get('description')
        video = request.files.get('video')

        if not all([title, description, video]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        # Save video file
        filename = secure_filename(video.filename)
        video_path = os.path.join('static', 'uploads', filename)
        video.save(video_path)

        # Save to database
        sql = """
            INSERT INTO teleclasses (title, description, video_url)
            VALUES (?, ?, ?)
        """
        db.session.execute(sql, [title, description, video_path])
        db.session.commit()

        return jsonify({"status": "success", "message": "Class uploaded successfully"})
    except Exception as e:
        logger.error(f"Error uploading class: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@routes.route("/generate_access_link", methods=["POST"])
def generate_access_link():
    """Generate access link for teleclasses"""
    try:
        data = request.form
        description = data.get('description')
        expiry = data.get('expiry')
        
        if not description:
            return jsonify({"status": "error", "message": "Description is required"}), 400
            
        # Generate unique access code
        access_code = secrets.token_urlsafe(16)
        
        # Save to database
        sql = """
            INSERT INTO teleclass_access (access_code, description, expiry_date)
            VALUES (?, ?, ?)
        """
        db.session.execute(sql, [access_code, description, expiry])
        db.session.commit()
        
        return jsonify({
            "status": "success",
            "access_link": f"https://aegis-ia.com/teleclasses/{access_code}"
        })
    except Exception as e:
        logger.error(f"Error generating access link: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@routes.route("/generate_referral_link", methods=["GET"])
def generate_referral_link():
    """Generate referral link for current user and get referral tree"""
    try:
        # Get current user
        user = User.query.filter_by(email='test@example.com').first()
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404

        # Check if user already has a referral link
        existing_link = ReferralLink.query.filter_by(user_id=user.id, active=True).first()
        if existing_link:
            referral_code = existing_link.code
        else:
            # Generate new referral link
            total_referrals = ReferralLink.query.filter_by(active=False).count()
            referral_number = str(total_referrals + 1).zfill(2)
            referral_code = f"linkreferidosaegiaiaapp{referral_number}"

            # Save to database
            new_link = ReferralLink(
                user_id=user.id,
                code=referral_code,
                active=True
            )
            db.session.add(new_link)
            db.session.commit()

        # Get direct referrals
        direct_referrals = []
        direct_refs = ReferralLink.query.filter_by(referred_by=user.id, active=False).all()
        for ref in direct_refs:
            ref_user = User.query.get(ref.user_id)
            if ref_user:
                direct_referrals.append({
                    "name": ref_user.full_name,
                    "date": ref.used_at.strftime("%Y-%m-%d") if ref.used_at else "N/A"
                })

        # Get indirect referrals (referrals of referrals)
        indirect_referrals = []
        for direct_ref in direct_refs:
            indirect_refs = ReferralLink.query.filter_by(referred_by=direct_ref.user_id, active=False).all()
            for ref in indirect_refs:
                ref_user = User.query.get(ref.user_id)
                referrer = User.query.get(direct_ref.user_id)
                if ref_user and referrer:
                    indirect_referrals.append({
                        "name": ref_user.full_name,
                        "date": ref.used_at.strftime("%Y-%m-%d") if ref.used_at else "N/A",
                        "referred_by": referrer.full_name
                    })

        # Calculate total referrals
        total_referrals = len(direct_referrals) + len(indirect_referrals)

        return jsonify({
            "status": "success",
            "referral_link": referral_code,
            "direct_referrals": direct_referrals,
            "indirect_referrals": indirect_referrals,
            "total_referrals": total_referrals
        })
    except Exception as e:
        logger.error(f"Error generating referral link: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@routes.route("/request_teleclass_access", methods=["POST"])
def request_teleclass_access():
    """Handle teleclass access requests"""
    try:
        data = request.get_json()
        if not data or not data.get('reason'):
            return jsonify({"status": "error", "message": "Please provide a reason for your request"}), 400

        # Get current user
        user = User.query.filter_by(email='test@example.com').first()
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404

        # Save access request to database
        sql = """
            INSERT INTO teleclass_requests (user_id, reason, status)
            VALUES (?, ?, 'pending')
        """
        db.session.execute(sql, [user.id, data['reason']])
        db.session.commit()

        # Send notification to admin (you can implement email notification here)
        logger.info(f"New teleclass access request from {user.email}: {data['reason']}")

        return jsonify({
            "status": "success",
            "message": "Your request has been submitted successfully"
        })
    except Exception as e:
        logger.error(f"Error submitting teleclass access request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@routes.route("/settings/wallet", methods=["GET", "POST"])
def settings_wallet_route():
    """Settings wallet page route"""
    try:
        if request.method == "POST":
            data = request.form
            user = User.query.filter_by(email='test@example.com').first()
            if not user:
                return jsonify({"status": "error", "message": "User not found"}), 404
                
            # Verify current password
            if not data.get('current_password'):
                return jsonify({"status": "error", "message": "Current password is required"}), 400
                
            if user.password_hash != data.get('current_password'):  # In production, use proper password verification
                return jsonify({"status": "error", "message": "Current password is incorrect"}), 400
                
            # Update wallet address
            user.wallet_address = data.get('wallet_address')
            db.session.commit()
            
            return jsonify({
                "status": "success",
                "message": "Wallet address updated successfully"
            })
            
        # GET request - render wallet settings page
        user = User.query.filter_by(email='test@example.com').first()
        return render_template("settings/wallet.html", user=user)
        
    except Exception as e:
        logger.error(f"Error in wallet settings: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@routes.route("/get_current_symbol")
def get_current_symbol():
    """Get current trading symbol"""
    try:
        trading_mode = current_app.config.get("TRADING_MODE", "spot")
        symbol = "XBTUSD"  # Use spot symbol for now
        return jsonify({"symbol": symbol})
    except Exception as e:
        logger.error(f"Error getting current symbol: {e}")
        return jsonify({"error": str(e)}), 500
