import os
import secrets
import time
import flask
from flask import Flask, render_template, request, redirect, url_for, flash, current_app, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from functools import wraps
import urllib.parse
import subprocess
import requests
from requests.auth import HTTPBasicAuth
import logging
from urllib.parse import urlencode
import shlex
from datetime import datetime, timedelta
from flask import jsonify
import random
import string
from flask import jsonify, request, session, redirect, url_for, flash
from flask_login import login_required, current_user
from werkzeug.security import check_password_hash
from functools import wraps
import random
import string
from flask_wtf.csrf import CSRFProtect
from sqlalchemy import func
import uuid
from PIL import Image
from flask import send_from_directory
import importlib
import werkzeug
importlib.reload(werkzeug)
from werkzeug.utils import secure_filename
import stripe
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import json
import shutil
from jinja2 import Undefined
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from functools import wraps
from datetime import datetime
import json
import os
import secrets
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import openai
import anthropic
import json
from custom_tools import CustomTools
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')


print("testa")


temp_tokens = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
ADMIN_USER = "gad8g8hbnawdhx"
ADMIN_PASS = "82q93fdfrdg"
CONTACTS_FILE = 'contacts.json'
SESSION_TOKEN = secrets.token_hex(16)  # Generate secure session token

# Initialize clients and tools
tools = CustomTools()

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)


def format_messages_for_claude(history, system_prompt):
    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    for msg in history:
        formatted_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return formatted_messages

def stream_claude_response(message, history, system_prompt):
    messages = format_messages_for_claude(history, system_prompt)
    
    try:
        # Create a messages stream with Claude
        with claude_client.messages.stream(
            model="claude-3-sonnet-20240229",
            messages=messages,
            max_tokens=4096
        ) as stream:
            # Stream the response text
            for text in stream.text_stream:
                yield f"data: {json.dumps({'delta': text})}\n\n"
                
    except Exception as e:
        error_msg = str(e)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

def stream_openai_response(model, message, history, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    
    try:
        stream = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=4096
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield f"data: {json.dumps({'delta': chunk.choices[0].delta.content})}\n\n"
    
    except Exception as e:
        error_msg = str(e)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
        
@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(tools.tools)

@app.route('/tools', methods=['POST'])
def add_tool():
    tool_data = request.json
    tool_id = tool_data.pop('id', None)
    if not tool_id:
        return jsonify({'error': 'Tool ID required'}), 400
    
    tools.tools[tool_id] = tool_data
    tools.save_tools()
    return jsonify({'success': True})

@app.route('/tools/<tool_id>', methods=['DELETE'])
def delete_tool(tool_id):
    if tool_id in tools.tools:
        del tools.tools[tool_id]
        tools.save_tools()
        return jsonify({'success': True})
    return jsonify({'error': 'Tool not found'}), 404

@app.route('/tools/<tool_id>', methods=['PUT'])
def update_tool(tool_id):
    if tool_id not in tools.tools:
        return jsonify({'error': 'Tool not found'}), 404
    
    tool_data = request.json
    tools.tools[tool_id].update(tool_data)
    tools.save_tools()
    return jsonify({'success': True})

@app.route('/custom-instructions', methods=['GET', 'POST'])
def handle_custom_instructions():
    instruction_file = 'custom_instructions.json'
    if request.method == 'GET':
        try:
            with open(instruction_file, 'r') as f:
                return jsonify(json.load(f))
        except FileNotFoundError:
            return jsonify({'instructions': ''})
    else:
        instructions = request.json.get('instructions', '')
        with open(instruction_file, 'w') as f:
            json.dump({'instructions': instructions}, f)
        return jsonify({'success': True})

def process_message(message):
    # Check if message starts with a tool command
    first_word = message.split()[0] if message else ''
    if first_word in [tool['command'] for tool in tools.tools.values()]:
        command = first_word
        query = message[len(command):].strip()
        return tools.execute_tool(command, query)
    return message

# Update the existing chat endpoint to handle tools
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        model = data['model']
        message = data['message']
        chat_history = data.get('history', [])
        system_prompt = data.get('systemPrompt', '')
        custom_instructions = data.get('customInstructions', '')

        # Process message for tools
        processed_message = process_message(message)
        if processed_message != message:
            # If the message was processed by a tool, return the result directly
            def generate():
                yield f"data: {json.dumps({'delta': processed_message})}\n\n"
            return Response(stream_with_context(generate()), content_type='text/event-stream')

        # Combine system prompt with custom instructions
        combined_prompt = f"{system_prompt}\n\nCustom Instructions:\n{custom_instructions}" if custom_instructions else system_prompt

        if 'claude' in model.lower():
            return Response(
                stream_with_context(stream_claude_response(message, chat_history, combined_prompt)),
                content_type='text/event-stream'
            )
        else:
            return Response(
                stream_with_context(stream_openai_response(model, message, chat_history, combined_prompt)),
                content_type='text/event-stream'
            )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/zgbai')
def indesdfsdfsdfx():
    return render_template('ai.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Handle text files
        if file.filename.lower().endswith(('.txt', '.md', '.py', '.json')):
            content = file.read().decode('utf-8')
            return jsonify({
                'success': True,
                'type': 'text',
                'data': content
            })
        
        # Handle images - convert to base64
        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            content = file.read()
            import base64
            img_base64 = base64.b64encode(content).decode('utf-8')
            mime_type = file.content_type or 'image/jpeg'
            return jsonify({
                'success': True,
                'type': 'image',
                'data': f"data:{mime_type};base64,{img_base64}"
            })
        
        return jsonify({'error': 'Unsupported file type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ensure required directories exist
if not os.path.exists('static'):
    os.makedirs('static')

ADMIN_USER = "gad8g8hbnawdhx"
ADMIN_PASS = "82q93fdfrdg"
CONTACTS_FILE = 'contacts.json'
SESSION_TOKEN = secrets.token_hex(16)  # Generate secure session token

# Ensure required directories exist
if not os.path.exists('static'):
    os.makedirs('static')

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                'Access Denied', 401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'}
            )
        return f(*args, **kwargs)
    return decorated

def check_auth(username, password):
    return username == ADMIN_USER and password == ADMIN_PASS

def save_contact(data):
    """Save contact form submission with enhanced data"""
    contacts = []
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, 'r', encoding='utf-8') as f:
            try:
                contacts = json.load(f)
            except json.JSONDecodeError:
                contacts = []
    
    # Enhanced submission data
    submission = {
        'id': secrets.token_hex(8),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'name': data.get('name'),
        'lastName': data.get('lastName'),
        'email': data.get('email'),
        'phone': data.get('phone'),
        'restaurant': data.get('restaurant'),
        'message': data.get('message'),
        'ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent'),
        'referrer': request.referrer,
        'status': 'new'  # For tracking in admin panel
    }
    
    contacts.append(submission)
    
    with open(CONTACTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(contacts, f, indent=4, ensure_ascii=False)
    
    return submission['id']

def get_analytics():
    """Calculate analytics for admin dashboard"""
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, 'r', encoding='utf-8') as f:
            try:
                contacts = json.load(f)
                total = len(contacts)
                new = sum(1 for c in contacts if c.get('status') == 'new')
                today = sum(1 for c in contacts if c.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d')))
                return {
                    'total_submissions': total,
                    'new_submissions': new,
                    'submissions_today': today
                }
            except json.JSONDecodeError:
                pass
    return {
        'total_submissions': 0,
        'new_submissions': 0,
        'submissions_today': 0
    }

@app.route('/')
def home():
    return render_template('mrk.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = {
            'name': request.form.get('name', '').strip(),
            'lastName': request.form.get('lastName', '').strip(),
            'email': request.form.get('email', '').strip(),
            'phone': request.form.get('phone', '').strip(),
            'restaurant': request.form.get('restaurant', '').strip(),
            'message': request.form.get('message', '').strip()
        }
        
        # Validate inputs
        if not all([data['name'], data['lastName'], data['email'], data['phone'], data['restaurant'], data['message']]):
            return jsonify({
                'success': False,
                'message': 'Please fill in all fields'
            }), 400
        
        if '@' not in data['email'] or '.' not in data['email']:
            return jsonify({
                'success': False,
                'message': 'Please provide a valid email address'
            }), 400
        
        # Save submission
        submission_id = save_contact(data)
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your message. We will get back to you soon!',
            'id': submission_id
        })
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An unexpected error occurred. Please try again later.'
        }), 500

@app.route('/newsletter', methods=['POST'])
def newsletter():
    try:
        email = request.form.get('email', '').strip()
        
        # Validate email
        if not email:
            return jsonify({
                'success': False,
                'message': 'Please provide an email address'
            }), 400
        
        if '@' not in email or '.' not in email:
            return jsonify({
                'success': False,
                'message': 'Please provide a valid email address'
            }), 400
        
        # TODO: Add newsletter subscription logic here
        
        return jsonify({
            'success': True,
            'message': 'Thank you for subscribing to our newsletter!'
        })
        
    except Exception as e:
        print(f"Error in newsletter: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred. Please try again later.'
        }), 500

@app.route('/admin')
@require_auth
def admin():
    """Enhanced admin dashboard"""
    analytics = get_analytics()
    
    # Load contacts with pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    contacts = []
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, 'r', encoding='utf-8') as f:
            try:
                contacts = json.load(f)
            except json.JSONDecodeError:
                contacts = []
    
    # Sort contacts by timestamp (newest first)
    contacts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Paginate contacts
    total_pages = (len(contacts) + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    contacts_page = contacts[start_idx:start_idx + per_page]
    
    return render_template(
        'admin.html',
        contacts=contacts_page,
        analytics=analytics,
        page=page,
        total_pages=total_pages,
        per_page=per_page
    )

@app.route('/admin/mark-read/<submission_id>')
@require_auth  
def mark_read(submission_id):
    """Mark submission as read"""
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, 'r', encoding='utf-8') as f:
            contacts = json.load(f)
            
        for contact in contacts:
            if contact.get('id') == submission_id:
                contact['status'] = 'read'
                
        with open(CONTACTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(contacts, f, indent=4)
    
    return redirect(url_for('admin'))

@app.route('/admin/delete/<submission_id>')
@require_auth
def delete_submission(submission_id):
    """Delete submission"""
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, 'r', encoding='utf-8') as f:
            contacts = json.load(f)
            
        contacts = [c for c in contacts if c.get('id') != submission_id]
                
        with open(CONTACTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(contacts, f, indent=4)
    
    return redirect(url_for('admin'))


db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Configure Stripe
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
endpoint_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')

MAILGUN_API_KEY = os.environ.get('MAILGUN_API_KEY')
MAILGUN_DOMAIN = os.environ.get('MAILGUN_DOMAIN')
MAILGUN_BASE_URL = 'https://api.mailgun.net/v3'

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        return 'Invalid signature', 400

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']

        # Retrieve customer email
        customer = stripe.Customer.retrieve(session['customer'])
        customer_email = customer['email']

        # Send thank you email
        send_thank_you_email(customer_email)

    return jsonify(success=True)

def send_thank_you_email(to_email):
    return requests.post(
        f"{MAILGUN_BASE_URL}/{MAILGUN_DOMAIN}/messages",
        auth=("api", MAILGUN_API_KEY),
        data={"from": f"Expantion <expantion@{MAILGUN_DOMAIN}>",
              "to": [to_email],
              "subject": "Thank you for your purchase!",
              "text": "We appreciate your business and hope you enjoy your product.",
              "html": "<strong>We appreciate your business and hope you enjoy your product.</strong>"})
    
@app.after_request
def add_csp_headers(response):
    csp = "default-src *; " \
          "script-src * 'unsafe-inline'; " \
          "style-src * 'unsafe-inline'; " \
          "font-src *; " \
          "img-src *; " \
          "connect-src *;"
    
    response.headers['Content-Security-Policy'] = csp
    return response

app.after_request(add_csp_headers)

@login_manager.unauthorized_handler
def unauthorized():
    flash('You must be logged in to view this page.', 'warning')
    return redirect(url_for('home'))

class Payment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    affiliate_id = db.Column(db.Integer, db.ForeignKey('affiliate.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    affiliate = db.relationship('Affiliate', backref=db.backref('payments', lazy=True))

class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    affiliate_id = db.Column(db.Integer, db.ForeignKey('affiliate.id'), nullable=False)
    plan_type = db.Column(db.String(20), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    affiliate = db.relationship('Affiliate', backref=db.backref('sales', lazy=True))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    is_system_user = db.Column(db.Boolean, default=False)
    is_affiliate = db.Column(db.Boolean, default=False)  # New field
    miab_url = db.Column(db.String(200))
    miab_email = db.Column(db.String(100))
    miab_password = db.Column(db.String(100))
    plan = db.Column(db.String(20))
    user_limit = db.Column(db.Integer)
    tickets = db.relationship('Ticket', backref='user', lazy=True)

class Ticket(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='Open')
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    messages = db.relationship('Message', backref='ticket', lazy=True)
    temp_password_hash = db.Column(db.String(100))
    referral = db.Column(db.String(20), nullable=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    is_admin = db.Column(db.Boolean, default=False)
    ticket_id = db.Column(db.Integer, db.ForeignKey('ticket.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    temp_user = db.Column(db.String(100), nullable=True)
    image_filename = db.Column(db.String(255), nullable=True)

class Affiliate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    referral_code = db.Column(db.String(20), unique=True, nullable=False)
    user_count = db.Column(db.Integer, default=0)
    clicks = db.Column(db.Integer, default=0)
    total_time_on_page = db.Column(db.Integer, default=0)  # in seconds



@app.before_request
def track_referral():
    if 'referral' not in session and request.args.get('referral'):
        session['referral'] = request.args.get('referral')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_affiliate(referral_code):
    return Affiliate.query.filter_by(referral_code=referral_code).first()

def get_affiliate_tickets(user_id):
    return Ticket.query.filter_by(user_id=user_id).order_by(Ticket.created_at.desc()).limit(5).all()

def get_referral_stats(affiliate_id):
    try:
        affiliate = Affiliate.query.get(affiliate_id)
        
        if affiliate is None:
            app.logger.error(f"Affiliate with ID {affiliate_id} not found")
            return None

        now = datetime.utcnow()
        first_day_of_current_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        first_day_of_last_month = (first_day_of_current_month - timedelta(days=1)).replace(day=1)

        total = Ticket.query.filter_by(referral=affiliate.referral_code).count()
        this_month = Ticket.query.filter(Ticket.referral == affiliate.referral_code, 
                                         Ticket.created_at >= first_day_of_current_month).count()
        last_month = Ticket.query.filter(Ticket.referral == affiliate.referral_code,
                                         Ticket.created_at >= first_day_of_last_month,
                                         Ticket.created_at < first_day_of_current_month).count()

        total_earnings = db.session.query(func.sum(Sale.amount)).filter_by(affiliate_id=affiliate_id).scalar() or 0
        total_paid = db.session.query(func.sum(Payment.amount)).filter_by(affiliate_id=affiliate_id).scalar() or 0
        
        recent_payments = Payment.query.filter_by(affiliate_id=affiliate_id).order_by(Payment.created_at.desc()).limit(5).all()

        return {
            'total': total,
            'this_month': this_month,
            'last_month': last_month,
            'user_count': affiliate.user_count,
            'clicks': affiliate.clicks,
            'total_time_on_page': affiliate.total_time_on_page,
            'total_earnings': total_earnings,
            'total_paid': total_paid,
            'balance_due': total_earnings - total_paid,
            'recent_payments': [
                {
                    'amount': payment.amount,
                    'created_at': payment.created_at.strftime('%Y-%m-%d')
                } for payment in recent_payments
            ]
        }
    except Exception as e:
        app.logger.error(f"Error in get_referral_stats: {str(e)}")
        return None
        
@app.teardown_request
def update_session_time(exception=None):
    if 'referral' in session and not current_user.is_authenticated and 'session_start_time' in session:
        current_time = time.time()
        session_duration = current_time - session['session_start_time']
        
        # Update the total time for the affiliate
        referral_code = session['referral']
        affiliate = Affiliate.query.filter_by(referral_code=referral_code).first()
        if affiliate:
            affiliate.total_time_on_page += int(session_duration)
            db.session.commit()
        
        # Reset the session start time
        session['session_start_time'] = current_time


@app.before_request
def handle_trailing_slash_and_query():
    rurl = request.url
    parsed_url = urlparse(rurl)
    path = parsed_url.path
    query = parse_qs(parsed_url.query)
    
    if path.endswith('/') and query:
        # Remove the trailing slash
        path = path.rstrip('/')
        # Reconstruct the URL
        new_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            path,
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment
        ))
        return redirect(new_url, code=301)
    
    # If there's no query string but there's a trailing slash, let Flask handle it
    return None

@app.before_request
def track_affiliate():
    # Track clicks
    if 'referral' in request.args and 'referral' not in session:
        referral_code = request.args['referral']
        affiliate = Affiliate.query.filter_by(referral_code=referral_code).first()
        if affiliate:
            affiliate.clicks += 1
            db.session.commit()
        session['referral'] = referral_code
        session['session_start_time'] = time.time()

    # Track time for unauthenticated sessions
    if 'referral' in session and not current_user.is_authenticated:
        if 'session_start_time' in session:
            current_time = time.time()
            session_duration = current_time - session['session_start_time']
            
            # Update the total time for the affiliate
            referral_code = session['referral']
            affiliate = Affiliate.query.filter_by(referral_code=referral_code).first()
            if affiliate:
                affiliate.total_time_on_page += int(session_duration)
                db.session.commit()
            
            # Reset the session start time
            session['session_start_time'] = current_time
            
# New route to handle file uploads
@app.route('/thanks')
def thanks():
    return render_template('thanks.html')


with app.app_context():
    db.create_all()
    if not User.query.filter_by(email='uNQLfd@caPCpL.com').first():
        admin_user = User(email='uNQLfd@caPCpL.com', 
                          password=generate_password_hash('XofwDnJUxXFB'),
                          is_system_user=True)
        db.session.add(admin_user)
        db.session.commit()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def temp_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'temp_user' not in session and not current_user.is_authenticated:
            flash('You need to log in with your temporary credentials to view this ticket.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function


def validate_temp_credentials(username, password):
    # Extract ticket ID from the username
    try:
        ticket_id = int(username.split('_')[1])
    except (IndexError, ValueError):
        return False

    # Query the ticket using the ticket ID
    ticket = Ticket.query.get(ticket_id)

    if not ticket:
        return False

    # Compare hashed password
    return check_password_hash(ticket.temp_password_hash, password)


def system_user_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_system_user:
            flash('You need to be a system user to access this page.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function
    
@app.before_request
def track_referral():
    if 'referral' not in session and request.args.get('referral'):
        session['referral'] = request.args.get('referral')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def get_miab_users(user):
    try:
        response = requests.get(
            f"{user.miab_url}/admin/mail/users?format=json",
            auth=(user.miab_email, user.miab_password)
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        app.logger.error(f"Error fetching MIAB users: {str(e)}")
        return []

def get_miab_usage_data(user):
    try:
        response = requests.get(
            f"{user.miab_url}/admin/system/status",
            auth=(user.miab_email, user.miab_password)
        )
        response.raise_for_status()
        status_data = response.json()

        # Extract storage and bandwidth data
        storage_used = status_data['disk']['used'] / (1024 ** 3)  # Convert to GB
        storage_limit = status_data['disk']['total'] / (1024 ** 3)  # Convert to GB
        bandwidth_used = status_data['network']['sent'] / (1024 ** 3)  # Convert to GB
        bandwidth_limit = 1000  # Assuming a 1TB bandwidth limit, adjust as needed

        return storage_used, storage_limit, bandwidth_used, bandwidth_limit
    except requests.RequestException as e:
        app.logger.error(f"Error fetching MIAB usage data: {str(e)}")
        return 0, 100, 0, 1000  # Default values

def get_ticket_stats(user):
    total_tickets = Ticket.query.filter_by(user_id=user.id).count()
    open_tickets = Ticket.query.filter_by(user_id=user.id, status='Open').count()
    closed_tickets = Ticket.query.filter_by(user_id=user.id, status='Closed').count()
    in_progress_tickets = Ticket.query.filter_by(user_id=user.id, status='In Progress').count()

    return {
        'total': total_tickets,
        'open': open_tickets,
        'closed': closed_tickets,
        'in_progress': in_progress_tickets
    }


def get_sales_stats(affiliate_id):
    basic_sales = Sale.query.filter_by(affiliate_id=affiliate_id, plan_type='basic').count()
    pro_sales = Sale.query.filter_by(affiliate_id=affiliate_id, plan_type='pro').count()
    enterprise_sales = Sale.query.filter_by(affiliate_id=affiliate_id, plan_type='enterprise').count()
    total_earnings = db.session.query(db.func.sum(Sale.amount)).filter_by(affiliate_id=affiliate_id).scalar() or 0
    total_paid = db.session.query(db.func.sum(Payment.amount)).filter_by(affiliate_id=affiliate_id).scalar() or 0
    
    return {
        'basic_sales': basic_sales,
        'pro_sales': pro_sales,
        'enterprise_sales': enterprise_sales,
        'total_earnings': total_earnings,
        'total_paid': total_paid,
        'balance_due': total_earnings - total_paid
    }


@app.route('/affiliate/claim_earnings/<int:affiliate_id>', methods=['POST'])
@login_required
def claim_earnings(affiliate_id):
    affiliate = Affiliate.query.get_or_404(affiliate_id)
    if current_user.id != affiliate.user_id and not current_user.is_system_user:
        flash('You do not have permission to perform this action.', 'error')
        return redirect(url_for('affiliate'))

    description = f"Earnings claim request from affiliate {affiliate.email}"
    new_ticket = Ticket(title="Affiliate Earnings Claim", description=description, user_id=affiliate.user_id, status='Open')
    db.session.add(new_ticket)
    db.session.commit()

    flash('Your earnings claim request has been submitted. We will process it shortly.', 'success')
    return redirect(url_for('affiliate'))

def serialize_referrals(referrals):
    """Converts referral objects into JSON serializable structures."""
    serialized_referrals = []
    for referral in referrals:
        # Use get_referral_stats to populate stats
        stats = get_referral_stats(referral.referral_code)
        if not stats:
            stats = {
                'total': 0,
                'this_month': 0,
                'last_month': 0,
                'user_count': 0,
                'clicks': 0,
                'total_time_on_page': 0
            }
        
        serialized_referrals.append({
            'email': referral.email,
            'referral_code': referral.referral_code,
            'stats': stats
        })
    return serialized_referrals

@app.route('/affiliate_dashboard', methods=['GET', 'POST'])
@login_required
def affiliate_dashboard():
    if current_user.is_system_user:
        affiliates = Affiliate.query.all()
        
        if request.method == 'POST':
            email = request.form.get('email')
            referral_code = request.form.get('referral_code')

            existing_affiliate = Affiliate.query.filter_by(email=email).first()
            if existing_affiliate:
                flash('An affiliate with this email already exists.', 'error')
            else:
                new_affiliate = Affiliate(email=email, referral_code=referral_code)
                db.session.add(new_affiliate)
                db.session.commit()
                flash('Affiliate added successfully', 'success')

            return redirect(url_for('affiliate'))

        return render_template('affiliate.html', affiliates=affiliates)
    
    else:
        user_affiliate = Affiliate.query.filter_by(email=current_user.email).first()
        referral_stats = get_referral_stats(user_affiliate.referral_code) if user_affiliate else None
        
        return render_template('affiliate.html', 
                               user_affiliate=user_affiliate, 
                               referral_stats=[{'stats': referral_stats}] if referral_stats else None)



@app.route('/affiliate', methods=['GET', 'POST'])
@login_required
def affiliate():
    if current_user.is_system_user:
        affiliates = Affiliate.query.all()
        
        if request.method == 'POST':
            email = request.form.get('email')
            referral_code = request.form.get('referral_code')
            
            if not email or not referral_code:
                flash('Email and referral code are required.', 'error')
            else:
                existing_affiliate = Affiliate.query.filter_by(email=email).first()
                if existing_affiliate:
                    flash('An affiliate with this email already exists.', 'error')
                else:
                    try:
                        new_affiliate = Affiliate(email=email, referral_code=referral_code)
                        db.session.add(new_affiliate)
                        db.session.commit()
                        flash('Affiliate added successfully', 'success')
                    except SQLAlchemyError as e:
                        db.session.rollback()
                        app.logger.error(f"Error adding affiliate: {str(e)}")
                        flash('An error occurred while adding the affiliate. Please try again.', 'error')
            
            return redirect(url_for('affiliate'))
        
        return render_template('affiliate.html', affiliates=affiliates)
    
    else:
        user_affiliate = Affiliate.query.filter_by(email=current_user.email).first()
        
        if user_affiliate:
            try:
                referral_stats = get_referral_stats(user_affiliate.id)
                if referral_stats is None:
                    flash('Error retrieving referral statistics. Please try again later.', 'error')
                    referral_stats = {}
            except Exception as e:
                app.logger.error(f"Error getting referral stats: {str(e)}")
                flash('An error occurred while fetching referral statistics. Please try again later.', 'error')
                referral_stats = {}
        else:
            referral_stats = {}
        
        return render_template('affiliate.html', 
                               user_affiliate=user_affiliate, 
                               referral_stats=referral_stats)

@app.template_filter('format_currency')
def format_currency(value):
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    elif isinstance(value, str):
        try:
            return f"{float(value):.2f}"
        except ValueError:
            return "0.00"
    elif isinstance(value, Undefined):
        return "0.00"
    else:
        return "0.00"









@app.route('/a.webp')
def serdve_webap():
    return send_file('a.webp', mimetype='image/webp')

@app.route('/v.webp')
def serdve_webp():
    return send_file('v.webp', mimetype='image/webp')

@app.route('/z.webp')
def serve_webp():
    return send_file('z.webp', mimetype='image/webp')

@app.route('/c.webp')
def sesrve_webp():
    return send_file('c.webp', mimetype='image/webp')


app.config['UPLOAD_FOLDER'] = 'instance'

@app.route('/upload_db', methods=['GET', 'POST'])
@login_required
@system_user_required
def upload_db():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and file.filename.endswith('.db'):
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure the upload folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(temp_path)
            
            # Close the current database connection
            db.session.remove()
            db.engine.dispose()
            
            try:
                # Replace the current database file
                current_db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
                shutil.move(temp_path, current_db_path)
                
                # Reconnect to the new database
                db.engine.dispose()
                db.session.remove()
                db.create_all()
                
                flash('Database successfully uploaded and replaced', 'success')
            except Exception as e:
                flash(f'Error replacing database: {str(e)}', 'error')
            
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid file type. Please upload a .db file.', 'error')
            return redirect(request.url)
    
    return render_template('upload_db.html')

@app.route('/ref/<referral_code>')
def affiliate_redirect(referral_code):
    # Generate a temporary token
    token = secrets.token_urlsafe(16)
    
    # Store the token with the referral code and expiration time
    temp_tokens[token] = {
        'referral_code': referral_code,
        'expires': datetime.utcnow() + timedelta(minutes=5)  # Token expires in 5 minutes
    }
    
    # Redirect to home page with the token
    return redirect(url_for('home', token=token))


@app.before_request
def track_affiliate():
    token = request.args.get('token')
    
    if token and token in temp_tokens:
        token_data = temp_tokens[token]
        
        # Check if the token is still valid
        if datetime.utcnow() <= token_data['expires']:
            referral_code = token_data['referral_code']
            affiliate = Affiliate.query.filter_by(referral_code=referral_code).first()
            
            if affiliate and 'referral' not in session:
                # Increment clicks only for new sessions
                affiliate.clicks += 1
                db.session.commit()
                
                # Set session variables
                session['referral'] = referral_code
                session['session_start_time'] = datetime.utcnow().timestamp()
        
        # Remove the used token
        del temp_tokens[token]

    # Track time for unauthenticated sessions
    if 'referral' in session and not current_user.is_authenticated:
        if 'session_start_time' in session:
            current_time = datetime.utcnow().timestamp()
            session_duration = current_time - session['session_start_time']
            
            # Update the total time for the affiliate
            referral_code = session['referral']
            affiliate = Affiliate.query.filter_by(referral_code=referral_code).first()
            if affiliate:
                affiliate.total_time_on_page += int(session_duration)
                db.session.commit()
            
            # Reset the session start time
            session['session_start_time'] = current_time


@app.route('/affiliate/edit/<int:affiliate_id>', methods=['GET', 'POST'])
@login_required
def edit_affiliate(affiliate_id):
    if not current_user.is_system_user:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('affiliate'))

    affiliate = Affiliate.query.get_or_404(affiliate_id)

    if request.method == 'POST':
        if 'add_payment' in request.form:
            amount = float(request.form['amount'])
            description = request.form['description']
            new_payment = Payment(affiliate_id=affiliate.id, amount=amount, description=description)
            db.session.add(new_payment)
            db.session.commit()
            flash('Payment added successfully', 'success')
        elif 'update_sale' in request.form:
            plan_type = request.form['plan_type']
            amount = float(request.form['amount'])
            new_sale = Sale(affiliate_id=affiliate.id, plan_type=plan_type, amount=amount)
            db.session.add(new_sale)
            db.session.commit()
            flash('Sale updated successfully', 'success')
        elif 'update_user_count' in request.form:
            new_user_count = int(request.form['user_count'])
            affiliate.user_count = new_user_count
            db.session.commit()
            flash('Invoiced user count updated successfully', 'success')

    # Get statistics
    stats = get_referral_stats(affiliate.id)
    sales_stats = get_sales_stats(affiliate.id)
    payments = Payment.query.filter_by(affiliate_id=affiliate.id).order_by(Payment.created_at.desc()).all()

    return render_template('edit_affiliate.html', affiliate=affiliate, stats=stats, sales_stats=sales_stats, payments=payments)

@app.teardown_request
def update_session_time(exception=None):
    if 'referral' in session and not current_user.is_authenticated and 'session_start_time' in session:
        current_time = datetime.utcnow().timestamp()
        session_duration = current_time - session['session_start_time']
        
        # Update the total time for the affiliate
        referral_code = session['referral']
        affiliate = Affiliate.query.filter_by(referral_code=referral_code).first()
        if affiliate:
            affiliate.total_time_on_page += int(session_duration)
            db.session.commit()
        
        # Reset the session start time
        session['session_start_time'] = current_time

@app.route('/update_user_count', methods=['POST'])
@login_required
def update_user_count():
    if not current_user.is_system_user:
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.json
    affiliate_id = data.get('affiliate_id')
    change = data.get('change')  # 1 for increase, -1 for decrease

    affiliate = Affiliate.query.get(affiliate_id)
    if not affiliate:
        return jsonify({'error': 'Affiliate not found'}), 404

    affiliate.user_count = max(0, affiliate.user_count + change)
    db.session.commit()

    return jsonify({'success': True, 'new_count': affiliate.user_count})

@app.route('/track_click', methods=['POST'])
def track_click():
    referral_code = request.json.get('referral_code')
    affiliate = Affiliate.query.filter_by(referral_code=referral_code).first()
    if affiliate:
        affiliate.clicks += 1
        db.session.commit()
    return jsonify({'success': True})

@app.route('/track_time', methods=['POST'])
def track_time():
    referral_code = request.json.get('referral_code')
    time_spent = request.json.get('time_spent')
    affiliate = Affiliate.query.filter_by(referral_code=referral_code).first()
    if affiliate:
        affiliate.total_time_on_page += time_spent
        db.session.commit()
    return jsonify({'success': True})



@app.route('/subscription')
@login_required
def subscription():
    return render_template('subscription.html', user=current_user)

@app.route('/upload_image', methods=['POST'])
@temp_login_required
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Open the image using Pillow
        img = Image.open(file.stream)

        # Resize the image if it's larger than 1000x1000
        max_size = (1000, 1000)
        img.thumbnail(max_size)

        # Save the resized image
        img.save(file_path)

        return jsonify({'filename': unique_filename}), 200
    return jsonify({'error': 'File type not allowed'}), 400

def get_affiliate(referral_code):
    return Affiliate.query.filter_by(referral_code=referral_code).first()


app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

@app.route('/explorer')
@login_required
def explorer():
    return render_template('explorer.html')



@app.route('/stataroos')
@login_required
def stataroos():
    miab_users = []
    try:
        curl_command = f'curl -X GET --user "{current_user.miab_email}:{current_user.miab_password}" {current_user.miab_url}/admin/mail/users?format=json'
        logger.info(f"Executing cURL command: {curl_command}")
        process = subprocess.Popen(shlex.split(curl_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"cURL command failed with return code {process.returncode}")
        miab_users = json.loads(stdout.decode('utf-8'))
        logger.info(f"Retrieved MIAB users: {miab_users}")
    except Exception as e:
        logger.error(f"Error fetching MIAB users: {str(e)}")
        flash(f'Error fetching MIAB users: {str(e)}', 'error')

    # Calculate user count and domain information
    user_count = sum(len(domain['users']) for domain in miab_users)
    total_domains = len(miab_users)

    return render_template('stataroos.html', 
                           user=current_user, 
                           miab_users=miab_users,
                           user_count=user_count,
                           total_domains=total_domains)


@app.route('/api/files', methods=['GET', 'POST', 'DELETE'])
@login_required
def handle_files():
    path = request.args.get('path', '/')
    action = request.args.get('action')

    if not os.path.abspath(path).startswith('/'):
        return jsonify({'error': 'Invalid path'}), 400

    if request.method == 'GET':
        items = []
        for item in os.scandir(path):
            items.append({
                'name': item.name,
                'is_dir': item.is_dir(),
                'size': item.stat().st_size if item.is_file() else 0,
                'mtime': item.stat().st_mtime
            })
        return jsonify(items)

    elif request.method == 'POST':
        if action == 'upload':
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(os.path.join(path, filename))
            return jsonify({'message': 'File uploaded successfully'})
        elif action == 'create_folder':
            folder_name = request.json['name']
            os.mkdir(os.path.join(path, folder_name))
            return jsonify({'message': 'Folder created successfully'})

    elif request.method == 'DELETE':
        item_path = os.path.join(path, request.json['name'])
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)
        return jsonify({'message': 'Item deleted successfully'})

@app.route('/api/file', methods=['GET', 'PUT'])
@login_required
def handle_file():
    path = request.args.get('path')
    
    if request.method == 'GET':
        return send_file(path)
    elif request.method == 'PUT':
        with open(path, 'w') as f:
            f.write(request.data.decode('utf-8'))
        return jsonify({'message': 'File saved successfully'})

@app.route('/mailsolutions')
def homer():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')
    
@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_system_user:
        tickets = Ticket.query.all()
        affiliates = Affiliate.query.all()
        return render_template('dashboard.html', tickets=tickets, affiliates=affiliates)
    elif current_user.is_affiliate:
        return redirect(url_for('affiliate'))
    else:
        tickets = Ticket.query.filter_by(user_id=current_user.id).all()
        return render_template('dashboard.html', tickets=tickets)

from urllib.parse import urlparse, urlunparse, parse_qs

@app.route('/download_db')
@login_required
@system_user_required
def download_db():
    if not current_user.is_system_user:
        flash('You do not have permission to download the database.', 'error')
        return redirect(url_for('dashboard'))
    
    db_path = 'instance/users.db'  # Adjust this path if your database is located elsewhere
    
    if not os.path.exists(db_path):
        flash('Database file not found.', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        return send_file(db_path,
                         as_attachment=True,
                         download_name='users_backup.db',
                         mimetype='application/octet-stream')
    except Exception as e:
        flash(f'Error downloading database: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/limits')
@login_required
def limits():
    # Fetch the current user's data
    user_data = get_user_limits_data(current_user)

    # Fetch ticket statistics
    ticket_stats = get_ticket_stats(current_user)

    return render_template('limits.html', user=user_data, ticket_stats=ticket_stats)

def get_user_limits_data(user):
    # Fetch MIAB user data
    miab_users = get_miab_users(user)

    # Calculate current users
    current_users = sum(len(domain['users']) for domain in miab_users)

    # Get storage and bandwidth data from MIAB
    storage_used, storage_limit, bandwidth_used, bandwidth_limit = get_miab_usage_data(user)

    return {
        'current_users': current_users,
        'user_limit': user.user_limit,
        'storage_used': storage_used,
        'storage_limit': storage_limit,
        'bandwidth_used': bandwidth_used,
        'bandwidth_limit': bandwidth_limit
    }


@app.route('/home')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    # Clear the referral and session start time when logging out
    session.pop('referral', None)
    session.pop('session_start_time', None)
    logout_user()
    return redirect(url_for('login'))

@app.route('/add_user', methods=['GET', 'POST'])
@login_required
@system_user_required
def add_user():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        is_system_user = 'is_system_user' in request.form
        is_affiliate = 'is_affiliate' in request.form
        miab_url = request.form['miab_url']
        miab_email = request.form['miab_email']
        miab_password = request.form['miab_password']
        plan = request.form['plan']

        user_limit = None
        if plan == 'basic':
            user_limit = 20
        elif plan == 'pro':
            user_limit = 150
        elif plan == 'custom':
            user_limit = int(request.form['custom_user_limit'])

        if User.query.filter_by(email=email).first():
            flash('Email already registered')
        else:
            new_user = User(
                email=email, 
                password=generate_password_hash(password),
                is_system_user=is_system_user,
                is_affiliate=is_affiliate,
                miab_url=miab_url,
                miab_email=miab_email,
                miab_password=miab_password,
                plan=plan,
                user_limit=user_limit
            )
            db.session.add(new_user)
            db.session.commit()
            flash('User added successfully')
        return redirect(url_for('dashboard'))
    return render_template('add_user.html')


import json


GALLERY_FOLDER = os.path.join('static', 'gallery')

@app.route('/gallery<int:image_number>.png')
def gallery_image(image_number):
    if 1 <= image_number <= 6:
        return send_from_directory(GALLERY_FOLDER, f'gallery{image_number}.png')
    else:
        abort(404)  # Return a 404 error if the image number is out of range

# Ensure the GALLERY_FOLDER exists
if not os.path.exists(GALLERY_FOLDER):
    os.makedirs(GALLERY_FOLDER)


@app.route('/add_miab_user', methods=['POST'])
@login_required
def add_miab_user():
    email = request.form['email']
    password = request.form['password']

    logger.info(f"Attempting to add MIAB user: {email}")

    try:
        # Attempt to fetch current users, but continue even if it fails
        try:
            response = requests.get(
                f"{current_user.miab_url}/admin/mail/users?format=json",
                auth=(current_user.miab_email, current_user.miab_password)
            )
            response.raise_for_status()
            users_data = response.json()
            current_user_count = sum(len(domain['users']) for domain in users_data)
        except Exception as e:
            logger.error(f"Error fetching current users: {str(e)}")
            current_user_count = 0  # Assume 0 if we can't fetch the count

        # Proceed with adding the user regardless of previous errors
        add_user_response = requests.post(
            f"{current_user.miab_url}/admin/mail/users/add",
            auth=(current_user.miab_email, current_user.miab_password),
            data={'email': email, 'password': password}
        )

        # Log the response for debugging, but don't raise an exception
        logger.info(f"Add user response status: {add_user_response.status_code}")
        logger.info(f"Add user response content: {add_user_response.text}")

        # Assume success unless we can confirm otherwise
        success = True
        try:
            response_data = add_user_response.json()
            if isinstance(response_data, dict) and response_data.get('error'):
                success = False
                logger.warning(f"MIAB API reported an error: {response_data['error']}")
        except json.JSONDecodeError:
            logger.warning("Could not parse MIAB API response as JSON")

        if success:
            new_count = current_user_count + 1
            success_message = f'MIAB user added successfully. New user count: {new_count}/{current_user.user_limit if current_user.user_limit else "Unlimited"}'
            flash(success_message, 'success')
            logger.info(success_message)
        else:
            warning_message = 'Attempted to add MIAB user, but encountered an issue. The user may or may not have been added.'
            flash(warning_message, 'warning')
            logger.warning(warning_message)

    except Exception as e:
        error_message = f"Unexpected error while adding MIAB user: {str(e)}"
        logger.error(error_message)
        flash(error_message, 'error')

    return redirect(url_for('miab_users'))

@app.route('/miab_users')
@login_required
def miab_users():
    if not current_user.miab_url:
        flash('MIAB API credentials not set', 'error')
        return redirect(url_for('dashboard'))

    try:
        response = requests.get(
            f"{current_user.miab_url}/admin/mail/users?format=json",
            auth=(current_user.miab_email, current_user.miab_password)
        )
        response.raise_for_status()
        users_data = response.json()

        # Print the JSON response
        print("MIAB API Response:")
        print(json.dumps(users_data, indent=2))

        # Count users across all domains
        current_user_count = sum(len(domain['users']) for domain in users_data)

        can_add_users = True
        limit_message = ""

        if current_user.user_limit is not None:
            if current_user_count >= current_user.user_limit:
                can_add_users = False
                limit_message = f'User limit reached: {current_user_count}/{current_user.user_limit} users'
            else:
                limit_message = f'Current usage: {current_user_count}/{current_user.user_limit} users'
        else:
            limit_message = f'Current usage: {current_user_count} users (No limit)'

        print(f"User count: {current_user_count}/{current_user.user_limit if current_user.user_limit else 'Unlimited'}")

        # Flatten the user list for display
        flat_users = [
            user for domain in users_data 
            for user in domain['users']
        ]

        return render_template('miab_users.html', 
                               users=flat_users, 
                               can_add_users=can_add_users, 
                               limit_message=limit_message,
                               current_user_count=current_user_count,
                               user_limit=current_user.user_limit)

    except requests.RequestException as e:
        error_message = f'Error fetching MIAB users: {str(e)}'
        flash(error_message, 'error')
        print(error_message)
        return render_template('miab_users.html', 
                               users=[], 
                               can_add_users=False, 
                               limit_message="Unable to fetch user data",
                               current_user_count=0,
                               user_limit=current_user.user_limit)



@app.route('/remove_miab_user', methods=['POST'])
@login_required
def remove_miab_user():
    email = request.form['email']

    try:
        # First, check if the user is an admin
        response = requests.get(
            f"{current_user.miab_url}/admin/mail/users?format=json",
            auth=(current_user.miab_email, current_user.miab_password)
        )
        response.raise_for_status()
        users_data = response.json()

        is_admin = any(
            user['email'] == email and 'admin' in user['privileges']
            for domain in users_data
            for user in domain['users']
        )

        if is_admin:
            flash('Cannot remove admin users', 'error')
            return redirect(url_for('miab_users'))

        # If not an admin, proceed with removal
        response = requests.post(
            f"{current_user.miab_url}/admin/mail/users/remove",
            auth=(current_user.miab_email, current_user.miab_password),
            data={'email': email}
        )
        response.raise_for_status()
        flash('MIAB user removed successfully', 'success')
    except requests.RequestException as e:
        flash(f'Error removing MIAB user: {str(e)}', 'error')

    return redirect(url_for('miab_users'))
@app.route('/manage_users')
@login_required
@system_user_required
def manage_users():
    users = User.query.all()
    return render_template('manage_users.html', users=users)

@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
@system_user_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    if request.method == 'POST':
        user.email = request.form['email']
        if request.form['password']:
            user.password = generate_password_hash(request.form['password'])
        user.is_system_user = 'is_system_user' in request.form
        user.miab_url = request.form['miab_url']
        user.miab_email = request.form['miab_email']
        if request.form['miab_password']:
            user.miab_password = request.form['miab_password']
        db.session.commit()
        flash('User updated successfully')
        return redirect(url_for('manage_users'))
    return render_template('edit_user.html', user=user)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
@system_user_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.is_system_user:
        flash('Cannot delete admin users', 'error')
    else:
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully', 'success')
    return redirect(url_for('manage_users'))


@app.route('/tickets')
@login_required
def tickets():
    if current_user.is_system_user:
        tickets = Ticket.query.all()
    else:
        tickets = Ticket.query.filter_by(user_id=current_user.id).all()
    return render_template('tickets.html', tickets=tickets)

@app.route('/view_ticket/<int:ticket_id>')
@temp_login_required
def view_ticket(ticket_id):
    ticket = Ticket.query.get_or_404(ticket_id)
    
    if current_user.is_authenticated:
        if not current_user.is_system_user and ticket.user_id != current_user.id:
            flash('You do not have permission to view this ticket.', 'error')
            return redirect(url_for('home'))
        can_send_message = current_user.is_system_user
    else:
        # Check if the logged-in temporary user has the right to view this ticket
        temp_ticket_id = get_ticket_id(session.get('temp_user'))
        if temp_ticket_id is None or temp_ticket_id != ticket_id:
            flash('You do not have permission to view this ticket.', 'error')
            return redirect(url_for('home'))
        can_send_message = False
    
    return render_template('view_ticket.html', 
                           ticket=ticket, 
                           can_send_message=can_send_message, 
                           get_affiliate=get_affiliate)

@app.route('/create_ticket', methods=['GET', 'POST'])
@login_required
def create_ticket():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        referral = session.get('referral')
        new_ticket = Ticket(title=title, description=description, user_id=current_user.id, referral=referral)
        db.session.add(new_ticket)
        db.session.commit()
        flash('Ticket created successfully', 'success')
        return redirect(url_for('dashboard'))
    return render_template('create_ticket.html')

@app.route('/create_home_ticket', methods=['POST'])
def create_home_ticket():
    name = request.form.get('name')
    email = request.form.get('email')
    reason = request.form.get('reason')
    plan = request.form.get('plan')
    domain = request.form.get('domain')
    message = request.form.get('message')
    referral = session.get('referral')

    # Generate temporary password
    temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    hashed_temp_password = generate_password_hash(temp_password)

    new_ticket = Ticket(
        title=f"Contact Request: {reason}",
        description=f"Name: {name}\nEmail: {email}\nReason: {reason}\nPlan: {plan}\nDomain: {domain}\nMessage: {message}",
        status='Open',
        user_id=current_user.id if current_user.is_authenticated else None,
        temp_password_hash=hashed_temp_password,
        referral=referral
    )

    db.session.add(new_ticket)
    db.session.commit()

    # Generate temporary username using the ticket's ID
    temp_username = f"user_{new_ticket.id}"

    return jsonify({
        'username': temp_username,
        'password': temp_password,
        'ticket_id': new_ticket.id
    })

def validate_temp_credentials(username, password):
    # Extract ticket ID from the username
    try:
        ticket_id = int(username.split('_')[1])
    except (IndexError, ValueError):
        return False

    # Query the ticket using the ticket ID
    ticket = Ticket.query.get(ticket_id)

    if not ticket:
        return False

    # Compare hashed password
    return check_password_hash(ticket.temp_password_hash, password)
    
@app.route('/temp_ticket_login', methods=['POST'])
def temp_ticket_login():
    username = request.json.get('username')
    password = request.json.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    if validate_temp_credentials(username, password):
        ticket_id = int(username.split('_')[1])
        session['temp_user'] = username
        return jsonify({"message": "Login successful", "ticket_id": ticket_id}), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

def validate_temp_credentials(username, password):
    try:
        ticket_id = int(username.split('_')[1])
    except (IndexError, ValueError):
        return False

    ticket = Ticket.query.get(ticket_id)

    if not ticket:
        return False

    return check_password_hash(ticket.temp_password_hash, password)
    
@app.route('/ticket/<int:ticket_id>')
@login_required
def ticket_details(ticket_id):
    ticket = Ticket.query.get_or_404(ticket_id)
    
    if current_user.is_system_user:
        can_send_message = True
    elif ticket.user_id == current_user.id:
        can_send_message = False
    else:
        flash('You do not have permission to view this ticket.', 'error')
        return redirect(url_for('tickets'))
    
    return render_template('ticket_details.html', 
                           ticket=ticket, 
                           can_send_message=can_send_message, 
                           get_affiliate=get_affiliate)


@app.route('/ticket/<int:ticket_id>/add_message', methods=['POST'])
@temp_login_required
def add_message(ticket_id):
    ticket = Ticket.query.get_or_404(ticket_id)

    # Check permissions
    if current_user.is_authenticated:
        if not current_user.is_system_user and ticket.user_id != current_user.id:
            return jsonify({'error': 'Permission denied'}), 403
    else:
        temp_ticket_id = get_ticket_id(session.get('temp_user'))
        if temp_ticket_id != ticket_id:
            return jsonify({'error': 'Permission denied'}), 403

    data = request.json  # Change this to handle JSON data
    message_content = data.get('message', '').strip()
    image_filename = data.get('image_filename')

    # Check if the message is empty
    if not message_content and not image_filename:
        return jsonify({'error': 'Message cannot be empty'}), 400

    message = Message(
        content=message_content,
        is_admin=current_user.is_authenticated and current_user.is_system_user,
        ticket_id=ticket.id,
        user_id=current_user.id if current_user.is_authenticated else None,
        temp_user=session.get('temp_user') if not current_user.is_authenticated else None,
        image_filename=image_filename
    )
    db.session.add(message)
    db.session.commit()

    return jsonify({
        'id': message.id,
        'content': message.content,
        'created_at': message.created_at.isoformat(),
        'is_admin': message.is_admin,
        'image_filename': message.image_filename
    }), 201

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/pricing')
def pricing():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head><script async src="https://cdn.tolt.io/tolt.js" data-tolt="YOUR-TOLT-PUBLIC-ID"></script>

        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pricing</title>
        <script async src="https://js.stripe.com/v3/pricing-table.js"></script><script>
  var toltScript = document.createElement('script');
  toltScript.src = 'https://cdn.tolt.io/tolt.js';
  toltScript.setAttribute('data-tolt', 'c58e85f6-afa3-4f06-b729-b8c9989537f1');
  document.head.appendChild(toltScript);
</script><script>
const updatePricingTables = () => {
var stripePricingTables = document.querySelectorAll("stripe-pricing-table");
if (window.tolt_referral !== null && stripePricingTables.length > 0) {
stripePricingTables.forEach(stripePricingTable => {
stripePricingTable.setAttribute("client-reference-id", window.tolt_referral);
})
}
}setTimeout(updatePricingTables, 1000);
setTimeout(updatePricingTables, 2200);
setTimeout(updatePricingTables, 3200);window.addEventListener("tolt_referral_ready", () => {
if (window.tolt_referral) {
updatePricingTables()
}
})
</script>
    </head>
    <body>
<script async src="https://js.stripe.com/v3/pricing-table.js"></script>
<stripe-pricing-table pricing-table-id="prctbl_1Q6qvVAHFs365FbAINeO0zIo"
publishable-key="pk_live_51Q6eTgAHFs365FbAxVrWOervK5xeMvqMh0aCUwuTElJ4hzTI30Cadh1E046AZlDQCGftcnhofyXaaZBeKlVQwjWh00qzgZhtw1">
</stripe-pricing-table>    </body>
    </html>
    '''
@app.route('/pricing-pro')
def pricingpro():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head><script async src="https://cdn.tolt.io/tolt.js" data-tolt="YOUR-TOLT-PUBLIC-ID"></script>

        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pricing</title>
        <script async src="https://js.stripe.com/v3/pricing-table.js"></script><script>
  var toltScript = document.createElement('script');
  toltScript.src = 'https://cdn.tolt.io/tolt.js';
  toltScript.setAttribute('data-tolt', 'c58e85f6-afa3-4f06-b729-b8c9989537f1');
  document.head.appendChild(toltScript);
</script><script>
const updatePricingTables = () => {
var stripePricingTables = document.querySelectorAll("stripe-pricing-table");
if (window.tolt_referral !== null && stripePricingTables.length > 0) {
stripePricingTables.forEach(stripePricingTable => {
stripePricingTable.setAttribute("client-reference-id", window.tolt_referral);
})
}
}setTimeout(updatePricingTables, 1000);
setTimeout(updatePricingTables, 2200);
setTimeout(updatePricingTables, 3200);window.addEventListener("tolt_referral_ready", () => {
if (window.tolt_referral) {
updatePricingTables()
}
})
</script>
    </head>
    <body>
<script async src="https://js.stripe.com/v3/pricing-table.js"></script>
<stripe-pricing-table pricing-table-id="prctbl_1Q6r24AHFs365FbANoY8N3iI"
publishable-key="pk_live_51Q6eTgAHFs365FbAxVrWOervK5xeMvqMh0aCUwuTElJ4hzTI30Cadh1E046AZlDQCGftcnhofyXaaZBeKlVQwjWh00qzgZhtw1">
</stripe-pricing-table>    </body>
    </html>
    '''


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'f.png', mimetype='image/png')



@app.route('/ticket/<int:ticket_id>/get_messages')
@temp_login_required
def get_messages(ticket_id):
    ticket = Ticket.query.get_or_404(ticket_id)

    if current_user.is_authenticated:
        if not current_user.is_system_user and ticket.user_id != current_user.id:
            return jsonify({'error': 'Permission denied'}), 403
    else:
        temp_ticket_id = get_ticket_id(session.get('temp_user'))
        if temp_ticket_id != ticket_id:
            return jsonify({'error': 'Permission denied'}), 403

    messages = Message.query.filter_by(ticket_id=ticket_id).order_by(Message.created_at).all()
    return jsonify([{
        'id': message.id,
        'content': message.content,
        'created_at': message.created_at.isoformat(),
        'is_admin': message.is_admin,
        'user_type': 'admin' if message.is_admin else ('user' if message.user_id else 'temp_user'),
        'image_url': url_for('uploaded_file', filename=message.image_filename) if message.image_filename else None
    } for message in messages])

@app.route('/update_ticket/<int:ticket_id>', methods=['POST'])
@login_required
@system_user_required
def update_ticket(ticket_id):
    ticket = Ticket.query.get_or_404(ticket_id)
    ticket.status = request.form['status']
    db.session.commit()
    flash('Ticket status updated', 'success')
    return redirect(url_for('ticket_details', ticket_id=ticket.id))

def get_ticket_id(username):
    if username is None or '_' not in username:
        return None
    try:
        return int(username.split('_')[1])
    except (IndexError, ValueError):
        return None

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)
