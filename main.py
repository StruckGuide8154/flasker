import os
import flask
from flask import Flask, render_template, request, redirect, url_for, flash, current_app
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
from datetime import datetime
from datetime import datetime
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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

@login_manager.unauthorized_handler
def unauthorized():
    flash('You must be logged in to view this page.', 'warning')
    return redirect(url_for('home'))
    
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    is_system_user = db.Column(db.Boolean, default=False)
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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Changed to nullable
    messages = db.relationship('Message', backref='ticket', lazy=True)
    temp_password_hash = db.Column(db.String(100))

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    ticket_id = db.Column(db.Integer, db.ForeignKey('ticket.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Changed to nullable
    temp_user = db.Column(db.String(100), nullable=True)  # New field for temporary users
    image_filename = db.Column(db.String(255), nullable=True)  # New field for image filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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


@app.route('/')
@login_required
def dashboard():
    if current_user.is_system_user:
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
            miab_users = []
            flash(f'Error fetching MIAB users: {str(e)}', 'error')
    else:
        miab_users = []

    tickets = Ticket.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', miab_users=miab_users, tickets=tickets)

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
        # For 'unlimited' and 'null', user_limit remains None

        if User.query.filter_by(email=email).first():
            flash('Email already registered')
        else:
            new_user = User(
                email=email, 
                password=generate_password_hash(password),
                is_system_user=is_system_user,
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

    return render_template('view_ticket.html', ticket=ticket, can_send_message=can_send_message)

@app.route('/create_ticket', methods=['GET', 'POST'])
@login_required
def create_ticket():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']

        # Check if the user is authenticated before accessing current_user.id
        if not current_user.is_authenticated:
            flash('You need to log in to create a ticket.', 'error')
            return redirect(url_for('home'))

        new_ticket = Ticket(title=title, description=description, user_id=current_user.id)
        db.session.add(new_ticket)
        db.session.commit()
        flash('Ticket created successfully', 'success')
        return redirect(url_for('tickets'))
    return render_template('create_ticket.html')

@app.route('/create_home_ticket', methods=['POST'])
def create_home_ticket():
    # Retrieve form data
    name = request.form.get('name')
    email = request.form.get('email')
    reason = request.form.get('reason')
    plan = request.form.get('plan')
    domain = request.form.get('domain')
    message = request.form.get('message')

    # Generate temporary password
    temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    # Hash the temporary password
    hashed_temp_password = generate_password_hash(temp_password)

    # Create a new ticket with the temporary password hash
    new_ticket = Ticket(
        title=f"Contact Request: {reason}",
        description=f"Name: {name}\nEmail: {email}\nReason: {reason}\nPlan: {plan}\nDomain: {domain}\nMessage: {message}",
        status='Open',
        user_id=current_user.id if current_user.is_authenticated else None,
        temp_password_hash=hashed_temp_password
    )

    db.session.add(new_ticket)
    db.session.commit()

    # Generate temporary username using the ticket's ID
    temp_username = f"user_{new_ticket.id}"

    # Return the temporary username and password to the client
    return jsonify({
        'username': temp_username,
        'password': temp_password,
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

    logger.info(f"Attempting login for username: {username}")

    if validate_temp_credentials(username, password):
        ticket_id = int(username.split('_')[1])
        session['temp_user'] = username
        logger.info(f"Login successful for username: {username}")
        return jsonify({"message": "Login successful", "ticket_id": ticket_id}), 200
    else:
        logger.warning(f"Invalid credentials for username: {username}")
        return jsonify({"error": "Invalid username or password"}), 401
    
@app.route('/ticket/<int:ticket_id>')
@login_required
def ticket_details(ticket_id):
    ticket = Ticket.query.get_or_404(ticket_id)
    if not current_user.is_system_user and ticket.user_id != current_user.id:
        flash('You do not have permission to view this ticket.', 'error')
        return redirect(url_for('tickets'))
    return render_template('ticket_details.html', ticket=ticket)

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

    data = request.form
    message_content = data.get('message', '')
    image_filename = data.get('image_filename', None)

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
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pricing</title>
        <script async src="https://js.stripe.com/v3/pricing-table.js"></script>
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
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pricing</title>
        <script async src="https://js.stripe.com/v3/pricing-table.js"></script>
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
