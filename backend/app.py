from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate  # Removed because schema managed manually now
import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()

app = Flask(__name__)
# Restrict CORS origins to Vite dev server & any additional origins via env (comma separated)
additional_origins = [o.strip() for o in os.getenv('CORS_ORIGINS', '').split(',') if o.strip()]
allowed_origins = ['http://localhost:5173', 'http://127.0.0.1:5173', *additional_origins]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Secret key (needed if later you use sessions/JWT signing seeds etc.)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "dev-secret")

# DB config
raw_db_url = os.getenv('DATABASE_URL')
if raw_db_url and raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = raw_db_url or 'sqlite:///local.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
# Migrations intentionally disabled for manual DDL management
# migrate = Migrate(app, db)

# ---------- Models (reflect existing tables) ----------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def set_password(self, raw_password: str):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

# ---------- Routes ----------
@app.route('/')
def test_connection():
    """A simple test route to confirm the backend is running."""
    return jsonify({"message": "Backend is connected to Frontend!"})

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json(force=True)
    email = (data.get('email') or '').strip().lower()
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User registered", "user_id": user.id}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json(force=True)
    email = (data.get('email') or '').strip().lower()
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    # For now just return success (you can add JWT later)
    return jsonify({"message": "Login successful", "user_id": user.id})

# Generic JSON error handler (optional improvement)
@app.errorhandler(500)
def internal_error(e):  # pragma: no cover simple handler
    return jsonify({"error": "Internal server error"}), 500

# ---------- Startup Checks (no automatic DDL) ----------
with app.app_context():
    try:
        # Lightweight existence check for required table(s)
        inspector = db.inspect(db.engine)
        existing = inspector.get_table_names()
        if 'users' not in existing:
            print("[WARN] 'users' table not found. Create it manually before using auth endpoints.")
        else:
            print("[OK] Found 'users' table.")
    except Exception as ex:  # pragma: no cover
        print("[ERROR] Could not inspect database:", ex)

if __name__ == '__main__':
    app.run(debug=True)



