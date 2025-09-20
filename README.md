# Code4Edtech Hackathon Auth Demo

Simple full‑stack example: Flask (Python) + PostgreSQL + React (Vite) for user registration and login.

## Backend (Flask)
Location: `backend/app.py`

Features:
- User model with email + hashed password (Werkzeug)
- Register & Login endpoints
- CORS restricted to Vite dev origins
- Auto-detects `DATABASE_URL` (Postgres) or falls back to local SQLite

### Environment Variables (.env at repo root)
```
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/your_db
SECRET_KEY=change_this_dev_secret
CORS_ORIGINS=http://localhost:5173
```

### Install Python Dependencies (inside your virtualenv)
```powershell
python -m pip install flask flask_sqlalchemy flask_migrate psycopg2-binary python-dotenv flask-cors
```

### (Optional) Initialize DB Migrations
```powershell
set FLASK_APP=backend/app.py
flask db init
flask db migrate -m "Initial users table"
flask db upgrade
```

### Run Backend
```powershell
python backend/app.py
```
Server runs at: http://127.0.0.1:5000

## Frontend (React + Vite)
Location: `frontend/`

### Install Node Dependencies
```powershell
cd frontend
npm install
```

### Run Frontend Dev Server
```powershell
npm run dev
```
Opens at: http://localhost:5173

The Vite dev server proxies `/api/*` to `http://127.0.0.1:5000` (see `vite.config.js`).

## Test Auth Flow (PowerShell)
```powershell
# Register
curl -X POST http://127.0.0.1:5000/api/auth/register -H "Content-Type: application/json" -d '{"email":"test@example.com","password":"Pass123!"}'
# Login
curl -X POST http://127.0.0.1:5000/api/auth/login -H "Content-Type: application/json" -d '{"email":"test@example.com","password":"Pass123!"}'
```
Or use the React UI at http://localhost:5173

## Project Structure (excerpt)
```
backend/
  app.py
frontend/
  index.html
  vite.config.js
  package.json
  src/
    main.jsx
    App.jsx
    components/
      AuthForm.jsx
    lib/
      api.js
```

## Next Steps / Ideas
- Add JWT auth (e.g. PyJWT) and return token on login
- Protect routes & create /api/users/me endpoint
- Form validation & improved error messages on frontend
- Password reset flow

## License
MIT
