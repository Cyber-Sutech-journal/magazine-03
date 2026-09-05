from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from http import cookies
import sqlite3
import ssl
import threading
import secrets
import hashlib
import hmac
import html
from datetime import datetime, timezone

HOST = "127.0.0.1"
HTTP_PORT = 8000
HTTPS_PORT = 8443
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cybersootec.db"

# In-memory sessions are enough for this small educational lab.
SESSIONS = {}
SESSION_LOCK = threading.Lock()

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

def hash_password(password):
    # PBKDF2-HMAC-SHA256; password is never stored in plaintext.
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 180_000)
    return f"pbkdf2_sha256$180000${salt.hex()}${digest.hex()}"

def verify_password(password, stored):
    try:
        algorithm, rounds, salt_hex, digest_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(rounds),
        ).hex()
        return hmac.compare_digest(candidate, digest_hex)
    except (ValueError, TypeError):
        return False

def create_session(user_id, username):
    token = secrets.token_urlsafe(32)
    with SESSION_LOCK:
        SESSIONS[token] = {"user_id": user_id, "username": username}
    return token

def get_session(handler):
    raw = handler.headers.get("Cookie", "")
    jar = cookies.SimpleCookie()
    try:
        jar.load(raw)
    except cookies.CookieError:
        return None
    token = jar["session"].value if "session" in jar else None
    if not token:
        return None
    with SESSION_LOCK:
        return SESSIONS.get(token)

def redirect(handler, location, session_token=None):
    handler.send_response(303)
    handler.send_header("Location", location)
    handler.send_header("Cache-Control", "no-store")
    if session_token:
        c = cookies.SimpleCookie()
        c["session"] = session_token
        c["session"]["Path"] = "/"
        c["session"]["HttpOnly"] = True
        # SameSite=Lax is appropriate for this local lab.
        c["session"]["SameSite"] = "Lax"
        handler.send_header("Set-Cookie", c["session"].OutputString())
    handler.end_headers()

class CyberSootecHandler(BaseHTTPRequestHandler):
    server_version = "CyberSootecLab/2.0"

    def log_message(self, fmt, *args):
        # Avoid dumping form bodies/passwords into the terminal.
        print(f"[{self.log_date_time_string()}] {self.command} {self.path}")

    def send_bytes(self, data, content_type, status=200, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text, status=200):
        self.send_bytes(text.encode("utf-8"), "text/html; charset=utf-8", status)

    def serve_file(self, filename):
        path = (BASE_DIR / filename).resolve()
        if BASE_DIR not in path.parents:
            self.send_error(403)
            return
        if not path.is_file():
            self.send_error(404, "File not found")
            return
        self.send_bytes(path.read_bytes(), MIME_TYPES.get(path.suffix.lower(), "application/octet-stream"))

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        routes = {
            "/": "index.html",
            "/index.html": "index.html",
            "/login": "login.html",
            "/register": "register.html",
            "/style.css": "style.css",
        }

        if route in routes:
            self.serve_file(routes[route])
            return

        if route == "/dashboard":
            session = get_session(self)
            if not session:
                redirect(self, "/login?error=login_required")
                return
            self.serve_file("dashboard.html")
            return

        if route == "/api/me":
            session = get_session(self)
            if not session:
                self.send_text('{"authenticated":false}', 401)
                return
            payload = (
                '{"authenticated":true,"username":"'
                + html.escape(session["username"], quote=True)
                + '"}'
            )
            self.send_bytes(payload.encode(), "application/json; charset=utf-8")
            return

        if route.startswith("/assets/"):
            relative = route.removeprefix("/assets/")
            self.serve_file(Path("assets") / relative)
            return

        self.send_error(404, "Page not found")

    def do_POST(self):
        route = urlparse(self.path).path

        if route not in ("/login", "/register"):
            self.send_error(404, "Page not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(body, keep_blank_values=True)

        username = form.get("username", [""])[0].strip()
        email = form.get("email", [""])[0].strip()
        password = form.get("password", [""])[0]

        if route == "/register":
            if not username or not email or not password:
                redirect(self, "/register?error=missing")
                return

            if len(username) > 50 or len(email) > 120 or len(password) > 200:
                redirect(self, "/register?error=invalid")
                return

            try:
                with db() as conn:
                    cur = conn.execute(
                        """INSERT INTO users (username,email,password_hash,created_at)
                           VALUES (?,?,?,?)""",
                        (
                            username,
                            email,
                            hash_password(password),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    user_id = cur.lastrowid
                    conn.commit()
            except sqlite3.IntegrityError:
                redirect(self, "/register?error=exists")
                return

            token = create_session(user_id, username)
            redirect(self, "/dashboard", token)
            return

        # Login
        if not username or not password:
            redirect(self, "/login?error=missing")
            return

        with db() as conn:
            user = conn.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()

        if not user or not verify_password(password, user["password_hash"]):
            redirect(self, "/login?error=invalid")
            return

        token = create_session(user["id"], user["username"])
        redirect(self, "/dashboard", token)

def run_http():
    server = ThreadingHTTPServer((HOST, HTTP_PORT), CyberSootecHandler)
    print(f"[HTTP ]  http://{HOST}:{HTTP_PORT}")
    server.serve_forever()

def run_https():
    server = ThreadingHTTPServer((HOST, HTTPS_PORT), CyberSootecHandler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(BASE_DIR / "cert.pem"), str(BASE_DIR / "key.pem"))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"[HTTPS]  https://{HOST}:{HTTPS_PORT}")
    server.serve_forever()

if __name__ == "__main__":
    init_db()
    print("CyberSootec v2 — SQLite authentication lab")
    t1 = threading.Thread(target=run_http, daemon=True)
    t2 = threading.Thread(target=run_https, daemon=True)
    t1.start()
    t2.start()
    print("HTTP  : http://127.0.0.1:8000")
    print("HTTPS : https://127.0.0.1:8443")
    print("Database: cybersootec.db")
    print("Press Ctrl+C to stop.\n")
    try:
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("\nStopped.")
