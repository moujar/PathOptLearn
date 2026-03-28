import sqlite3
import datetime
import random
import bcrypt

DB_PATH = "adaptive.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            username          TEXT UNIQUE NOT NULL,
            email             TEXT UNIQUE NOT NULL,
            password_hash     TEXT NOT NULL,
            verified          INTEGER DEFAULT 0,
            verification_code TEXT,
            code_expiry       TEXT,
            created_at        TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            name       TEXT NOT NULL,
            created_at TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            course_id  INTEGER NOT NULL,
            video_id   TEXT,
            title      TEXT,
            score      INTEGER,
            total      INTEGER,
            passed     INTEGER,
            timestamp  TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (course_id)  REFERENCES courses(id)
        )
    """)

    conn.commit()
    conn.close()


# ── AUTH ────────────────────────────────────────────────────
def register_student(username, email, password):
    """Returns (True, student_id) or (False, error_message)."""
    if not username.strip() or not password.strip() or not email.strip():
        return False, "All fields are required."
    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    code   = str(random.randint(100000, 999999))
    expiry = (datetime.datetime.now() + datetime.timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = get_conn()
        cur  = conn.execute(
            """INSERT INTO students
               (username, email, password_hash, verified, verification_code, code_expiry, created_at)
               VALUES (?, ?, ?, 0, ?, ?, ?)""",
            (username.strip(), email.strip().lower(), hashed, code, expiry,
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        student_id = cur.lastrowid
        conn.commit()
        conn.close()
        return True, (student_id, code)
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already taken."
        if "email" in str(e):
            return False, "An account with this email already exists."
        return False, "Registration failed."


def verify_student(student_id, code):
    """Returns (True, None) or (False, error_message)."""
    conn = get_conn()
    row  = conn.execute(
        "SELECT verification_code, code_expiry, verified FROM students WHERE id = ?",
        (student_id,)
    ).fetchone()
    conn.close()

    if not row:
        return False, "Account not found."
    stored_code, expiry, verified = row

    if verified:
        return True, None

    if datetime.datetime.now() > datetime.datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S"):
        return False, "Code expired. Please register again."

    if code.strip() != stored_code:
        return False, "Wrong code. Check your email."

    conn = get_conn()
    conn.execute("UPDATE students SET verified = 1 WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()
    return True, None


def resend_verification(student_id):
    """Generates a new code and updates the DB. Returns the new code."""
    code   = str(random.randint(100000, 999999))
    expiry = (datetime.datetime.now() + datetime.timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
    conn   = get_conn()
    conn.execute(
        "UPDATE students SET verification_code = ?, code_expiry = ? WHERE id = ?",
        (code, expiry, student_id)
    )
    conn.commit()
    conn.close()
    return code


def login_student(username, password):
    """Returns (True, student_id) or (False, error_message)."""
    conn = get_conn()
    row  = conn.execute(
        "SELECT id, password_hash, verified FROM students WHERE username = ?",
        (username.strip(),)
    ).fetchone()
    conn.close()

    if not row:
        return False, "Username not found."
    student_id, password_hash, verified = row

    if not bcrypt.checkpw(password.encode(), password_hash.encode()):
        return False, "Wrong password."

    if not verified:
        return False, "UNVERIFIED"

    return True, student_id


def get_student_email(student_id):
    conn = get_conn()
    row  = conn.execute("SELECT email FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()
    return row[0] if row else None


# ── COURSES ─────────────────────────────────────────────────
def create_course(student_id, name):
    conn      = get_conn()
    cur       = conn.execute(
        "INSERT INTO courses (student_id, name, created_at) VALUES (?, ?, ?)",
        (student_id, name.strip(), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    course_id = cur.lastrowid
    conn.commit()
    conn.close()
    return course_id


def get_courses(student_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, created_at FROM courses WHERE student_id = ? ORDER BY id DESC",
        (student_id,)
    ).fetchall()
    conn.close()
    return rows


def delete_course(course_id, student_id):
    conn = get_conn()
    conn.execute("DELETE FROM courses WHERE id = ? AND student_id = ?", (course_id, student_id))
    conn.execute("DELETE FROM progress WHERE course_id = ? AND student_id = ?", (course_id, student_id))
    conn.commit()
    conn.close()


# ── PROGRESS ────────────────────────────────────────────────
def log_progress(student_id, course_id, video_id, title, score, total, passed):
    conn = get_conn()
    conn.execute("""
        INSERT INTO progress (student_id, course_id, video_id, title, score, total, passed, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student_id, course_id, video_id, title,
        score, total, int(passed),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def get_course_progress(student_id, course_id):
    conn = get_conn()
    rows = conn.execute("""
        SELECT video_id, title, score, total, passed, timestamp
        FROM progress
        WHERE student_id = ? AND course_id = ?
        ORDER BY id ASC
    """, (student_id, course_id)).fetchall()
    conn.close()
    return rows


def get_course_stats(student_id, course_id):
    rows = get_course_progress(student_id, course_id)
    if not rows:
        return {"videos_watched": 0, "avg_score": 0, "pass_rate": 0, "last_activity": None}
    videos_watched = len(rows)
    avg_score      = sum(int(r[2] / r[3] * 100) for r in rows if r[3] > 0) // videos_watched
    pass_rate      = int(sum(1 for r in rows if r[4]) / videos_watched * 100)
    last_activity  = rows[-1][5]
    return {
        "videos_watched": videos_watched,
        "avg_score":      avg_score,
        "pass_rate":      pass_rate,
        "last_activity":  last_activity,
    }


init_db()
