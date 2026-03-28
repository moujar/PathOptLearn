import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db import login_student, register_student, verify_student, resend_verification
from mailer import send_verification_email

st.set_page_config(page_title="Login — Adaptive Learning", page_icon="🎓", layout="centered")

# Already logged in → go to dashboard
if st.session_state.get("student_id") and st.session_state.get("verified"):
    st.switch_page("pages/2_Dashboard.py")


# ── VERIFICATION SCREEN ──────────────────────────────────────
if st.session_state.get("pending_verification"):
    student_id = st.session_state["pending_student_id"]
    email      = st.session_state["pending_email"]
    username   = st.session_state["pending_username"]

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.title("🎓 Verify your email")
        st.markdown(f"We sent a 6-digit code to **{email}**.")
        st.markdown("*Check your spam folder if you don't see it.*")
        st.markdown("---")

        with st.form("verify_form"):
            code      = st.text_input("Verification code", max_chars=6, placeholder="123456")
            submitted = st.form_submit_button("Verify", type="primary", use_container_width=True)

        if submitted:
            ok, error = verify_student(student_id, code)
            if ok:
                st.session_state["student_id"]           = student_id
                st.session_state["username"]             = username
                st.session_state["verified"]             = True
                st.session_state["pending_verification"] = False
                st.success("Email verified! Welcome 🎉")
                st.switch_page("pages/2_Dashboard.py")
            else:
                st.error(error)

        st.markdown("")
        if st.button("Resend code", use_container_width=True):
            new_code  = resend_verification(student_id)
            ok, error = send_verification_email(email, username, new_code)
            if ok:
                st.success("New code sent!")
            else:
                st.error(f"Could not send email: {error}")

        if st.button("← Back to login", use_container_width=True):
            st.session_state["pending_verification"] = False
            st.rerun()

    st.stop()


# ── MAIN LOGIN / REGISTER UI ─────────────────────────────────
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.title("🎓 Adaptive Learning")
    st.markdown("##### Personalized video-based learning powered by AI")
    st.markdown("---")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    # ── LOGIN ────────────────────────────────────────────────
    with tab_login:
        with st.form("login_form"):
            username  = st.text_input("Username")
            password  = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please fill in both fields.")
            else:
                ok, result = login_student(username, password)
                if ok:
                    st.session_state["student_id"] = result
                    st.session_state["username"]   = username.strip()
                    st.session_state["verified"]   = True
                    st.switch_page("pages/2_Dashboard.py")
                elif result == "UNVERIFIED":
                    # Resend code and send to verify screen
                    conn = __import__('sqlite3').connect("adaptive.db")
                    row  = conn.execute(
                        "SELECT id, email FROM students WHERE username = ?",
                        (username.strip(),)
                    ).fetchone()
                    conn.close()
                    if row:
                        sid, email = row
                        new_code   = resend_verification(sid)
                        send_verification_email(email, username.strip(), new_code)
                        st.session_state["pending_verification"] = True
                        st.session_state["pending_student_id"]   = sid
                        st.session_state["pending_email"]        = email
                        st.session_state["pending_username"]     = username.strip()
                        st.rerun()
                else:
                    st.error(result)

    # ── REGISTER ─────────────────────────────────────────────
    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_email    = st.text_input("Email address")
            new_password = st.text_input("Choose a password", type="password")
            confirm_pw   = st.text_input("Confirm password", type="password")
            submitted_r  = st.form_submit_button("Create Account", type="primary", use_container_width=True)

        if submitted_r:
            if new_password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                ok, result = register_student(new_username, new_email, new_password)
                if ok:
                    student_id, code = result
                    with st.spinner("Sending verification email..."):
                        mail_ok, mail_err = send_verification_email(
                            new_email.strip(), new_username.strip(), code
                        )
                    if not mail_ok:
                        st.warning(f"Could not send email ({mail_err}). Your code is: **{code}**")

                    st.session_state["pending_verification"] = True
                    st.session_state["pending_student_id"]   = student_id
                    st.session_state["pending_email"]        = new_email.strip()
                    st.session_state["pending_username"]     = new_username.strip()
                    st.rerun()
                else:
                    st.error(result)
