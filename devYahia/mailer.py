import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── SMTP CONFIG ──────────────────────────────────────────────
SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587
SENDER_EMAIL  = "yahiasaqer@gmail.com"
SENDER_PASSWORD = "zjrc uncz tojk ppho"   # App Password


def send_verification_email(recipient_email: str, username: str, code: str) -> tuple:
    """
    Send a verification code email.
    Returns (True, None) on success or (False, error_string) on failure.
    """
    subject = "Your Adaptive Learning verification code"

    body = f"""Hello {username},

Your verification code is:

    {code}

This code expires in 15 minutes.

If you did not create an account, you can ignore this email.

— Adaptive Learning System
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = recipient_email
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        return True, None
    except smtplib.SMTPAuthenticationError:
        return False, "Email authentication failed. Check SMTP credentials."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"
