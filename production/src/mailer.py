import smtplib
import sys
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, os.path.dirname(__file__))
from config import SMTP_HOST, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD


def send_verification_email(recipient_email: str, username: str, code: str) -> tuple:
    """
    Send a verification code email via Brevo SMTP.
    Returns (True, None) on success or (False, error_string) on failure.
    """
    body = f"""Hello {username},

Your verification code is:

    {code}

This code expires in 15 minutes.

If you did not create an account, you can ignore this email.

— PathOptLearn
"""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your PathOptLearn verification code"
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
        return False, "Email authentication failed. Check SMTP credentials in .env."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"
