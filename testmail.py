import smtplib
import socket
import traceback
from email.message import EmailMessage
from fastapi import APIRouter

router = APIRouter()

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

SMTP_USERNAME = "hahishapp@gmail.com"
SMTP_PASSWORD = "oschbpcxwcgmrfzv"

TEST_RECEIVER = "urugendoturimo@gmail.com"


async def send_test_email():

    print("=" * 50)
    print("1. TESTING DNS")
    print("=" * 50)

    try:
        ip = socket.gethostbyname(SMTP_HOST)
        print(f"DNS OK: {SMTP_HOST} -> {ip}")
    except Exception as e:
        print("DNS FAILED:", repr(e))
        traceback.print_exc()
        return False

    print("=" * 50)
    print("2. TESTING TCP CONNECTION - PORT 587")
    print("=" * 50)

    try:
        sock = socket.create_connection(
            (SMTP_HOST, SMTP_PORT),
            timeout=15
        )

        print(f"TCP CONNECTION OK: {SMTP_HOST}:{SMTP_PORT}")

        sock.close()

    except Exception as e:
        print(f"TCP CONNECTION FAILED: {SMTP_HOST}:{SMTP_PORT}")
        print("ERROR:", repr(e))
        traceback.print_exc()
        return False

    print("=" * 50)
    print("3. TESTING SMTP + TLS")
    print("=" * 50)

    try:

        message = EmailMessage()

        message["Subject"] = "HahishApp SMTP Test"
        message["From"] = SMTP_USERNAME
        message["To"] = TEST_RECEIVER

        message.set_content(
            """
Hello,

This is a test email from HahishApp.

If you received this email, Gmail SMTP is working from Render.

HahishApp
"""
        )

        with smtplib.SMTP(
            SMTP_HOST,
            SMTP_PORT,
            timeout=30
        ) as smtp:

            print("SMTP connection established")

            smtp.set_debuglevel(1)

            print("Starting TLS...")

            smtp.starttls()

            print("TLS OK")

            print("Logging into Gmail...")

            smtp.login(
                SMTP_USERNAME,
                SMTP_PASSWORD
            )

            print("GMAIL LOGIN OK")

            print("Sending email...")

            smtp.send_message(message)

            print("EMAIL SENT SUCCESSFULLY")

        return True

    except Exception as e:

        print("=" * 50)
        print("SMTP TEST FAILED")
        print("=" * 50)

        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", repr(e))

        traceback.print_exc()

        return False


@router.get("/test-email")
async def test_email():

    result = await send_test_email()

    return {
        "success": result
    }
@router.get("/test-email-ports")
async def test_email_ports():

    results = {}

    for port in [465, 587]:

        try:
            print(f"Testing smtp.gmail.com:{port}")

            sock = socket.create_connection(
                ("smtp.gmail.com", port),
                timeout=15
            )

            sock.close()

            print(f"PORT {port}: CONNECTED")

            results[str(port)] = "CONNECTED"

        except Exception as e:

            print(f"PORT {port}: FAILED")
            print("ERROR:", repr(e))

            results[str(port)] = f"FAILED: {repr(e)}"

    return results
