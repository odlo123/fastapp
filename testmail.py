import os
import httpx

from fastapi import APIRouter

router = APIRouter()

BREVO_API_URL ="https://api.brevo.com/v3/smtp/email"

BREVO_API_KEY = "xsmtpsib-48acbef24463246121e533bd0de661f71923c8390ead31912c6bff9d625530bf-x7M5a3hylV8n7aO7"
BREVO_SENDER_EMAIL = "hahishapp@gmail.com"
BREVO_SENDER_NAME = "HahishApp"


TEST_RECEIVER = "urugendoturimo@gmail.com"


async def send_email(
    to: str,
    subject: str,
    html_content: str
):

    if not BREVO_API_KEY:
        print("BREVO_API_KEY is missing")
        return False

    if not BREVO_SENDER_EMAIL:
        print("BREVO_SENDER_EMAIL is missing")
        return False

    payload = {
        "sender": {
            "name": BREVO_SENDER_NAME,
            "email": BREVO_SENDER_EMAIL
        },
        "to": [
            {
                "email": to
            }
        ],
        "subject": subject,
        "htmlContent": html_content
    }

    headers = {
        "accept": "application/json",
        "api-key": BREVO_API_KEY,
        "content-type": "application/json"
    }

    try:

        print("=" * 50)
        print("SENDING EMAIL THROUGH BREVO")
        print("=" * 50)

        async with httpx.AsyncClient(
            timeout=30
        ) as client:

            response = await client.post(
                BREVO_API_URL,
                json=payload,
                headers=headers
            )

        print("BREVO STATUS:", response.status_code)
        print("BREVO RESPONSE:", response.text)

        response.raise_for_status()

        print("EMAIL SENT SUCCESSFULLY")

        return True

    except Exception as e:

        print("=" * 50)
        print("BREVO EMAIL ERROR")
        print("=" * 50)

        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", str(e))

        return False


@router.get("/test-brevo")
async def test_brevo():

    result = await send_email(
        to=TEST_RECEIVER,
        subject="HahishApp Brevo Test",
        html_content="""
        <html>
        <body>

            <h2 style="color:#2E7D32">
                HahishApp
            </h2>

            <p>
                This is a test email sent using
                the Brevo HTTPS API.
            </p>

            <p>
                If you received this message,
                HahishApp email delivery is working.
            </p>

        </body>
        </html>
        """
    )

    return {
        "success": result
    }
