
import hmac
import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()


def get_current_role(
    x_api_key: str = Header(default="", alias="X-API-Key"),
) -> str:
    public_key = os.getenv("PUBLIC_API_KEY", "")
    staff_key = os.getenv("STAFF_API_KEY", "")

    if not public_key or not staff_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if hmac.compare_digest(x_api_key, staff_key):
        return "staff"

    if hmac.compare_digest(x_api_key, public_key):
        return "public"

    raise HTTPException(
        status_code=401,
        detail="Missing or invalid API key.",
    )
