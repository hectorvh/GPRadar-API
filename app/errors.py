"""API errors: one exception type, rendered as {"error": {"code", "message"}}."""

from __future__ import annotations

STATUS = {
    "bad_request": 400,
    "not_found": 404,
    "method_not_allowed": 405,
    "too_large": 413,
    "unsupported_type": 415,
    "invalid_metadata": 422,
    "rate_limited": 429,
    "internal_error": 500,
    "busy": 503,
    "timeout": 504,
}


class ApiError(Exception):
    """An error with a stable code and a message safe to show to the client."""

    def __init__(self, code, message, headers=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = STATUS[code]
        self.headers = headers or {}

    def body(self):
        return {"error": {"code": self.code, "message": self.message}}
