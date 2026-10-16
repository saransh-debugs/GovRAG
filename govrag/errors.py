"""Error types. Each carries a fixed snake_case code that is safe to log and audit.

Messages never include user input or document text, so an exception can't leak PII
into logs, spans, or HTTP responses.
"""

from __future__ import annotations


class GovRagError(Exception):
    status: int = 500
    code: str = "internal"

    def __init__(self, code: str | None = None, *, retry_after_s: float | None = None) -> None:
        self.code = code or self.code
        self.retry_after_s = retry_after_s
        super().__init__(self.code)


class Unauthorized(GovRagError):
    status, code = 401, "unauthorized"


class Forbidden(GovRagError):
    status, code = 403, "forbidden"


class BadRequest(GovRagError):
    status, code = 422, "bad_request"


class NotFound(GovRagError):
    status, code = 404, "not_found"


class Conflict(GovRagError):
    """Optimistic concurrency: someone else changed it first. Re-read and retry."""

    status, code = 409, "conflict"


class Overloaded(GovRagError):
    """Load shedding. The client should retry after retry_after_s."""

    status, code = 503, "overloaded"

    def __init__(self, code: str | None = None, *, retry_after_s: float = 1.0) -> None:
        super().__init__(code, retry_after_s=retry_after_s)


class Unavailable(GovRagError):
    """A dependency (DB, LLM, audit) is down. We fail closed instead of degrading security."""

    status, code = 503, "unavailable"


class IntegrityError(GovRagError):
    """Something that must never happen happened (e.g. an HMAC token collision)."""

    status, code = 500, "integrity"
