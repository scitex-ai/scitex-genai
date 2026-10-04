"""Offline controls for the genuine default single-POST HTTP carrier."""

import io
from email.message import Message
from urllib.error import HTTPError

from scitex_genai.decision import DecisionBudget, SystemOneTarget, _transport

_KEY = "test-only-credential-not-a-real-provider-key"
_TARGET = SystemOneTarget(
    "offline", "model", "account", "https://provider.invalid/v1/systemone"
)
_BUDGET = DecisionBudget(timeout_s=2, max_response_bytes=4096)


def test_default_transport_posts_once_with_bounded_read_and_no_env_proxy(monkeypatch):
    # Arrange
    calls = []
    handlers = []
    headers = Message()
    headers["X-Request-ID"] = "original-id"
    response = io.BytesIO(b"abcdef")
    response.code, response.headers = 200, headers

    class Opener:
        def open(self, request, *, timeout):
            calls.append(
                (
                    request.get_method(),
                    request.full_url,
                    request.data,
                    request.get_header("Authorization") == "Bearer " + _KEY,
                    timeout,
                )
            )
            return response

    def build_opener(*values):
        handlers.extend(values)
        return Opener()

    monkeypatch.setattr(_transport, "build_opener", build_opener)
    # Act
    result = _transport._HTTPSinglePost().post(
        target=_TARGET,
        body=b"original body",
        budget=DecisionBudget(2, 3),
        api_key=_KEY,
    )
    # Assert
    assert (
        calls,
        handlers[0].proxies,
        type(handlers[1]),
        result.body,
        result.body_complete,
        result.headers,
        response.closed,
    ) == (
        [("POST", _TARGET.endpoint, b"original body", True, 2)],
        {},
        _transport._NoRedirect,
        b"abcd",
        False,
        (("X-Request-ID", "original-id"),),
        True,
    )


def test_redirect_is_not_followed():
    # Arrange
    handler = _transport._NoRedirect()
    # Act
    redirected = handler.redirect_request(
        None, None, 302, "redirect", {}, "https://other.invalid/"
    )
    # Assert
    assert redirected is None


def test_default_http_error_body_is_retained_and_closed(monkeypatch):
    # Arrange
    calls = []
    body = io.BytesIO(b"original unauthorized body")
    error = HTTPError(_TARGET.endpoint, 401, "Unauthorized", Message(), body)

    class Opener:
        def open(self, request, *, timeout):
            calls.append(request.get_method())
            raise error

    monkeypatch.setattr(_transport, "build_opener", lambda *args: Opener())
    # Act
    result = _transport._HTTPSinglePost().post(
        target=_TARGET,
        body=b"request",
        budget=_BUDGET,
        api_key=_KEY,
    )
    # Assert
    assert (calls, result.status, result.body, result.body_complete, body.closed) == (
        ["POST"],
        401,
        b"original unauthorized body",
        True,
        True,
    )
