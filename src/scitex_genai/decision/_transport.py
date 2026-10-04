"""Default single-POST HTTPS transport without redirects, proxies or retries."""

from __future__ import annotations

from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from ._types import DecisionBudget, SystemOneResponse, SystemOneTarget


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _HTTPSinglePost:
    def __init__(self, *, opener_factory=None):
        # An explicit private collaborator permits offline carrier controls
        # without replacing process-global urllib functions.
        self._opener_factory = opener_factory or build_opener

    def post(
        self,
        *,
        target: SystemOneTarget,
        body: bytes,
        budget: DecisionBudget,
        api_key: str,
    ) -> SystemOneResponse:
        request = Request(
            target.endpoint,
            data=body,
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        opener = self._opener_factory(ProxyHandler({}), _NoRedirect())
        try:
            response = opener.open(request, timeout=budget.timeout_s)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read(budget.max_response_bytes + 1)
            return SystemOneResponse(
                status=response.code,
                body=raw,
                headers=tuple(response.headers.items()),
                body_complete=len(raw) <= budget.max_response_bytes,
            )
