"""Finite credential-echo admission for original Systemone responses."""

import json


def _echoes_credential(response, api_key):
    """Withhold exact credential echoes; never redact an original response.

    Check raw UTF-8 bytes, header fields and decoded JSON keys/strings, including
    duplicates. This is not a universal secret or arbitrary-encoding detector.
    """
    if api_key.encode("utf-8") in response.body or any(
        api_key in value for pair in response.headers for value in pair
    ):
        return True
    try:
        pending = [json.loads(response.body.decode("utf-8"), object_pairs_hook=list)]
    except (ValueError, RecursionError):
        return False
    while pending:
        value = pending.pop()
        if type(value) is str and api_key in value:
            return True
        if type(value) in (list, tuple):
            pending.extend(value)
    return False
