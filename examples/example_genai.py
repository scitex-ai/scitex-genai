#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal scitex-genai LLM completion example.

Run (requires an OpenAI API key in OPENAI_API_KEY):
    python examples/example_genai.py

Skips gracefully when the provider SDK or API key is unavailable.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping live call.", file=sys.stderr)
        return 0

    try:
        from scitex_genai import GenAI
    except ImportError as e:
        print(f"scitex-genai not installed: {e}", file=sys.stderr)
        return 1

    ai = GenAI(provider="openai", model="gpt-4o-mini")
    response = ai.complete("Summarise neural networks in one sentence.")
    print(response)
    summary = ai.get_cost_summary()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
