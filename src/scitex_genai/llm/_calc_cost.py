#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 01:37:36 (ywatanabe)"
# File: ./scitex_repo/src/scitex/ai/_gen_ai/_calc_cost.py

"""
Functionality:
    - Calculates usage costs for AI model API calls
    - Handles token-based pricing for different models
Input:
    - Model name
    - Number of input and output tokens used
Output:
    - Total cost in USD based on token usage
Prerequisites:
    - MODELS parameter dictionary with pricing information
    - pandas package
"""

from typing import Any, Union

import pandas as pd

from ._PARAMS import MODELS


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculates API usage cost based on token count.

    Example
    -------
    >>> cost = calc_cost("gpt-4", 100, 50)
    >>> print(f"${cost:.4f}")
    $0.0030

    Parameters
    ----------
    model : str
        Name of the AI model
    input_tokens : int
        Number of input tokens used
    output_tokens : int
        Number of output tokens used

    Returns
    -------
    float
        Total cost in USD. Returns 0.0 for unknown models (e.g. self-hosted
        or OpenAI-compatible endpoints that have no pricing row).
    """
    models_df = pd.DataFrame(MODELS)
    indi = models_df["name"] == model

    if not indi.any():
        # Self-hosted / unknown models have no pricing data; treat as free.
        return 0.0

    costs = models_df[["input_cost", "output_cost"]][indi]
    cost = (
        input_tokens * costs["input_cost"] + output_tokens * costs["output_cost"]
    ) / 1_000_000

    return cost.iloc[0]


# def calc_cost(model, input_tokens, output_tokens):
#     indi = MODELS["name"] == model
#     costs = MODELS[["input_cost", "output_cost"]][indi]
#     cost = (
#         input_tokens * costs["input_cost"]
#         + output_tokens * costs["output_cost"]
#     ) / 1_000_000
#     return cost.iloc[0]


# EOF
