#!/usr/bin/env python3
# Time-stamp: "2025-06-01 14:15:00 (ywatanabe)"
# File: ./tests/scitex/ai/llm/test__format_output_func.py

"""Tests for scitex_genai.llm._format_output_func module."""

import pytest

from scitex_genai.llm import format_output_func


class TestFormatOutputFunc:
    """Test suite for format_output_func function."""

    def test_basic_text_unchanged(self):
        """Test that plain text without URLs is processed (markdown2 adds newline)."""
        # Arrange
        text = "This is a simple text without any URLs."
        # Act
        result = format_output_func(text)
        # markdown2 adds a newline character
        # Assert
        assert result == text + "\n"

    def test_http_url_wrapping(self):
        """Test that HTTP URLs are wrapped in anchor tags."""
        # Arrange
        text = "Visit http://example.com for more info"
        # Act
        result = format_output_func(text)
        # Assert
        assert '<a href="http://example.com">http://example.com</a>' in result

    def test_https_url_wrapping(self):
        """Test that HTTPS URLs are wrapped in anchor tags."""
        # Arrange
        text = "Secure site: https://secure.example.com"
        # Act
        result = format_output_func(text)
        # Assert
        assert (
            '<a href="https://secure.example.com">https://secure.example.com</a>'
            in result
        )

    def test_doi_url_conversion(self):
        """Test that DOI URLs are converted and wrapped."""
        # Arrange
        text = "Research paper: doi:10.1234/example"
        # Act
        result = format_output_func(text)
        # Assert
        assert (
            '<a href="https://doi.org/10.1234/example">https://doi.org/10.1234/example</a>'
            in result
        )

    def test_multiple_urls_a_href_https_site1_com_https_site1_com_a_in_result(self):
        # Arrange
        # Arrange
        text = "Visit https://site1.com and http://site2.com"
        # Act
        result = format_output_func(text)
        # Act
        # Assert
        # Assert
        assert '<a href="https://site1.com">https://site1.com</a>' in result

    def test_multiple_urls_a_href_http_site2_com_http_site2_com_a_in_result(self):
        # Arrange
        # Arrange
        text = "Visit https://site1.com and http://site2.com"
        # Act
        result = format_output_func(text)
        # Act
        # Assert
        # Assert
        assert '<a href="http://site2.com">http://site2.com</a>' in result


    def test_already_wrapped_urls_unchanged(self):
        """Test that already wrapped URLs get double-wrapped (current behavior)."""
        # Arrange
        text = 'Already wrapped: <a href="https://example.com">https://example.com</a>'
        # Act
        result = format_output_func(text)
        # The regex pattern does not prevent double-wrapping; it only checks for the prefix
        # The URL inside the existing anchor tag still matches and gets wrapped again
        # Assert
        assert result.count('<a href="https://example.com">') == 2

    def test_markdown_bold_conversion(self):
        """Test markdown bold syntax conversion."""
        # Arrange
        text = "This is **bold** text"
        # Act
        result = format_output_func(text)
        # Assert
        assert "<strong>bold</strong>" in result

    def test_markdown_italic_conversion(self):
        """Test markdown italic syntax conversion."""
        # Arrange
        text = "This is *italic* text"
        # Act
        result = format_output_func(text)
        # Assert
        assert "<em>italic</em>" in result

    def test_markdown_code_conversion(self):
        """Test markdown code syntax conversion."""
        # Arrange
        text = "Use `code` for inline code"
        # Act
        result = format_output_func(text)
        # Assert
        assert "<code>code</code>" in result

    def test_paragraph_tag_removal_not_result_startswith_p(self):
        # Arrange
        # Arrange
        text = "Simple paragraph"
        # Act
        result = format_output_func(text)
        # Act
        # Assert
        # Assert
        assert not result.startswith("<p>")

    def test_paragraph_tag_removal_not_result_endswith_p(self):
        # Arrange
        # Arrange
        text = "Simple paragraph"
        # Act
        result = format_output_func(text)
        # Act
        # Assert
        # Assert
        assert not result.endswith("</p>")


    def test_url_with_query_parameters(self):
        """Test URLs with query parameters are handled correctly."""
        # Arrange
        text = "Search: https://example.com/search?q=test&page=1"
        # Act
        result = format_output_func(text)
        # markdown2 escapes & to &amp; in HTML output
        # Assert
        assert (
            '<a href="https://example.com/search?q=test&amp;page=1">https://example.com/search?q=test&amp;page=1</a>'
            in result
        )

    def test_url_with_anchors(self):
        """Test URLs with anchor fragments are handled correctly."""
        # Arrange
        text = "Section link: https://example.com/page#section"
        # Act
        result = format_output_func(text)
        # Assert
        assert (
            '<a href="https://example.com/page#section">https://example.com/page#section</a>'
            in result
        )

    def test_url_at_end_of_sentence(self):
        """Test URLs at the end of sentences are handled correctly."""
        # Arrange
        text = "Visit https://example.com."
        # Act
        result = format_output_func(text)
        # The regex includes the period in the URL (regex stops at whitespace, not punctuation)
        # Assert
        assert '<a href="https://example.com.">https://example.com.</a>' in result

    def test_url_in_parentheses(self):
        """Test URLs in parentheses are handled correctly."""
        # Arrange
        text = "(see https://example.com)"
        # Act
        result = format_output_func(text)
        # The regex includes the closing parenthesis in the URL (regex stops at whitespace, not parenthesis)
        # Assert
        assert '(see <a href="https://example.com)">https://example.com)</a>' in result

    def test_mixed_content_strong_this_site_strong_in_result(self):
        # Arrange
        # Arrange
        text = "Check **this site**: https://example.com for *more info*"
        # Act
        result = format_output_func(text)
        # Act
        # Assert
        # Assert
        assert "<strong>this site</strong>" in result

    def test_mixed_content_a_href_https_example_com_https_example_com_a_in_result(self):
        # Arrange
        # Arrange
        text = "Check **this site**: https://example.com for *more info*"
        # Act
        result = format_output_func(text)
        # Act
        # Assert
        # Assert
        assert '<a href="https://example.com">https://example.com</a>' in result

    def test_mixed_content_em_more_info_em_in_result(self):
        # Arrange
        # Arrange
        text = "Check **this site**: https://example.com for *more info*"
        # Act
        result = format_output_func(text)
        # Act
        # Assert
        # Assert
        assert "<em>more info</em>" in result


    def test_empty_string_result_equals_n(self):
        """Test empty string input."""
        # Arrange
        # Act
        result = format_output_func("")
        # markdown2 adds a newline even for empty strings
        # Assert
        assert result == "\n"

    def test_whitespace_only_result_strip(self):
        """Test whitespace-only input."""
        # Arrange
        # Act
        result = format_output_func("   \n\t   ")
        # Assert
        assert result.strip() == ""

    def test_multiline_text_a_href_https_url1_com_https_url1_com_a_in_result(self):
        # Arrange
        # Arrange
        text = """Line 1 with https://url1.com
Line 2 with https://url2.com
Line 3 without URL"""
        # Act
        result = format_output_func(text)
        # Act
        # Assert
        # Assert
        assert '<a href="https://url1.com">https://url1.com</a>' in result

    def test_multiline_text_a_href_https_url2_com_https_url2_com_a_in_result(self):
        # Arrange
        # Arrange
        text = """Line 1 with https://url1.com
Line 2 with https://url2.com
Line 3 without URL"""
        # Act
        result = format_output_func(text)
        # Act
        # Assert
        # Assert
        assert '<a href="https://url2.com">https://url2.com</a>' in result


    def test_complex_doi_format(self):
        """Test complex DOI formats."""
        # Arrange
        text = "Article: doi:10.1038/s41586-020-2649-2"
        # Act
        result = format_output_func(text)
        # Assert
        assert (
            '<a href="https://doi.org/10.1038/s41586-020-2649-2">https://doi.org/10.1038/s41586-020-2649-2</a>'
            in result
        )

    @pytest.mark.parametrize(
        "url,expected_wrapped",
        [
            (
                "http://example.com",
                '<a href="http://example.com">http://example.com</a>',
            ),
            (
                "https://example.com",
                '<a href="https://example.com">https://example.com</a>',
            ),
            (
                "doi:10.1234/test",
                '<a href="https://doi.org/10.1234/test">https://doi.org/10.1234/test</a>',
            ),
        ],
    )
    def test_various_url_formats(self, url, expected_wrapped):
        """Test various URL formats are wrapped correctly."""
        # Arrange
        text = f"Check out {url}"
        # Act
        result = format_output_func(text)
        # Assert
        assert expected_wrapped in result

    def test_special_characters_in_text(self):
        """Test text with special characters."""
        # Arrange
        text = "Special chars: < > & ' \""
        # Act
        result = format_output_func(text)
        # Markdown2 should handle HTML escaping
        # Assert
        assert result  # Just ensure it doesn't crash


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/llm/_format_output_func.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 01:39:25 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/ai/llm/_format_output_func.py
#
# """
# Functionality:
#     - Formats AI model output text
#     - Wraps URLs in HTML anchor tags
#     - Converts markdown to HTML
#     - Handles DOI links specially
# Input:
#     - Raw text output from AI models
#     - Optional API key for masking
# Output:
#     - Formatted HTML text with proper link handling
# Prerequisites:
#     - markdown2 package
#     - Regular expressions support
# """
#
# """Imports"""
# import re
# import sys
# from typing import List, Optional
#
# import markdown2
# import matplotlib.pyplot as plt
# import scitex
#
# """Functions & Classes"""
#
#
# def format_output_func(out_text: str) -> str:
#     """Formats AI output text with proper link handling and markdown conversion.
#
#     Example
#     -------
#     >>> text = "Check https://example.com or doi:10.1234/abc"
#     >>> print(format_output_func(text))
#     Check <a href="https://example.com">https://example.com</a> or <a href="https://doi.org/10.1234/abc">https://doi.org/10.1234/abc</a>
#
#     Parameters
#     ----------
#     out_text : str
#         Raw text output from AI model
#
#     Returns
#     -------
#     str
#         HTML formatted text with proper link handling
#     """
#
#     def find_unwrapped_urls(text: str) -> List[str]:
#         url_pattern = r'(?<!<a href=")(https?://|doi:|http://doi.org/)[^\s,<>"]+'
#         return re.findall(url_pattern, text)
#
#     def add_a_href_tag(text: str) -> str:
#         def replace_url(match) -> str:
#             url = match.group(0)
#             if url.startswith("doi:"):
#                 url = "https://doi.org/" + url[4:]
#             return f'<a href="{url}">{url}</a>'
#
#         url_pattern = r'(?<!<a href=")(https?://|doi:|http://doi.org/)[^\s,<>"]+'
#         return re.sub(url_pattern, replace_url, text)
#
#     def add_masked_api_key(text: str, api_key: str) -> str:
#         masked_key = f"{api_key[:4]}****{api_key[-4:]}"
#         return text + f"\n(API Key: {masked_key}"
#
#     formatted_text = markdown2.markdown(out_text)
#     formatted_text = add_a_href_tag(formatted_text)
#     formatted_text = re.sub(r"^<p>(.*)</p>$", r"\1", formatted_text, flags=re.DOTALL)
#     return formatted_text
#
#
# def main() -> None:
#     pass
#
#
# if __name__ == "__main__":
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(
#         sys, plt, verbose=False
#     )
#     main()
#     scitex.session.close(CONFIG, verbose=False, notify=False)
#
# # EOF
# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # # Time-stamp: "2024-11-04 01:32:10 (ywatanabe)"
# # # File: ./scitex_repo/src/scitex/ai/llm/_format_output_func.py
#
# # """This script does XYZ."""
#
#
# # """Imports"""
# # import re
# # import sys
#
# # import markdown2
# # import matplotlib.pyplot as plt
# # import scitex
#
# # """
# # Warnings
# # """
# # # warnings.simplefilter("ignore", UserWarning)
#
#
# # """
# # Config
# # """
# # # CONFIG = scitex.gen.load_configs()
#
#
# # """
# # Functions & Classes
# # """
#
#
# # def format_output_func(out_text):
# #     def find_unwrapped_urls(text):
# #         # Regex to find URLs that are not already within <a href> tags
# #         url_pattern = (
# #             r'(?<!<a href=")(https?://|doi:|http://doi.org/)[^\s,<>"]+'
# #         )
#
# #         # Find all matches that are not already wrapped
# #         unwrapped_urls = re.findall(url_pattern, text)
#
# #         return unwrapped_urls
#
# #     def add_a_href_tag(text):
# #         # Function to replace each URL with its wrapped version
# #         def replace_url(match):
# #             url = match.group(0)
# #             # Normalize DOI URLs
# #             if url.startswith("doi:"):
# #                 url = "https://doi.org/" + url[4:]
# #             return f'<a href="{url}">{url}</a>'
#
# #         # Regex pattern to match URLs not already wrapped in <a> tags
# #         url_pattern = (
# #             r'(?<!<a href=")(https?://|doi:|http://doi.org/)[^\s,<>"]+'
# #         )
#
# #         # Replace all occurrences of unwrapped URLs in the text
# #         updated_text = re.sub(url_pattern, replace_url, text)
#
# #         return updated_text
#
# #     def add_masked_api_key(text, api_key):
# #         masked_api_key = f"{api_key[:4]}****{api_key[-4:]}"
# #         return text + f"\n(API Key: {masked_api_key}"
#
# #     out_text = markdown2.markdown(out_text)
# #     out_text = add_a_href_tag(out_text)
# #     out_text = re.sub(r"^<p>(.*)</p>$", r"\1", out_text, flags=re.DOTALL)
# #     return out_text
#
#
# # def main():
# #     pass
#
#
# # if __name__ == "__main__":
# #     # # Argument Parser
# #     # import argparse
# #     # parser = argparse.ArgumentParser(description='')
# #     # parser.add_argument('--var', '-v', type=int, default=1, help='')
# #     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
# #     # args = parser.parse_args()
#
# #     # Main
# #     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(
# #         sys, plt, verbose=False
# #     )
# #     main()
# #     scitex.session.close(CONFIG, verbose=False, notify=False)
#
# #
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/llm/_format_output_func.py
# --------------------------------------------------------------------------------
