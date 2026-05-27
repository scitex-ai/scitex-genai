#!/usr/bin/env python3
# Time-stamp: "2025-06-01 14:15:00 (ywatanabe)"
# File: ./tests/scitex/ai/llm/test__format_output_func.py

"""Tests for scitex_genai.llm._format_output_func module.

Each test exercises one observable behaviour of ``format_output_func``
with Arrange / Act / Assert markers and a single assertion. The
function under test is a pure string transformer (markdown2 + a regex
for URL anchoring), so no test-doubles are needed — we feed real input
and inspect real output.
"""

import pytest

from scitex_genai.llm import format_output_func


class TestFormatOutputFunc:
    """Test suite for format_output_func function."""

    def test_plain_text_without_urls_is_returned_with_trailing_newline(self):
        """Plain text passes through markdown2 and gains a single trailing newline."""
        # Arrange
        text = "This is a simple text without any URLs."

        # Act
        result = format_output_func(text)

        # Assert
        assert result == text + "\n"

    def test_http_url_is_wrapped_in_anchor_tag(self):
        """HTTP URLs are wrapped in <a href="..."> anchor tags."""
        # Arrange
        text = "Visit http://example.com for more info"
        expected_anchor = '<a href="http://example.com">http://example.com</a>'

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_https_url_is_wrapped_in_anchor_tag(self):
        """HTTPS URLs are wrapped in <a href="..."> anchor tags."""
        # Arrange
        text = "Secure site: https://secure.example.com"
        expected_anchor = (
            '<a href="https://secure.example.com">https://secure.example.com</a>'
        )

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_doi_url_is_converted_to_https_doi_org_anchor(self):
        """``doi:`` URLs are normalised to https://doi.org/... before wrapping."""
        # Arrange
        text = "Research paper: doi:10.1234/example"
        expected_anchor = (
            '<a href="https://doi.org/10.1234/example">'
            "https://doi.org/10.1234/example</a>"
        )

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_multiple_urls_first_is_wrapped(self):
        """When two URLs appear, the first https URL is wrapped."""
        # Arrange
        text = "Visit https://site1.com and http://site2.com"
        expected_anchor = '<a href="https://site1.com">https://site1.com</a>'

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_multiple_urls_second_is_wrapped(self):
        """When two URLs appear, the second http URL is also wrapped."""
        # Arrange
        text = "Visit https://site1.com and http://site2.com"
        expected_anchor = '<a href="http://site2.com">http://site2.com</a>'

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_already_wrapped_urls_get_double_wrapped(self):
        """Pre-wrapped URLs are double-wrapped (current regex limitation, documented behaviour)."""
        # Arrange
        text = 'Already wrapped: <a href="https://example.com">https://example.com</a>'

        # Act
        result = format_output_func(text)

        # Assert
        assert result.count('<a href="https://example.com">') == 2

    def test_markdown_bold_is_converted_to_strong_tag(self):
        """``**bold**`` becomes ``<strong>bold</strong>``."""
        # Arrange
        text = "This is **bold** text"

        # Act
        result = format_output_func(text)

        # Assert
        assert "<strong>bold</strong>" in result

    def test_markdown_italic_is_converted_to_em_tag(self):
        """``*italic*`` becomes ``<em>italic</em>``."""
        # Arrange
        text = "This is *italic* text"

        # Act
        result = format_output_func(text)

        # Assert
        assert "<em>italic</em>" in result

    def test_markdown_inline_code_is_converted_to_code_tag(self):
        r"""``\`code\``` becomes ``<code>code</code>``."""
        # Arrange
        text = "Use `code` for inline code"

        # Act
        result = format_output_func(text)

        # Assert
        assert "<code>code</code>" in result

    def test_wrapping_paragraph_open_tag_is_stripped(self):
        """Simple paragraphs do not start with ``<p>``."""
        # Arrange
        text = "Simple paragraph"

        # Act
        result = format_output_func(text)

        # Assert
        assert not result.startswith("<p>")

    def test_wrapping_paragraph_close_tag_is_stripped(self):
        """Simple paragraphs do not end with ``</p>``."""
        # Arrange
        text = "Simple paragraph"

        # Act
        result = format_output_func(text)

        # Assert
        assert not result.endswith("</p>")

    def test_url_with_query_parameters_is_wrapped_with_amp_entity(self):
        """``&`` in URL query string is HTML-escaped to ``&amp;`` in the anchor."""
        # Arrange
        text = "Search: https://example.com/search?q=test&page=1"
        expected_anchor = (
            '<a href="https://example.com/search?q=test&amp;page=1">'
            "https://example.com/search?q=test&amp;page=1</a>"
        )

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_url_with_anchor_fragment_is_wrapped_intact(self):
        """URLs with ``#section`` fragments are wrapped without losing the fragment."""
        # Arrange
        text = "Section link: https://example.com/page#section"
        expected_anchor = (
            '<a href="https://example.com/page#section">'
            "https://example.com/page#section</a>"
        )

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_url_at_end_of_sentence_includes_trailing_period(self):
        """The regex stops at whitespace, so a trailing period is captured into the URL."""
        # Arrange
        text = "Visit https://example.com."
        expected_anchor = '<a href="https://example.com.">https://example.com.</a>'

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_url_in_parentheses_captures_trailing_paren(self):
        """The regex stops at whitespace, so a trailing ``)`` is captured into the URL."""
        # Arrange
        text = "(see https://example.com)"
        expected_substring = (
            '(see <a href="https://example.com)">https://example.com)</a>'
        )

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_substring in result

    def test_mixed_content_emits_strong_tag(self):
        """Mixed markdown+URL input still produces the ``<strong>`` tag for bold text."""
        # Arrange
        text = "Check **this site**: https://example.com for *more info*"

        # Act
        result = format_output_func(text)

        # Assert
        assert "<strong>this site</strong>" in result

    def test_mixed_content_emits_anchor_tag(self):
        """Mixed markdown+URL input still produces the URL anchor."""
        # Arrange
        text = "Check **this site**: https://example.com for *more info*"
        expected_anchor = '<a href="https://example.com">https://example.com</a>'

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_mixed_content_emits_em_tag(self):
        """Mixed markdown+URL input still produces the ``<em>`` tag for italic text."""
        # Arrange
        text = "Check **this site**: https://example.com for *more info*"

        # Act
        result = format_output_func(text)

        # Assert
        assert "<em>more info</em>" in result

    def test_empty_string_yields_single_newline(self):
        """markdown2 appends a newline even for empty input."""
        # Arrange
        text = ""

        # Act
        result = format_output_func(text)

        # Assert
        assert result == "\n"

    def test_whitespace_only_input_strips_to_empty_string(self):
        """Whitespace-only input round-trips to a string whose strip() is empty."""
        # Arrange
        text = "   \n\t   "

        # Act
        result = format_output_func(text)

        # Assert
        assert result.strip() == ""

    def test_multiline_text_wraps_first_url(self):
        """In multiline input, the first per-line URL is anchored."""
        # Arrange
        text = (
            "Line 1 with https://url1.com\n"
            "Line 2 with https://url2.com\n"
            "Line 3 without URL"
        )
        expected_anchor = '<a href="https://url1.com">https://url1.com</a>'

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_multiline_text_wraps_second_url(self):
        """In multiline input, the second per-line URL is also anchored."""
        # Arrange
        text = (
            "Line 1 with https://url1.com\n"
            "Line 2 with https://url2.com\n"
            "Line 3 without URL"
        )
        expected_anchor = '<a href="https://url2.com">https://url2.com</a>'

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

    def test_complex_doi_format_with_hyphens_and_digits_is_normalised(self):
        """A real-world DOI (with hyphens and digits) is rewritten as https://doi.org/..."""
        # Arrange
        text = "Article: doi:10.1038/s41586-020-2649-2"
        expected_anchor = (
            '<a href="https://doi.org/10.1038/s41586-020-2649-2">'
            "https://doi.org/10.1038/s41586-020-2649-2</a>"
        )

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_anchor in result

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
    def test_various_url_formats_are_wrapped_in_anchor_tags(
        self, url, expected_wrapped
    ):
        """http, https and doi URLs all produce the expected anchor tag."""
        # Arrange
        text = f"Check out {url}"

        # Act
        result = format_output_func(text)

        # Assert
        assert expected_wrapped in result

    def test_special_characters_in_text_return_nonempty_output(self):
        """Text containing HTML special characters does not crash and returns a non-empty string."""
        # Arrange
        text = "Special chars: < > & ' \""

        # Act
        result = format_output_func(text)

        # Assert
        assert isinstance(result, str) and len(result) > 0


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
