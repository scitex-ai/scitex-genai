import pytest

from scitex_genai.availability import provider_route


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@provider.invalid/v1",
        "https://provider.invalid/v1?token=secret",
        "file:///tmp/provider",
    ],
)
def test_endpoint_rejects_embedded_credentials(url):
    # Arrange
    provider, model = "custom", "model"
    # Act
    # Assert
    with pytest.raises(ValueError, match="without embedded credentials"):
        provider_route(
            provider, model, endpoint_url=url, protocol="openai-chat-completions"
        )
