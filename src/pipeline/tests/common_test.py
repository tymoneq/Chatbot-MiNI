import logging

import pytest

from pipeline.common import get_llm_client

## get config test


def test_get_llm_client_with_key(monkeypatch):
    """Test function-level env var extraction."""
    # 1. Set the mock environment variable
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-openrouter-key")

    # 2. Call the function (no reload needed since os.getenv is inside)
    client = get_llm_client()

    # 3. Assert the client got the right key
    assert client.api_key == "mock-openrouter-key"
    assert client.base_url == "https://openrouter.ai/api/v1/"


@pytest.mark.usefixtures("caplog")
def test_get_llm_client_missing_key_warning(monkeypatch, caplog):
    """Test the warning when the key is missing."""

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-fallback-key")

    with caplog.at_level(logging.WARNING):
        client = get_llm_client()

    assert "OPENROUTER_API_KEY not found in environment variables." in caplog.text
    # The client initializes using the fallback key
    assert client.api_key == "dummy-fallback-key"
