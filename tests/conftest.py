import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # entrypoints (tavily_maxer.py, webapp.py)
sys.path.insert(0, str(_ROOT / "lib"))  # supporting modules (quant, charts, portfolio, ...)


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch):
    """Every test gets dummy provider keys so pydantic field validation on
    TavilySearch/ChatNebius construction passes without touching the network.
    Tests that exercise the missing-key path explicitly delete these."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-nebius-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
