"""Public access (PUBLIC_ROLE), tolerant API_KEYS parsing and the server-provided Grafana link."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import auth.middleware as am
    monkeypatch.setenv("API_KEYS", ' "owner-key : Admin" ')
    am._key_store.cache_clear()
    from api.main import app
    yield TestClient(app)
    am._key_store.cache_clear()


def test_without_public_role_a_key_is_required(client, monkeypatch):
    monkeypatch.delenv("PUBLIC_ROLE", raising=False)
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "owner-key"}).status_code == 200


def test_public_reader_can_read_but_not_upload(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_ROLE", "reader")
    assert client.get("/documents").status_code == 200                      # no key
    assert client.get("/documents", headers={"X-API-Key": "stale-key"}).status_code == 200
    r = client.post("/ingest", files={"file": ("a.txt", b"hello world", "text/plain")})
    assert r.status_code == 403 and "writer" in r.json()["detail"]
    assert client.get("/config/public").json()["public_access"] is True


def test_invalid_public_role_is_ignored(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_ROLE", "root")
    assert client.get("/documents").status_code == 401


def test_bare_key_counts_as_admin(monkeypatch):
    import auth.middleware as am
    monkeypatch.setenv("API_KEYS", "just-a-key")
    assert am._load_keys() == {"just-a-key": "admin"}


def test_grafana_url_is_published(client, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "grafana_url", "https://grafana.example.com/")
    assert client.get("/config/public").json()["grafana_url"] == "https://grafana.example.com"
