"""Tests for the dataflowr REST API.

Uses FastAPI's TestClient (backed by httpx) and mocks network calls so all
tests run offline.
"""

import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient

from dataflowr.api import app

client = TestClient(app)


# ── GET / ────────────────────────────────────────────────────────────────────

def test_root_returns_overview():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["num_modules"] == 32
    assert data["num_sessions"] == 9
    assert data["num_homeworks"] == 5
    assert "endpoints" in data


# ── GET /modules ─────────────────────────────────────────────────────────────

def test_list_modules_returns_all():
    r = client.get("/modules")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 32


def test_list_modules_session_filter():
    r = client.get("/modules?session=7")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["id"] == "12"


def test_list_modules_tag_filter():
    r = client.get("/modules?tag=attention")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()]
    assert "12" in ids


def test_list_modules_gpu_filter_true():
    r = client.get("/modules?gpu=true")
    assert r.status_code == 200
    data = r.json()
    assert all(m["requires_gpu"] for m in data)
    ids = [m["id"] for m in data]
    assert "18b" in ids
    assert "2a" not in ids


def test_list_modules_gpu_filter_false():
    r = client.get("/modules?gpu=false")
    assert r.status_code == 200
    data = r.json()
    assert all(not m["requires_gpu"] for m in data)


def test_list_modules_session_10_empty():
    r = client.get("/modules?session=10")
    assert r.status_code == 200
    assert r.json() == []


# ── GET /modules/{id} ────────────────────────────────────────────────────────

def test_get_module_found():
    r = client.get("/modules/12")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "12"
    assert data["session"] == 7
    assert len(data["notebooks"]) == 4


def test_get_module_case_insensitive():
    r = client.get("/modules/2A")
    assert r.status_code == 200
    assert r.json()["id"] == "2a"


def test_get_module_not_found():
    r = client.get("/modules/999")
    assert r.status_code == 404


def test_get_module_not_found_suggests_alternatives():
    r = client.get("/modules/flas")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "flash" in detail.lower()


# ── GET /modules/{id}/notebooks ──────────────────────────────────────────────

def test_get_module_notebooks():
    r = client.get("/modules/12/notebooks")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 4
    kinds = {nb["kind"] for nb in data}
    assert "practical" in kinds
    assert "solution" in kinds


def test_get_module_notebooks_kind_filter():
    r = client.get("/modules/12/notebooks?kind=practical")
    assert r.status_code == 200
    data = r.json()
    assert all(nb["kind"] == "practical" for nb in data)


def test_get_module_notebooks_not_found():
    r = client.get("/modules/999/notebooks")
    assert r.status_code == 404


# ── GET /modules/{id}/notebooks/{kind}/content ───────────────────────────────

def test_get_notebook_content():
    with mock.patch(
        "dataflowr.api.fetch_notebook_content",
        return_value="# Notebook content",
    ):
        r = client.get("/modules/12/notebooks/practical/content")
    assert r.status_code == 200
    assert "Notebook content" in r.text


def test_get_notebook_content_invalid_kind():
    r = client.get("/modules/12/notebooks/nonexistent/content")
    assert r.status_code == 404
    assert "nonexistent" in r.json()["detail"]


def test_get_notebook_content_module_not_found():
    r = client.get("/modules/999/notebooks/practical/content")
    assert r.status_code == 404


# ── GET /sessions ─────────────────────────────────────────────────────────────

def test_list_sessions_returns_all():
    r = client.get("/sessions")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 9


def test_list_sessions_includes_sessions_1_to_9():
    r = client.get("/sessions")
    numbers = [s["number"] for s in r.json()]
    assert set(numbers) == set(range(1, 10))


# ── GET /sessions/{n} ────────────────────────────────────────────────────────

def test_get_session_found():
    r = client.get("/sessions/7")
    assert r.status_code == 200
    data = r.json()
    assert data["number"] == 7
    assert "12" in data["modules"]
    assert any(m["id"] == "12" for m in data["modules_detail"])


def test_get_session_10_not_found():
    r = client.get("/sessions/10")
    assert r.status_code == 404


def test_get_session_not_found():
    r = client.get("/sessions/99")
    assert r.status_code == 404


# ── GET /homeworks ────────────────────────────────────────────────────────────

def test_list_homeworks():
    r = client.get("/homeworks")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 5


# ── GET /homeworks/{id} ───────────────────────────────────────────────────────

def test_get_homework_found():
    r = client.get("/homeworks/1")
    assert r.status_code == 200
    assert r.json()["title"] == "MLP from Scratch"


def test_get_homework_not_found():
    r = client.get("/homeworks/99")
    assert r.status_code == 404


# ── GET /search ───────────────────────────────────────────────────────────────

def test_search_returns_results():
    r = client.get("/search?q=attention")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] >= 1
    assert any(m["id"] == "12" for m in data["results"])


def test_search_no_results():
    r = client.get("/search?q=xyzxyzxyzthisdoesnotexist")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["results"] == []


def test_search_missing_query_param():
    r = client.get("/search")
    assert r.status_code == 422   # FastAPI validation error


# ── GET /modules/{id}/slides ──────────────────────────────────────────────────

def test_get_slides_no_slides_for_module():
    r = client.get("/modules/12/slides")
    assert r.status_code == 404
    assert "No slides" in r.json()["detail"]


def test_get_slides_content():
    with mock.patch(
        "dataflowr.api.fetch_slide_content",
        return_value="# Slide content",
    ):
        r = client.get("/modules/3/slides")
    assert r.status_code == 200
    assert "Slide content" in r.text


# ── GET /modules/{id}/page ───────────────────────────────────────────────────

def test_get_page_content():
    with mock.patch(
        "dataflowr.api.fetch_module_markdown",
        return_value="# Module page",
    ):
        r = client.get("/modules/12/page")
    assert r.status_code == 200
    assert "Module page" in r.text
