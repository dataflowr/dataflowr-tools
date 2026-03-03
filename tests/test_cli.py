"""Tests for the dataflowr CLI.

All tests use typer.testing.CliRunner and mock network calls, so they run
offline.
"""

import json
import unittest.mock as mock

import pytest
from typer.testing import CliRunner

from dataflowr.cli import app

runner = CliRunner()


# ── info ─────────────────────────────────────────────────────────────────────

def test_info_shows_title():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Deep Learning Do It Yourself" in result.output


def test_info_shows_counts():
    result = runner.invoke(app, ["info"])
    assert "34" in result.output   # modules
    assert "9" in result.output    # sessions
    assert "6" in result.output    # homeworks


def test_info_json():
    result = runner.invoke(app, ["info", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["num_modules"] == 34
    assert data["num_sessions"] == 9
    assert data["num_homeworks"] == 6


# ── modules list ─────────────────────────────────────────────────────────────

def test_modules_list_shows_all():
    result = runner.invoke(app, ["modules", "list"])
    assert result.exit_code == 0
    assert "12" in result.output
    assert "flash" in result.output


def test_modules_list_session_filter():
    result = runner.invoke(app, ["modules", "list", "--session", "7"])
    assert result.exit_code == 0
    assert "12" in result.output
    # Session 7 only has module 12 — other sessions' modules should not appear
    assert "18b" not in result.output


def test_modules_list_tag_filter():
    result = runner.invoke(app, ["modules", "list", "--tag", "attention"])
    assert result.exit_code == 0
    assert "12" in result.output


def test_modules_list_gpu_filter():
    result = runner.invoke(app, ["modules", "list", "--gpu"])
    assert result.exit_code == 0
    assert "18b" in result.output
    assert "2a" not in result.output


def test_modules_list_json_is_valid():
    result = runner.invoke(app, ["modules", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 34
    assert all("id" in m for m in data)


def test_modules_list_session_10_empty():
    result = runner.invoke(app, ["modules", "list", "--session", "10"])
    assert result.exit_code == 0
    assert "flash" not in result.output


# ── module get ───────────────────────────────────────────────────────────────

def test_module_get_shows_details():
    result = runner.invoke(app, ["module", "12"])
    assert result.exit_code == 0
    assert "Attention" in result.output
    assert "GPT" in result.output


def test_module_get_case_insensitive():
    result = runner.invoke(app, ["module", "2A"])
    assert result.exit_code == 0
    assert "PyTorch Tensors" in result.output


def test_module_get_not_found_exit_code():
    result = runner.invoke(app, ["module", "999"])
    assert result.exit_code == 1


def test_module_get_not_found_message():
    result = runner.invoke(app, ["module", "999"])
    assert "not found" in result.output.lower()


def test_module_get_suggests_typos():
    result = runner.invoke(app, ["module", "flas"])
    # "flas" should suggest "flash"
    assert "flash" in result.output.lower()


def test_module_get_json():
    result = runner.invoke(app, ["module", "12", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["id"] == "12"
    assert data["session"] == 7


# ── sessions ─────────────────────────────────────────────────────────────────

def test_sessions_list_shows_all_9():
    result = runner.invoke(app, ["sessions", "list"])
    assert result.exit_code == 0
    assert "Advanced Topics" not in result.output
    for n in range(1, 10):
        assert str(n) in result.output


def test_session_get_shows_modules():
    result = runner.invoke(app, ["sessions", "get", "7"])
    assert result.exit_code == 0
    assert "Attention" in result.output
    assert "Transformers replaced RNNs" in result.output


def test_session_get_not_found():
    result = runner.invoke(app, ["sessions", "get", "99"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_session_get_json():
    result = runner.invoke(app, ["sessions", "get", "7", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["session"]["number"] == 7
    assert "12" in [m["id"] for m in data["modules"]]


# ── homeworks ────────────────────────────────────────────────────────────────

def test_homeworks_list_shows_all():
    result = runner.invoke(app, ["homeworks", "list"])
    assert result.exit_code == 0
    assert "MLP from Scratch" in result.output
    assert "Flash Attention" in result.output


def test_homeworks_get_shows_details():
    result = runner.invoke(app, ["homeworks", "get", "1"])
    assert result.exit_code == 0
    assert "MLP" in result.output


def test_homeworks_get_not_found():
    result = runner.invoke(app, ["homeworks", "get", "99"])
    assert result.exit_code == 1


# ── search ───────────────────────────────────────────────────────────────────

def test_search_returns_results():
    result = runner.invoke(app, ["search", "attention"])
    assert result.exit_code == 0
    assert "12" in result.output


def test_search_no_results_exit_0():
    result = runner.invoke(app, ["search", "xyzxyzxyzthisdoesnotexist"])
    assert result.exit_code == 0
    assert "No modules found" in result.output


def test_search_json():
    result = runner.invoke(app, ["search", "attention", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert any(m["id"] == "12" for m in data)


# ── notebook (mocked network) ─────────────────────────────────────────────────

def _mock_fetch(content="# Notebook content"):
    return mock.patch(
        "dataflowr.cli.fetch_notebook_content",
        return_value=content,
    )


def test_notebook_command_fetches_content():
    with _mock_fetch("# Practical notebook content"):
        result = runner.invoke(app, ["notebook", "12"])
    assert result.exit_code == 0
    assert "Practical notebook content" in result.output


def test_notebook_command_kind_intro():
    with _mock_fetch("# Intro content"):
        result = runner.invoke(app, ["notebook", "2a", "--kind", "intro"])
    assert result.exit_code == 0
    assert "Intro content" in result.output


def test_notebook_command_invalid_kind():
    result = runner.invoke(app, ["notebook", "12", "--kind", "nonexistent"])
    assert result.exit_code == 1
    assert "No 'nonexistent' notebooks" in result.output


def test_notebook_command_module_not_found():
    result = runner.invoke(app, ["notebook", "999"])
    assert result.exit_code == 1


# ── quiz (no network — module without quiz) ──────────────────────────────────

def test_quiz_no_quiz_for_module():
    result = runner.invoke(app, ["quiz", "6"])
    assert result.exit_code == 0
    assert "No quizzes available" in result.output
