"""Tests for catalog integrity and Course model methods.

All tests are offline — no network access required.
"""

import pytest
from dataflowr import COURSE
from dataflowr.models import Module, Notebook, NotebookKind


# ── Catalog structural integrity ────────────────────────────────────────────

def test_session_count():
    assert len(COURSE.sessions) == 9


def test_all_sessions_1_through_9_present():
    numbers = {s.number for s in COURSE.sessions}
    assert numbers == set(range(1, 10))


def test_external_modules_have_no_session():
    for mid in ("flash", "llm_gen", "graph0"):
        assert COURSE.modules[mid].session is None


def test_module_count():
    assert len(COURSE.modules) == 32


def test_no_duplicate_module_ids():
    # dict keys are inherently unique, but verify id field matches the key
    for key, module in COURSE.modules.items():
        assert module.id == key, f"Key {key!r} != module.id {module.id!r}"


def test_all_session_modules_exist():
    """Every module id referenced in a session must exist in COURSE.modules."""
    for session in COURSE.sessions:
        for mid in session.modules:
            assert mid in COURSE.modules, (
                f"Session {session.number} references unknown module {mid!r}"
            )


def test_all_prerequisites_exist():
    """Every prerequisite id must point to a real module."""
    for module in COURSE.modules.values():
        for prereq in module.prerequisites:
            assert prereq in COURSE.modules, (
                f"Module {module.id!r} has unknown prerequisite {prereq!r}"
            )


def test_homework_count():
    assert len(COURSE.homeworks) == 5


def test_all_homework_ids_unique():
    ids = [hw.id for hw in COURSE.homeworks]
    assert len(ids) == len(set(ids))


# ── Module.folder property ───────────────────────────────────────────────────

@pytest.mark.parametrize("module_id,expected_folder", [
    ("1",    "Module1"),
    ("12",   "Module12"),
    ("2a",   "Module2"),
    ("2b",   "Module2"),
    ("18b",  "Module18"),
    ("11a",  "Module11"),
    ("14b",  "Module14"),
])
def test_module_folder(module_id, expected_folder):
    m = COURSE.get_module(module_id)
    assert m is not None
    assert m.folder == expected_folder


# ── Notebook.raw_url property ────────────────────────────────────────────────

def test_notebook_raw_url_substitution():
    nb = Notebook(
        filename="Module12/GPT_hist.ipynb",
        title="microGPT",
        kind=NotebookKind.practical,
        github_url="https://github.com/dataflowr/notebooks/blob/master/Module12/GPT_hist.ipynb",
    )
    assert "raw.githubusercontent.com" in nb.raw_url
    assert "/blob/" not in nb.raw_url
    assert nb.raw_url.startswith("https://raw.githubusercontent.com/")


# ── Course.get_module ────────────────────────────────────────────────────────

def test_get_module_exact_match():
    m = COURSE.get_module("12")
    assert m is not None
    assert m.id == "12"
    assert "Transformer" in m.title


def test_get_module_case_insensitive():
    m = COURSE.get_module("2A")
    assert m is not None
    assert m.id == "2a"


def test_get_module_case_insensitive_multi():
    assert COURSE.get_module("18B").id == "18b"
    assert COURSE.get_module("FLASH").id == "flash"


def test_get_module_not_found_returns_none():
    assert COURSE.get_module("999") is None
    assert COURSE.get_module("xyz") is None


# ── Course.suggest_module_ids ────────────────────────────────────────────────

def test_suggest_module_ids_typo():
    suggestions = COURSE.suggest_module_ids("2A")
    assert "2a" in suggestions


def test_suggest_module_ids_close_match():
    suggestions = COURSE.suggest_module_ids("flas")
    assert "flash" in suggestions


def test_suggest_module_ids_no_match():
    suggestions = COURSE.suggest_module_ids("zzzzz")
    assert suggestions == []


# ── Course.search ────────────────────────────────────────────────────────────

def test_search_returns_relevant_module():
    results = COURSE.search("attention")
    ids = [m.id for m in results]
    assert "12" in ids


def test_search_by_tag():
    results = COURSE.search("diffusion")
    # "diffusion" appears in module 18b's title and tags (not in description)
    assert any(
        "diffusion" in m.title.lower() or "diffusion" in " ".join(m.tags).lower()
        for m in results
    )


def test_search_no_results():
    results = COURSE.search("xyzxyzxyzthisdoesnotexist")
    assert results == []


def test_search_case_insensitive():
    lower = COURSE.search("attention")
    upper = COURSE.search("ATTENTION")
    assert {m.id for m in lower} == {m.id for m in upper}


# ── Course.get_session_modules ───────────────────────────────────────────────

def test_get_session_modules_session_7():
    modules = COURSE.get_session_modules(7)
    ids = [m.id for m in modules]
    assert ids == ["12"]


def test_get_session_modules_session_2():
    modules = COURSE.get_session_modules(2)
    ids = [m.id for m in modules]
    assert "2a" in ids
    assert "2b" in ids
    assert "2c" in ids


def test_get_session_modules_session_10_empty():
    assert COURSE.get_session_modules(10) == []


def test_get_session_modules_invalid_session():
    assert COURSE.get_session_modules(999) == []
