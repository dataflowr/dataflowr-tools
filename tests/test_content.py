"""Tests for content parsing helpers in dataflowr.content.

Network calls are mocked via unittest.mock so these tests run offline.
"""

import json
import unittest.mock as mock
import urllib.error
import urllib.request

import pytest

from dataflowr.content import (
    _clean_franklin,
    _clean_remark,
    _is_exercise_markdown,
    _is_placeholder_code,
    fetch_notebook_content,
    fetch_notebook_exercises,
    fetch_transcript_note,
    list_transcript_notes,
    search_transcript_notes,
)


# ── _clean_franklin ──────────────────────────────────────────────────────────

def test_clean_franklin_strips_at_def():
    text = "@def title = \"Module 3\"\n\nSome content here."
    result = _clean_franklin(text)
    assert "@def" not in result
    assert "Some content here." in result


def test_clean_franklin_strips_toc():
    text = "## Section\n\n\\toc\n\nParagraph."
    result = _clean_franklin(text)
    assert "\\toc" not in result
    assert "Paragraph." in result


def test_clean_franklin_strips_html_block():
    text = "Before\n\n~~~\n<div>some html</div>\n~~~\n\nAfter"
    result = _clean_franklin(text)
    assert "<div>" not in result
    assert "Before" in result
    assert "After" in result


def test_clean_franklin_preserves_normal_markdown():
    text = "# Heading\n\nSome **bold** and *italic* text.\n\n- item 1\n- item 2"
    result = _clean_franklin(text)
    assert "# Heading" in result
    assert "**bold**" in result
    assert "- item 1" in result


def test_clean_franklin_multiple_frontmatter_lines():
    text = "@def title = \"T\"\n@def tags = [\"a\"]\n\n# Content"
    result = _clean_franklin(text)
    assert "@def" not in result
    assert "# Content" in result


# ── _clean_remark ────────────────────────────────────────────────────────────

_REMARK_HTML = """<html><body>
<textarea id="source">
# Slide 1

Some text here.

---

class: center
count: false

# Slide 2

.center.footer[CC BY-SA 4.0]

More text.
</textarea>
</body></html>"""


def test_clean_remark_extracts_textarea_content():
    result = _clean_remark(_REMARK_HTML)
    assert "Some text here." in result
    assert "More text." in result


def test_clean_remark_strips_class_directive():
    result = _clean_remark(_REMARK_HTML)
    assert "class: center" not in result


def test_clean_remark_strips_count_directive():
    result = _clean_remark(_REMARK_HTML)
    assert "count: false" not in result


def test_clean_remark_strips_footer_macro():
    result = _clean_remark(_REMARK_HTML)
    assert ".center.footer[" not in result


def test_clean_remark_fallback_without_textarea():
    # When no <textarea> is found the raw html is used as fallback
    plain = "# Just some text\nNo textarea here."
    result = _clean_remark(plain)
    assert "Just some text" in result


# ── _is_exercise_markdown ────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "**Exercise:** implement the forward pass.",
    "TODO: complete this function.",
    "Your turn: fill in the blanks below.",
    "Question: what does this line do?",
    "Task: write a training loop.",
    "Implement the following:",
    "Write a function that computes the loss.",
])
def test_is_exercise_markdown_positive(text):
    assert _is_exercise_markdown(text)


def test_is_exercise_markdown_negative_regular_text():
    assert not _is_exercise_markdown("This is a normal explanation of PyTorch tensors.")


def test_is_exercise_markdown_case_insensitive():
    assert _is_exercise_markdown("EXERCISE: do this")
    assert _is_exercise_markdown("TODO: fix me")


# ── _is_placeholder_code ────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    "# Your code here",
    "# TODO: implement this",
    "raise NotImplementedError",
    "pass",
    "...",
    "",
    "   ",
])
def test_is_placeholder_code_positive(code):
    assert _is_placeholder_code(code)


@pytest.mark.parametrize("code", [
    "import torch\nimport numpy as np",
    "def forward(self, x):\n    return self.linear(x)",
    "loss = criterion(output, target)\nloss.backward()",
])
def test_is_placeholder_code_negative(code):
    assert not _is_placeholder_code(code)


# ── fetch_notebook_content (mocked) ─────────────────────────────────────────

def _make_notebook_bytes(cells):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb).encode()


def _mock_urlopen(content: bytes):
    """Return a context-manager mock that yields a response with .read()."""
    resp = mock.MagicMock()
    resp.read.return_value = content
    resp.__enter__ = lambda s: s
    resp.__exit__ = mock.MagicMock(return_value=False)
    return mock.patch("urllib.request.urlopen", return_value=resp)


def test_fetch_notebook_content_markdown_file():
    raw_url = "https://raw.githubusercontent.com/dataflowr/notebooks/master/README.md"
    content = b"# Title\n\nSome text."
    with _mock_urlopen(content):
        result = fetch_notebook_content.__wrapped__(raw_url, True)
    assert result == "# Title\n\nSome text."


def test_fetch_notebook_content_notebook_markdown_cells():
    cells = [
        {"cell_type": "markdown", "source": ["# Hello\n", "world"]},
        {"cell_type": "markdown", "source": ["Second cell"]},
    ]
    raw_url = "https://raw.githubusercontent.com/dataflowr/notebooks/master/Module1/nb.ipynb"
    with _mock_urlopen(_make_notebook_bytes(cells)):
        result = fetch_notebook_content.__wrapped__(raw_url, False)
    assert "# Hello\nworld" in result
    assert "Second cell" in result


def test_fetch_notebook_content_includes_code_cells():
    cells = [
        {"cell_type": "markdown", "source": ["Exercise"]},
        {"cell_type": "code", "source": ["x = 1"]},
    ]
    raw_url = "https://raw.githubusercontent.com/dataflowr/notebooks/master/Module1/nb.ipynb"
    with _mock_urlopen(_make_notebook_bytes(cells)):
        result = fetch_notebook_content.__wrapped__(raw_url, True)
    assert "```python" in result
    assert "x = 1" in result


def test_fetch_notebook_content_excludes_code_cells_when_flagged():
    cells = [
        {"cell_type": "markdown", "source": ["Exercise"]},
        {"cell_type": "code", "source": ["x = 1"]},
    ]
    raw_url = "https://raw.githubusercontent.com/dataflowr/notebooks/master/Module1/nb.ipynb"
    with _mock_urlopen(_make_notebook_bytes(cells)):
        result = fetch_notebook_content.__wrapped__(raw_url, False)
    assert "x = 1" not in result
    assert "Exercise" in result


def test_fetch_notebook_content_skips_empty_cells():
    cells = [
        {"cell_type": "markdown", "source": ["   "]},
        {"cell_type": "markdown", "source": ["Real content"]},
    ]
    raw_url = "https://raw.githubusercontent.com/dataflowr/notebooks/master/Module1/nb.ipynb"
    with _mock_urlopen(_make_notebook_bytes(cells)):
        result = fetch_notebook_content.__wrapped__(raw_url, True)
    assert "Real content" in result
    # The empty cell should not add a blank entry
    assert result.strip() == "Real content"


def test_fetch_notebook_content_raises_on_network_error():
    raw_url = "https://raw.githubusercontent.com/dataflowr/notebooks/master/Module1/nb.ipynb"
    error = urllib.error.URLError("timed out")
    with mock.patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(RuntimeError, match="Failed to fetch"):
            fetch_notebook_content.__wrapped__(raw_url, True)


# ── fetch_notebook_exercises (mocked) ───────────────────────────────────────

def test_fetch_notebook_exercises_finds_exercise_cells():
    cells = [
        {"cell_type": "markdown", "source": ["Introduction"]},
        {"cell_type": "markdown", "source": ["**Exercise:** implement forward."]},
        {"cell_type": "code", "source": ["# Your code here"]},
        {"cell_type": "markdown", "source": ["Some explanation"]},
    ]
    raw_url = "https://raw.githubusercontent.com/dataflowr/notebooks/master/Module1/nb.ipynb"
    with _mock_urlopen(_make_notebook_bytes(cells)):
        result = fetch_notebook_exercises.__wrapped__(raw_url)
    assert "Exercise" in result
    assert "Your code here" in result
    assert "Introduction" not in result
    assert "Some explanation" not in result


def test_fetch_notebook_exercises_empty_notebook():
    raw_url = "https://raw.githubusercontent.com/dataflowr/notebooks/master/Module1/nb.ipynb"
    with _mock_urlopen(_make_notebook_bytes([])):
        result = fetch_notebook_exercises.__wrapped__(raw_url)
    assert "No exercise cells found" in result


# ── Transcript knowledge base ────────────────────────────────────────────────


def test_list_transcript_notes_local(tmp_path):
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    (kb_dir / "dropout.md").write_text("# Dropout")
    (kb_dir / "backpropagation.md").write_text("# Backpropagation")

    with mock.patch.dict(
        "dataflowr.content._REPO_PATHS",
        {"dataflowr/transcripts": tmp_path},
    ):
        result = list_transcript_notes.__wrapped__()
    concepts = [r["concept"] for r in result]
    assert "dropout" in concepts
    assert "backpropagation" in concepts
    assert all(r["name"].endswith(".md") for r in result)


def test_fetch_transcript_note_local(tmp_path):
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    (kb_dir / "training loop.md").write_text("# Training Loop\nContent here.")

    with mock.patch.dict(
        "dataflowr.content._REPO_PATHS",
        {"dataflowr/transcripts": tmp_path},
    ):
        result = fetch_transcript_note.__wrapped__("training loop")
    assert "Training Loop" in result
    assert "Content here." in result


def test_search_transcript_notes_substring_match():
    notes = [
        {"name": "backpropagation.md", "concept": "backpropagation"},
        {"name": "backpropagation through time.md", "concept": "backpropagation through time"},
        {"name": "dropout.md", "concept": "dropout"},
        {"name": "backward pass.md", "concept": "backward pass"},
    ]
    with mock.patch("dataflowr.content.list_transcript_notes", return_value=notes):
        results = search_transcript_notes("backprop")
    concepts = [r["concept"] for r in results]
    assert "backpropagation" in concepts
    assert "backpropagation through time" in concepts
    assert "dropout" not in concepts


def test_search_transcript_notes_exact_match_first():
    notes = [
        {"name": "dropout.md", "concept": "dropout"},
        {"name": "dropblock.md", "concept": "dropblock"},
    ]
    with mock.patch("dataflowr.content.list_transcript_notes", return_value=notes):
        results = search_transcript_notes("dropout")
    assert results[0]["concept"] == "dropout"


def test_search_transcript_notes_no_match():
    notes = [
        {"name": "dropout.md", "concept": "dropout"},
    ]
    with mock.patch("dataflowr.content.list_transcript_notes", return_value=notes):
        results = search_transcript_notes("xyznonexistent")
    assert results == []


def test_search_transcript_notes_multi_word():
    notes = [
        {"name": "training loop.md", "concept": "training loop"},
        {"name": "training.md", "concept": "training"},
        {"name": "loop.md", "concept": "loop"},
    ]
    with mock.patch("dataflowr.content.list_transcript_notes", return_value=notes):
        results = search_transcript_notes("training loop")
    assert results[0]["concept"] == "training loop"
