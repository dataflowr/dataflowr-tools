"""
Fetch actual course content from GitHub (notebooks) and the dataflowr/website repo.

All functions use only stdlib — no new dependencies.

Local clone support
-------------------
Set environment variables to serve content from local git clones instead of
fetching from GitHub. This eliminates GitHub API rate limits and enables
offline use. Any subset of repos can be cloned; missing ones fall back to
the network automatically.

    DATAFLOWR_WEBSITE_PATH   → path to cloned dataflowr/website
    DATAFLOWR_SLIDES_PATH    → path to cloned dataflowr/slides
    DATAFLOWR_QUIZ_PATH      → path to cloned dataflowr/quiz
    DATAFLOWR_FLASH_PATH     → path to cloned dataflowr/gpu_llm_flash-attention
    DATAFLOWR_LLM_GEN_PATH   → path to cloned dataflowr/llm_controlled-generation
    DATAFLOWR_LLM_EFF_PATH   → path to cloned dataflowr/llm_efficiency
    DATAFLOWR_NOTEBOOKS_PATH → path to cloned dataflowr/notebooks

Example (VM deployment with all repos cloned under /opt/dataflowr/repos):
    export DATAFLOWR_WEBSITE_PATH=/opt/dataflowr/repos/website
    export DATAFLOWR_SLIDES_PATH=/opt/dataflowr/repos/slides
    export DATAFLOWR_QUIZ_PATH=/opt/dataflowr/repos/quiz
    export DATAFLOWR_FLASH_PATH=/opt/dataflowr/repos/gpu_llm_flash-attention
    export DATAFLOWR_LLM_GEN_PATH=/opt/dataflowr/repos/llm_controlled-generation
    export DATAFLOWR_NOTEBOOKS_PATH=/opt/dataflowr/repos/notebooks
"""

import functools
import json
import os
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

WEBSITE_REPO = "dataflowr/website"
_GITHUB_API = f"https://api.github.com/repos/{WEBSITE_REPO}/contents"

SLIDES_REPO = "dataflowr/slides"
_SLIDES_GITHUB_API = f"https://api.github.com/repos/{SLIDES_REPO}/contents"

TRANSCRIPTS_REPO = "dataflowr/transcripts"
_TRANSCRIPTS_GITHUB_API = f"https://api.github.com/repos/{TRANSCRIPTS_REPO}/contents/knowledge_base"
_TRANSCRIPTS_RAW_BASE = f"https://raw.githubusercontent.com/{TRANSCRIPTS_REPO}/main/knowledge_base"

# ── Local repo path registry ────────────────────────────────────────────────
# Maps "owner/repo" → local Path, populated from environment variables at
# import time. Any repo not present falls back to GitHub network access.

_REPO_PATHS: dict[str, Path] = {
    repo: Path(path)
    for repo, path in {
        "dataflowr/website":                   os.environ.get("DATAFLOWR_WEBSITE_PATH"),
        "dataflowr/slides":                    os.environ.get("DATAFLOWR_SLIDES_PATH"),
        "dataflowr/quiz":                      os.environ.get("DATAFLOWR_QUIZ_PATH"),
        "dataflowr/gpu_llm_flash-attention":   os.environ.get("DATAFLOWR_FLASH_PATH"),
        "dataflowr/llm_controlled-generation": os.environ.get("DATAFLOWR_LLM_GEN_PATH"),
        "dataflowr/llm_efficiency":            os.environ.get("DATAFLOWR_LLM_EFF_PATH"),
        "dataflowr/notebooks":                 os.environ.get("DATAFLOWR_NOTEBOOKS_PATH"),
        "dataflowr/transcripts":               os.environ.get("DATAFLOWR_TRANSCRIPTS_PATH"),
    }.items()
    if path is not None
}


def _read_local(repo: str, file_path: str) -> str | None:
    """Read a file from a local repo clone. Returns None if unavailable."""
    local = _REPO_PATHS.get(repo)
    if local is None:
        return None
    full = local / file_path
    try:
        return full.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _raw_url_to_local(raw_url: str) -> str | None:
    """Try to serve a raw.githubusercontent.com URL from a local clone.

    Parses https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}
    and returns the local file contents when a clone is configured.
    Returns None if no local clone is available or the file is missing.
    """
    prefix = "https://raw.githubusercontent.com/"
    if not raw_url.startswith(prefix):
        return None
    rest = raw_url[len(prefix):]
    parts = rest.split("/", 3)
    if len(parts) < 4:
        return None
    owner, repo, _, path = parts
    path = path.split("?")[0]  # strip query string if present
    return _read_local(f"{owner}/{repo}", path)


# ── Notebook content ─────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def fetch_notebook_content(raw_url: str, include_code: bool = True) -> str:
    """Fetch a .ipynb or .md from its raw GitHub URL and return readable text.

    Checks local repo clones first (via DATAFLOWR_*_PATH env vars) before
    making a network request.

    Args:
        raw_url: Raw GitHub URL (e.g. raw.githubusercontent.com/...)
        include_code: If True, include code cells as fenced code blocks (ipynb only).

    Returns:
        Markdown text. For .md files, returns the file as-is.
        For .ipynb files, returns markdown + (optionally) code cells joined by blank lines.
    """
    local = _raw_url_to_local(raw_url)
    if local is not None:
        content = local.encode("utf-8")
    else:
        req = urllib.request.Request(raw_url, headers={"User-Agent": "dataflowr-mcp/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch {raw_url}: {e.reason}") from e

    # Plain markdown file — return as-is
    if raw_url.split("?")[0].endswith(".md"):
        return content.decode("utf-8", errors="replace")

    # Jupyter notebook
    nb = json.loads(content.decode("utf-8", errors="replace"))
    parts = []
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if cell_type == "markdown":
            parts.append(source)
        elif cell_type == "code" and include_code:
            parts.append(f"```python\n{source}\n```")

    return "\n\n".join(parts)


# Keywords that mark a markdown cell as an exercise prompt.
_EXERCISE_KEYWORDS = (
    "exercise", "todo", "your turn", "fill in", "complete the",
    "question", "task", "implement", "write a", "write the",
)

# Markers that indicate a code cell is a student placeholder (not a solution).
_PLACEHOLDER_MARKERS = (
    "# your code here", "# todo", "raise notimplementederror",
)


def _is_exercise_markdown(source: str) -> bool:
    lower = source.lower()
    return any(kw in lower for kw in _EXERCISE_KEYWORDS)


def _is_placeholder_code(source: str) -> bool:
    """Return True if a code cell looks like a student skeleton rather than a solution."""
    stripped = source.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if any(marker in lower for marker in _PLACEHOLDER_MARKERS):
        return True
    # A cell consisting solely of `pass` or `...`
    if stripped in ("pass", "..."):
        return True
    return False


@functools.lru_cache(maxsize=256)
def fetch_notebook_exercises(raw_url: str) -> str:
    """Fetch only the exercise cells from a Jupyter notebook.

    Checks local repo clones first (via DATAFLOWR_*_PATH env vars) before
    making a network request.

    Returns exercise prompt markdown cells and their immediately following
    placeholder code cells (cells the student needs to fill in).
    Skips all expository text, import cells, and solution code.

    Args:
        raw_url: Raw GitHub URL of the .ipynb file.

    Returns:
        Markdown string with exercise prompts and skeleton code blocks,
        or a message if no exercise cells are found.
    """
    local = _raw_url_to_local(raw_url)
    if local is not None:
        content = local.encode("utf-8")
    else:
        req = urllib.request.Request(raw_url, headers={"User-Agent": "dataflowr-mcp/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch {raw_url}: {e.reason}") from e

    nb = json.loads(content.decode("utf-8", errors="replace"))
    cells = nb.get("cells", [])
    parts: list[str] = []
    i = 0

    while i < len(cells):
        cell = cells[i]
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown" and _is_exercise_markdown(source):
            # Include the exercise prompt.
            parts.append(source.strip())
            # Consume the code cells that immediately follow (the skeleton to fill in).
            i += 1
            while i < len(cells):
                next_cell = cells[i]
                if next_cell.get("cell_type") != "code":
                    break
                next_source = "".join(next_cell.get("source", []))
                parts.append(f"```python\n{next_source}\n```")
                i += 1
                # Stop after the first non-placeholder code cell so we don't
                # consume expository code that follows.
                if not _is_placeholder_code(next_source):
                    break
            continue

        if cell_type == "code" and _is_placeholder_code(source):
            # Standalone placeholder without a preceding exercise header.
            parts.append(f"```python\n{source}\n```")

        i += 1

    if not parts:
        return "No exercise cells found in this notebook."

    return "\n\n".join(parts)


# ── Module page markdown ─────────────────────────────────────────────────────


class _TextExtractor(HTMLParser):
    """Strip HTML tags, skipping script/style/nav/footer/header blocks."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._texts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._texts.append(data.strip())

    def get_text(self) -> str:
        return "\n".join(self._texts)


@functools.lru_cache(maxsize=128)
def fetch_module_markdown(website_url: str) -> str:
    """Fetch module content as markdown.

    - For dataflowr.github.io URLs: fetches the raw .md source from dataflowr/website.
    - For github.com URLs: fetches README.md from that repo.

    Checks local repo clones first (via DATAFLOWR_*_PATH env vars). For
    github.com URLs the network fallback uses raw.githubusercontent.com
    (no GitHub API, no rate limit).

    Args:
        website_url: The module's website_url field value.

    Returns:
        Markdown text (Franklin syntax stripped for website sources).
    """
    if "github.com/" in website_url:
        repo_path = website_url.rstrip("/").split("github.com/")[-1]

        # Try local clone first
        local = _read_local(repo_path, "README.md")
        if local is not None:
            return local

        # Use raw.githubusercontent.com — no GitHub API, no rate limit
        raw_url = f"https://raw.githubusercontent.com/{repo_path}/main/README.md"
        req = urllib.request.Request(raw_url, headers={"User-Agent": "dataflowr/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch {raw_url}: {e.reason}") from e

    # dataflowr.github.io/website URL — fetch module .md from website repo
    slug = website_url.rstrip("/").split("/modules/")[-1]
    path = f"modules/{slug}.md"

    # Try local clone first
    local = _read_local("dataflowr/website", path)
    if local is not None:
        return _clean_franklin(local)

    # Fall back to GitHub API
    api_url = f"{_GITHUB_API}/{path}"
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "dataflowr/0.1",
            "Accept": "application/vnd.github.raw+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch {api_url}: {e.reason}") from e
    return _clean_franklin(content)


def _clean_franklin(text: str) -> str:
    """Strip Franklin static-site syntax from markdown source."""
    result = []
    in_html_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@def "):       # Franklin frontmatter
            continue
        if stripped == "\\toc":               # Franklin table-of-contents macro
            continue
        if stripped == "~~~":                 # Franklin raw-HTML block delimiter
            in_html_block = not in_html_block
            continue
        if in_html_block:
            continue
        result.append(line)
    return "\n".join(result).strip()


@functools.lru_cache(maxsize=1)
def list_website_modules() -> list[dict]:
    """List all module source files in the dataflowr/website repo.

    Uses local clone when DATAFLOWR_WEBSITE_PATH is set, otherwise
    queries the GitHub API.

    Returns:
        List of dicts with 'name' (filename) and 'slug' (module slug).
    """
    local = _REPO_PATHS.get("dataflowr/website")
    if local is not None:
        modules_dir = local / "modules"
        if modules_dir.is_dir():
            return [
                {"name": f.name, "slug": f.stem}
                for f in sorted(modules_dir.glob("*.md"))
            ]

    # Fall back to GitHub API
    api_url = f"{_GITHUB_API}/modules"
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "dataflowr/0.1", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch website module list: {e.reason}") from e
    return [
        {"name": e["name"], "slug": e["name"].removesuffix(".md")}
        for e in entries
        if e["type"] == "file" and e["name"].endswith(".md")
    ]


# ── Slides ───────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=128)
def fetch_slide_content(slides_url: str) -> str:
    """Fetch slide content from the dataflowr/slides repo.

    Uses local clone when DATAFLOWR_SLIDES_PATH is set, otherwise
    queries the GitHub API.

    Args:
        slides_url: The module's slides URL (e.g. .../module1.html or .../14-03-dropout.html).

    Returns:
        Cleaned markdown text extracted from the Remark.js HTML.
    """
    filename = slides_url.rstrip("/").split("/")[-1]

    local = _read_local("dataflowr/slides", filename)
    if local is not None:
        return _clean_remark(local)

    # Fall back to GitHub API
    api_url = f"{_SLIDES_GITHUB_API}/{filename}"
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "dataflowr/0.1",
            "Accept": "application/vnd.github.raw+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch slides {filename}: {e.reason}") from e
    return _clean_remark(html)


def _clean_remark(html: str) -> str:
    """Extract and clean Remark.js slide content from HTML."""
    # Extract content from <textarea id="source">
    match = re.search(r'<textarea[^>]*>(.*?)</textarea>', html, re.DOTALL)
    text = match.group(1) if match else html

    result = []
    for line in text.splitlines():
        stripped = line.strip()
        # Skip Remark.js directives
        if any(stripped.startswith(k) for k in ("class:", "count:", "layout:", "template:")):
            continue
        # Skip footer macros
        if stripped.startswith(".center.footer[") or stripped.startswith(".footer["):
            continue
        # Strip inline Remark.js styling: .red[x], .bold[x], .grey[x], .center[x], etc.
        line = re.sub(r'\.\w+\[([^\]]*)\]', r'\1', line)
        # Strip remaining HTML tags
        line = re.sub(r'<[^>]+>', '', line)
        result.append(line)
    return "\n".join(result).strip()


@functools.lru_cache(maxsize=1)
def list_slide_files() -> list[dict]:
    """List all slide HTML files in the dataflowr/slides repo.

    Uses local clone when DATAFLOWR_SLIDES_PATH is set, otherwise
    queries the GitHub API.

    Returns:
        List of dicts with 'name' (filename) and 'slug' (name without .html).
    """
    local = _REPO_PATHS.get("dataflowr/slides")
    if local is not None and local.is_dir():
        return [
            {"name": f.name, "slug": f.stem}
            for f in sorted(local.glob("*.html"))
        ]

    # Fall back to GitHub API
    req = urllib.request.Request(
        _SLIDES_GITHUB_API,
        headers={"User-Agent": "dataflowr/0.1", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch slides file list: {e.reason}") from e
    return [
        {"name": e["name"], "slug": e["name"].removesuffix(".html")}
        for e in entries
        if e["type"] == "file" and e["name"].endswith(".html")
    ]


# ── Quiz ─────────────────────────────────────────────────────────────────────


QUIZ_REPO = "dataflowr/quiz"
_QUIZ_GITHUB_API = f"https://api.github.com/repos/{QUIZ_REPO}/contents/dl-quiz/src"
_QUIZ_RAW_BASE = f"https://raw.githubusercontent.com/{QUIZ_REPO}/main/dl-quiz/src"


@functools.lru_cache(maxsize=256)
def _fetch_quiz_file_raw(filename: str) -> bytes:
    """Fetch a single quiz TOML file (cached).

    Uses local clone when DATAFLOWR_QUIZ_PATH is set, otherwise
    fetches from raw.githubusercontent.com.
    """
    local = _REPO_PATHS.get("dataflowr/quiz")
    if local is not None:
        file_path = local / "dl-quiz" / "src" / filename
        try:
            return file_path.read_bytes()
        except OSError:
            pass

    # Fall back to raw GitHub (not rate-limited)
    url = f"{_QUIZ_RAW_BASE}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "dataflowr/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch quiz file {filename}: {e.reason}") from e


def parse_quiz_questions(quiz_files: list[str]) -> list[dict]:
    """Parse quiz TOML files into a list of structured question dicts.

    Each dict contains:
        index         (int)       1-based global question number
        text          (str)       question text
        choices       (list[str]) answer options
        answer_index  (int)       0-based index of the correct choice
        answer_number (int)       1-based number of the correct choice
        context       (str)       explanation shown after answering

    Returns an empty list on Python < 3.11 (no tomllib).
    """
    try:
        import tomllib as _tomllib
    except ImportError:
        return []

    questions = []
    idx = 1
    for filename in quiz_files:
        raw = _fetch_quiz_file_raw(filename)
        data = _tomllib.loads(raw.decode("utf-8", errors="replace"))
        for q in data.get("questions", []):
            prompt = q.get("prompt", {})
            answer_idx = q.get("answer", {}).get("answer", -1)
            choices = prompt.get("choices", [])
            questions.append({
                "index": idx,
                "text": prompt.get("prompt", "").strip(),
                "choices": choices,
                "answer_index": answer_idx,
                "answer_number": answer_idx + 1 if answer_idx >= 0 else -1,
                "context": q.get("context", "").strip(),
            })
            idx += 1
    return questions


def check_quiz_answer(quiz_files: list[str], question_number: int, answer_number: int) -> dict:
    """Check a student's answer for a specific quiz question.

    Args:
        quiz_files:       List of TOML filenames for the module.
        question_number:  1-based question index (global across all files).
        answer_number:    1-based answer choice selected by the student.

    Returns:
        On success: dict with keys correct (bool), correct_number (int),
            correct_choice (str), student_choice (str), context (str).
        On error: dict with key 'error' (str).
    """
    questions = parse_quiz_questions(quiz_files)
    if not questions:
        return {"error": "Quiz parsing requires Python 3.11+ (tomllib)."}
    if not (1 <= question_number <= len(questions)):
        return {"error": f"Question {question_number} not found. Valid range: 1–{len(questions)}."}
    q = questions[question_number - 1]
    choices = q["choices"]
    if not (1 <= answer_number <= len(choices)):
        return {"error": f"Answer {answer_number} is not valid. Choose 1–{len(choices)}."}
    correct = (answer_number - 1) == q["answer_index"]
    return {
        "correct": correct,
        "correct_number": q["answer_number"],
        "correct_choice": choices[q["answer_index"]] if 0 <= q["answer_index"] < len(choices) else "?",
        "student_choice": choices[answer_number - 1],
        "context": q["context"],
    }


def fetch_quiz_content(quiz_files: list[str]) -> str:
    """Fetch and format quiz questions from TOML files in the dataflowr/quiz repo.

    Args:
        quiz_files: List of TOML filenames (e.g. ['quiz_21.toml', 'quiz_22.toml']).

    Returns:
        Formatted markdown with questions, choices, correct answers, and context.
        Falls back to raw TOML text on Python 3.10 (no tomllib).
    """
    try:
        import tomllib as _tomllib
    except ImportError:
        _tomllib = None

    parts = []
    for filename in quiz_files:
        raw = _fetch_quiz_file_raw(filename)

        if _tomllib is None:
            parts.append(f"### {filename}\n\n```toml\n{raw.decode('utf-8', errors='replace')}\n```")
            continue

        data = _tomllib.loads(raw.decode("utf-8", errors="replace"))
        questions = data.get("questions", [])
        for i, q in enumerate(questions, 1):
            prompt = q.get("prompt", {})
            question_text = prompt.get("prompt", "").strip()
            choices = prompt.get("choices", [])
            answer_idx = q.get("answer", {}).get("answer", -1)
            context = q.get("context", "").strip()

            block = [f"**Q{i}.** {question_text}\n"]
            for j, choice in enumerate(choices):
                marker = "✓" if j == answer_idx else " "
                block.append(f"  {marker} {j + 1}. {choice}")
            if context:
                block.append(f"\n*{context}*")
            parts.append("\n".join(block))

    return "\n\n---\n\n".join(parts)


@functools.lru_cache(maxsize=1)
def list_quiz_files() -> list[dict]:
    """List all quiz TOML files in the dataflowr/quiz repo.

    Uses local clone when DATAFLOWR_QUIZ_PATH is set, otherwise
    queries the GitHub API.

    Returns:
        List of dicts with 'name' (filename) and 'slug' (name without .toml).
    """
    local = _REPO_PATHS.get("dataflowr/quiz")
    if local is not None:
        quiz_dir = local / "dl-quiz" / "src"
        if quiz_dir.is_dir():
            return [
                {"name": f.name, "slug": f.stem}
                for f in sorted(quiz_dir.glob("*.toml"))
            ]

    # Fall back to GitHub API
    req = urllib.request.Request(
        _QUIZ_GITHUB_API,
        headers={"User-Agent": "dataflowr/0.1", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch quiz file list: {e.reason}") from e
    return [
        {"name": e["name"], "slug": e["name"].removesuffix(".toml")}
        for e in entries
        if e["type"] == "file" and e["name"].endswith(".toml")
    ]


# ── Website page content ─────────────────────────────────────────────────────


@functools.lru_cache(maxsize=128)
def fetch_page_content(url: str) -> str:
    """Fetch a course website page and return its text content (HTML stripped).

    Args:
        url: Full URL of the course page (e.g. module.website_url).

    Returns:
        Plain text extracted from the page, with script/style/nav removed.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "dataflowr-mcp/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch {url}: {e.reason}") from e

    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


# ── Transcripts knowledge base ────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def list_transcript_notes() -> list[dict]:
    """List all concept notes in the dataflowr/transcripts knowledge base.

    Uses local clone when DATAFLOWR_TRANSCRIPTS_PATH is set, otherwise
    queries the GitHub API.

    Returns:
        List of dicts with 'name' (filename) and 'concept' (name without .md).
    """
    local = _REPO_PATHS.get("dataflowr/transcripts")
    if local is not None:
        kb_dir = local / "knowledge_base"
        if kb_dir.is_dir():
            return [
                {"name": f.name, "concept": f.stem}
                for f in sorted(kb_dir.glob("*.md"))
            ]

    # Fall back to GitHub API
    req = urllib.request.Request(
        _TRANSCRIPTS_GITHUB_API,
        headers={"User-Agent": "dataflowr/0.1", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch transcript notes list: {e.reason}") from e
    return [
        {"name": e["name"], "concept": e["name"].removesuffix(".md")}
        for e in entries
        if e["type"] == "file" and e["name"].endswith(".md")
    ]


@functools.lru_cache(maxsize=256)
def fetch_transcript_note(concept: str) -> str:
    """Fetch a single concept note from the transcript knowledge base.

    Args:
        concept: The concept name (e.g. 'training loop', 'backpropagation').

    Returns:
        The full markdown content of the note.
    """
    filename = f"{concept}.md"

    local = _read_local("dataflowr/transcripts", f"knowledge_base/{filename}")
    if local is not None:
        return local

    # Fall back to raw GitHub (not rate-limited)
    url = f"{_TRANSCRIPTS_RAW_BASE}/{urllib.request.quote(filename)}"
    req = urllib.request.Request(url, headers={"User-Agent": "dataflowr/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch transcript note '{concept}': {e.reason}") from e


def search_transcript_notes(query: str) -> list[dict]:
    """Search concept notes by fuzzy-matching against concept names.

    Matches are ranked: exact match first, then starts-with, then contains,
    then word-boundary partial matches.

    Args:
        query: Search string (e.g. 'backprop', 'training', 'dropout').

    Returns:
        List of matching dicts with 'name' and 'concept' keys, sorted by relevance.
    """
    notes = list_transcript_notes()
    query_lower = query.lower()
    query_words = query_lower.split()

    exact = []
    starts_with = []
    contains = []
    word_match = []

    for note in notes:
        concept = note["concept"].lower()
        if concept == query_lower:
            exact.append(note)
        elif concept.startswith(query_lower):
            starts_with.append(note)
        elif query_lower in concept:
            contains.append(note)
        elif all(w in concept for w in query_words):
            word_match.append(note)

    return exact + starts_with + contains + word_match
