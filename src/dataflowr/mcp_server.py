"""
dataflowr MCP Server

Makes the dataflowr course natively available to AI agents (Claude, Cursor, etc.)
via the Model Context Protocol, using the official MCP Python SDK (FastMCP).

Run with stdio (default — for Claude Desktop, Cursor, VS Code, Claude Code):
    python -m dataflowr.mcp_server

Run with HTTP transport (for remote/shared deployments):
    python -m dataflowr.mcp_server --http
    # → POST http://localhost:8001/mcp

Tools exposed:
    - list_modules          list all modules (filterable by session/tag/GPU)
    - get_module            full module details by id
    - search_modules        keyword search across titles, descriptions, tags
    - list_sessions         list all sessions
    - get_session           session details + all module content
    - get_notebook_url      get GitHub/Colab URL for a specific notebook
    - list_homeworks        list all homeworks
    - get_homework          full details for a homework
    - get_slide_content     fetch lecture slides from dataflowr/slides
    - get_quiz_content      fetch quiz questions from dataflowr/quiz
    - get_notebook_content  fetch actual notebook cells from GitHub
    - get_notebook_exercises fetch only exercise prompts + skeleton code
    - get_page_content      fetch module source markdown from dataflowr/website
    - get_course_overview   full course structure as context
    - sync_catalog          compare catalog against website + slides repos
    - get_prerequisites     prerequisite modules for a given module
    - check_quiz_answer     validate a student's quiz answer
    - suggest_next          what to study after a given module
    - search_transcripts    fuzzy search 318 concept notes from lecture transcripts
    - get_transcript_note   fetch full content of a transcript concept note

Prompts exposed:
    - explain_module        tutoring session for a specific module
    - quiz_student          interactive quiz session for a module
    - debug_help            Socratic debugging help for a practical notebook
    - learning_path         personalised prerequisite chain to a target module
"""

from mcp.server.fastmcp import FastMCP

from .catalog import COURSE
from .content import (
    check_quiz_answer as _check_quiz_answer,
    fetch_module_markdown,
    fetch_notebook_content,
    fetch_notebook_exercises,
    fetch_quiz_content,
    fetch_slide_content,
    fetch_transcript_note,
    list_quiz_files,
    list_slide_files,
    list_website_modules,
    search_transcript_notes,
)

mcp = FastMCP(
    "dataflowr",
    instructions=(
        "You are a tutor for the Deep Learning DIY course at dataflowr.github.io. "
        "The course has 25+ modules across 10 sessions covering PyTorch from scratch: "
        "tensors, autodiff, CNNs, RNNs, Transformers, GANs, VAEs, diffusion models, and more. "
        "Recommended workflow: search_modules → get_module → get_page_content → "
        "get_notebook_content → get_quiz_content. "
        "For questions about what the professor says, lecture quotes, or spoken explanations, "
        "ALWAYS use the transcript tools: search_transcripts → get_transcript_note. "
        "These tools provide 318 concept notes extracted from lecture transcripts with "
        "timestamped quotes. Never say transcripts are unavailable — use search_transcripts first."
    ),
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _module_to_text(module) -> str:
    """Render a module as readable text for an LLM."""
    lines = [
        f"## Module {module.id}: {module.title}",
        "",
        f"**Session:** {module.session if module.session is not None else '— (external course)'}",
        f"**Description:** {module.description}",
        f"**Tags:** {', '.join(module.tags)}",
        f"**Requires GPU:** {'Yes' if module.requires_gpu else 'No'}",
        f"**Website:** {module.website_url}",
    ]
    if module.slides_url:
        lines.append(f"**Slides:** {module.slides_url}")

    if module.notebooks:
        lines.append("")
        lines.append("**Notebooks:**")
        for nb in module.notebooks:
            lines.append(f"- `{nb.filename}` ({nb.kind.value}): {nb.title}")
            lines.append(f"  - GitHub: {nb.github_url}")
            if nb.colab_url:
                lines.append(f"  - Colab: {nb.colab_url}")

    if module.prerequisites:
        lines.append(f"**Prerequisites:** Modules {', '.join(module.prerequisites)}")

    return "\n".join(lines)


def _session_to_text(session, include_modules: bool = True) -> str:
    lines = [
        f"## Session {session.number}: {session.title}",
        "",
        f"**Modules:** {', '.join(session.modules)}",
    ]

    if session.things_to_remember:
        lines.append("")
        lines.append("**Things to remember:**")
        for thing in session.things_to_remember:
            lines.append(f"- {thing}")

    if include_modules:
        lines.append("")
        lines.append("**Module details:**")
        for module_id in session.modules:
            module = COURSE.get_module(module_id)
            if module:
                lines.append("")
                lines.append(_module_to_text(module))

    return "\n".join(lines)


# ── Tools ──────────────────────────────────────────────────────────────────


@mcp.tool()
def list_modules(
    session: int | None = None,
    tag: str | None = None,
    gpu_only: bool = False,
) -> str:
    """List all modules in the dataflowr Deep Learning DIY course.

    Can be filtered by session number, tag, or GPU requirement.
    Returns module IDs, titles, descriptions, and tags.
    """
    modules = list(COURSE.modules.values())
    if session is not None:
        modules = [m for m in modules if m.session == session]
    if tag is not None:
        tag_lower = tag.lower()
        modules = [m for m in modules if any(tag_lower in t.lower() for t in m.tags)]
    if gpu_only:
        modules = [m for m in modules if m.requires_gpu]

    lines = [f"Found {len(modules)} modules:\n"]
    for m in modules:
        gpu = " [GPU]" if m.requires_gpu else ""
        session_label = f"Session {m.session}" if m.session is not None else "external"
        lines.append(f"- Module {m.id} ({session_label}){gpu}: **{m.title}**")
        lines.append(f"  {m.description[:100]}")
        lines.append(f"  Tags: {', '.join(m.tags[:4])}")
    return "\n".join(lines)


@mcp.tool()
def get_module(module_id: str) -> str:
    """Get full details for a specific module.

    Includes description, notebooks with GitHub and Colab links, tags, and prerequisites.
    Use this when a student asks about a specific topic or module.
    Module IDs: '12' (Attention/Transformers), '2a' (PyTorch tensors), '18b' (diffusion), etc.
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return (
            f"Module '{module_id}' not found.{hint} "
            f"Use the list_modules tool to see all {len(COURSE.modules)} available modules."
        )
    return _module_to_text(module)


@mcp.tool()
def search_modules(query: str) -> str:
    """Search modules by keyword across titles, descriptions, and tags.

    Use this to find relevant modules for a student's question or topic.
    E.g. query='attention' finds the Transformer module;
    'generative' finds GANs, autoencoders, flows, diffusion.
    """
    results = COURSE.search(query)
    if not results:
        return (
            f"No modules found for '{query}'. "
            f"Try different keywords, or use list_modules to browse all {len(COURSE.modules)} modules."
        )
    lines = [f"Found {len(results)} module(s) for '{query}':\n"]
    for m in results:
        lines.append(_module_to_text(m))
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def list_sessions() -> str:
    """List all course sessions with their titles and module IDs.

    Sessions group modules into ~2-3 hour teaching blocks.
    """
    lines = [f"The course has {len(COURSE.sessions)} sessions:\n"]
    for s in COURSE.sessions:
        lines.append(f"**Session {s.number}: {s.title}**")
        lines.append(f"  Modules: {', '.join(s.modules)}")
    return "\n".join(lines)


@mcp.tool()
def get_session(session_number: int) -> str:
    """Get full details for a session, including all its modules, notebooks, and key takeaways.

    Use this when a student wants to understand what a session covers.
    """
    session = next((s for s in COURSE.sessions if s.number == session_number), None)
    if not session:
        valid = sorted(s.number for s in COURSE.sessions)
        return (
            f"Session {session_number} not found. "
            f"Valid sessions: {', '.join(str(v) for v in valid)}. "
            f"Use list_sessions to see all sessions."
        )
    return _session_to_text(session, include_modules=True)


@mcp.tool()
def get_notebook_url(module_id: str, kind: str = "practical") -> str:
    """Get the GitHub and Colab URL for a specific notebook.

    Use this when a student wants to open or run a specific exercise.
    kind: 'intro' | 'practical' | 'solution' | 'bonus' | 'homework' (default: practical)
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"Module '{module_id}' not found.{hint}"
    notebooks = [nb for nb in module.notebooks if nb.kind.value == kind]
    if not notebooks:
        kinds = sorted({nb.kind.value for nb in module.notebooks})
        return (
            f"No '{kind}' notebooks for Module {module.id}. "
            f"Available kinds: {kinds}. "
            f"Try get_notebook_url with kind set to one of: {', '.join(kinds)}."
        )
    lines = []
    for nb in notebooks:
        lines.append(f"**{nb.title}**")
        lines.append(f"- GitHub: {nb.github_url}")
        if nb.colab_url:
            lines.append(f"- Colab: {nb.colab_url}")
    return "\n".join(lines)


@mcp.tool()
def list_homeworks() -> str:
    """List all graded homeworks with descriptions and notebook links."""
    lines = [f"The course has {len(COURSE.homeworks)} graded homeworks:\n"]
    for hw in COURSE.homeworks:
        lines.append(f"**HW{hw.id}: {hw.title}**")
        lines.append(f"  {hw.description}")
        lines.append(f"  Website: {hw.website_url}")
        for nb in hw.notebooks:
            if nb.kind.value != "solution":
                lines.append(f"  Notebook: {nb.github_url}")
    return "\n".join(lines)


@mcp.tool()
def get_homework(hw_id: int) -> str:
    """Get full details for a specific homework, including description, website URL, and notebook links."""
    hw = next((h for h in COURSE.homeworks if h.id == hw_id), None)
    if not hw:
        available = [h.id for h in COURSE.homeworks]
        return (
            f"Homework {hw_id} not found. Available homework IDs: {available}. "
            f"Use list_homeworks to see all homeworks."
        )
    lines = [
        f"## HW{hw.id}: {hw.title}",
        "",
        f"{hw.description}",
        "",
        f"**Website:** {hw.website_url}",
    ]
    if hw.notebooks:
        lines.append("")
        lines.append("**Notebooks:**")
        for nb in hw.notebooks:
            if nb.kind.value != "solution":
                lines.append(f"- `{nb.filename}`: {nb.title}")
                lines.append(f"  - GitHub: {nb.github_url}")
                if nb.colab_url:
                    lines.append(f"  - Colab: {nb.colab_url}")
    return "\n".join(lines)


@mcp.tool()
def get_slide_content(module_id: str) -> str:
    """Fetch the lecture slide content for a module from the dataflowr/slides GitHub repo.

    Returns the slide text (Remark.js markdown, cleaned).
    Use this when a student wants to review lecture slides or study the theory for a module.
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"Module '{module_id}' not found.{hint}"
    if not module.slides_url:
        has_slides = sorted(mid for mid, m in COURSE.modules.items() if m.slides_url)
        return (
            f"No slides available for module '{module_id}'. "
            f"Modules with slides: {', '.join(has_slides)}."
        )
    return fetch_slide_content(module.slides_url)


@mcp.tool()
def get_quiz_content(module_id: str) -> str:
    """Fetch the quiz questions for a module from the dataflowr/quiz GitHub repo.

    Returns multiple-choice questions with choices, correct answers, and explanations.
    Use this when a student wants to self-test their understanding of a module.
    Modules with quizzes: '2a' (tensors), '2b' (autograd), '3' (loss functions).
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"Module '{module_id}' not found.{hint}"
    if not module.quiz_files:
        has_quiz = sorted(mid for mid, m in COURSE.modules.items() if m.quiz_files)
        return (
            f"No quizzes available for module '{module_id}'. "
            f"Modules with quizzes: {', '.join(has_quiz)}."
        )
    return fetch_quiz_content(module.quiz_files)


@mcp.tool()
def get_notebook_content(
    module_id: str,
    kind: str = "practical",
    include_code: bool = True,
) -> str:
    """Fetch the actual content of a course notebook from GitHub.

    Returns markdown explanations and (optionally) code cells.
    Use this when a student wants to understand what a notebook covers,
    needs help with an exercise, or asks about specific code.
    kind: 'intro' | 'practical' | 'solution' | 'bonus' | 'homework' (default: practical)
    include_code: set False for explanations only (default: True)
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"Module '{module_id}' not found.{hint}"
    notebooks = [nb for nb in module.notebooks if nb.kind.value == kind]
    if not notebooks:
        kinds = sorted({nb.kind.value for nb in module.notebooks})
        return (
            f"No '{kind}' notebooks for Module {module.id}. "
            f"Available kinds: {kinds}. "
            f"Try get_notebook_content with kind set to one of: {', '.join(kinds)}."
        )
    parts = []
    for nb in notebooks:
        parts.append(f"# {nb.title}\n")
        parts.append(fetch_notebook_content(nb.raw_url, include_code=include_code))
    result = "\n\n---\n\n".join(parts)
    other_kinds = sorted({nb.kind.value for nb in module.notebooks if nb.kind.value != kind})
    if other_kinds:
        result += f"\n\n> **Note:** this module also has notebooks of kind: {', '.join(other_kinds)}. Call get_notebook_content with the appropriate kind to access them."
    return result


@mcp.tool()
def get_notebook_exercises(module_id: str, kind: str = "practical") -> str:
    """Fetch only the exercise cells from a module's notebook.

    Returns the exercise prompt (markdown) and the skeleton code the student must fill in.
    Skips all expository text, imports, and solution code.
    Prefer this over get_notebook_content when helping a student with an exercise —
    it gives the task without spoiling surrounding context.
    kind: 'intro' | 'practical' | 'solution' | 'bonus' | 'homework' (default: practical)
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"Module '{module_id}' not found.{hint}"
    notebooks = [nb for nb in module.notebooks if nb.kind.value == kind]
    if not notebooks:
        kinds = sorted({nb.kind.value for nb in module.notebooks})
        return (
            f"No '{kind}' notebooks for Module {module.id}. "
            f"Available kinds: {kinds}. "
            f"Try get_notebook_exercises with kind set to one of: {', '.join(kinds)}."
        )
    parts = []
    for nb in notebooks:
        parts.append(f"# {nb.title}\n")
        parts.append(fetch_notebook_exercises(nb.raw_url))
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def get_page_content(module_id: str) -> str:
    """Fetch the text content of the course website page for a module.

    Returns the lecture notes, explanations, and learning objectives as plain text.
    Use this for conceptual questions about a topic.
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"Module '{module_id}' not found.{hint}"
    return fetch_module_markdown(module.website_url)


@mcp.tool()
def get_course_overview() -> str:
    """Get a complete overview of the dataflowr course: all sessions, modules, and their relationships.

    Use this to understand the full structure or to give a student a learning path.
    """
    lines = [
        f"# {COURSE.title}",
        "",
        f"{COURSE.description}",
        "",
        f"- Website: {COURSE.website_url}",
        f"- GitHub: {COURSE.github_url}",
        f"- {len(COURSE.modules)} modules across {len(COURSE.sessions)} sessions",
        f"- {len(COURSE.homeworks)} graded homeworks",
        "",
        "## Sessions Overview",
    ]
    for s in COURSE.sessions:
        lines.append("")
        lines.append(_session_to_text(s, include_modules=False))
    lines.append("")
    lines.append("## Homeworks")
    for hw in COURSE.homeworks:
        lines.append(f"- **HW{hw.id}: {hw.title}** — {hw.description[:80]}")
    return "\n".join(lines)


@mcp.tool()
def sync_catalog() -> str:
    """Compare the local catalog against the dataflowr/website, /slides, and /quiz GitHub repos.

    Lists modules/slides present in the repos but missing from the catalog,
    and catalog entries that have no corresponding source file.
    """
    # Website sync
    website_files = list_website_modules()
    website_slugs = {f["slug"] for f in website_files}
    catalog_slugs = {
        m.website_url.rstrip("/").split("/modules/")[-1]
        for m in COURSE.modules.values()
    }
    web_in_repo_not_catalog = sorted(website_slugs - catalog_slugs)
    web_in_catalog_not_repo = sorted(catalog_slugs - website_slugs)

    # Slides sync
    slide_files = list_slide_files()
    slide_filenames = {f["name"] for f in slide_files}
    catalog_slide_filenames = {
        m.slides_url.rstrip("/").split("/")[-1]
        for m in COURSE.modules.values()
        if m.slides_url
    }
    slides_in_repo_not_catalog = sorted(slide_filenames - catalog_slide_filenames)
    slides_in_catalog_not_repo = sorted(catalog_slide_filenames - slide_filenames)

    # Quiz sync
    quiz_files_list = list_quiz_files()
    quiz_filenames = {f["name"] for f in quiz_files_list}
    catalog_quiz_filenames = {
        qf
        for m in COURSE.modules.values()
        for qf in m.quiz_files
    }
    quizzes_in_repo_not_catalog = sorted(quiz_filenames - catalog_quiz_filenames)
    quizzes_in_catalog_not_repo = sorted(catalog_quiz_filenames - quiz_filenames)

    lines = [
        "## Catalog sync: dataflowr/website + dataflowr/slides + dataflowr/quiz",
        "",
        "### Website repo (dataflowr/website)",
        f"Repo: {len(website_slugs)} module files | Catalog: {len(catalog_slugs)} modules",
        "",
    ]
    if web_in_repo_not_catalog:
        lines.append("In repo but NOT in catalog:")
        for slug in web_in_repo_not_catalog:
            lines.append(f"- `{slug}`")
    else:
        lines.append("All website repo modules are in the catalog. ✓")
    if web_in_catalog_not_repo:
        lines.append("\nIn catalog but NOT in repo:")
        for slug in web_in_catalog_not_repo:
            lines.append(f"- `{slug}`")
    else:
        lines.append("All catalog modules have a source file in the repo. ✓")

    lines += [
        "",
        "### Slides repo (dataflowr/slides)",
        f"Repo: {len(slide_filenames)} slide files | Catalog: {len(catalog_slide_filenames)} slides referenced",
        "",
    ]
    if slides_in_repo_not_catalog:
        lines.append("In repo but NOT referenced in catalog:")
        for name in slides_in_repo_not_catalog:
            lines.append(f"- `{name}`")
    else:
        lines.append("All slide files are referenced in the catalog. ✓")
    if slides_in_catalog_not_repo:
        lines.append("\nReferenced in catalog but NOT in repo:")
        for name in slides_in_catalog_not_repo:
            lines.append(f"- `{name}`")
    else:
        lines.append("All catalog slide references point to existing files. ✓")

    lines += [
        "",
        "### Quiz repo (dataflowr/quiz)",
        f"Repo: {len(quiz_filenames)} quiz files | Catalog: {len(catalog_quiz_filenames)} quizzes referenced",
        "",
    ]
    if quizzes_in_repo_not_catalog:
        lines.append("In repo but NOT referenced in catalog:")
        for name in quizzes_in_repo_not_catalog:
            lines.append(f"- `{name}`")
    else:
        lines.append("All quiz files are referenced in the catalog. ✓")
    if quizzes_in_catalog_not_repo:
        lines.append("\nReferenced in catalog but NOT in repo:")
        for name in quizzes_in_catalog_not_repo:
            lines.append(f"- `{name}`")
    else:
        lines.append("All catalog quiz references point to existing files. ✓")

    return "\n".join(lines)


@mcp.tool()
def get_prerequisites(module_id: str) -> str:
    """Get the prerequisite modules for a given module, with full details for each.

    Use this when a student asks 'what should I know before studying X?' or
    seems to be missing background knowledge.
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return (
            f"Module '{module_id}' not found.{hint} "
            f"Use list_modules to see all {len(COURSE.modules)} available modules."
        )
    if not module.prerequisites:
        return (
            f"Module {module.id} ({module.title}) has no prerequisites — "
            f"it's a good starting point!"
        )
    lines = [
        f"## Prerequisites for Module {module.id}: {module.title}",
        "",
        "Before starting this module, you should be comfortable with:",
        "",
    ]
    for prereq_id in module.prerequisites:
        prereq = COURSE.get_module(prereq_id)
        if prereq:
            lines.append(f"### Module {prereq.id}: {prereq.title}")
            lines.append(f"{prereq.description}")
            lines.append(f"- Session: {prereq.session if prereq.session is not None else '— (external course)'}")
            lines.append(f"- Tags: {', '.join(prereq.tags)}")
            lines.append(f"- Website: {prereq.website_url}")
            lines.append("")
        else:
            lines.append(f"- Module {prereq_id} (details not in catalog)")
    return "\n".join(lines)


@mcp.tool()
def check_quiz_answer(module_id: str, question_number: int, answer_number: int) -> str:
    """Check whether a student's answer to a quiz question is correct.

    Returns: correct (bool), the right answer with its text, and an explanation.
    Call get_quiz_content first to see the questions and choices,
    then use this tool to validate the student's response.
    question_number and answer_number are 1-based.
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"Module '{module_id}' not found.{hint}"
    if not module.quiz_files:
        has_quiz = sorted(mid for mid, m in COURSE.modules.items() if m.quiz_files)
        return (
            f"No quizzes available for module '{module_id}'. "
            f"Modules with quizzes: {', '.join(has_quiz)}."
        )
    result = _check_quiz_answer(module.quiz_files, question_number, answer_number)
    if "error" in result:
        return f"Error: {result['error']}"
    if result["correct"]:
        lines = [f"✓ Correct! The answer is {result['correct_number']}. {result['correct_choice']}"]
    else:
        lines = [
            f"✗ Incorrect. You chose {answer_number}. {result['student_choice']}",
            f"The correct answer is {result['correct_number']}. {result['correct_choice']}",
        ]
    if result.get("context"):
        lines.append(f"\nExplanation: {result['context']}")
    return "\n".join(lines)


@mcp.tool()
def suggest_next(module_id: str) -> str:
    """Suggest what to study after completing a given module.

    Returns: (1) modules that directly list this one as a prerequisite,
    (2) the next modules in the same session, and (3) the start of the next session.
    Use this when a student finishes a module and asks 'what should I do next?'
    """
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return (
            f"Module '{module_id}' not found.{hint} "
            f"Use list_modules to see all {len(COURSE.modules)} available modules."
        )

    direct_next = [m for m in COURSE.modules.values() if module.id in m.prerequisites]

    session_modules = COURSE.get_session_modules(module.session) if module.session is not None else []
    session_ids = [m.id for m in session_modules]
    same_session_next: list = []
    if module.id in session_ids:
        idx = session_ids.index(module.id)
        same_session_next = session_modules[idx + 1: idx + 3]

    next_session_start: list = []
    next_session = next(
        (s for s in COURSE.sessions if module.session is not None and s.number == module.session + 1),
        None,
    )
    if next_session and next_session.modules:
        first_next = COURSE.get_module(next_session.modules[0])
        if first_next:
            next_session_start = [first_next]

    lines = [f"## What to study after Module {module.id}: {module.title}", ""]

    if direct_next:
        lines.append("### Modules that directly build on this one:")
        lines.append("")
        for m in direct_next:
            session_label = f"Session {m.session}" if m.session is not None else "external"
            lines.append(f"**Module {m.id}: {m.title}** ({session_label})")
            lines.append(f"{m.description}")
            lines.append(f"- Tags: {', '.join(m.tags)}")
            lines.append(f"- Website: {m.website_url}")
            lines.append("")

    if same_session_next:
        lines.append("### Continue in the same session:")
        lines.append("")
        for m in same_session_next:
            lines.append(f"**Module {m.id}: {m.title}**")
            lines.append(f"{m.description}")
            lines.append(f"- Website: {m.website_url}")
            lines.append("")

    if next_session_start:
        lines.append("### Start of the next session:")
        lines.append("")
        for m in next_session_start:
            session_label = f"Session {m.session}" if m.session is not None else "external"
            lines.append(f"**Module {m.id}: {m.title}** ({session_label})")
            lines.append(f"{m.description}")
            lines.append(f"- Website: {m.website_url}")
            lines.append("")

    if not direct_next and not same_session_next and not next_session_start:
        lines.append(
            "This is one of the final modules in the course. "
            "Consider tackling the graded homeworks or exploring the bonus modules."
        )

    return "\n".join(lines)


# ── Transcript knowledge base ─────────────────────────────────────────────


@mcp.tool()
def search_transcripts(query: str) -> str:
    """Search the knowledge base of 318 concept notes extracted from lecture transcripts.

    Use this tool whenever the user asks what the professor says, wants lecture quotes,
    or asks about spoken explanations from the course videos.
    Fuzzy-matches against concept names (e.g. 'backprop', 'training loop', 'dropout').
    Returns matching concept names. Then call get_transcript_note to read the full content
    with timestamped lecture quotes.
    """
    results = search_transcript_notes(query)
    if not results:
        return (
            f"No transcript notes found for '{query}'. "
            f"Try different keywords (e.g. 'gradient', 'convolution', 'loss')."
        )
    lines = [f"Found {len(results)} concept note(s) for '{query}':\n"]
    for note in results:
        lines.append(f"- **{note['concept']}**")
    return "\n".join(lines)


@mcp.tool()
def get_transcript_note(concept: str) -> str:
    """Fetch the full content of a concept note from the transcript knowledge base.

    Returns the note with concept summary, timestamped lecture quotes, and cross-references.
    Use search_transcripts first to find the exact concept name.
    concept: the concept name, e.g. 'training loop', 'backpropagation', 'dropout'
    """
    try:
        return fetch_transcript_note(concept)
    except RuntimeError:
        results = search_transcript_notes(concept)
        if results:
            suggestions = [r["concept"] for r in results[:5]]
            return (
                f"Concept '{concept}' not found. Did you mean: {', '.join(suggestions)}? "
                f"Use search_transcripts to find the exact name."
            )
        return (
            f"Concept '{concept}' not found. "
            f"Use search_transcripts to find available concepts."
        )


# ── Prompts ────────────────────────────────────────────────────────────────


@mcp.prompt()
def explain_module(module_id: str) -> str:
    """Start a tutoring session for a specific module.

    The agent fetches the module content, checks prerequisites, and explains
    key concepts step-by-step using the Socratic method.
    """
    return (
        f"Please be my tutor for Module {module_id} of the dataflowr "
        f"Deep Learning DIY course.\n\n"
        f"Follow these steps:\n"
        f"1. Call `get_prerequisites` for '{module_id}' and briefly tell me what "
        f"   background I need — flag anything I should review first.\n"
        f"2. Call `get_module` for '{module_id}' to get the full module details.\n"
        f"3. Call `get_page_content` for '{module_id}' to read the lecture notes.\n"
        f"4. Explain the key concepts clearly, using concrete examples and "
        f"   analogies. Assume I know the prerequisites but nothing more.\n"
        f"5. After your explanation, ask me a question to check my understanding "
        f"   before continuing — don't lecture without pausing.\n\n"
        f"Use the Socratic method throughout: prefer questions over statements."
    )


@mcp.prompt()
def quiz_student(module_id: str) -> str:
    """Start an interactive quiz session for a module.

    The agent presents one question at a time, waits for the student's answer,
    then gives feedback before moving to the next question.
    """
    return (
        f"I want to test my knowledge of Module {module_id} from the dataflowr course.\n\n"
        f"Follow these steps:\n"
        f"1. Call `get_quiz_content` for '{module_id}'.\n"
        f"   - If no quiz exists, call `get_page_content` for '{module_id}' and "
        f"     write 4-5 multiple-choice questions yourself.\n"
        f"2. Present ONE question at a time. Do not reveal the answer yet.\n"
        f"3. Wait for my answer before saying whether I'm right.\n"
        f"4. After my answer, give a brief explanation — right or wrong.\n"
        f"5. After all questions, summarise my score and list any concepts "
        f"   I should revisit.\n\n"
        f"Start with question 1 now."
    )


@mcp.prompt()
def debug_help(module_id: str) -> str:
    """Help a student debug their code for a module's practical notebook.

    The agent reads the exercises and uses Socratic questioning to guide
    the student toward the solution without giving it away.
    """
    return (
        f"I'm working on the practical notebook for Module {module_id} of the "
        f"dataflowr course and I'm stuck.\n\n"
        f"Follow these steps:\n"
        f"1. Call `get_notebook_content` for '{module_id}', kind='practical', "
        f"   include_code=false — read the exercises without seeing solutions.\n"
        f"2. Ask me to describe exactly what I'm trying to do and what error "
        f"   or unexpected behaviour I'm seeing.\n"
        f"3. Guide me with hints and questions rather than giving the answer. "
        f"   Use the Socratic method.\n"
        f"4. Only show solution code if I'm still stuck after three attempts.\n\n"
        f"Start by asking me what I'm working on."
    )


@mcp.prompt()
def learning_path(target_module_id: str, known_modules: str = "") -> str:
    """Build a personalised learning path toward a target module.

    The agent walks the full prerequisite chain and orders the modules
    the student still needs to cover.
    known_modules: comma-separated IDs of modules already completed (optional)
    """
    known_text = (
        f"I have already completed: {known_modules}."
        if known_modules
        else "I'm starting from scratch."
    )
    return (
        f"I want to reach Module {target_module_id} of the dataflowr course. "
        f"{known_text}\n\n"
        f"Please build me a personalised learning path:\n"
        f"1. Call `get_prerequisites` for '{target_module_id}'.\n"
        f"2. For each prerequisite, call `get_prerequisites` again "
        f"   (recursively) to build the full dependency chain.\n"
        f"3. Remove any modules I've already completed.\n"
        f"4. Present the ordered list I need to study, with a one-line "
        f"   description and website link for each module.\n"
        f"5. Group them into logical study sessions.\n\n"
        f"Be concrete — I want to know exactly what to do next."
    )


# ── Entry point ────────────────────────────────────────────────────────────


def run_mcp_server():
    """Run the MCP server.

    Default: stdio transport (for Claude Desktop, Cursor, VS Code, Claude Code).
    Pass --http to use Streamable HTTP transport instead.
    Port defaults to 8001, or the PORT environment variable (set by cloud platforms).
    """
    import os
    import sys
    if "--http" in sys.argv:
        import uvicorn
        port = int(os.environ.get("PORT", 8001))
        uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    run_mcp_server()
