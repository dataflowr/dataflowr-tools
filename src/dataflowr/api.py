"""
dataflowr REST API

Run with:
    uvicorn dataflowr.api:app --reload

Endpoints:
    GET /                        course overview
    GET /modules                 list all modules
    GET /modules/{id}            get module by id
    GET /modules/{id}/notebooks  get notebooks for a module
    GET /sessions                list all sessions
    GET /sessions/{n}            get session with its modules
    GET /homeworks               list all homeworks
    GET /search?q=...            search modules
    GET /transcripts/search?q=.. search transcript concept notes
    GET /transcripts/{concept}   get a transcript concept note
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional

from .catalog import COURSE
from .content import (fetch_notebook_content, fetch_module_markdown, list_website_modules,
                       fetch_slide_content, list_slide_files,
                       fetch_quiz_content, list_quiz_files,
                       parse_quiz_questions, check_quiz_answer,
                       search_transcript_notes, fetch_transcript_note)
from .models import Module, Session, Homework

app = FastAPI(
    title="dataflowr API",
    description="REST API for the Deep Learning DIY course — dataflowr.github.io",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/", summary="Course overview")
def root():
    """Return a summary of the course."""
    return {
        "title": COURSE.title,
        "description": COURSE.description,
        "website_url": COURSE.website_url,
        "github_url": COURSE.github_url,
        "num_modules": len(COURSE.modules),
        "num_sessions": len(COURSE.sessions),
        "num_homeworks": len(COURSE.homeworks),
        "endpoints": {
            "modules":        "/modules",
            "sessions":       "/sessions",
            "homeworks":      "/homeworks",
            "search":         "/search?q=<query>",
            "quiz_questions": "/modules/{id}/quiz/questions",
            "quiz_check":     "POST /modules/{id}/quiz/check",
            "transcripts_search": "/transcripts/search?q=<query>",
            "transcript_note":   "/transcripts/{concept}",
            "docs":              "/docs",
        },
    }


# ── Modules ────────────────────────────────────────────────────────────────

@app.get("/modules", summary="List all modules")
def list_modules(
    session: Optional[int] = Query(None, description="Filter by session number"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    gpu: Optional[bool] = Query(None, description="Filter by GPU requirement"),
):
    """Return all modules, with optional filters."""
    modules = list(COURSE.modules.values())

    if session is not None:
        modules = [m for m in modules if m.session == session]

    if tag is not None:
        tag_lower = tag.lower()
        modules = [m for m in modules if any(tag_lower in t.lower() for t in m.tags)]

    if gpu is not None:
        modules = [m for m in modules if m.requires_gpu == gpu]

    return [_module_summary(m) for m in modules]


@app.get("/modules/{module_id}", summary="Get a module by ID")
def get_module(module_id: str):
    """Return full details for a module (e.g. '12', '2a', '18b')."""
    return _get_module_or_404(module_id).model_dump()


@app.get("/modules/{module_id}/notebooks", summary="Get notebooks for a module")
def get_module_notebooks(module_id: str, kind: Optional[str] = None):
    """Return all notebooks for a module, optionally filtered by kind.

    Kind values: intro, practical, solution, bonus, homework
    """
    module = _get_module_or_404(module_id)
    notebooks = module.notebooks
    if kind:
        notebooks = [nb for nb in notebooks if nb.kind.value == kind]

    return [nb.model_dump() for nb in notebooks]


@app.get("/modules/{module_id}/notebooks/{kind}/content",
         summary="Get notebook content", response_class=PlainTextResponse)
def get_notebook_content(
    module_id: str,
    kind: str,
    include_code: bool = Query(True, description="Include code cells"),
):
    """Fetch the actual cells of a notebook from GitHub as plain text.

    Kind values: intro, practical, solution, bonus, homework
    """
    module = _get_module_or_404(module_id)
    notebooks = [nb for nb in module.notebooks if nb.kind.value == kind]
    if not notebooks:
        kinds = sorted({nb.kind.value for nb in module.notebooks})
        raise HTTPException(
            status_code=404,
            detail=f"No '{kind}' notebooks for module '{module_id}'. "
                   f"Available kinds: {kinds}. Valid values: intro, practical, solution, bonus, homework.",
        )
    parts = []
    for nb in notebooks:
        parts.append(f"# {nb.title}\n")
        parts.append(fetch_notebook_content(nb.raw_url, include_code=include_code))
    return "\n\n---\n\n".join(parts)


@app.get("/modules/{module_id}/slides",
         summary="Get slide content for a module", response_class=PlainTextResponse)
def get_slides_content(module_id: str):
    """Fetch the Remark.js slide content for a module from the dataflowr/slides GitHub repo."""
    module = _get_module_or_404(module_id)
    if not module.slides_url:
        has_slides = sorted(mid for mid, m in COURSE.modules.items() if m.slides_url)
        raise HTTPException(
            status_code=404,
            detail=f"No slides available for module '{module_id}'. "
                   f"Modules with slides: {has_slides}.",
        )
    return fetch_slide_content(module.slides_url)


@app.get("/modules/{module_id}/quiz",
         summary="Get quiz questions for a module", response_class=PlainTextResponse)
def get_quiz_content(module_id: str):
    """Fetch and format quiz questions for a module from the dataflowr/quiz GitHub repo."""
    module = _get_module_or_404(module_id)
    if not module.quiz_files:
        has_quiz = sorted(mid for mid, m in COURSE.modules.items() if m.quiz_files)
        raise HTTPException(
            status_code=404,
            detail=f"No quizzes available for module '{module_id}'. "
                   f"Modules with quizzes: {has_quiz}.",
        )
    return fetch_quiz_content(module.quiz_files)


class QuizAnswerRequest(BaseModel):
    question: int  # 1-based question number
    answer: int    # 1-based answer choice


@app.get("/modules/{module_id}/quiz/questions",
         summary="Get quiz questions for a module (no answers revealed)")
def get_quiz_questions(module_id: str):
    """Return all quiz questions for a module without revealing the correct answers.

    Each question has: index, text, choices (list).
    Use POST /modules/{id}/quiz/check to verify an answer.
    """
    module = _get_module_or_404(module_id)
    if not module.quiz_files:
        has_quiz = sorted(mid for mid, m in COURSE.modules.items() if m.quiz_files)
        raise HTTPException(
            status_code=404,
            detail=f"No quizzes available for module '{module_id}'. "
                   f"Modules with quizzes: {has_quiz}.",
        )
    questions = parse_quiz_questions(module.quiz_files)
    if not questions:
        raise HTTPException(
            status_code=501,
            detail="Quiz parsing requires Python 3.11+ (tomllib). "
                   "Use GET /modules/{id}/quiz for raw content.",
        )
    return [
        {"index": q["index"], "text": q["text"], "choices": q["choices"]}
        for q in questions
    ]


@app.post("/modules/{module_id}/quiz/check",
          summary="Check a quiz answer")
def check_quiz(module_id: str, body: QuizAnswerRequest):
    """Check a student's answer for a quiz question.

    Body: {"question": <1-based question number>, "answer": <1-based choice number>}

    Returns whether the answer is correct, the correct choice, and the explanation.
    """
    module = _get_module_or_404(module_id)
    if not module.quiz_files:
        raise HTTPException(status_code=404, detail=f"No quizzes available for module '{module_id}'.")
    result = check_quiz_answer(module.quiz_files, body.question, body.answer)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/modules/{module_id}/page",
         summary="Get module source markdown from the website repo", response_class=PlainTextResponse)
def get_page_content(module_id: str):
    """Fetch the raw markdown source for a module from the dataflowr/website GitHub repo."""
    return fetch_module_markdown(_get_module_or_404(module_id).website_url)


# ── Sessions ───────────────────────────────────────────────────────────────

@app.get("/sessions", summary="List all sessions")
def list_sessions():
    """Return all sessions with their module IDs."""
    return [_session_summary(s) for s in COURSE.sessions]


@app.get("/sessions/{session_number}", summary="Get a session with its modules")
def get_session(session_number: int):
    """Return a session and the full details of all its modules."""
    session = next((s for s in COURSE.sessions if s.number == session_number), None)
    if not session:
        valid = sorted(s.number for s in COURSE.sessions)
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_number} not found. Valid sessions: {valid}.",
        )

    modules = COURSE.get_session_modules(session_number)
    return {
        **session.model_dump(),
        "modules_detail": [m.model_dump() for m in modules],
    }


# ── Homeworks ──────────────────────────────────────────────────────────────

@app.get("/homeworks", summary="List all homeworks")
def list_homeworks():
    """Return all homeworks."""
    return [hw.model_dump() for hw in COURSE.homeworks]


@app.get("/homeworks/{hw_id}", summary="Get a homework by ID")
def get_homework(hw_id: int):
    """Return full details for a homework."""
    hw = next((h for h in COURSE.homeworks if h.id == hw_id), None)
    if not hw:
        available = [h.id for h in COURSE.homeworks]
        raise HTTPException(
            status_code=404,
            detail=f"Homework {hw_id} not found. Available homework IDs: {available}.",
        )
    return hw.model_dump()


# ── Search ─────────────────────────────────────────────────────────────────

@app.get("/search", summary="Search modules")
def search(q: str = Query(..., description="Search query")):
    """Search modules by keyword across title, description, and tags."""
    results = COURSE.search(q)
    return {
        "query": q,
        "count": len(results),
        "results": [_module_summary(m) for m in results],
    }


# ── Transcripts ────────────────────────────────────────────────────────────

@app.get("/transcripts/search", summary="Search transcript concept notes")
def search_transcripts(q: str = Query(..., description="Search query (e.g. 'backprop', 'training loop')")):
    """Search the transcript knowledge base by concept name."""
    results = search_transcript_notes(q)
    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.get("/transcripts/{concept}", summary="Get a transcript concept note",
         response_class=PlainTextResponse)
def get_transcript(concept: str):
    """Fetch the full content of a concept note from the transcript knowledge base."""
    try:
        return fetch_transcript_note(concept)
    except RuntimeError:
        results = search_transcript_notes(concept)
        suggestions = [r["concept"] for r in results[:5]] if results else []
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise HTTPException(
            status_code=404,
            detail=f"Concept '{concept}' not found.{hint} "
                   f"Use GET /transcripts/search?q=<query> to find concepts.",
        )


# ── Catalog sync ───────────────────────────────────────────────────────────

@app.get("/catalog/sync", summary="Compare catalog against dataflowr/website, /slides, and /quiz repos")
def catalog_sync():
    """Compare catalog modules against the website, slides, and quiz repos."""
    # Website sync
    website_files = list_website_modules()
    website_slugs = {f["slug"] for f in website_files}
    catalog_slugs = {
        m.website_url.rstrip("/").split("/modules/")[-1]
        for m in COURSE.modules.values()
    }

    # Slides sync
    slide_files = list_slide_files()
    slide_filenames = {f["name"] for f in slide_files}
    catalog_slide_filenames = {
        m.slides_url.rstrip("/").split("/")[-1]
        for m in COURSE.modules.values()
        if m.slides_url
    }

    # Quiz sync
    quiz_files_list = list_quiz_files()
    quiz_filenames = {f["name"] for f in quiz_files_list}
    catalog_quiz_filenames = {
        qf
        for m in COURSE.modules.values()
        for qf in m.quiz_files
    }

    return {
        "website": {
            "repo_count": len(website_slugs),
            "catalog_count": len(catalog_slugs),
            "in_repo_not_catalog": sorted(website_slugs - catalog_slugs),
            "in_catalog_not_repo": sorted(catalog_slugs - website_slugs),
        },
        "slides": {
            "repo_count": len(slide_filenames),
            "catalog_count": len(catalog_slide_filenames),
            "in_repo_not_catalog": sorted(slide_filenames - catalog_slide_filenames),
            "in_catalog_not_repo": sorted(catalog_slide_filenames - slide_filenames),
        },
        "quiz": {
            "repo_count": len(quiz_filenames),
            "catalog_count": len(catalog_quiz_filenames),
            "in_repo_not_catalog": sorted(quiz_filenames - catalog_quiz_filenames),
            "in_catalog_not_repo": sorted(catalog_quiz_filenames - quiz_filenames),
        },
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_module_or_404(module_id: str) -> Module:
    """Return the Module for *module_id*, or raise HTTPException 404."""
    module = COURSE.get_module(module_id)
    if not module:
        suggestions = COURSE.suggest_module_ids(module_id)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise HTTPException(
            status_code=404,
            detail=f"Module '{module_id}' not found.{hint} "
                   f"Use GET /modules to list all {len(COURSE.modules)} available modules.",
        )
    return module


def _module_summary(m: Module) -> dict:
    return {
        "id": m.id,
        "title": m.title,
        "description": m.description,
        "session": m.session,
        "tags": m.tags,
        "requires_gpu": m.requires_gpu,
        "num_notebooks": len(m.notebooks),
        "website_url": m.website_url,
        "slides_url": m.slides_url,
    }


def _session_summary(s: Session) -> dict:
    return {
        "number": s.number,
        "title": s.title,
        "modules": s.modules,
        "num_modules": len(s.modules),
    }
