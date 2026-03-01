"""Data models for the dataflowr course."""

import difflib
import re
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class NotebookKind(str, Enum):
    intro = "intro"
    practical = "practical"
    solution = "solution"
    bonus = "bonus"
    homework = "homework"


class Notebook(BaseModel):
    filename: str
    title: str
    kind: NotebookKind
    github_url: str
    colab_url: Optional[str] = None
    requires_gpu: bool = False

    @property
    def raw_url(self) -> str:
        return self.github_url.replace(
            "github.com", "raw.githubusercontent.com"
        ).replace("/blob/", "/")


class Module(BaseModel):
    id: str                          # e.g. "2a", "9b", "12"
    title: str
    description: str
    session: Optional[int] = None    # which session this belongs to (None for external/standalone modules)
    website_url: str
    slides_url: Optional[str] = None
    video_url: Optional[str] = None
    notebooks: list[Notebook] = []
    prerequisites: list[str] = []   # list of module ids
    tags: list[str] = []
    requires_gpu: bool = False
    quiz_files: list[str] = []      # TOML quiz files in dataflowr/quiz repo

    @property
    def folder(self) -> str:
        """GitHub folder name, e.g. 'Module12'."""
        num = re.sub(r'[a-z]+$', '', self.id)
        return f"Module{num}"


class Session(BaseModel):
    number: int
    title: str
    modules: list[str]              # list of module ids
    things_to_remember: list[str] = []


class Homework(BaseModel):
    id: int
    title: str
    description: str
    website_url: str
    notebooks: list[Notebook] = []


class Course(BaseModel):
    title: str
    description: str
    github_url: str
    website_url: str
    modules: dict[str, Module]      # keyed by module id
    sessions: list[Session]
    homeworks: list[Homework]

    def get_module(self, module_id: str) -> Optional[Module]:
        # Exact match first, then case-insensitive fallback
        if module_id in self.modules:
            return self.modules[module_id]
        module_id_lower = module_id.lower().strip()
        for k, v in self.modules.items():
            if k.lower() == module_id_lower:
                return v
        return None

    def suggest_module_ids(self, query: str, n: int = 3) -> list[str]:
        """Return close module ID matches for a mistyped query (e.g. '2A' → '2a')."""
        query_lower = query.lower().strip()
        keys = list(self.modules.keys())
        return difflib.get_close_matches(query_lower, keys, n=n, cutoff=0.4)

    def search(self, query: str) -> list[Module]:
        """Simple keyword search across module titles, descriptions, tags."""
        q = query.lower()
        results = []
        for module in self.modules.values():
            searchable = " ".join([
                module.title,
                module.description,
                " ".join(module.tags),
                " ".join(nb.title for nb in module.notebooks),
            ]).lower()
            if q in searchable:
                results.append(module)
        return results

    def get_session_modules(self, session_number: int) -> list[Module]:
        for session in self.sessions:
            if session.number == session_number:
                return [
                    self.modules[mid]
                    for mid in session.modules
                    if mid in self.modules
                ]
        return []
