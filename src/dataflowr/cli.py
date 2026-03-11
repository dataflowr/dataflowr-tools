"""
dataflowr CLI

Usage:
    dataflowr modules list
    dataflowr modules list --session 7
    dataflowr module get 12
    dataflowr module notebooks 12
    dataflowr sessions list
    dataflowr session get 7
    dataflowr search "attention transformer"
    dataflowr homeworks list
"""

import json
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
from typing import Optional

from .catalog import COURSE
from .content import (fetch_notebook_content, fetch_module_markdown, list_website_modules,
                       fetch_slide_content, list_slide_files,
                       fetch_quiz_content, list_quiz_files,
                       parse_quiz_questions, check_quiz_answer,
                       search_transcript_notes, fetch_transcript_note)

app = typer.Typer(
    name="dataflowr",
    help="CLI for the Deep Learning DIY course (dataflowr.github.io)",
    no_args_is_help=True,
)
modules_app = typer.Typer(help="Browse course modules", no_args_is_help=True)
sessions_app = typer.Typer(help="Browse course sessions", no_args_is_help=True)
homeworks_app = typer.Typer(help="Browse homeworks", no_args_is_help=True)

app.add_typer(modules_app, name="modules")
app.add_typer(sessions_app, name="sessions")
app.add_typer(homeworks_app, name="homeworks")

console = Console()

_NOTEBOOK_ICONS = {
    "intro": "📖",
    "practical": "✏️ ",
    "solution": "✅",
    "bonus": "🌟",
    "homework": "📝",
}


def _resolve_module_or_exit(module_id: str):
    """Return the Module for *module_id*, or print an error and exit."""
    module = COURSE.get_module(module_id)
    if not module:
        rprint(f"[red]Module '{module_id}' not found.[/red]")
        if suggestions := COURSE.suggest_module_ids(module_id):
            rprint(f"[yellow]Did you mean: {', '.join(suggestions)}?[/yellow]")
        rprint(f"[dim]Run [bold]dataflowr modules list[/bold] to see all {len(COURSE.modules)} modules.[/dim]")
        raise typer.Exit(1)
    return module


# ── Modules ────────────────────────────────────────────────────────────────

@modules_app.command("list")
def modules_list(
    session: Optional[int] = typer.Option(None, "--session", "-s",
                                           help="Filter by session number"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    gpu_only: bool = typer.Option(False, "--gpu", help="Show only GPU-required modules"),
):
    """List all modules, optionally filtered by session, tag, or GPU requirement."""
    if session is not None:
        modules = COURSE.get_session_modules(session)
    else:
        modules = list(COURSE.modules.values())

    if tag is not None:
        tag_lower = tag.lower()
        modules = [m for m in modules if any(tag_lower in t.lower() for t in m.tags)]

    if gpu_only:
        modules = [m for m in modules if m.requires_gpu]

    if json_output:
        typer.echo(json.dumps([m.model_dump() for m in modules], indent=2))
        return

    table = Table(title="Dataflowr Modules", show_lines=True)
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Session", style="dim", width=8)
    table.add_column("Title", style="bold")
    table.add_column("Tags", style="dim")
    table.add_column("GPU", width=4)

    for m in modules:
        table.add_row(
            m.id,
            str(m.session) if m.session is not None else "—",
            m.title,
            ", ".join(m.tags[:3]),
            "⚡" if m.requires_gpu else "",
        )
    console.print(table)


@app.command("module")
def module_get(
    module_id: str = typer.Argument(help="Module ID (e.g. 12, 2a, 18b)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get full details for a module."""
    module = _resolve_module_or_exit(module_id)

    if json_output:
        typer.echo(module.model_dump_json(indent=2))
        return

    gpu_badge = " [yellow]⚡ GPU required[/yellow]" if module.requires_gpu else ""
    header = Text()
    header.append(f"Module {module.id}: ", style="dim")
    header.append(module.title, style="bold white")

    console.print(Panel(header, expand=False))
    console.print(f"\n[bold]Description:[/bold] {module.description}{gpu_badge}")
    console.print(f"[bold]Session:[/bold]     {module.session if module.session is not None else '— (external course)'}")
    console.print(f"[bold]Tags:[/bold]        {', '.join(module.tags)}")
    console.print(f"[bold]Website:[/bold]     [link={module.website_url}]{module.website_url}[/link]")

    if module.slides_url:
        console.print(f"[bold]Slides:[/bold]      [link={module.slides_url}]{module.slides_url}[/link]")

    if module.notebooks:
        console.print("\n[bold]Notebooks:[/bold]")
        for nb in module.notebooks:
            icon = _NOTEBOOK_ICONS.get(nb.kind.value, "📄")
            gpu = " ⚡" if nb.requires_gpu else ""
            console.print(f"  {icon} [cyan]{nb.filename}[/cyan]{gpu}")
            console.print(f"     {nb.title}")
            console.print(f"     GitHub: [link={nb.github_url}]{nb.github_url}[/link]")
            if nb.colab_url:
                console.print(f"     Colab:  [link={nb.colab_url}]{nb.colab_url}[/link]")


# ── Sessions ───────────────────────────────────────────────────────────────

@sessions_app.command("list")
def sessions_list(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List all sessions with their modules."""
    if json_output:
        typer.echo(json.dumps([s.model_dump() for s in COURSE.sessions], indent=2))
        return

    table = Table(title="Dataflowr Sessions", show_lines=True)
    table.add_column("Session", style="cyan", width=8)
    table.add_column("Title", style="bold")
    table.add_column("Modules")

    for s in COURSE.sessions:
        table.add_row(str(s.number), s.title, ", ".join(s.modules))
    console.print(table)


@sessions_app.command("get")
def session_get(
    session_number: int = typer.Argument(help="Session number"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get full details for a session including all its modules."""
    session = next((s for s in COURSE.sessions if s.number == session_number), None)
    if not session:
        valid = sorted(s.number for s in COURSE.sessions)
        rprint(f"[red]Session {session_number} not found.[/red]")
        rprint(f"[dim]Valid sessions: {', '.join(str(n) for n in valid)}. "
               f"Run [bold]dataflowr sessions list[/bold] for details.[/dim]")
        raise typer.Exit(1)

    if json_output:
        modules = COURSE.get_session_modules(session_number)
        output = {
            "session": session.model_dump(),
            "modules": [m.model_dump() for m in modules],
        }
        typer.echo(json.dumps(output, indent=2))
        return

    console.print(Panel(
        f"[bold]Session {session.number}: {session.title}[/bold]",
        expand=False,
    ))
    modules = COURSE.get_session_modules(session_number)
    for m in modules:
        gpu = " ⚡" if m.requires_gpu else ""
        console.print(f"\n  [cyan bold]Module {m.id}[/cyan bold]{gpu}: {m.title}")
        console.print(f"  {m.description}")
        console.print(f"  [link={m.website_url}]{m.website_url}[/link]")

    if session.things_to_remember:
        console.print("\n[bold]Things to remember:[/bold]")
        for thing in session.things_to_remember:
            console.print(f"  • {thing}")


# ── Homeworks ──────────────────────────────────────────────────────────────

@homeworks_app.command("list")
def homeworks_list(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List all homeworks."""
    if json_output:
        typer.echo(json.dumps([h.model_dump() for h in COURSE.homeworks], indent=2))
        return

    table = Table(title="Dataflowr Homeworks", show_lines=True)
    table.add_column("HW", style="cyan", width=4)
    table.add_column("Title", style="bold")
    table.add_column("Description")

    for hw in COURSE.homeworks:
        table.add_row(str(hw.id), hw.title, hw.description[:80] + "...")
    console.print(table)


@homeworks_app.command("get")
def homework_get(
    hw_id: int = typer.Argument(help="Homework ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get full details for a homework."""
    hw = next((h for h in COURSE.homeworks if h.id == hw_id), None)
    if not hw:
        valid = ', '.join(str(h.id) for h in COURSE.homeworks)
        rprint(f"[red]Homework {hw_id} not found.[/red]")
        rprint(f"[dim]Available: {valid}. Run [bold]dataflowr homeworks list[/bold] to see all homeworks.[/dim]")
        raise typer.Exit(1)

    if json_output:
        typer.echo(hw.model_dump_json(indent=2))
        return

    console.print(Panel(f"[bold]HW{hw.id}: {hw.title}[/bold]", expand=False))
    console.print(f"\n[bold]Description:[/bold] {hw.description}")
    console.print(f"[bold]Website:[/bold]     [link={hw.website_url}]{hw.website_url}[/link]")

    if hw.notebooks:
        console.print("\n[bold]Notebooks:[/bold]")
        for nb in hw.notebooks:
            if nb.kind.value != "solution":
                icon = _NOTEBOOK_ICONS.get(nb.kind.value, "📄")
                gpu = " ⚡" if nb.requires_gpu else ""
                console.print(f"  {icon} [cyan]{nb.filename}[/cyan]{gpu}")
                console.print(f"     {nb.title}")
                console.print(f"     GitHub: [link={nb.github_url}]{nb.github_url}[/link]")
                if nb.colab_url:
                    console.print(f"     Colab:  [link={nb.colab_url}]{nb.colab_url}[/link]")


# ── Notebook content / Page content ────────────────────────────────────────

@app.command()
def notebook(
    module_id: str = typer.Argument(help="Module ID (e.g. 12, 2a, 18b)"),
    kind: str = typer.Option("practical", "--kind", "-k",
                              help="Notebook kind: intro, practical, solution, bonus, homework"),
    no_code: bool = typer.Option(False, "--no-code", help="Exclude code cells"),
):
    """Fetch and print notebook content from GitHub."""
    module = _resolve_module_or_exit(module_id)

    notebooks = [nb for nb in module.notebooks if nb.kind.value == kind]
    if not notebooks:
        kinds = sorted({nb.kind.value for nb in module.notebooks})
        rprint(f"[red]No '{kind}' notebooks for module '{module_id}'.[/red]")
        if kinds:
            rprint(f"[dim]Available kinds: {', '.join(kinds)}. "
                   f"Try: [bold]dataflowr notebook {module_id} --kind {kinds[0]}[/bold][/dim]")
        else:
            rprint(f"[dim]Module '{module_id}' has no notebooks.[/dim]")
        raise typer.Exit(1)

    for nb in notebooks:
        console.print(f"\n[bold]# {nb.title}[/bold]\n")
        content = fetch_notebook_content(nb.raw_url, include_code=not no_code)
        console.print(content)


@app.command()
def page(
    module_id: str = typer.Argument(help="Module ID (e.g. 12, 2a, 18b)"),
):
    """Fetch and print the course website page content for a module."""
    module = _resolve_module_or_exit(module_id)
    content = fetch_module_markdown(module.website_url)
    console.print(content)


@app.command()
def slides(
    module_id: str = typer.Argument(help="Module ID (e.g. 12, 2a, 18b)"),
):
    """Fetch and print slide content for a module from the dataflowr/slides GitHub repo."""
    module = _resolve_module_or_exit(module_id)
    if not module.slides_url:
        has_slides = sorted(mid for mid, m in COURSE.modules.items() if m.slides_url)
        rprint(f"[yellow]No slides available for module '{module_id}'.[/yellow]")
        rprint(f"[dim]Modules with slides: {', '.join(has_slides)}[/dim]")
        raise typer.Exit(0)
    content = fetch_slide_content(module.slides_url)
    console.print(content)


@app.command()
def quiz(
    module_id: str = typer.Argument(help="Module ID (e.g. 2a, 2b, 3)"),
    show: bool = typer.Option(False, "--show", help="Display all questions with answers instead of interactive mode"),
):
    """Take an interactive quiz for a module, with answer checking.

    Presents questions one by one, accepts your answer, and gives immediate
    feedback with the explanation. Shows your final score at the end.

    Use --show to display all questions and correct answers at once.
    """
    module = _resolve_module_or_exit(module_id)
    if not module.quiz_files:
        has_quiz = sorted(mid for mid, m in COURSE.modules.items() if m.quiz_files)
        rprint(f"[yellow]No quizzes available for module '{module_id}'.[/yellow]")
        rprint(f"[dim]Modules with quizzes: {', '.join(has_quiz)}[/dim]")
        raise typer.Exit(0)

    if show:
        content = fetch_quiz_content(module.quiz_files)
        console.print(content)
        return

    # Interactive mode
    questions = parse_quiz_questions(module.quiz_files)
    if not questions:
        rprint("[yellow]Interactive mode requires Python 3.11+ (tomllib). Showing display mode instead.[/yellow]")
        console.print(fetch_quiz_content(module.quiz_files))
        return

    console.print(Panel(
        f"[bold]Quiz — Module {module.id}: {module.title}[/bold]\n"
        f"[dim]{len(questions)} question(s). Type the number of your answer.[/dim]",
        expand=False,
    ))

    score = 0
    for q in questions:
        console.print(f"\n[bold cyan]Q{q['index']}.[/bold cyan] {q['text']}\n")
        for j, choice in enumerate(q["choices"], 1):
            console.print(f"  [dim]{j}.[/dim] {choice}")

        while True:
            raw = typer.prompt(f"\nYour answer (1–{len(q['choices'])})")
            try:
                answer_num = int(raw.strip())
                if 1 <= answer_num <= len(q["choices"]):
                    break
            except ValueError:
                pass
            rprint(f"[red]Please enter a number between 1 and {len(q['choices'])}.[/red]")

        result = check_quiz_answer(module.quiz_files, q["index"], answer_num)
        if result.get("correct"):
            score += 1
            rprint("[green]✓ Correct![/green]")
        else:
            rprint(
                f"[red]✗ Incorrect.[/red] "
                f"The right answer was [bold]{result['correct_number']}. {result['correct_choice']}[/bold]"
            )
        if result.get("context"):
            rprint(f"[dim]{result['context']}[/dim]")

    pct = score / len(questions) * 100
    color = "green" if pct >= 70 else "yellow" if pct >= 50 else "red"
    console.print(
        f"\n[bold]Final score: [{color}]{score}/{len(questions)} ({pct:.0f}%)[/{color}][/bold]"
    )
    if pct < 70:
        console.print(
            f"[dim]Review the module at: [link={module.website_url}]{module.website_url}[/link][/dim]"
        )


# ── Transcripts ────────────────────────────────────────────────────────────

transcripts_app = typer.Typer(help="Browse transcript knowledge base", no_args_is_help=True)
app.add_typer(transcripts_app, name="transcripts")


@transcripts_app.command("search")
def transcripts_search(
    query: str = typer.Argument(help="Search query (e.g. 'backprop', 'training loop')"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Search the transcript knowledge base by concept name."""
    results = search_transcript_notes(query)

    if not results:
        rprint(f"[yellow]No concept notes found for '{query}'.[/yellow]")
        rprint("[dim]Try different keywords (e.g. 'gradient', 'convolution', 'loss').[/dim]")
        raise typer.Exit(0)

    if json_output:
        typer.echo(json.dumps(results, indent=2))
        return

    rprint(f"\nFound [bold]{len(results)}[/bold] concept note(s) for '[cyan]{query}[/cyan]':\n")
    for note in results:
        rprint(f"  [cyan bold]{note['concept']}[/cyan bold]")


@transcripts_app.command("get")
def transcripts_get(
    concept: str = typer.Argument(help="Concept name (e.g. 'training loop', 'dropout')"),
):
    """Fetch and display a concept note from the transcript knowledge base."""
    try:
        content = fetch_transcript_note(concept)
        console.print(content)
    except RuntimeError:
        results = search_transcript_notes(concept)
        rprint(f"[red]Concept '{concept}' not found.[/red]")
        if results:
            suggestions = [r["concept"] for r in results[:5]]
            rprint(f"[yellow]Did you mean: {', '.join(suggestions)}?[/yellow]")
        rprint("[dim]Use [bold]dataflowr transcripts search <query>[/bold] to find concepts.[/dim]")
        raise typer.Exit(1)


# ── Search ─────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Search modules by keyword (title, description, tags)."""
    results = COURSE.search(query)

    if not results:
        rprint(f"[yellow]No modules found for '{query}'.[/yellow]")
        rprint(f"[dim]Try different keywords, or run [bold]dataflowr modules list[/bold] "
               f"to browse all {len(COURSE.modules)} modules.[/dim]")
        raise typer.Exit(0)

    if json_output:
        typer.echo(json.dumps([m.model_dump() for m in results], indent=2))
        return

    rprint(f"\nFound [bold]{len(results)}[/bold] module(s) for '[cyan]{query}[/cyan]':\n")
    for m in results:
        gpu = " ⚡" if m.requires_gpu else ""
        session_label = f"Session {m.session}" if m.session is not None else "external"
        rprint(f"  [cyan bold]Module {m.id}[/cyan bold]{gpu} ({session_label}): {m.title}")
        rprint(f"    [dim]{m.description[:100]}...[/dim]")
        rprint(f"    Tags: {', '.join(m.tags[:4])}")
        rprint(f"    [link={m.website_url}]{m.website_url}[/link]\n")


# ── Info ───────────────────────────────────────────────────────────────────

@app.command()
def info(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show course overview."""
    if json_output:
        summary = {
            "title": COURSE.title,
            "description": COURSE.description,
            "website_url": COURSE.website_url,
            "github_url": COURSE.github_url,
            "num_modules": len(COURSE.modules),
            "num_sessions": len(COURSE.sessions),
            "num_homeworks": len(COURSE.homeworks),
        }
        typer.echo(json.dumps(summary, indent=2))
        return

    console.print(Panel(
        f"[bold blue]{COURSE.title}[/bold blue]\n\n"
        f"{COURSE.description}\n\n"
        f"[bold]{len(COURSE.modules)}[/bold] modules  •  "
        f"[bold]{len(COURSE.sessions)}[/bold] sessions  •  "
        f"[bold]{len(COURSE.homeworks)}[/bold] homeworks\n\n"
        f"Website: [link={COURSE.website_url}]{COURSE.website_url}[/link]\n"
        f"GitHub:  [link={COURSE.github_url}]{COURSE.github_url}[/link]",
        title="dataflowr",
        expand=False,
    ))


@app.command()
def sync(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Compare the catalog against the dataflowr/website, /slides, and /quiz GitHub repos."""
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
    quiz_files = list_quiz_files()
    quiz_filenames = {f["name"] for f in quiz_files}
    catalog_quiz_filenames = {
        qf
        for m in COURSE.modules.values()
        for qf in m.quiz_files
    }
    quizzes_in_repo_not_catalog = sorted(quiz_filenames - catalog_quiz_filenames)
    quizzes_in_catalog_not_repo = sorted(catalog_quiz_filenames - quiz_filenames)

    if json_output:
        typer.echo(json.dumps({
            "website": {
                "repo_count": len(website_slugs),
                "catalog_count": len(catalog_slugs),
                "in_repo_not_catalog": web_in_repo_not_catalog,
                "in_catalog_not_repo": web_in_catalog_not_repo,
            },
            "slides": {
                "repo_count": len(slide_filenames),
                "catalog_count": len(catalog_slide_filenames),
                "in_repo_not_catalog": slides_in_repo_not_catalog,
                "in_catalog_not_repo": slides_in_catalog_not_repo,
            },
            "quiz": {
                "repo_count": len(quiz_filenames),
                "catalog_count": len(catalog_quiz_filenames),
                "in_repo_not_catalog": quizzes_in_repo_not_catalog,
                "in_catalog_not_repo": quizzes_in_catalog_not_repo,
            },
        }, indent=2))
        return

    console.print("\n[bold cyan]── Website repo (dataflowr/website) ──[/bold cyan]")
    console.print(f"Repo: {len(website_slugs)} module files  •  Catalog: {len(catalog_slugs)} modules\n")
    if web_in_repo_not_catalog:
        console.print("[bold yellow]In website repo but NOT in catalog:[/bold yellow]")
        for slug in web_in_repo_not_catalog:
            console.print(f"  [yellow]+[/yellow] {slug}")
    else:
        console.print("[green]All website repo modules are in the catalog. ✓[/green]")
    if web_in_catalog_not_repo:
        console.print("\n[bold red]In catalog but NOT in website repo:[/bold red]")
        for slug in web_in_catalog_not_repo:
            console.print(f"  [red]-[/red] {slug}")
    else:
        console.print("[green]All catalog modules have a source file in the repo. ✓[/green]")

    console.print("\n[bold cyan]── Slides repo (dataflowr/slides) ──[/bold cyan]")
    console.print(f"Repo: {len(slide_filenames)} slide files  •  Catalog: {len(catalog_slide_filenames)} slides referenced\n")
    if slides_in_repo_not_catalog:
        console.print("[bold yellow]In slides repo but NOT referenced in catalog:[/bold yellow]")
        for name in slides_in_repo_not_catalog:
            console.print(f"  [yellow]+[/yellow] {name}")
    else:
        console.print("[green]All slide files are referenced in the catalog. ✓[/green]")
    if slides_in_catalog_not_repo:
        console.print("\n[bold red]Referenced in catalog but NOT in slides repo:[/bold red]")
        for name in slides_in_catalog_not_repo:
            console.print(f"  [red]-[/red] {name}")
    else:
        console.print("[green]All catalog slide references point to existing files. ✓[/green]")

    console.print("\n[bold cyan]── Quiz repo (dataflowr/quiz) ──[/bold cyan]")
    console.print(f"Repo: {len(quiz_filenames)} quiz files  •  Catalog: {len(catalog_quiz_filenames)} quizzes referenced\n")
    if quizzes_in_repo_not_catalog:
        console.print("[bold yellow]In quiz repo but NOT referenced in catalog:[/bold yellow]")
        for name in quizzes_in_repo_not_catalog:
            console.print(f"  [yellow]+[/yellow] {name}")
    else:
        console.print("[green]All quiz files are referenced in the catalog. ✓[/green]")
    if quizzes_in_catalog_not_repo:
        console.print("\n[bold red]Referenced in catalog but NOT in quiz repo:[/bold red]")
        for name in quizzes_in_catalog_not_repo:
            console.print(f"  [red]-[/red] {name}")
    else:
        console.print("[green]All catalog quiz references point to existing files. ✓[/green]")


def main():
    app()


if __name__ == "__main__":
    main()
