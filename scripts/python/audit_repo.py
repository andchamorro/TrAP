"""Repository audit: one inventory row per tracked file.

Writes ``docs/guides/inventory.tsv`` with the columns ``path``, ``class``,
``evidence``, ``action`` and ``owner``.  ``evidence`` is recomputed on every
run from the tree: where ``docs/guides/manuscript_provenance.md`` names the
file, how many other tracked files reference it, and its TODO/FIXME, revision-
coupling and ``print`` counts.  ``class``, ``action`` and ``owner`` are hand
decisions: when the inventory already exists they are carried over unchanged,
and only files new since the last run get a rule-based suggestion.

Classes (see ``docs/guides/archive_index.md`` for the ARCHIVE destinations):
REPORTED, PIPELINE, SUPPORT, EXPLORATORY-KEEP, ARCHIVE, REMOVE, ASK.

Usage:
    python scripts/python/audit_repo.py audit          # refresh the inventory
    python scripts/python/audit_repo.py check          # fail on missing/empty rows
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import subprocess

from loguru import logger
import typer

app = typer.Typer()

REPO = Path(__file__).resolve().parents[2]
INVENTORY = REPO / "docs" / "guides" / "inventory.tsv"
PROVENANCE = REPO / "docs" / "guides" / "manuscript_provenance.md"
COLUMNS = ["path", "class", "evidence", "action", "owner"]
CLASSES = {"REPORTED", "PIPELINE", "SUPPORT", "EXPLORATORY-KEEP", "ARCHIVE", "REMOVE", "ASK"}
EXPLORATORY_SECTION = "Not behind any reported result"

TODO_RE = re.compile(r"\b(TODO|FIXME)\b")
#: Guards the comment convention in ``docs/development/contributing.md``: a comment
#: should state the scientific reason, not cite a revision round, a person, a private
#: plan section or an agent instruction file.  The first alternative is spelled in
#: parts so this guard does not itself trip the rule it enforces.
REVISION_COUPLING_RE = re.compile(
    r"review" r"er|\bR[0-9]-[0-9]|CLAUDE\.md|plan §", re.IGNORECASE
)
PRINT_RE = re.compile(r"^\s*print\(", re.MULTILINE)

SUPPORT_PREFIXES = ("tests/", "docs/", "envs/", "recipe/", ".github/", "references/", "reports/")
SUPPORT_FILES = {
    ".gitignore",
    ".zenodo.json",
    "LICENSE",
    "Makefile",
    "README.md",
    "pyproject.toml",
    "setup.cfg",
}


def git(*args: str) -> str:
    """Run a git command in the repository root and return its stdout."""
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout


def tracked_files() -> list[str]:
    """Return every tracked path, sorted."""
    return sorted(git("ls-files").split())


def provenance_sections() -> dict[str, str]:
    """Map each ``##`` section title of the provenance guide to its body."""
    sections: dict[str, str] = {}
    title = "preamble"
    for line in PROVENANCE.read_text().splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
            sections[title] = ""
        else:
            sections[title] = sections.get(title, "") + line + "\n"
    return sections


def provenance_role(path: str, sections: dict[str, str], unique_names: set[str]) -> str:
    """Return ``reported``, ``exploratory`` or ``""`` for a path in the provenance guide.

    A file counts as named when its full path appears, or when its basename appears
    as a code span and no other tracked file shares that basename (the guide's
    tables abbreviate sibling paths, e.g. ``43_seed_heldout.slurm``).
    """
    name = Path(path).name
    roles = set()
    for title, body in sections.items():
        if path in body or (name in unique_names and f"`{name}`" in body):
            roles.add("exploratory" if title == EXPLORATORY_SECTION else "reported")
    if "reported" in roles:
        return "reported"
    return roles.pop() if roles else ""


def reference_pattern(path: str) -> str:
    """Build the Perl regex (git grep -P) that finds references to ``path`` from other files."""
    name = re.escape(Path(path).name)
    if path.startswith("trap/") and path.endswith(".py"):
        module = path[:-3].replace("/__init__", "")
        dotted = re.escape(module.replace("/", "."))
        parent, _, leaf = module.rpartition("/")
        from_import = rf"from {re.escape(parent.replace('/', '.'))} import .*\b{leaf}\b"
        relative = rf"from \.+{leaf}\b|from \. import .*\b{leaf}\b"
        return rf"{name}|{dotted}\b|{from_import}|{relative}"
    return name


def referencing_files(path: str) -> list[str]:
    """Return tracked files, other than ``path`` and the inventory, that reference it."""
    hits = git("grep", "-l", "-P", reference_pattern(path)).split()
    return [h for h in hits if h not in (path, "docs/guides/inventory.tsv")]


def marker_counts(path: str) -> dict[str, int]:
    """Count TODO/FIXME, revision-coupling markers and ``print`` calls in a text file."""
    try:
        text = (REPO / path).read_text()
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return {}
    counts = {
        "todo": len(TODO_RE.findall(text)),
        "revision_coupling": len(REVISION_COUPLING_RE.findall(text)),
        "print": len(PRINT_RE.findall(text)) if path.endswith(".py") else 0,
    }
    return {k: v for k, v in counts.items() if v}


def suggest_class(path: str, role: str, n_refs: int) -> str:
    """Rule-based first guess, used only for files without a hand classification."""
    if role == "reported":
        return "REPORTED"
    if role == "exploratory":
        return "EXPLORATORY-KEEP"
    if path in SUPPORT_FILES or path.startswith(SUPPORT_PREFIXES):
        return "SUPPORT"
    # Revision-round analyses back the response letter, not a manuscript item, so they
    # are not REPORTED; they are kept because a negative result nobody can regenerate
    # is not evidence.  Each route is self-contained and nothing in the
    # pipeline imports it, so the reference count would otherwise send them to ASK.
    if path.startswith("scripts/analyses/"):
        return "EXPLORATORY-KEEP"
    if path.startswith("scripts/draft/"):
        return "ARCHIVE"
    if n_refs == 0:
        return "ASK"
    return "PIPELINE"


def read_inventory() -> dict[str, dict[str, str]]:
    """Load the existing inventory keyed by path, or an empty dict."""
    if not INVENTORY.exists():
        return {}
    with INVENTORY.open(newline="") as fh:
        return {row["path"]: row for row in csv.DictReader(fh, delimiter="\t")}


@app.command()
def audit() -> None:
    """Refresh the evidence column and add rows for files new since the last run."""
    files = tracked_files()
    previous = read_inventory()
    sections = provenance_sections()
    names = [Path(f).name for f in files]
    unique_names = {n for n in names if names.count(n) == 1}

    rows = []
    for path in files:
        role = provenance_role(path, sections, unique_names)
        refs = referencing_files(path)
        evidence = [f"provenance:{role}"] if role else []
        evidence.append(f"refs={len(refs)}")
        evidence += [f"{k}={v}" for k, v in marker_counts(path).items()]
        old = previous.get(path, {})
        rows.append(
            {
                "path": path,
                "class": old.get("class") or suggest_class(path, role, len(refs)),
                "evidence": "; ".join(evidence),
                "action": old.get("action", ""),
                "owner": old.get("owner", ""),
            }
        )

    with INVENTORY.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    dropped = sorted(set(previous) - set(files))
    if dropped:
        logger.warning(f"{len(dropped)} rows dropped (no longer tracked): {', '.join(dropped)}")
    logger.success(f"Wrote {len(rows)} rows to {INVENTORY.relative_to(REPO)}")


@app.command()
def check() -> None:
    """Exit non-zero unless every tracked file has exactly one row with a valid class."""
    files = set(tracked_files()) - {"docs/guides/inventory.tsv"}
    inventory = read_inventory()
    missing = sorted(files - set(inventory))
    invalid = sorted(p for p, row in inventory.items() if row["class"] not in CLASSES)
    for path in missing:
        logger.error(f"no inventory row: {path}")
    for path in invalid:
        logger.error(f"invalid class {inventory[path]['class']!r}: {path}")
    if missing or invalid:
        raise typer.Exit(code=1)
    logger.success(f"{len(files)} tracked files classified")


if __name__ == "__main__":
    app()
