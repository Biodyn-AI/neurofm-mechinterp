"""Checks of the manuscript in paper/ after its numbers, tables and figures have been regenerated.

1. No leftover editing markers (TODO, TBD, XXX) in the manuscript and highlights.
2. Every macro placeholder is resolved (numbers.tex and tables/*.tex contain no "[??]").
3. The LaTeX build (in a temporary copy of the package; the checked PDF is copied back as manuscript.pdf) has
   no undefined references or citations and no overfull boxes wider than 10 pt. The build date (printed on the
   first page and stored in the PDF) and the PDF's ID are fixed, so that the same sources built with the same
   TeX distribution give the same manuscript.pdf.
4. Every highlight is at most 85 characters.
Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "paper"
MARKERS = [r"\bTODO\b", r"\bTBD\b", r"\bXXX\b"]
BUILD_DATE = "1794182400"  # 2026-11-09 00:00 UTC, as seconds since 1970 (SOURCE_DATE_EPOCH)


def tables() -> list[Path]:
    """The table files (macOS stores resource forks as ._* files on non-Apple volumes; they are skipped)."""
    return sorted(p for p in (PKG / "tables").glob("*.tex") if not p.name.startswith("._"))


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def check_markers() -> list[str]:
    problems = []
    for name in ("manuscript.tex", "highlights.tex"):
        path = PKG / name
        if not path.exists():
            continue
        text = strip_comments(path.read_text())
        for pattern in MARKERS:
            for match in re.finditer(pattern, text):
                problems.append(f"{name}:{text.count(chr(10), 0, match.start()) + 1}: '{match.group(0)}'")
    return problems


def check_placeholders() -> list[str]:
    problems = []
    for path in [PKG / "numbers.tex", *tables()]:
        if path.exists():
            problems += [f"{path.name}: {line.strip()[:120]}" for line in path.read_text().splitlines() if "[??]" in line]
    return problems


def check_build() -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp) / "package"
        shutil.copytree(PKG, build, ignore=shutil.ignore_patterns("._*", "*.aux", "*.log", "*.bbl", "*.blg", "*.fls",
                                                                  "*.fdb_latexmk", "*.out", "*.synctex.gz"))
        env = {**os.environ, "SOURCE_DATE_EPOCH": BUILD_DATE, "FORCE_SOURCE_DATE": "1"}
        result = subprocess.run(["latexmk", "-pdf", "-bibtex", "-interaction=nonstopmode", "-halt-on-error",
                                 r"-usepretex=\pdftrailerid{neurofm-mechinterp manuscript}", "manuscript.tex"],
                                cwd=build, capture_output=True, text=True, env=env)
        log = (build / "manuscript.log").read_text(errors="ignore") if (build / "manuscript.log").exists() else ""
        if result.returncode == 0 and (build / "manuscript.pdf").exists():
            shutil.copy(build / "manuscript.pdf", PKG / "manuscript.pdf")
    problems = []
    if result.returncode != 0:
        problems.append("latexmk failed: " + result.stdout[-2000:])
    for pattern in (r"Reference `[^']+' on page \d+ undefined", r"Citation `[^']+' on page \d+ undefined",
                    r"There were undefined references", r"Undefined control sequence"):
        problems += sorted(set(re.findall(pattern, log)))
    for match in re.finditer(r"Overfull \\hbox \((\d+\.?\d*)pt too wide\)", log):
        if float(match.group(1)) > 10:
            problems.append(match.group(0))
    return problems


def check_highlights() -> list[str]:
    path = PKG / "highlights.tex"
    if not path.exists():
        return ["highlights.tex missing"]
    items = re.findall(r"\\item\s+(.+)", strip_comments(path.read_text()))
    return [f"highlight too long ({len(i)} chars): {i}" for i in items if len(i) > 85]


def main() -> int:
    failures = {
        "editing markers": check_markers(),
        "unresolved numbers": check_placeholders(),
        "LaTeX build": check_build(),
        "highlights": check_highlights(),
    }
    for name, problems in failures.items():
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"[{status}] {name}")
        for problem in problems[:40]:
            print("    " + problem)
    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
