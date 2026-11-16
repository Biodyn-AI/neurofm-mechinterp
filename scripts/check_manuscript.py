"""Checks of the manuscript in paper/ (PLOS ONE format) after its numbers, tables and figures have been regenerated.

1. No leftover editing markers (TODO, TBD, XXX) in the manuscript and S1 Appendix.
2. Every macro placeholder is resolved (numbers.tex and tables/*.tex contain no "[??]").
3. The LaTeX build of manuscript.tex and then S1_Appendix.tex (in a temporary copy of the package; the checked PDFs
   are copied back) has no undefined references or citations and no overfull boxes wider than 10 pt. The build date
   (printed in the footer and stored in the PDF) and the PDF's ID are fixed, so that the same sources built with the
   same TeX distribution give the same PDFs.
4. PLOS requirements: no figures and no footnotes in the manuscript file, an abstract of at most 300 words, and one
   figure file per figure (figures/Fig<n>.tif: RGB, LZW, 300-600 dpi, 2.63-7.5 in wide, at most 8.75 in high,
   at most 10 MB).
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

from PIL import Image

PKG = Path(__file__).resolve().parents[1] / "paper"
DOCUMENTS = ("manuscript", "S1_Appendix")            # S1_Appendix reads the main text's labels from manuscript.aux
MARKERS = [r"\bTODO\b", r"\bTBD\b", r"\bXXX\b"]
BUILD_DATE = "1794700800"  # 2026-11-15 00:00 UTC, as seconds since 1970 (SOURCE_DATE_EPOCH)


def tables() -> list[Path]:
    """The table files (macOS stores resource forks as ._* files on non-Apple volumes; they are skipped)."""
    return sorted(p for p in (PKG / "tables").glob("*.tex") if not p.name.startswith("._"))


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def check_markers() -> list[str]:
    problems = []
    for name in DOCUMENTS:
        text = strip_comments((PKG / f"{name}.tex").read_text())
        for pattern in MARKERS:
            for match in re.finditer(pattern, text):
                problems.append(f"{name}.tex:{text.count(chr(10), 0, match.start()) + 1}: '{match.group(0)}'")
    return problems


def check_placeholders() -> list[str]:
    problems = []
    for path in [PKG / "numbers.tex", *tables()]:
        problems += [f"{path.name}: {line.strip()[:120]}" for line in path.read_text().splitlines() if "[??]" in line]
    return problems


def check_build() -> list[str]:
    problems = []
    env = {**os.environ, "SOURCE_DATE_EPOCH": BUILD_DATE, "FORCE_SOURCE_DATE": "1"}
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp) / "package"
        shutil.copytree(PKG, build, ignore=shutil.ignore_patterns("._*", "*.aux", "*.log", "*.bbl", "*.blg", "*.fls",
                                                                  "*.fdb_latexmk", "*.out", "*.synctex.gz", "*.tif"))
        for name in DOCUMENTS:
            result = subprocess.run(["latexmk", "-pdf", "-bibtex", "-interaction=nonstopmode", "-halt-on-error",
                                     rf"-usepretex=\pdftrailerid{{neurofm-mechinterp {name}}}", f"{name}.tex"],
                                    cwd=build, capture_output=True, text=True, env=env)
            log = (build / f"{name}.log").read_text(errors="ignore") if (build / f"{name}.log").exists() else ""
            if result.returncode != 0:
                problems.append(f"latexmk failed for {name}.tex: " + result.stdout[-2000:])
                continue
            shutil.copy(build / f"{name}.pdf", PKG / f"{name}.pdf")
            for pattern in (r"Reference `[^']+' on page \d+ undefined", r"Citation `[^']+' on page \d+ undefined",
                            r"There were undefined references", r"Undefined control sequence", r"multiply defined"):
                problems += [f"{name}: {m}" for m in sorted(set(re.findall(pattern, log)))]
            for match in re.finditer(r"Overfull \\hbox \((\d+\.?\d*)pt too wide\)", log):
                if float(match.group(1)) > 10:
                    problems.append(f"{name}: {match.group(0)}")
    return problems


def check_plos() -> list[str]:
    problems = []
    text = strip_comments((PKG / "manuscript.tex").read_text())
    if r"\includegraphics" in text:
        problems.append("the manuscript file includes a figure (PLOS: figures are uploaded as separate files)")
    if re.search(r"\\footnote(?![a-z])", text):
        problems.append("the manuscript uses a footnote (not permitted by PLOS)")
    pdf_text = subprocess.run(["pdftotext", "-layout", str(PKG / "manuscript.pdf"), "-"], capture_output=True, text=True).stdout
    abstract = pdf_text[pdf_text.index("Abstract") + len("Abstract"):pdf_text.index("Introduction")]
    abstract = re.sub(r"\s\d+(?=\s)", " ", re.sub(r"\S+ \d+, \d{4}\s+\d+/\d+", " ", abstract))  # line and page numbers
    if len(abstract.split()) > 300:
        problems.append(f"abstract of {len(abstract.split())} words (PLOS: at most 300)")
    labels = re.findall(r"\\begin\{figure\}.*?\\label\{([^}]*)\}", text, re.S)
    for number, label in enumerate(labels, 1):
        path = PKG / "figures" / f"Fig{number}.tif"
        if not path.exists():
            problems.append(f"{path.name} missing for {label}")
            continue
        image = Image.open(path)
        dpi = float(image.info.get("dpi", (0, 0))[0])
        width, height = (image.size[0] / dpi, image.size[1] / dpi) if dpi else (0.0, 0.0)
        if image.mode != "RGB" or image.info.get("compression") != "tiff_lzw" or not 300 <= dpi <= 600 \
                or not 2.63 <= width <= 7.5 or height > 8.75 or path.stat().st_size > 10e6:
            problems.append(f"{path.name}: mode {image.mode}, {image.info.get('compression')}, {dpi:.0f} dpi, "
                            f"{width:.2f} x {height:.2f} in, {path.stat().st_size / 1e6:.1f} MB")
    return problems


def main() -> int:
    failures = {
        "editing markers": check_markers(),
        "unresolved numbers": check_placeholders(),
        "LaTeX build": check_build(),
        "PLOS requirements": check_plos(),
    }
    for name, problems in failures.items():
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"[{status}] {name}")
        for problem in problems[:40]:
            print("    " + problem)
    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
