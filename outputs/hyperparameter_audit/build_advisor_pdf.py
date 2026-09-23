from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
MD = ROOT / "GuardFed_AD2plus_All_Hyperparameters_for_Advisor.md"
TEX = ROOT / "GuardFed_AD2plus_All_Hyperparameters_for_Advisor.tex"


def esc(text: str) -> str:
    text = text.replace("\\", r"\textbackslash{}").replace("&", r"\&")
    text = text.replace("%", r"\% ").replace("#", r"\#")
    text = text.replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
    text = text.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return text


def inline(text: str) -> str:
    links = re.compile(r"\[([^]]+)\]\([^)]*\)")
    text = links.sub(lambda m: m.group(1), text)
    parts = re.split(r"(`[^`]+`|\*\*[^*]+\*\*|\$[^$]+\$)", text)
    out = []
    for part in parts:
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            out.append(r"\texttt{" + esc(part[1:-1]) + "}")
        elif part.startswith("**") and part.endswith("**"):
            out.append(r"\textbf{" + inline(part[2:-2]) + "}")
        elif part.startswith("$") and part.endswith("$"):
            out.append(r"\(" + part[1:-1] + r"\)")
        else:
            out.append(esc(part))
    return "".join(out)


def table_row(line: str):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_tex(rows):
    n = max(len(row) for row in rows)
    widths = {1: 0.94, 2: 0.46, 3: 0.30, 4: 0.22, 5: 0.18, 6: 0.15, 7: 0.13, 8: 0.115, 9: 0.105}
    width = widths.get(n, max(0.07, 0.98 / n))
    spec = "|".join([f"p{{{width:.3f}\\linewidth}}"] * n)
    size = r"\tiny" if n >= 8 else (r"\scriptsize" if n >= 6 else r"\small")
    out = [r"{" + size, r"\renewcommand{\arraystretch}{1.18}", r"\begin{longtable}{|" + spec + r"|}", r"\hline"]
    for idx, row in enumerate(rows):
        row = row + [""] * (n - len(row))
        cells = [inline(cell) for cell in row[:n]]
        out.append(" & ".join(cells) + r" \\\hline")
        if idx == 0:
            out.append(r"\endhead")
    out.extend([r"\end{longtable}", r"}"])
    return out


def convert(lines):
    out = []
    i = 0
    in_quote = False
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if not line.strip():
            if in_quote:
                out.append(r"\end{quote}")
                in_quote = False
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i].rstrip("\n"))
                i += 1
            out.append(r"\begin{lstlisting}")
            out.extend(code)
            out.append(r"\end{lstlisting}")
            i += 1
            continue
        if line.strip() == "---":
            out.append(r"\medskip")
            i += 1
            continue
        # The Markdown source also contains single-line display math embedded
        # in prose, for example: "公平预算为：$$B=0.06.$$".
        # Convert each $$...$$ segment before normal text escaping.
        if "$$" in line and not line.lstrip().startswith("$$"):
            pieces = re.split(r"(\$\$.*?\$\$)", line)
            rendered = []
            for piece in pieces:
                if piece.startswith("$$") and piece.endswith("$$"):
                    formula = piece[2:-2]
                    if formula.endswith("."):
                        formula = formula[:-1]
                        punctuation = "."
                    else:
                        punctuation = ""
                    rendered.append(r"\(" + formula + r"\)" + punctuation)
                elif piece:
                    rendered.append(inline(piece))
            out.append("".join(rendered) + "\n")
            i += 1
            continue
        if line.strip().startswith("$$") and line.strip().endswith("$$") and line.strip() != "$$":
            formula = line.strip()[2:-2]
            if formula.endswith("."):
                formula = formula[:-1]
                punctuation = "."
            else:
                punctuation = ""
            out.append(r"\[" + formula + r"\]" + punctuation)
            i += 1
            continue
        if line.startswith("$$"):
            formula = []
            if line.strip() != "$$":
                formula.append(line.strip()[2:])
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("$$"):
                formula.append(lines[i].rstrip("\n"))
                i += 1
            if i < len(lines) and lines[i].strip().endswith("$$"):
                tail = lines[i].strip()
                if tail != "$$":
                    formula.append(tail[:-2])
                i += 1
            out.append(r"\[" + "\n".join(formula) + r"\]")
            continue
        if line.startswith("|") and i + 1 < len(lines):
            separator = lines[i + 1].strip()
            separator_cells = separator.strip("|").split("|")
            is_separator = (
                separator.startswith("|")
                and len(separator_cells) > 0
                and all("-" in cell and cell.strip().replace(":", "").replace("-", "") == "" for cell in separator_cells)
            )
            if not is_separator:
                out.append(inline(line) + "\n")
                i += 1
                continue
            rows = [table_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(table_row(lines[i]))
                i += 1
            out.extend(table_tex(rows))
            continue
        match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if match:
            level = len(match.group(1))
            title = inline(match.group(2))
            if level == 1:
                out.append(r"\title{" + title + "}")
                out.append(r"\maketitle")
            elif level == 2:
                out.append(r"\section{" + title + "}")
            else:
                out.append(r"\subsection{" + title + "}")
            i += 1
            continue
        if line.startswith("> "):
            if not in_quote:
                out.append(r"\begin{quote}")
                in_quote = True
            out.append(inline(line[2:]) + r"\\")
            i += 1
            continue
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i].rstrip("\n")))
                i += 1
            out.append(r"\begin{itemize}")
            out.extend(r"\item " + inline(item) for item in items)
            out.append(r"\end{itemize}")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i].rstrip("\n")))
                i += 1
            out.append(r"\begin{enumerate}")
            out.extend(r"\item " + inline(item) for item in items)
            out.append(r"\end{enumerate}")
            continue
        out.append(inline(line) + "\n")
        i += 1
    if in_quote:
        out.append(r"\end{quote}")
    return out


header = r"""\documentclass[UTF8,a4paper,10pt]{ctexart}
\usepackage[a4paper,margin=1.55cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{array,longtable,booktabs}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setcounter{secnumdepth}{0}
\lstset{basicstyle=\ttfamily\small,backgroundcolor=\color{gray!8},frame=single,breaklines=true,columns=fullflexible}
\begin{document}
"""
footer = r"\end{document}" + "\n"
TEX.write_text(header + "\n".join(convert(MD.read_text(encoding="utf-8").splitlines(True))) + footer, encoding="utf-8")
print(TEX)
