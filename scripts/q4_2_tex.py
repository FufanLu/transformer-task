"""Write report/q4_2_generations.tex from the Q4.2 generation files.

Expects outputs/q4_2/{ours,pythia}_T{temp}/generation*.jsonl for each temperature.
"""

import glob
import json
import unicodedata

TEMPS = ["0.01", "0.5", "1.0"]
MODELS = [("ours", "Our model (final checkpoint from Q3.5)"), ("pythia", "Pythia-1.4B")]
REPLACE = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "--", "\u2014": "---", "\u2026": "..."}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}


def tex_escape(text: str) -> str:
    text = "".join(REPLACE.get(c, c) for c in text)
    out = []
    for c in text:
        if c in SPECIAL:
            out.append(SPECIAL[c])
        elif c in "\n\r\t":
            out.append(" ")
        elif ord(c) < 128 or unicodedata.category(c).startswith("L"):
            out.append(c)
        else:
            out.append(f"[U+{ord(c):04X}]")
    return " ".join("".join(out).split())


def load(model: str, temp: str) -> list[dict]:
    path = glob.glob(f"outputs/q4_2/{model}_T{temp}/generation*.jsonl")[0]
    return [json.loads(line) for line in open(path)]


def main():
    prefixes = [row["prefix"] for row in load("pythia", TEMPS[0])]
    lines = []
    for model, title in MODELS:
        lines.append(f"\\paragraph{{{title}}}")
        for i, prefix in enumerate(prefixes):
            lines.append(f"\\noindent\\textit{{Prefix {i + 1}.}}\\begin{{itemize}}")
            for temp in TEMPS:
                generation = load(model, temp)[i]["generation"]
                lines.append(f"  \\item $T={temp}$: \\textcolor{{blue!60!black}}{{{tex_escape(generation)}}}")
            lines.append("\\end{itemize}")
    open("report/q4_2_generations.tex", "w").write("\n".join(lines) + "\n")
    open("report/q4_2_prefixes.tex", "w").write(
        "\\begin{enumerate}\n" + "\n".join(f"  \\item {tex_escape(p)}" for p in prefixes) + "\n\\end{enumerate}\n"
    )
    print("wrote report/q4_2_generations.tex and report/q4_2_prefixes.tex")


if __name__ == "__main__":
    main()
