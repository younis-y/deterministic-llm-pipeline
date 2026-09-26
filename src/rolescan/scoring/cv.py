"""CV variant loading.

The LLM picks which CV to send and what to change in it, so it needs to see
what each variant actually says. This reads the variant files, strips LaTeX to
readable text, and trims each to a budget so a six-variant prompt stays cheap.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rolescan.models import CVVariant

__all__ = ["CVLibrary", "strip_latex"]

log = logging.getLogger(__name__)

_SUFFIXES = (".tex", ".md", ".txt")
_KEY = re.compile(r"[^a-z0-9]+")


def _fold(name: str) -> str:
    """Compare CV filenames loosely.

    The enum may differ in punctuation from actual filenames: CV_AI-LLM-Engineering
    in the enum matches cv_ai_llm_engineering.tex on disk. Hyphens, underscores
    and case are not worth a bug.
    """
    return _KEY.sub("", name.casefold())


#: Per-variant prompt budget, deliberately below what the files hold. The
#: stripped variants run 3,746-3,879 characters, so roughly the last third of
#: each CV never reaches the model. That is the intended trade: six variants in
#: one system prompt, on every scored posting, several times a day. The tail of
#: a CV is education and interests, which is the part the fit score needs
#: least. Raise it only against a measured token bill.
_PER_VARIANT_CHARS = 2600

_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_PREAMBLE = re.compile(r"^.*?\\begin\{document\}", re.DOTALL)
_ENVIRONMENT = re.compile(r"\\(begin|end)\{[^}]*\}")
_COMMAND_ARG = re.compile(r"\\(?:textbf|textit|emph|href|large|Huge|textcolor)\s*")
_COMMAND = re.compile(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?")
_BRACES = re.compile(r"[{}]")
_ITEM = re.compile(r"\\item\s*")
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def strip_latex(source: str) -> str:
    """LaTeX to plain text, good enough for a prompt.

    Not a real parser and does not need to be. It drops the preamble, unwraps
    formatting commands, and keeps the words.
    """
    text = _COMMENT.sub("", source)
    if "\\begin{document}" in text:
        text = _PREAMBLE.sub("", text, count=1)
    text = text.replace("\\end{document}", "")
    text = _ITEM.sub("\n- ", text)
    text = _ENVIRONMENT.sub("", text)
    text = _COMMAND_ARG.sub("", text)
    text = _COMMAND.sub(" ", text)
    text = _BRACES.sub("", text)
    text = text.replace("\\\\", "\n").replace("~", " ").replace("\\&", "&")
    text = _WS.sub(" ", text)
    return _BLANKS.sub("\n\n", text).strip()


class CVLibrary:
    """The candidate's CV variants, keyed by variant enum."""

    def __init__(self, variants: dict[CVVariant, str]) -> None:
        self.variants = variants

    def __bool__(self) -> bool:
        return bool(self.variants)

    def __len__(self) -> int:
        return len(self.variants)

    @classmethod
    def load(cls, directory: Path | None) -> CVLibrary:
        """Read every recognised variant file from `directory`.

        Missing files are not an error. A missing directory just means the LLM
        works from the profile summary alone and gives less specific tailoring.
        """
        if directory is None or not directory.is_dir():
            return cls({})
        found: dict[CVVariant, str] = {}
        by_key: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.casefold() in _SUFFIXES:
                by_key.setdefault(_fold(path.stem), path)
        for variant in CVVariant:
            source = by_key.get(_fold(variant.value))
            if source is None:
                continue
            try:
                raw = source.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                log.warning("could not read %s: %s", source, e)
                continue
            body = (
                strip_latex(raw) if source.suffix.casefold() == ".tex" else raw.strip()
            )
            found[variant] = body[:_PER_VARIANT_CHARS]
        if not found:
            log.info("no CV variants found in %s", directory)
        return cls(found)

    def prompt_block(self) -> str:
        """Render the variants for inclusion in the system prompt."""
        if not self.variants:
            names = ", ".join(v.value for v in CVVariant)
            return (
                "The candidate's CV variants are not available to read. Choose "
                f"the most plausible from: {names}. Keep tailoring advice "
                "general, and say so in the reason."
            )
        parts = [
            f'<cv name="{variant.value}">\n{text}\n</cv>'
            for variant, text in self.variants.items()
        ]
        return "\n\n".join(parts)
