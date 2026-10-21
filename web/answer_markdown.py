"""Minimal, safe Markdown for model answers in server-rendered pages.

Mirrors ``renderMarkdown`` in static/ask.js: escape everything, then
**bold**, *italic*, `code`, and "- " / "1. " list lines. Kept tiny on
purpose; answers are short and the subset is what Claude actually uses.
"""

import re

from markupsafe import Markup, escape

_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITAL = re.compile(r"(^|[^*])\*([^*\s][^*]*)\*")
_UND = re.compile(r"(^|\W)_([^_\s][^_]*)_(?=\W|$)")
_UL = re.compile(r"^\s*[-*•]\s+(.*)$")
_OL = re.compile(r"^\s*\d+[.)]\s+(.*)$")


def _inline(t: str) -> str:
    t = _CODE.sub(r"<code>\1</code>", t)
    t = _BOLD.sub(r"<strong>\1</strong>", t)
    t = _ITAL.sub(r"\1<em>\2</em>", t)
    return _UND.sub(r"\1<em>\2</em>", t)


def render(text) -> Markup:
    out, kind = [], None
    for raw in str(escape(text or "")).split("\n"):
        m_ul, m_ol = _UL.match(raw), _OL.match(raw)
        item = m_ul or m_ol
        if item:
            k = "ul" if m_ul else "ol"
            if kind != k:
                if kind:
                    out.append(f"</{kind}>")
                out.append(f"<{k}>")
                kind = k
            out.append(f"<li>{_inline(item.group(1))}</li>")
            continue
        if kind:
            out.append(f"</{kind}>")
            kind = None
        if raw.strip():
            out.append(f"<p>{_inline(raw)}</p>")
    if kind:
        out.append(f"</{kind}>")
    return Markup("".join(out))
