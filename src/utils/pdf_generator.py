"""Transcript PDF export for ``/generate_pdf``.

The input is the parsed transcript (``src.transcript_lines``): one dict per
line with ``time``, ``player``, ``character`` and ``text``; a line the parser
did not recognise (a restart marker, say) has ``time`` set to None and is
printed as-is. The layout is a title, a column header, then one paragraph per
line in a monospaced face so the time and speaker columns line up. Pages
get a thin frame drawn in code, so the export needs no image assets.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

from src.utils.timefmt import clock

OUTPUT_DIR = os.path.join(".logs", "pdfs")
PAGE_MARGIN = 0.5 * inch
TITLE = "Session Transcript"

TITLE_STYLE = ParagraphStyle(
    "TranscriptTitle",
    fontName="Helvetica-Bold",
    fontSize=22,
    leading=26,
    alignment=1,  # centred
    textColor=colors.HexColor("#3b2f2f"),
)
LINE_STYLE = ParagraphStyle(
    "TranscriptLine",
    fontName="Courier",
    fontSize=9.5,
    leading=12,
)
HEADER_TEXT = f"{'Time':<10} {'Speaker':<30} Said"
RULE = "-" * 92


def draw_page_frame(canvas, doc) -> None:
    """Page decoration: a hairline border inside the margins and the page
    number at the foot. Registered for every page."""
    width, height = A4
    inset = PAGE_MARGIN / 2
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#b9a98a"))
    canvas.setLineWidth(0.6)
    canvas.rect(inset, inset, width - 2 * inset, height - 2 * inset)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b6b6b"))
    canvas.drawCentredString(width / 2, inset + 6, str(doc.page))
    canvas.restoreState()


def speaker_label(entry) -> str:
    """``Player (Character)``, or whichever of the two is known."""
    player, character = entry.get("player"), entry.get("character")
    if player and character and character != player:
        return f"{player} ({character})"
    return player or character or ""


def format_line(entry) -> str:
    """The text for one transcript entry, escaped for reportlab markup.

    ``Paragraph`` parses its input as XML, so a transcribed ``<`` or ``&``
    has to be escaped or the build raises."""
    text = entry.get("text", "")
    if entry.get("time") is None:
        return escape(text)
    return escape(f"{clock(entry['time']):<10} {speaker_label(entry):<30} {text}")


def _new_output_path() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    handle, path = tempfile.mkstemp(suffix=".pdf", dir=OUTPUT_DIR)
    os.close(handle)
    return path


def build_pdf(entries) -> str:
    """Lay out and write the PDF (blocking: reportlab). Returns its path."""
    path = _new_output_path()
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=2 * PAGE_MARGIN,
        bottomMargin=2 * PAGE_MARGIN,
        title=TITLE,
    )
    story = [
        Paragraph(TITLE, TITLE_STYLE),
        Spacer(1, 20),
        Preformatted(HEADER_TEXT, LINE_STYLE),
        Preformatted(RULE, LINE_STYLE),
        Spacer(1, 8),
    ]
    for entry in entries:
        story.append(Paragraph(format_line(entry), LINE_STYLE))
        story.append(Spacer(1, 6))
    doc.build(story, onFirstPage=draw_page_frame, onLaterPages=draw_page_frame)
    return path


async def pdf_generator(entries, logo_path=None) -> str:
    """Build the transcript PDF off the event loop and return its path.

    :param entries: parsed transcript lines, oldest first.
    :param logo_path: unused; kept so existing callers need not change.
    """
    return await asyncio.to_thread(build_pdf, entries)
