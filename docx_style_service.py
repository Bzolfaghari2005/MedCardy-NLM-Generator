"""
docx_style_service.py – Word styles, RTL helpers, font management,
and note callout styles for the Word Booklet Maker (جزوه‌ساز Word).
"""
from __future__ import annotations

import platform
import subprocess
from typing import Optional

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.styles.style import BaseStyle

from settings import (
    BOOKLET_BODY_FONT_SIZE_PT,
    BOOKLET_DEFAULT_FONT_EMOJI,
    BOOKLET_DEFAULT_FONT_ENGLISH,
    BOOKLET_DEFAULT_FONT_HEADING,
    BOOKLET_DEFAULT_FONT_PERSIAN,
    BOOKLET_DEFAULT_FONT_TABLE,
    BOOKLET_FALLBACK_FONT_PERSIAN,
    BOOKLET_H1_FONT_SIZE_PT,
    BOOKLET_H2_FONT_SIZE_PT,
    BOOKLET_H3_FONT_SIZE_PT,
    BOOKLET_H4_FONT_SIZE_PT,
    BOOKLET_HEADING_COLOR_HEX,
    BOOKLET_LINE_SPACING,
    BOOKLET_SPACE_AFTER_PT,
    BOOKLET_TABLE_HEADER_COLOR_HEX,
)


# ══════════════════════════════════════════════════════════════════════════════
# Font availability checker
# ══════════════════════════════════════════════════════════════════════════════

_font_cache: dict[str, bool] = {}


def is_font_available(font_name: str) -> bool:
    if font_name in _font_cache:
        return _font_cache[font_name]

    system = platform.system()
    available = False

    if system == "Windows":
        try:
            import winreg
            base_keys = [
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\FontSubstitutes",
            ]
            for key_path in base_keys:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                    i = 0
                    while True:
                        name, _, _ = winreg.EnumValue(key, i)
                        if font_name.lower() in name.lower():
                            available = True
                            break
                        i += 1
                except (OSError, FileNotFoundError):
                    break
                finally:
                    try:
                        winreg.CloseKey(key)
                    except Exception:
                        pass
                if available:
                    break
        except ImportError:
            pass

    elif system in ("Linux", "Darwin"):
        try:
            result = subprocess.run(
                ["fc-list", ":family=" + font_name],
                capture_output=True, text=True, timeout=5,
            )
            available = bool(result.stdout.strip())
        except Exception:
            pass

    _font_cache[font_name] = available
    return available


def resolve_font(preferred: str, fallback: str = "Arial") -> tuple[str, bool]:
    """Return (font_name, was_fallback_used)."""
    if is_font_available(preferred):
        return preferred, False
    if is_font_available(fallback):
        return fallback, True
    return "Arial", True


# ══════════════════════════════════════════════════════════════════════════════
# XML element helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_or_add(parent, tag: str):
    child = parent.find(qn(tag))
    if child is None:
        child = OxmlElement(tag)
        parent.append(child)
    return child


def _set_xml_bool(parent, tag: str, val: bool = True):
    el = _get_or_add(parent, tag)
    if not val:
        el.attrib[qn("w:val")] = "0"
    elif qn("w:val") in el.attrib:
        del el.attrib[qn("w:val")]


def _hex_to_rgb(hex_str: str) -> RGBColor:
    hex_str = hex_str.lstrip("#")
    r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
    return RGBColor(r, g, b)


# ══════════════════════════════════════════════════════════════════════════════
# RTL paragraph / run helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_rtl_paragraph(paragraph, align: str = "JUSTIFY") -> None:
    """Set RTL direction and alignment on a paragraph."""
    pPr = paragraph._p.get_or_add_pPr()
    _set_xml_bool(pPr, "w:bidi", True)

    alignment_map = {
        "JUSTIFY": WD_ALIGN_PARAGRAPH.JUSTIFY,
        "RIGHT": WD_ALIGN_PARAGRAPH.RIGHT,
        "LEFT": WD_ALIGN_PARAGRAPH.LEFT,
        "CENTER": WD_ALIGN_PARAGRAPH.CENTER,
    }
    paragraph.alignment = alignment_map.get(align.upper(), WD_ALIGN_PARAGRAPH.JUSTIFY)


def set_rtl_run(run) -> None:
    """Set RTL on an individual run."""
    rPr = run._r.get_or_add_rPr()
    _set_xml_bool(rPr, "w:rtl", True)


def set_paragraph_spacing(paragraph, line_spacing: float = BOOKLET_LINE_SPACING,
                           space_after_pt: int = BOOKLET_SPACE_AFTER_PT) -> None:
    pf = paragraph.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line_spacing
    pf.space_after = Pt(space_after_pt)
    pf.space_before = Pt(0)


def apply_font(run, font_name: str, size_pt: Optional[int] = None,
               bold: bool = False, italic: bool = False,
               color_hex: Optional[str] = None) -> None:
    run.font.name = font_name
    # Eastern fonts (necessary for correct rendering in RTL Word docs)
    rPr = run._r.get_or_add_rPr()
    rFonts = _get_or_add(rPr, "w:rFonts")
    rFonts.set(qn("w:cs"), font_name)
    rFonts.set(qn("w:eastAsia"), font_name)

    if size_pt is not None:
        run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.italic = italic
    if color_hex:
        run.font.color.rgb = _hex_to_rgb(color_hex)


# ══════════════════════════════════════════════════════════════════════════════
# Style creation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_style(doc: Document, style_name: str, base_style_name: str = "Normal") -> BaseStyle:
    """Get or create a named paragraph style."""
    try:
        return doc.styles[style_name]
    except KeyError:
        return doc.styles.add_style(style_name, 1)  # 1 = WD_STYLE_TYPE.PARAGRAPH


# ══════════════════════════════════════════════════════════════════════════════
# Document-level styles setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_document_styles(
    doc: Document,
    settings: dict,
) -> dict[str, str]:
    """Create / override styles in *doc*.

    *settings* keys (all optional, fall back to module defaults):
      font_persian, font_english, font_heading, font_table,
      body_size, h1_size, h2_size, h3_size, h4_size,
      heading_color_hex, table_header_color_hex, line_spacing, space_after_pt

    Returns a dict of font names actually applied (may differ from requested
    if fonts are unavailable).
    """
    font_persian_req = settings.get("font_persian", BOOKLET_DEFAULT_FONT_PERSIAN)
    font_english = settings.get("font_english", BOOKLET_DEFAULT_FONT_ENGLISH)
    font_heading_req = settings.get("font_heading", BOOKLET_DEFAULT_FONT_HEADING)

    font_persian, persian_fallback = resolve_font(font_persian_req, BOOKLET_FALLBACK_FONT_PERSIAN)
    font_heading, heading_fallback = resolve_font(font_heading_req, font_persian)

    body_size = settings.get("body_size", BOOKLET_BODY_FONT_SIZE_PT)
    h1_size = settings.get("h1_size", BOOKLET_H1_FONT_SIZE_PT)
    h2_size = settings.get("h2_size", BOOKLET_H2_FONT_SIZE_PT)
    h3_size = settings.get("h3_size", BOOKLET_H3_FONT_SIZE_PT)
    h4_size = settings.get("h4_size", BOOKLET_H4_FONT_SIZE_PT)
    heading_color = settings.get("heading_color_hex", BOOKLET_HEADING_COLOR_HEX)
    line_spacing = settings.get("line_spacing", BOOKLET_LINE_SPACING)
    space_after = settings.get("space_after_pt", BOOKLET_SPACE_AFTER_PT)

    # ── Normal ────────────────────────────────────────────────────────────────
    normal = doc.styles["Normal"]
    normal.font.name = font_persian
    normal.font.size = Pt(body_size)
    nf_pPr = normal._element.get_or_add_pPr()
    nf_rPr = normal._element.get_or_add_rPr()
    nPr_fonts = _get_or_add(nf_rPr, "w:rFonts")
    nPr_fonts.set(qn("w:cs"), font_persian)
    nPr_fonts.set(qn("w:ascii"), font_english)
    nPr_fonts.set(qn("w:hAnsi"), font_english)

    # ── Body Text (RTL) ───────────────────────────────────────────────────────
    body = _ensure_style(doc, "NLM Body Text", "Normal")
    body.font.name = font_persian
    body.font.size = Pt(body_size)
    bpPr = body.paragraph_format
    bpPr.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    bpPr.line_spacing = line_spacing
    bpPr.space_after = Pt(space_after)
    bpPr.space_before = Pt(0)
    bpPr.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # RTL via XML
    bEl = body._element.get_or_add_pPr()
    _set_xml_bool(bEl, "w:bidi", True)

    # ── Headings ──────────────────────────────────────────────────────────────
    heading_defs = [
        ("Heading 1", h1_size, True),
        ("Heading 2", h2_size, True),
        ("Heading 3", h3_size, True),
        ("Heading 4", h4_size, True),
    ]
    for style_name, size, bold in heading_defs:
        try:
            st = doc.styles[style_name]
        except KeyError:
            st = doc.styles.add_style(style_name, 1)
        st.font.name = font_heading
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.color.rgb = _hex_to_rgb(heading_color)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.space_before = Pt(8)
        st.paragraph_format.keep_with_next = True
        hEl = st._element.get_or_add_pPr()
        _set_xml_bool(hEl, "w:bidi", True)
        hfEl = st._element.get_or_add_rPr()
        hfFonts = _get_or_add(hfEl, "w:rFonts")
        hfFonts.set(qn("w:cs"), font_heading)
        hfFonts.set(qn("w:ascii"), font_heading)
        hfFonts.set(qn("w:hAnsi"), font_heading)

    # ── Note / callout styles ─────────────────────────────────────────────────
    _create_note_styles(doc, font_persian, body_size, heading_color)

    # ── Code style ────────────────────────────────────────────────────────────
    code_st = _ensure_style(doc, "NLM Code", "Normal")
    code_st.font.name = "Courier New"
    code_st.font.size = Pt(max(body_size - 1, 9))
    code_pPr = code_st.paragraph_format
    code_pPr.space_after = Pt(4)

    return {
        "font_persian": font_persian,
        "font_persian_fallback": persian_fallback,
        "font_heading": font_heading,
        "font_heading_fallback": heading_fallback,
        "font_english": font_english,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Note callout styles
# ══════════════════════════════════════════════════════════════════════════════

_NOTE_STYLE_DEFS = {
    "GOLDEN_TIP":      ("NLM Note GoldenTip",    "F7C948", "⭐"),
    "IMPORTANT":       ("NLM Note Important",     "4CAF50", "✅"),
    "CLINICAL_EXAMPLE":("NLM Note Clinical",      "2196F3", "📌"),
    "CLINICAL_PEARL":  ("NLM Note Pearl",         "00BCD4", "💡"),
    "WARNING":         ("NLM Note Warning",       "F44336", "🚨"),
    "HIGH_YIELD":      ("NLM Note HighYield",     "FF5722", "🔥"),
    "DATA":            ("NLM Note Data",          "607D8B", "📊"),
    "SUMMARY":         ("NLM Note Summary",       "795548", "🏁"),
    "DRUG_NOTE":       ("NLM Note Drug",          "9C27B0", "💊"),
    "EXAM_FINDING":    ("NLM Note Exam",          "009688", "🩺"),
}


def get_note_style_name(note_type: str) -> str:
    return _NOTE_STYLE_DEFS.get(note_type, ("NLM Note Important", "4CAF50", "✅"))[0]


def _create_note_styles(doc: Document, font_persian: str, body_size: int, accent_color: str) -> None:
    for note_type, (style_name, color_hex, _emoji) in _NOTE_STYLE_DEFS.items():
        st = _ensure_style(doc, style_name, "Normal")
        st.font.name = font_persian
        st.font.size = Pt(body_size)
        pf = st.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.space_after = Pt(6)
        pf.space_before = Pt(2)
        pf.left_indent = Cm(0.5)
        pf.right_indent = Cm(0.3)

        # RTL
        pEl = st._element.get_or_add_pPr()
        _set_xml_bool(pEl, "w:bidi", True)

        # Left border (visual callout)
        pBdr = _get_or_add(pEl, "w:pBdr")
        right_bdr = OxmlElement("w:right")
        right_bdr.set(qn("w:val"), "single")
        right_bdr.set(qn("w:sz"), "18")
        right_bdr.set(qn("w:space"), "4")
        right_bdr.set(qn("w:color"), color_hex)
        pBdr.append(right_bdr)

        # Very light background shading
        shd = _get_or_add(pEl, "w:shd")
        # Lighten the color significantly for background
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
        light_r = min(255, r + (255 - r) * 9 // 10)
        light_g = min(255, g + (255 - g) * 9 // 10)
        light_b = min(255, b + (255 - b) * 9 // 10)
        light_hex = f"{light_r:02X}{light_g:02X}{light_b:02X}"
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), light_hex)


# ══════════════════════════════════════════════════════════════════════════════
# Table style
# ══════════════════════════════════════════════════════════════════════════════

def style_table(table, settings: dict, font_persian: str) -> None:
    """Apply RTL-friendly borders and header style to a python-docx table."""
    header_color_hex = settings.get("table_header_color_hex", BOOKLET_TABLE_HEADER_COLOR_HEX)
    font_table = settings.get("font_table", BOOKLET_DEFAULT_FONT_TABLE)
    body_size = settings.get("body_size", BOOKLET_BODY_FONT_SIZE_PT)

    table.style = "Table Grid"

    for i, row in enumerate(table.rows):
        for cell in row.cells:
            # Cell padding
            tc = cell._tc
            tcPr = _get_or_add(tc, "w:tcPr")
            tcMar = _get_or_add(tcPr, "w:tcMar")
            for side in ("w:top", "w:bottom", "w:left", "w:right"):
                m = OxmlElement(side)
                m.set(qn("w:w"), "80")
                m.set(qn("w:type"), "dxa")
                tcMar.append(m)

            for para in cell.paragraphs:
                # RTL
                pPr = para._p.get_or_add_pPr()
                _set_xml_bool(pPr, "w:bidi", True)
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT

                for run in para.runs:
                    run.font.name = font_persian
                    run.font.size = Pt(body_size)
                    rPr = run._r.get_or_add_rPr()
                    rFonts = _get_or_add(rPr, "w:rFonts")
                    rFonts.set(qn("w:cs"), font_persian)
                    rFonts.set(qn("w:ascii"), font_table)
                    rFonts.set(qn("w:hAnsi"), font_table)

            if i == 0:
                # Header row – shading + bold
                for cell in row.cells:
                    tc = cell._tc
                    tcPr = _get_or_add(tc, "w:tcPr")
                    shd = _get_or_add(tcPr, "w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), header_color_hex.lstrip("#"))
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Mark header row to repeat across pages
    try:
        trPr = _get_or_add(table.rows[0]._tr, "w:trPr")
        tblHeader = OxmlElement("w:tblHeader")
        trPr.append(tblHeader)
    except (IndexError, Exception):
        pass

    # Table width = 100% of page
    tblPr = _get_or_add(table._tbl, "w:tblPr")
    tblW = _get_or_add(tblPr, "w:tblW")
    tblW.set(qn("w:w"), "5000")
    tblW.set(qn("w:type"), "pct")

    # RTL table direction
    bidiVisual = _get_or_add(tblPr, "w:bidiVisual")


# ══════════════════════════════════════════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════════════════════════════════════════

def set_page_margins(section, settings: dict) -> None:
    from settings import (
        BOOKLET_MARGIN_BOTTOM_CM, BOOKLET_MARGIN_LEFT_CM,
        BOOKLET_MARGIN_RIGHT_CM, BOOKLET_MARGIN_TOP_CM,
    )
    section.top_margin = Cm(settings.get("margin_top_cm", BOOKLET_MARGIN_TOP_CM))
    section.bottom_margin = Cm(settings.get("margin_bottom_cm", BOOKLET_MARGIN_BOTTOM_CM))
    section.right_margin = Cm(settings.get("margin_right_cm", BOOKLET_MARGIN_RIGHT_CM))
    section.left_margin = Cm(settings.get("margin_left_cm", BOOKLET_MARGIN_LEFT_CM))
