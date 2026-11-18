"""Render a chat-message list (as passed to the model) into a human-readable PDF with the images.

    from medvlm_bench.promptpdf import messages_to_pdf
    messages_to_pdf(msgs, "prompt.pdf", title="UCSF-PDGM-0283 | icl_knn_k4", footer="p_pos=0.72")

Each message is drawn as a coloured box (system / user / assistant); images inside one message are laid
out in a grid with a caption ("image 1 - 448x448"). Needs `reportlab` (pip install reportlab).
"""
from __future__ import annotations

import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROLE_STYLE = {  # header colour, box background
    "system": (colors.HexColor("#55606B"), colors.HexColor("#EEF0F2")),
    "user": (colors.HexColor("#1F5FA8"), colors.HexColor("#E6EFF8")),
    "assistant": (colors.HexColor("#2A7550"), colors.HexColor("#E5F2EA")),
}


def _png(im, max_px=900) -> io.BytesIO:
    im = im.convert("RGB")
    if max(im.size) > max_px:
        s = max_px / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf


def messages_to_pdf(messages, path: str, title: str = "Model input", footer: str = "", max_cols: int = 3):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13)
    cap = ParagraphStyle("cap", parent=body, fontSize=7.5, leading=9, textColor=colors.HexColor("#4A5663"), alignment=1)
    head = ParagraphStyle("head", parent=body, fontName="Helvetica-Bold", fontSize=9, textColor=colors.white)
    h1 = ParagraphStyle("h1", parent=ss["Heading2"], fontSize=14, spaceAfter=2)

    page_w = A4[0] - 30 * mm
    inner_w = page_w - 8 * mm
    story = [Paragraph(escape(title), h1)]
    n_img_total = sum(1 for m in messages for p in m["content"] if p["type"] == "image")
    story.append(Paragraph(f"{len(messages)} messages, {n_img_total} images. Text is shown verbatim.",
                           ParagraphStyle("sub", parent=body, textColor=colors.HexColor("#4A5663"))))
    story.append(Spacer(1, 4 * mm))

    img_no = 0
    for mi, m in enumerate(messages, 1):
        role = m["role"]
        hc, bg = ROLE_STYLE.get(role, ROLE_STYLE["user"])
        label = f"{mi}.  {role.upper()}"
        if role == "assistant":
            label += "  (reply given in the prompt)"
        rows = [[Paragraph(label, head)]]
        pending: list = []

        def flush_images():
            nonlocal pending
            if not pending:
                return
            cols = min(max_cols, len(pending))
            cell_w = inner_w / cols
            img_w = min(cell_w - 4 * mm, 80 * mm if len(pending) == 1 else 50 * mm)
            cells = []
            for no, im in pending:
                w, h = im.size
                flow = Image(_png(im), width=img_w, height=img_w * h / w)
                cells.append([flow, Paragraph(f"image {no} &middot; {w}&times;{h} px", cap)])
            for i in range(0, len(cells), cols):  # one outer row per image row, so a long grid can break across pages
                row = cells[i:i + cols]
                t = Table([row + [""] * (cols - len(row))], colWidths=[cell_w] * cols)
                t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
                rows.append([t])
            pending = []

        for part in m["content"]:
            if part["type"] == "image":
                img_no += 1
                pending.append((img_no, part["image"]))
            else:
                flush_images()
                txt = escape(part["text"]).replace("\n", "<br/>")
                rows.append([Paragraph(txt, body)])
        flush_images()

        box = Table(rows, colWidths=[page_w])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), hc), ("BACKGROUND", (0, 1), (-1, -1), bg),
            ("BOX", (0, 0), (-1, -1), 0.8, hc),
            ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm), ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        # short messages stay in one piece; long ones (many images) may break across pages
        story.append(KeepTogether(box) if len(rows) <= 3 and n_img_total <= 12 else box)
        story.append(Spacer(1, 3 * mm))

    if isinstance(story[-1], Spacer):  # a trailing spacer can spill onto an empty last page
        story.pop()
    if footer:
        story.append(Paragraph(escape(footer), ParagraphStyle("foot", parent=body, textColor=colors.HexColor("#4A5663"))))

    def _page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#7A858F"))
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(path, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                      bottomMargin=16 * mm, title=title).build(story, onFirstPage=_page, onLaterPages=_page)
    return path