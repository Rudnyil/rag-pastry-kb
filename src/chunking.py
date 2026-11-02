"""Split a Markdown document into chunks, one per '## ' section.

Why by section: every section of the policy document answers one kind of question,
so a section is a natural unit of meaning. Sections longer than max_chars are split
on paragraph boundaries with a small overlap.
"""
from __future__ import annotations
import re


def split_markdown(text: str, source: str, max_chars: int = 900, overlap: int = 100) -> list[dict]:
    parts = re.split(r"(?m)^## ", text)
    chunks: list[dict] = []
    for part in parts[1:]:  # parts[0] is the document title and intro
        title, _, body = part.partition("\n")
        body = body.strip()
        if not body:
            continue
        for i, piece in enumerate(_split_long(body, max_chars, overlap)):
            chunks.append({
                "id": f"{source}:{_slug(title)}:{i}",
                "text": f"{title.strip()}\n{piece}",
                "metadata": {"source": source, "section": title.strip(), "part": i},
            })
    return chunks


def _split_long(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    pieces, current = [], ""
    for para in body.split("\n\n"):
        if current and len(current) + len(para) > max_chars:
            pieces.append(current.strip())
            current = current[-overlap:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
