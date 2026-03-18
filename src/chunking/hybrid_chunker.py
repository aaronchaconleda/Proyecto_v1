from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.chunking.chunk_metadata import ChunkRecord, make_chunk_id

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?\:\;])\s+|\n+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

CONTINUATION_STARTERS = {
    "y",
    "o",
    "u",
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "que",
    "en",
    "con",
    "para",
    "por",
    "al",
    "sin",
    "ademas",
    "además",
}


@dataclass
class TextUnit:
    text: str
    page_start: int
    page_end: int
    section: str
    token_count: int


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    if paragraphs:
        return paragraphs
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


def _split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text.strip()]


def _split_by_token_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        piece = "".join(
            token if idx == 0 or re.match(r"[,\.\!\?\:\;\)\]\}]", token) else f" {token}"
            for idx, token in enumerate(tokens[start:end])
        ).strip()
        if piece:
            chunks.append(piece)
        if end >= len(tokens):
            break
        start += step
    return chunks


def _build_paragraph_units(pages: Sequence[Tuple[int, str]]) -> List[TextUnit]:
    units: List[TextUnit] = []
    section_idx = 0
    for page_number, text in pages:
        for paragraph in _split_paragraphs(text):
            section_idx += 1
            units.append(
                TextUnit(
                    text=paragraph,
                    page_start=page_number,
                    page_end=page_number,
                    section=f"s{section_idx}",
                    token_count=_token_count(paragraph),
                )
            )
    return units


def _is_cross_page_continuation(left: TextUnit, right: TextUnit) -> bool:
    if right.page_start != left.page_end + 1:
        return False
    left_text = left.text.rstrip()
    right_text = right.text.lstrip()
    if not left_text or not right_text:
        return False
    if left_text[-1] in ".!?;:)":
        return False
    first_word = right_text.split(maxsplit=1)[0].strip("“\"'`([")
    if not first_word:
        return False
    if first_word[0].islower():
        return True
    if first_word.lower() in CONTINUATION_STARTERS:
        return True
    if left.token_count < 20:
        return True
    return False


def _merge_cross_page_units(units: Sequence[TextUnit]) -> List[TextUnit]:
    if not units:
        return []
    merged: List[TextUnit] = [units[0]]
    for current in units[1:]:
        prev = merged[-1]
        if _is_cross_page_continuation(prev, current):
            merged_text = f"{prev.text} {current.text}".strip()
            merged[-1] = TextUnit(
                text=merged_text,
                page_start=prev.page_start,
                page_end=current.page_end,
                section=prev.section,
                token_count=_token_count(merged_text),
            )
        else:
            merged.append(current)
    return merged


def _ensure_unit_cap(units: Sequence[TextUnit], chunk_size: int, overlap: int) -> List[TextUnit]:
    out: List[TextUnit] = []
    for unit in units:
        if unit.token_count <= chunk_size:
            out.append(unit)
            continue

        sentences = _split_sentences(unit.text)
        if len(sentences) <= 1:
            pieces = _split_by_token_window(unit.text, chunk_size=chunk_size, overlap=max(1, overlap // 2))
            for idx, piece in enumerate(pieces, start=1):
                out.append(
                    TextUnit(
                        text=piece,
                        page_start=unit.page_start,
                        page_end=unit.page_end,
                        section=f"{unit.section}.w{idx}",
                        token_count=_token_count(piece),
                    )
                )
            continue

        buffer: List[str] = []
        buffer_tokens = 0
        part_idx = 0
        for sentence in sentences:
            sent_tokens = _token_count(sentence)
            if sent_tokens > chunk_size:
                if buffer:
                    part_idx += 1
                    text_part = " ".join(buffer).strip()
                    out.append(
                        TextUnit(
                            text=text_part,
                            page_start=unit.page_start,
                            page_end=unit.page_end,
                            section=f"{unit.section}.p{part_idx}",
                            token_count=_token_count(text_part),
                        )
                    )
                    buffer = []
                    buffer_tokens = 0
                pieces = _split_by_token_window(sentence, chunk_size=chunk_size, overlap=max(1, overlap // 2))
                for idx, piece in enumerate(pieces, start=1):
                    out.append(
                        TextUnit(
                            text=piece,
                            page_start=unit.page_start,
                            page_end=unit.page_end,
                            section=f"{unit.section}.w{idx}",
                            token_count=_token_count(piece),
                        )
                    )
                continue

            if buffer_tokens + sent_tokens > chunk_size and buffer:
                part_idx += 1
                text_part = " ".join(buffer).strip()
                out.append(
                    TextUnit(
                        text=text_part,
                        page_start=unit.page_start,
                        page_end=unit.page_end,
                        section=f"{unit.section}.p{part_idx}",
                        token_count=_token_count(text_part),
                    )
                )
                buffer = [sentence]
                buffer_tokens = sent_tokens
            else:
                buffer.append(sentence)
                buffer_tokens += sent_tokens

        if buffer:
            part_idx += 1
            text_part = " ".join(buffer).strip()
            out.append(
                TextUnit(
                    text=text_part,
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                    section=f"{unit.section}.p{part_idx}",
                    token_count=_token_count(text_part),
                )
            )
    return out


def chunk_document_hybrid(
    *,
    doc_id: str,
    pages: Sequence[Tuple[int, str]],
    chunk_size: int,
    overlap: int,
) -> List[ChunkRecord]:
    if not pages:
        return []

    base_units = _build_paragraph_units(pages)
    merged_units = _merge_cross_page_units(base_units)
    units = _ensure_unit_cap(merged_units, chunk_size=chunk_size, overlap=overlap)
    if not units:
        return []

    token_offsets: List[int] = []
    total = 0
    for unit in units:
        token_offsets.append(total)
        total += unit.token_count

    results: List[ChunkRecord] = []
    i = 0
    while i < len(units):
        j = i
        current_units: List[TextUnit] = []
        current_tokens = 0
        while j < len(units):
            candidate = units[j]
            if current_units and current_tokens + candidate.token_count > chunk_size:
                break
            current_units.append(candidate)
            current_tokens += candidate.token_count
            j += 1
            if current_tokens >= chunk_size:
                break

        if not current_units:
            i += 1
            continue

        chunk_text = "\n\n".join(unit.text for unit in current_units).strip()
        page_start = current_units[0].page_start
        page_end = current_units[-1].page_end
        offset_start = token_offsets[i]
        offset_end = token_offsets[j - 1] + current_units[-1].token_count
        chunk_id = make_chunk_id(doc_id, page_start, offset_start, offset_end)
        results.append(
            ChunkRecord(
                doc_id=doc_id,
                chunk_id=chunk_id,
                page=page_start,
                page_end=page_end,
                section=current_units[0].section,
                offset_start=offset_start,
                offset_end=offset_end,
                text=chunk_text,
            )
        )

        if j >= len(units):
            break

        back = 0
        next_i = j
        while next_i - 1 >= i and back < overlap:
            back += units[next_i - 1].token_count
            next_i -= 1
        if next_i <= i:
            next_i = i + 1
        i = next_i

    return results
