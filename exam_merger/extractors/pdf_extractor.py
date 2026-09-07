"""
pdf_extractor.py — Layout-aware PDF text extraction.

Extraction strategy (per page):
  1. pdfplumber  — layout-aware, handles multi-column text, preferred.
  2. pypdf       — fallback when pdfplumber fails for a given page.

Boilerplate detection:
  - Repeated identical lines at positions 0, 1, -1, -2 across pages.
  - Lines that appear in the top/bottom 10% of the page bounding box
    on 3+ pages are treated as headers/footers.
"""

import os

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..models.chunk import Chunk

# ── Table helpers ─────────────────────────────────────────────────────────────

def table_to_markdown(table: list[list[str]]) -> str:
    if not table or not table[0]:
        return ''
    rows = []
    for r in table:
        if any(cell is not None and str(cell).strip() != '' for cell in r):
            rows.append([str(cell or '').replace('\n', ' ').strip() for cell in r])

    if not rows:
        return ''

    cols_count = max(len(r) for r in rows)
    md_lines = []
    # Pad/truncate all rows to the same column count
    padded = []
    for row in rows:
        if len(row) < cols_count:
            row = row + [''] * (cols_count - len(row))
        elif len(row) > cols_count:
            row = row[:cols_count]
        padded.append(row)

    md_lines.append('| ' + ' | '.join(padded[0]) + ' |')
    md_lines.append('| ' + ' | '.join(['---'] * cols_count) + ' |')
    for row in padded[1:]:
        md_lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(md_lines)


# ── Text chunking helper ──────────────────────────────────────────────────────

def split_text_to_limit(text: str, limit: int) -> list[str]:
    """Split text into sub-chunks of at most `limit` chars, splitting on paragraph breaks."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush():
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append('\n\n'.join(current_parts))
            current_parts = []
            current_len = 0

    for paragraph in text.split('\n\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > limit:
            flush()
            for start in range(0, len(paragraph), limit):
                chunks.append(paragraph[start:start + limit])
            continue

        projected = current_len + (2 + len(paragraph) if current_parts else len(paragraph))
        if current_parts and projected > limit:
            flush()

        current_parts.append(paragraph)
        current_len += (2 + len(paragraph)) if len(current_parts) > 1 else len(paragraph)

    flush()
    return [c for c in chunks if c.strip()]


# ── Boilerplate detection ─────────────────────────────────────────────────────

def detect_boilerplate(pages_lines: list[list[str]]) -> set[str]:
    """
    Detect lines that repeat identically across 3+ pages at the same
    positional slot (first 2 / last 2 lines of each page).  These are
    running headers/footers that add no semantic content.
    """
    boilerplate: set[str] = set()
    num_pages = len(pages_lines)
    if num_pages < 3:
        return boilerplate

    def _scan_positions(positions: list[int]) -> None:
        for line_idx in positions:
            consecutive = 1
            last_val = None
            for p in range(num_pages):
                lines = pages_lines[p]
                abs_idx = abs(line_idx)
                try:
                    current = lines[line_idx].strip() if len(lines) >= abs_idx else None
                except IndexError:
                    current = None

                if current and len(current) < 120:
                    if current == last_val:
                        consecutive += 1
                    else:
                        if consecutive >= 3 and last_val:
                            boilerplate.add(last_val)
                        consecutive = 1
                        last_val = current
                else:
                    if consecutive >= 3 and last_val:
                        boilerplate.add(last_val)
                    consecutive = 1
                    last_val = None
            if consecutive >= 3 and last_val:
                boilerplate.add(last_val)

    _scan_positions([0, 1, -1, -2])
    return boilerplate


def detect_boilerplate_by_bbox(pages_text_bbox: list[list[tuple]],
                                page_heights: list[float]) -> set[str]:
    """
    Additional heuristic: lines found exclusively in the top/bottom 10%
    of the page on 3+ pages are headers/footers.

    `pages_text_bbox` is a list of pages; each page is a list of
    (text, x0, top, x1, bottom) tuples from pdfplumber word extraction.
    """
    boilerplate: set[str] = set()
    from collections import Counter
    line_page_counts: Counter = Counter()
    line_zone_counts: Counter = Counter()  # how many pages the line is in header/footer zone

    num_pages = len(pages_text_bbox)
    for page_idx, words in enumerate(pages_text_bbox):
        if not words or page_idx >= len(page_heights):
            continue
        height = page_heights[page_idx]
        if height == 0:
            continue

        # Group words into logical lines by top coordinate (within 3pt tolerance)
        from collections import defaultdict
        line_buckets: dict[int, list[str]] = defaultdict(list)
        for (text, x0, top, x1, bottom) in words:
            bucket = int(top / 3) * 3
            line_buckets[bucket].append(text)

        for bucket, word_list in line_buckets.items():
            line_text = ' '.join(word_list).strip()
            if not line_text or len(line_text) > 120:
                continue
            line_page_counts[line_text] += 1
            # Check if line is in top or bottom 10% of page
            relative_pos = bucket / height
            if relative_pos < 0.10 or relative_pos > 0.90:
                line_zone_counts[line_text] += 1

    for line_text, count in line_page_counts.items():
        if count >= 3 and line_zone_counts.get(line_text, 0) >= 3:
            boilerplate.add(line_text)

    return boilerplate


# ── Per-page text extraction ──────────────────────────────────────────────────

def _extract_text_pdfplumber(path: str) -> tuple[list[str], list[list[str]],
                                                   list[dict], list[float]]:
    """
    Extract page text using pdfplumber (layout-aware).
    Returns:
        raw_texts      — list of str, one per page
        pages_lines    — list of line-lists, one per page (for boilerplate detection)
        tables_per_page— dict {page_idx: [table, ...]} of raw table data
        page_heights   — list of float page heights in pts
    """
    raw_texts: list[str] = []
    pages_lines: list[list[str]] = []
    tables_per_page: dict[int, list] = {}
    page_heights: list[float] = []

    with pdfplumber.open(path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_heights.append(float(page.height or 0))

            # Extract tables first, then strip table bounding boxes from text
            tables = page.extract_tables()
            if tables:
                tables_per_page[page_idx] = tables

            # Use extract_text with layout=True for better multi-column handling
            try:
                text = page.extract_text(layout=True) or ''
            except TypeError:
                # Older pdfplumber versions don't support layout=True
                text = page.extract_text() or ''

            raw_texts.append(text)
            lines = [line.strip() for line in text.split('\n')]
            pages_lines.append(lines)

    return raw_texts, pages_lines, tables_per_page, page_heights


def _extract_text_pypdf(path: str) -> tuple[list[str], list[list[str]]]:
    """Fallback: extract text via pypdf (no layout awareness)."""
    raw_texts: list[str] = []
    pages_lines: list[list[str]] = []

    reader = pypdf.PdfReader(path)
    for page in reader.pages:
        text = page.extract_text() or ''
        raw_texts.append(text)
        pages_lines.append([line.strip() for line in text.split('\n')])

    return raw_texts, pages_lines


# ── Main public API ───────────────────────────────────────────────────────────

def extract(path: str, describe_diagrams: bool = False) -> list[Chunk]:
    chunks: list[Chunk] = []
    filename = os.path.basename(path)

    if pypdf is None and pdfplumber is None:
        return []

    raw_texts: list[str] = []
    pages_lines: list[list[str]] = []
    tables_per_page: dict[int, list] = {}
    used_pdfplumber = False

    # ── Attempt pdfplumber (preferred — layout-aware) ──────────────────────
    if pdfplumber is not None:
        try:
            raw_texts, pages_lines, tables_per_page, _ = _extract_text_pdfplumber(path)
            used_pdfplumber = True
        except Exception as e:
            print(f'Warning: pdfplumber failed for {filename} ({e}), falling back to pypdf')

    # ── Fallback: pypdf ────────────────────────────────────────────────────
    if not used_pdfplumber:
        if pypdf is None:
            return []
        try:
            raw_texts, pages_lines = _extract_text_pypdf(path)
        except Exception as e:
            print(f'Error reading {filename} via pypdf: {e}')
            return []

        # When using pypdf fallback, still try pdfplumber for tables only
        if pdfplumber is not None:
            try:
                with pdfplumber.open(path) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        if tables:
                            tables_per_page[page_idx] = tables
            except Exception as e:
                print(f'Warning: pdfplumber table extraction failed for {filename}: {e}')

    # ── Boilerplate detection ──────────────────────────────────────────────
    boilerplate = detect_boilerplate(pages_lines)

    # ── Build chunks ───────────────────────────────────────────────────────
    limit = 10_000
    for page_idx, text in enumerate(raw_texts):
        location = f'page {page_idx + 1}'

        # Strip boilerplate lines
        cleaned_lines = [
            line for line in text.split('\n')
            if line.strip() not in boilerplate
        ]
        cleaned_text = '\n'.join(cleaned_lines).strip()

        # Table chunks (extracted separately for semantic precision)
        if page_idx in tables_per_page:
            for t_idx, table in enumerate(tables_per_page[page_idx]):
                table_md = table_to_markdown(table)
                if table_md:
                    chunks.append(Chunk(
                        text=table_md,
                        source_file=filename,
                        location=f'{location} - table {t_idx + 1}',
                        chunk_type='table',
                    ))

        # Text chunks
        if cleaned_text:
            if len(cleaned_text) > limit:
                for part_num, part in enumerate(split_text_to_limit(cleaned_text, limit), start=1):
                    chunks.append(Chunk(
                        text=part,
                        source_file=filename,
                        location=f'{location} (part {part_num})',
                        chunk_type='text',
                    ))
            else:
                chunks.append(Chunk(
                    text=cleaned_text,
                    source_file=filename,
                    location=location,
                    chunk_type='text',
                ))

    return chunks
