import re
import unicodedata
from ..models.chunk import Chunk

# Minimum characters a chunk must have to be kept (unless it's a table/diagram)
MIN_CHUNK_CHARS = 40

PAGE_PATTERNS = [
    re.compile(r'^page\s+\d+$', re.IGNORECASE),
    re.compile(r'^slide\s+\d+$', re.IGNORECASE),
    re.compile(r'^page\s+\d+\s+of\s+\d+$', re.IGNORECASE),
    re.compile(r'^\d+$'),
    re.compile(r'^\d+\s+of\s+\d+$'),
]

# Patterns that indicate stub/fragment content — single-line filler that adds no information
STUB_PATTERNS = [
    re.compile(r'^(see|refer to|cf\.?|cont\.?|cont\'d|continued|figure|fig\.?|table|appendix)\s', re.IGNORECASE),
    re.compile(r'^\[.*?\]$'),          # Lone bracket tags like [IMAGE] or [TABLE]
    re.compile(r'^[-•·▪▸◦]+\s*$'),    # Bare bullet symbols with no content
    re.compile(r'^#+\s*$'),            # Heading marker with no text
]

# Unicode replacements: normalise common typographic chars to ASCII equivalents
_UNICODE_REPLACEMENTS = [
    ('\u201c', '"'), ('\u201d', '"'),   # smart double quotes
    ('\u2018', "'"), ('\u2019', "'"),   # smart single quotes
    ('\u2013', '-'), ('\u2014', '-'),   # en-dash, em-dash
    ('\u2026', '...'),                  # ellipsis
    ('\u00a0', ' '),                    # non-breaking space
    ('\u2022', '-'),                    # bullet •
    ('\u25cf', '-'),                    # filled circle ●
    ('\u2212', '-'),                    # minus sign −
    ('\ufeff', ''),                     # BOM
]


def clean_text(text: str) -> str:
    # 1. NFKC unicode normalisation — fixes ligatures (ﬁ→fi), fullwidth chars, etc.
    text = unicodedata.normalize('NFKC', text)

    # 2. Replace known typographic characters
    for src, dst in _UNICODE_REPLACEMENTS:
        text = text.replace(src, dst)

    # 3. Trim lines and collapse multiple blank lines (keep at most one)
    lines = [line.strip() for line in text.split('\n')]

    cleaned_lines = []
    prev_was_empty = False
    for line in lines:
        if line == '':
            if not prev_was_empty:
                cleaned_lines.append('')
                prev_was_empty = True
        else:
            cleaned_lines.append(line)
            prev_was_empty = False

    # Remove leading/trailing empty lines
    while cleaned_lines and cleaned_lines[0] == '':
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == '':
        cleaned_lines.pop()

    return '\n'.join(cleaned_lines)


def is_boilerplate(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True

    # Match page/slide number patterns
    for pattern in PAGE_PATTERNS:
        if pattern.match(cleaned):
            return True

    return False


def is_stub_fragment(text: str, chunk_type: str) -> bool:
    """Return True if the chunk is too short or matches a known stub pattern."""
    # Tables and diagram descriptions are always kept regardless of length
    if chunk_type in ('table', 'diagram_description'):
        return False

    stripped = text.strip()

    # Too short to carry meaningful information
    if len(stripped) < MIN_CHUNK_CHARS:
        return True

    # Single-line stub patterns (only apply when entire content is one line)
    if '\n' not in stripped:
        for pattern in STUB_PATTERNS:
            if pattern.match(stripped):
                return True

    return False


def normalize(chunks: list[Chunk]) -> list[Chunk]:
    normalized_chunks = []

    for chunk in chunks:
        # 1. Clean text
        cleaned = clean_text(chunk.text)

        # 2. Skip empty or boilerplate
        if not cleaned or is_boilerplate(cleaned):
            continue

        # 3. Skip stub fragments that add no value
        if is_stub_fragment(cleaned, chunk.chunk_type):
            continue

        # 4. Update chunk text in-place
        chunk.text = cleaned
        normalized_chunks.append(chunk)

    return normalized_chunks
