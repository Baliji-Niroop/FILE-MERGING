"""
assembler.py — Assemble deduplicated ChunkGroups into a Markdown master document.

Improvements over the original:
- Case-insensitive topic deduplication (avoids ## Physics and ## physics)
- Smarter topic fallback: extracts the first meaningful sentence from the chunk
  text instead of using the bare filename.
- Generation timestamp in the document header.
- TOC anchors are correctly slugified for GitHub Markdown.
"""

import os
import re
from collections import Counter
from datetime import datetime
from ..models.chunk import Chunk, ChunkGroup


# ── Slugify ───────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s-]+', '-', slug)
    return slug.strip('-')


# ── Topic extraction helpers ──────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r'[.!?]\s')
_CLEAN_TOPIC = re.compile(r'[_\-]+')
_WHITESPACE = re.compile(r'\s+')


def _first_sentence(text: str, max_chars: int = 70) -> str:
    """
    Extract the first sentence (or first max_chars of text) from a chunk,
    cleaned up as a topic label.
    """
    # Take first non-empty line
    first_line = ''
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if not first_line:
        return ''

    # Split on sentence boundary
    match = _SENTENCE_SPLIT.search(first_line)
    candidate = first_line[:match.start() + 1] if match else first_line

    # Trim to max_chars at a word boundary
    if len(candidate) > max_chars:
        candidate = candidate[:max_chars].rsplit(' ', 1)[0]

    # Remove trailing punctuation
    candidate = candidate.rstrip('.,;:')
    return candidate.strip()


def get_topic_hint(chunk: Chunk) -> str:
    """Extract a human-readable topic hint from a chunk based on its location/type."""
    # 1. DOCX section heading path
    if chunk.location.startswith('Section: '):
        parts = chunk.location[9:].split(' > ')
        if parts:
            return parts[0].strip()

    # 2. DOCX bare heading
    if chunk.location.startswith('Heading '):
        return chunk.text.strip()[:70]

    # 3. PPTX slide title
    if chunk.location.endswith('- title') and chunk.chunk_type == 'text':
        return chunk.text.strip()[:70]

    return ''


def _stable_mode(values: list[str]) -> str:
    """Return the most frequent non-empty value; break ties alphabetically."""
    counts = Counter(v for v in values if v)
    if not counts:
        return ''
    max_count = max(counts.values())
    candidates = sorted(v for v, c in counts.items() if c == max_count)
    return candidates[0]


def _normalise_topic(topic: str) -> str:
    """Normalise topic for deduplication key: lowercase, collapse whitespace."""
    return _WHITESPACE.sub(' ', topic.strip()).lower()


# ── Assembler ─────────────────────────────────────────────────────────────────

def assemble(groups: list[ChunkGroup], source_files: list[str]) -> str:
    if not groups:
        return '# Master Reference Document\n\nNo content extracted.'

    # ── 1. Assign topic tags ──────────────────────────────────────────────
    # Use a normalised key so "Physics" and "physics" map to the same section.
    topic_order: list[str] = []           # display names in insertion order
    topic_key_to_display: dict[str, str] = {}   # norm_key → first-seen display name
    topics_map: dict[str, list[ChunkGroup]] = {}  # norm_key → groups

    for group in groups:
        hints = [get_topic_hint(m) for m in group.members if get_topic_hint(m)]

        if hints:
            topic_display = _stable_mode(hints)
        else:
            # Smarter fallback: first sentence of the representative text
            topic_display = _first_sentence(group.representative_text)

        # Last-resort fallback: source file base name (no extension)
        if not topic_display:
            sources = [os.path.splitext(m.source_file)[0] for m in group.members]
            topic_display = _stable_mode(sources) or 'General Reference'

        # Replace underscores/hyphens in filename-derived topics
        topic_display = _CLEAN_TOPIC.sub(' ', topic_display).strip()
        group.topic_tag = topic_display

        norm_key = _normalise_topic(topic_display)

        if norm_key not in topic_key_to_display:
            topic_key_to_display[norm_key] = topic_display
            topics_map[norm_key] = []
            topic_order.append(norm_key)

        topics_map[norm_key].append(group)

    # ── 2. Build Markdown ─────────────────────────────────────────────────
    subject_name = 'Exam Study Guide'
    if source_files:
        subject_name = os.path.basename(os.path.dirname(source_files[0]))
        if not subject_name or subject_name in ('.', ''):
            subject_name = 'Subject Master'

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    source_basenames = [os.path.basename(f) for f in source_files]

    md_lines: list[str] = []
    md_lines.append(f'# {subject_name} — Master Reference Document')
    md_lines.append(f'*Generated from: {", ".join(source_basenames)}*  ')
    md_lines.append(f'*Generated on: {timestamp}*\n')

    # Table of Contents
    md_lines.append('## Table of Contents')
    for norm_key in topic_order:
        display = topic_key_to_display[norm_key]
        slug = slugify(display)
        md_lines.append(f'- [{display}](#{slug})')
    md_lines.append('')

    # ── 3. Render topics ──────────────────────────────────────────────────
    for norm_key in topic_order:
        display = topic_key_to_display[norm_key]
        md_lines.append(f'## {display}\n')

        for group in topics_map[norm_key]:
            diagrams = [m for m in group.members if m.chunk_type == 'diagram_description']
            is_only_diagram = len(diagrams) == len(group.members)

            if is_only_diagram:
                for m in group.members:
                    md_lines.append(f'[DIAGRAM — {m.source_file}, {m.location}]: {m.text}\n')
            else:
                md_lines.append(group.representative_text)

                # Inline diagram descriptions
                for d in diagrams:
                    md_lines.append(f'\n[DIAGRAM — {d.source_file}, {d.location}]: {d.text}')

                # Source citations (deduplicated)
                citations: list[str] = []
                seen_cits: set[str] = set()
                for m in group.members:
                    cit = f'{m.source_file} ({m.location})'
                    if cit not in seen_cits:
                        citations.append(cit)
                        seen_cits.add(cit)
                if citations:
                    md_lines.append(f'\n*Sources: {", ".join(citations)}*')
                md_lines.append('')

        md_lines.append('---')

    return '\n'.join(md_lines)
