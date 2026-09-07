"""
main.py — Exam Reference Merger entry point.

Upgrades over original:
- Parallel file extraction via ThreadPoolExecutor (I/O-bound, safe with GIL).
- Rich progress output using tqdm bars per pipeline stage.
- Per-file error isolation: one bad file no longer aborts the whole run.
- Configurable worker count (default: min(4, cpu_count)).
- Unsupported file reporting is always on (no need for --verbose).
"""

import os
import sys
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adjust sys.path so we can run this file directly or as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from exam_merger.models.chunk import Chunk, ChunkGroup
from exam_merger.extractors import pdf_extractor, pptx_extractor, docx_extractor
from exam_merger.pipeline import normalizer, deduplicator, assembler
from exam_merger.utils import get_logger, progress

logger = get_logger()

SUPPORTED_EXTS = {'.pdf', '.pptx', '.docx'}
SKIPPABLE_EXTS = {'.doc', '.ppt', '.txt', '.png', '.jpg', '.jpeg', '.gif',
                  '.bmp', '.xls', '.xlsx', '.csv'}


# ── File collection ───────────────────────────────────────────────────────────

def _collect_files(input_dir: str, recursive: bool) -> list[str]:
    file_paths: list[str] = []
    skipped: list[str] = []

    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                full = os.path.join(root, file)
                if ext in SUPPORTED_EXTS:
                    file_paths.append(full)
                elif ext in SKIPPABLE_EXTS:
                    skipped.append(file)
    else:
        for file in os.listdir(input_dir):
            full = os.path.join(input_dir, file)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTS:
                file_paths.append(full)
            elif ext in SKIPPABLE_EXTS:
                skipped.append(file)

    if skipped:
        logger.warning(f'Skipping {len(skipped)} unsupported file(s): {", ".join(skipped[:5])}'
                       + (' ...' if len(skipped) > 5 else ''))

    return file_paths


# ── Single-file extraction (runs in thread pool) ──────────────────────────────

def _extract_one(path: str, describe_diagrams: bool) -> tuple[str, list[Chunk], str | None]:
    """
    Extract chunks from a single file.  Returns (path, chunks, error_msg).
    Designed to be called from a thread pool.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.pdf':
            chunks = pdf_extractor.extract(path, describe_diagrams)
        elif ext == '.pptx':
            chunks = pptx_extractor.extract(path, describe_diagrams)
        elif ext == '.docx':
            chunks = docx_extractor.extract(path, describe_diagrams)
        else:
            chunks = []
        return path, chunks, None
    except Exception:
        return path, [], traceback.format_exc()


# ── Main merge function ───────────────────────────────────────────────────────

def merge_folder(
    input_dir: str,
    output_path: str = None,
    describe_diagrams: bool = False,
    threshold: float = 0.82,
    recursive: bool = False,
    verbose: bool = False,
    max_workers: int = None,
) -> str:
    """
    Merge all supported documents in `input_dir` into a single Markdown file.

    Parameters
    ----------
    input_dir        : Directory containing .pdf / .pptx / .docx files.
    output_path      : Where to write the output Markdown.
                       Defaults to ./output/<folder>_master_reference.md
    describe_diagrams: Enable Claude-based image descriptions (needs API key).
    threshold        : Cosine similarity threshold for deduplication (0–1).
    recursive        : Recurse into subdirectories.
    verbose          : Print per-stage statistics.
    max_workers      : Max parallel extraction threads.
                       Defaults to min(4, os.cpu_count() or 1).

    Returns
    -------
    Path to the written Markdown file.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f'Input path is not a directory: {input_dir}')

    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    # ── 1. Collect files ──────────────────────────────────────────────────
    file_paths = _collect_files(input_dir, recursive)

    if not file_paths:
        raise ValueError(
            f'No supported documents (.pdf, .pptx, .docx) found in: {input_dir}'
        )

    if verbose:
        logger.info(f'Found {len(file_paths)} file(s) to merge:')
        for p in file_paths:
            logger.info(f'  - {os.path.basename(p)}')

    # ── 2. Extract chunks — parallel ──────────────────────────────────────
    if verbose:
        logger.info(f'Extracting content (up to {max_workers} parallel workers)...')

    all_chunks: list[Chunk] = []
    extraction_errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_one, path, describe_diagrams): path
            for path in file_paths
        }

        # Use tqdm to show a progress bar as futures complete
        for future in progress(as_completed(futures), description='Extracting'):
            path, chunks, error = future.result()
            basename = os.path.basename(path)
            if error:
                logger.error(f'  [ERROR] {basename}: {error.splitlines()[-1]}')
                extraction_errors.append(basename)
            else:
                all_chunks.extend(chunks)
                if verbose:
                    logger.info(f'  -> {basename}: {len(chunks)} chunks')

    if extraction_errors:
        logger.warning(
            f'{len(extraction_errors)} file(s) had extraction errors and were skipped: '
            + ', '.join(extraction_errors)
        )

    if not all_chunks:
        raise ValueError('No content was extracted from any of the files.')

    if verbose:
        logger.info(f'Total raw chunks extracted: {len(all_chunks)}')

    # ── 3. Normalize ──────────────────────────────────────────────────────
    if verbose:
        logger.info('Normalizing chunks...')
    normalized_chunks = normalizer.normalize(all_chunks)
    if verbose:
        dropped = len(all_chunks) - len(normalized_chunks)
        logger.info(f'  -> {len(normalized_chunks)} chunks remain '
                    f'({dropped} dropped as boilerplate/stubs)')

    # ── 4. Deduplicate ────────────────────────────────────────────────────
    if verbose:
        logger.info('Deduplicating chunks (vectorized cosine similarity)...')
    groups = deduplicator.deduplicate(normalized_chunks, threshold)
    if verbose:
        logger.info(f'  -> Grouped into {len(groups)} unique concept groups.')

    # ── 5. Assemble Markdown ──────────────────────────────────────────────
    if verbose:
        logger.info('Assembling master document...')
    markdown_output = assembler.assemble(groups, file_paths)

    # ── 6. Save output ────────────────────────────────────────────────────
    if not output_path:
        folder_name = os.path.basename(os.path.abspath(input_dir)) or 'merged'
        output_dir = os.path.normpath(
            os.path.join(os.path.abspath(input_dir), '..', 'output')
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{folder_name}_master_reference.md')
    else:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_output)

    if verbose:
        logger.info(f'\nSuccess! Master reference written to:\n  {output_path}')

    return output_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Exam Reference Merger — Clean, deduplicate, and merge study files.'
    )
    parser.add_argument(
        'input_folder',
        help='Path to folder containing exam materials (.pdf, .pptx, .docx)'
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: ./output/<folder>_master_reference.md)'
    )
    parser.add_argument(
        '--describe-diagrams',
        action='store_true',
        help='Enable Claude-based diagram descriptions (requires ANTHROPIC_API_KEY)'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float, default=0.82,
        help='Deduplication cosine similarity threshold (default: 0.82)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Scan subfolders recursively'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed per-stage statistics'
    )
    parser.add_argument(
        '--max-workers',
        type=int, default=None,
        help='Max parallel extraction threads (default: min(4, cpu_count))'
    )

    args = parser.parse_args()

    try:
        output = merge_folder(
            input_dir=args.input_folder,
            output_path=args.output,
            describe_diagrams=args.describe_diagrams,
            threshold=args.similarity_threshold,
            recursive=args.recursive,
            verbose=args.verbose,
            max_workers=args.max_workers,
        )
        print(f'\nOutput: {output}')
    except Exception as e:
        logger.error(f'\nFatal error: {e}')
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
