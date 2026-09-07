from collections import deque
from typing import Optional

import numpy as np
from ..models.chunk import Chunk, ChunkGroup

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    sk_cosine_similarity = None
    TfidfVectorizer = None

# ── Embedding model singleton ────────────────────────────────────────────────

_model = None
_EMBED_BATCH_SIZE = 64        # process this many chunks per forward pass
_EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'


class _FallbackEmbeddingModel:
    """Pure-offline TF-IDF char n-gram model — no network required."""

    def encode(self, texts: list[str], batch_size: int = _EMBED_BATCH_SIZE,
               show_progress_bar: bool = False) -> np.ndarray:
        if TfidfVectorizer is None:
            # Absolute last resort: identity embeddings (all chunks treated as unique)
            return np.eye(len(texts))
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray().astype(float)


def get_embedding_model():
    global _model
    if _model is None:
        if SentenceTransformer is not None:
            try:
                _model = SentenceTransformer(_EMBED_MODEL_NAME)
            except Exception:
                _model = _FallbackEmbeddingModel()
        else:
            _model = _FallbackEmbeddingModel()
    return _model


# ── Vectorised similarity helpers ────────────────────────────────────────────

def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation — safe against zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)      # avoid divide-by-zero
    return matrix / norms


def _encode_in_batches(model, texts: list[str]) -> np.ndarray:
    """
    Encode text in batches of _EMBED_BATCH_SIZE to control peak RAM.
    Returns a (N, D) float32 array of L2-normalised embeddings.
    """
    all_vecs = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start: start + _EMBED_BATCH_SIZE]
        vecs = np.asarray(
            model.encode(batch, batch_size=_EMBED_BATCH_SIZE, show_progress_bar=False),
            dtype=np.float32
        )
        all_vecs.append(vecs)
    combined = np.vstack(all_vecs)
    return _l2_normalise(combined)


def _build_adjacency_list(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """
    Build a sparse adjacency list of pairs whose cosine similarity >= threshold.

    Strategy:
    - Compute the full similarity matrix as a single vectorised dot product
      (L2-normalised embeddings → dot = cosine).  This stays in numpy C land
      and is ~10x faster than the Python BFS loop it replaces.
    - Apply threshold mask immediately → sparse boolean matrix.
    - Convert to adjacency list — only store edges that exist.

    Memory note: for N=1000, the float32 matrix is 4 MB.  That's fine.
    For N>3000 we could use chunked row-wise processing; N that large is
    unlikely in practice for exam materials.
    """
    n = len(embeddings)
    # dot product of L2-normalised vectors = cosine similarity
    sim_matrix = embeddings @ embeddings.T           # shape (N, N), float32
    np.fill_diagonal(sim_matrix, 0.0)                # exclude self-loops

    adjacency: list[list[int]] = [[] for _ in range(n)]
    rows, cols = np.where(sim_matrix >= threshold)
    for r, c in zip(rows.tolist(), cols.tolist()):
        if c > r:                                    # store each edge once
            adjacency[r].append(c)
            adjacency[c].append(r)

    return adjacency, sim_matrix


def _bfs_connected_components(adjacency: list[list[int]], n: int) -> list[list[int]]:
    """BFS over adjacency list — O(V + E) instead of O(V²)."""
    visited = [False] * n
    clusters = []
    for i in range(n):
        if not visited[i]:
            cluster = []
            queue = deque([i])
            visited[i] = True
            while queue:
                curr = queue.popleft()
                cluster.append(curr)
                for neighbour in adjacency[curr]:
                    if not visited[neighbour]:
                        visited[neighbour] = True
                        queue.append(neighbour)
            clusters.append(cluster)
    return clusters


# ── Two-tier merge logic ─────────────────────────────────────────────────────

def _merge_cluster(cluster_indices: list[int], chunks: list[Chunk],
                   sim_matrix: np.ndarray, threshold: float) -> ChunkGroup:
    """
    Within a connected-component cluster apply the two-tier merge rule:
      - sim >= 0.95  → near-duplicate; keep the longer version only
      - threshold <= sim < 0.95 → related; append as **Additional Detail**
    """
    if len(cluster_indices) == 1:
        c = chunks[cluster_indices[0]]
        return ChunkGroup(representative_text=c.text, members=[c])

    # Sort by text length ascending so shorter (intro/summary) chunks are
    # processed first and can be superseded by longer detailed versions.
    sorted_indices = sorted(cluster_indices, key=lambda idx: len(chunks[idx].text))

    # accepted_concepts: list of {"main_idx": int, "details": list[int]}
    accepted_concepts: list[dict] = []

    for idx in sorted_indices:
        best_c_idx = -1
        best_type: Optional[str] = None
        highest_sim = -1.0

        for c_idx, concept in enumerate(accepted_concepts):
            sim = float(sim_matrix[idx, concept['main_idx']])

            if sim >= 0.95:
                if best_type != 'duplicate' or sim > highest_sim:
                    highest_sim = sim
                    best_c_idx = c_idx
                    best_type = 'duplicate'
            elif sim >= threshold:
                if best_type != 'duplicate':
                    if best_type is None or sim > highest_sim:
                        highest_sim = sim
                        best_c_idx = c_idx
                        best_type = 'related'

        if best_c_idx != -1:
            if best_type == 'duplicate':
                # Replace with the longer chunk (idx is processed in length order
                # so idx is always >= length of current main)
                accepted_concepts[best_c_idx]['main_idx'] = idx
            else:
                accepted_concepts[best_c_idx]['details'].append(idx)
        else:
            accepted_concepts.append({'main_idx': idx, 'details': []})

    # Build representative text
    concept_texts = []
    for concept in accepted_concepts:
        main_chunk = chunks[concept['main_idx']]
        parts = [main_chunk.text]
        for detail_idx in concept['details']:
            detail_chunk = chunks[detail_idx]
            if detail_chunk.chunk_type == 'table':
                parts.append(f'\n**Supplemental Table:**\n{detail_chunk.text}')
            else:
                parts.append(f'\n**Additional Detail:**\n{detail_chunk.text}')
        concept_texts.append('\n\n'.join(parts))

    rep_text = '\n\n---\n\n'.join(concept_texts)
    all_members = [chunks[idx] for idx in cluster_indices]
    return ChunkGroup(representative_text=rep_text, members=all_members)


# ── Public API ───────────────────────────────────────────────────────────────

def deduplicate(chunks: list[Chunk], threshold: float = 0.82) -> list[ChunkGroup]:
    """
    Deduplicate and group chunks by semantic similarity.

    Improvements over the original:
    1. Batched encoding (batch_size=64) — controls peak RAM usage.
    2. L2-normalised dot product instead of sklearn cosine_similarity —
       same math, but purely vectorised in numpy C code (~10x faster).
    3. Adjacency list + BFS — O(V+E) instead of O(V²) clustering loop.
    4. Graceful two-stage fallback: sentence-transformers → TF-IDF → identity.
    """
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    embeddings: Optional[np.ndarray] = None

    # Stage 1: sentence-transformers
    try:
        model = get_embedding_model()
        embeddings = _encode_in_batches(model, texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
    except Exception as e:
        print(f'\nWarning: embedding model failed ({e}). Falling back to TF-IDF...')
        embeddings = None

    # Stage 2: TF-IDF fallback
    if embeddings is None:
        try:
            if TfidfVectorizer is None:
                raise ImportError('scikit-learn not available')
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
            raw = vectorizer.fit_transform(texts).toarray().astype(np.float32)
            embeddings = _l2_normalise(raw)
        except Exception as e2:
            print(f'Warning: TF-IDF fallback also failed ({e2}). Treating all chunks as unique.')
            return [
                ChunkGroup(representative_text=c.text, members=[c])
                for c in chunks
            ]

    # Build sparse adjacency list + keep sim_matrix for two-tier merge
    adjacency, sim_matrix = _build_adjacency_list(embeddings, threshold)

    # BFS connected components
    clusters = _bfs_connected_components(adjacency, len(chunks))

    # Merge within each cluster
    groups = [_merge_cluster(cluster, chunks, sim_matrix, threshold)
              for cluster in clusters]

    return groups
