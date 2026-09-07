from dataclasses import dataclass, field
import numpy as np

@dataclass
class Chunk:
    text: str                      # the actual extracted content
    source_file: str               # e.g. "Unit3_CAD.pptx"
    location: str                  # e.g. "slide 4" or "page 12"
    chunk_type: str                # "text" | "table" | "diagram_description" | "speaker_notes"
    embedding: np.ndarray = field(default=None, repr=False)  # filled in during dedup stage

    def __repr__(self) -> str:
        return f"Chunk(text={self.text[:30]!r}..., source={self.source_file}, loc={self.location}, type={self.chunk_type})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Chunk):
            return False
        return (
            self.text == other.text and
            self.source_file == other.source_file and
            self.location == other.location and
            self.chunk_type == other.chunk_type
        )

    def __hash__(self) -> int:
        return hash((self.text, self.source_file, self.location, self.chunk_type))

@dataclass
class ChunkGroup:
    representative_text: str       # the merged/best version of this concept
    members: list[Chunk]           # all chunks that got clustered together
    topic_tag: str = None          # assigned during assembly, e.g. "Unit I: CAD Fundamentals"

    def __repr__(self) -> str:
        return f"ChunkGroup(repr_len={len(self.representative_text)}, members={len(self.members)})"
