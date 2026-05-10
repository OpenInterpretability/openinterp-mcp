"""Atlas search — embed publications and find related entries.

Storage is sqlite at `~/.openinterp/atlas.db`. No vector-DB dependency. Cosine similarity is
trivially fast for the scale we expect (10k entries).
"""

from openinterp_mcp.atlas.search import search_atlas, refresh_index
from openinterp_mcp.atlas.vector_store import VectorStore

__all__ = ["search_atlas", "refresh_index", "VectorStore"]
