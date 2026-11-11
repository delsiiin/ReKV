"""Dependency-free StreamKV settings, shared by the runtime and CLI."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamKVConfig:
    n_local: int = 15000
    retrieve_size: int = 64
    compression_ratio: float = 1.
    compress_temp: float = 0.
    retrieve_temp: float = 0.
    segment_mode: str = 'uniform'
    min_segment_size: int = 8
    max_segment_size: int = 64
    segment_theta: float = .95
    use_segment_summary: bool = True
    retrieve_local: bool = True
    retrieve_local_size: int = 8
    retrieve_chunk_size: int = 1
    budget_tokens: int | None = None

    def __post_init__(self):
        if self.budget_tokens is not None and (
            isinstance(self.budget_tokens, bool)
            or not isinstance(self.budget_tokens, int)
            or self.budget_tokens <= 0
        ):
            raise ValueError('budget_tokens must be a positive integer')
