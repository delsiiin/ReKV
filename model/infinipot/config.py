"""The released run_sample.sh / README experiment profile."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InfiniPotConfig:
    block_size: int = 32
    compress_frame_num: int = 24
    compression_method: str = 'infinipot-v'
    max_frames_num: int = 768
    tar_ratio: float = 0.5
    query_ratio: float = 0.25
    adaptive_pooling: bool = False
    per_frame: bool = False

    def __post_init__(self):
        if self.block_size == 0 or self.block_size < -1:
            raise ValueError('block_size must be -1 (standard generation) or positive')
        if self.compress_frame_num < 0:
            raise ValueError('compress_frame_num must be nonnegative')
        if self.block_size > 0 and self.compress_frame_num >= self.block_size:
            raise ValueError('compress_frame_num must be smaller than block_size')
        if self.max_frames_num < 2:
            raise ValueError('max_frames_num must be at least 2')
        if self.compression_method not in ('infinipot-v', 'uniform', 'swa'):
            raise ValueError('compression_method must be infinipot-v, uniform or swa')
        if not math.isfinite(self.tar_ratio) or not 0 <= self.tar_ratio <= 1:
            raise ValueError('tar_ratio must be between 0 and 1')
        if not math.isfinite(self.query_ratio) or not 0 <= self.query_ratio <= 1:
            raise ValueError('query_ratio must be between 0 and 1')

    def validate_budget(self, token_per_frame, total_frames=None):
        # The reference topk cannot select a negative budget, and -0 slicing
        # does not denote an empty query. Reject rather than alter its scores.
        if (self.block_size > 0 and self.compress_frame_num > 0
                and self.compression_method == 'infinipot-v' and self.tar_ratio > 0):
            keep_frames = self.compress_frame_num
            if total_frames is not None:
                remaining = total_frames - self.block_size
                if remaining <= 0:
                    return
                # The first compression has the smallest effective budget.
                # Later non-final blocks are full again, and their keep budget
                # can only grow as the remaining video becomes shorter.
                keep_frames = max(keep_frames, self.block_size - remaining)
            query_frames = int(self.block_size * self.query_ratio)
            retained = keep_frames * token_per_frame
            tar_budget = retained - round((1 - self.tar_ratio) * retained)
            if query_frames < 1 or query_frames * token_per_frame > tar_budget:
                raise ValueError('TaR requires at least one query frame and a budget covering all query tokens')
