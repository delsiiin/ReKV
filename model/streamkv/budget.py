"""Guidance-ranked, fixed-size historical KV archive for budget mode."""
import torch

from .cache import ContextManager


class BudgetContextManager(ContextManager):
    def calculate_score(self, video_len, prompt):
        super().calculate_score(video_len, prompt)
        # Score old and new blocks with the same current guidance query so scores
        # from different segments do not need to be calibrated against each other.
        query = prompt.mean(dim=2).reshape(self.num_units, -1)
        self.archive_compression_score = torch.stack([
            self.block_k[u].get_cosine_similarity(query[u])
            for u in range(self.num_units)
        ])

    def compress_to_budget(self, budget_tokens):
        """Cap each batch item's archive; prompt/local/working buffers are excluded."""
        if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int):
            raise ValueError('budget_tokens must be an integer')
        capacity = budget_tokens // self.block_size
        if capacity < 1:
            raise ValueError(f'budget_tokens must fit one frame block ({self.block_size} tokens)')
        if self.num_global_block <= capacity:
            return

        scores = self.archive_compression_score
        # Stable ties preserve earlier blocks; chronological order is restored
        # before remapping retrieval and segment indices.
        kept = scores.argsort(dim=1, descending=True, stable=True)[:, :capacity]
        kept = kept.sort(dim=1).values
        for u in range(self.num_units):
            indices = kept[u].tolist()
            remap = {old: new for new, old in enumerate(indices)}
            old_blocks = self.global_blocks[u]
            old_cached = self.cached_blocks[u]
            for old in old_cached:
                if old not in remap:
                    old_blocks[old].offload()
            self.global_blocks[u] = [old_blocks[old] for old in indices]
            self.cached_blocks[u] = {remap[old]: stamp for old, stamp in old_cached.items()
                                     if old in remap}
            vectors = self.block_k[u]
            vectors.data[:capacity].copy_(vectors.get_data()[kept[u]])
            vectors.length = capacity
            self.chunk2block[u] = [mapped for chunk in self.chunk2block[u]
                                   if (mapped := [remap[old] for old in chunk if old in remap])]

        self.archive_compression_score = scores.gather(1, kept)
        self.num_global_block = capacity
        self.retrieved_block_indices = None
        self.layer_retrieval_score = None
