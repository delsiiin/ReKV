"""Shared ReKV block selection, without changes to the encoding cache."""
import torch


def select_blocks(query, *, n_init, topk, chunk_size, block_size, init_exc,
                  remainder_k, block_keys, num_global_block):
    batch, heads, _, dim = query.shape
    query = query.mean(dim=2).reshape(batch, heads * dim)
    logits = None
    if num_global_block <= topk:
        if not init_exc:
            keys = remainder_k[:, :, n_init:, :]
            kv_heads = keys.size(1)
            length = keys.size(2)
            assert length % block_size == 0
            blocks = length // block_size
            if blocks <= topk:
                indices = [list(range(blocks)) for _ in range(batch)]
            else:
                keys = keys.reshape(batch, kv_heads, 1, length, dim).expand(
                    batch, kv_heads, heads // kv_heads, length, dim)
                keys = keys.reshape(batch, heads, length, dim).transpose(1, 2)
                keys = keys.reshape(batch, blocks, block_size, heads * dim).mean(dim=-2)
                logits = torch.matmul(keys, query[:, :, None]).squeeze(-1)
        else:
            indices = [list(range(num_global_block)) for _ in range(batch)]
    else:
        # Preserve the original FP32 accumulation for offloaded block keys.
        logits = torch.stack([(query[u:u + 1].float() @ keys.float().T)[0]
                              for u, keys in enumerate(block_keys)])

    if logits is not None:
        assert topk % chunk_size == 0
        remainder = logits.shape[1] % chunk_size
        chunks = logits[:, :logits.shape[1] - remainder].reshape(
            batch, -1, chunk_size).mean(dim=-1)
        if remainder:
            chunks = torch.cat((chunks, logits[:, -remainder:].mean(-1, keepdim=True)), dim=1)
        indices = chunks.topk(topk // chunk_size, dim=1).indices.sort(dim=1)[0]
        indices = indices[:, :, None] * chunk_size + torch.arange(chunk_size, device=indices.device)
        indices = indices.reshape(batch, -1).cpu().tolist()
        indices = [[i for i in row if i < logits.shape[1]] for row in indices]
    return indices, logits
