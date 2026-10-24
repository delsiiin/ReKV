"""ReKV attention with per-layer ragged retrieval and device-resident compressed KV."""
from dataclasses import dataclass
import importlib.util
import torch
from model.attention.dot_production_attention import get_multi_stage_dot_production_attention
from .rotary import rotate_at
from .packed_attention import PackedRotaryCache


@dataclass
class RotaryCache:
    """Post-RoPE keys with their actual rotation positions; values are unrotated."""
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor

    def __iter__(self):
        return iter((self.key, self.value))

    def __getitem__(self, index):
        return (self.key, self.value)[index]


@dataclass
class EncodeLayer:
    past: RotaryCache
    captured: object = None
    score_key: object = None
    position_start: int = 0
    capture_scores: bool = True


@dataclass
class RetrieveLayer:
    snapshot: object
    layer: int
    topk: int
    hits: set
    profiler: object = None


def streaming_attention_forward(n_local, n_init, fattn=True, packed_attention=False, **kwargs):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)

    def forward(self, query, key_value, position_bias, use_cache, past_key_value,
                project_q, project_k, project_v, attention_out, dim_head, num_heads, num_heads_kv):
        window = getattr(self, '_streaming_n_local', n_local)
        batch, length, _ = query.shape
        q = project_q(query).view(batch, length, num_heads, dim_head).transpose(1, 2).contiguous()
        k = project_k(key_value).view(batch, -1, num_heads_kv, dim_head).transpose(1, 2).contiguous()
        v = project_v(key_value).view(batch, -1, num_heads_kv, dim_head).transpose(1, 2).contiguous()
        retrieving = isinstance(past_key_value, RetrieveLayer)
        encoding = isinstance(past_key_value, EncodeLayer)
        # QA normally fits in its deliberately enlarged window. Keep compressed
        # payloads through prefill and decode; fall back to the established
        # sliding/sink implementation if the window is exceeded.
        packed_past = past_key_value if isinstance(past_key_value, PackedRotaryCache) else None
        if (retrieving and packed_attention and q.is_cuda and batch == 1
                and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and torch.version.cuda is not None and dim_head <= 256
                and 0 < position_bias.inv_freq.numel()*2 <= dim_head
                and importlib.util.find_spec('triton') is not None
                and not torch.is_grad_enabled()):
            packed_past, ids = past_key_value.snapshot.retrieve_layer(
                past_key_value.layer, q, past_key_value.topk, past_key_value.profiler,
                position_bias=position_bias, packed=True)
            past_key_value.hits.update(ids)
        if packed_past is not None:
            total = packed_past.length + k.shape[-2]
            if total <= window and not torch.is_grad_enabled():
                positions = torch.arange(packed_past.length, total, device=q.device)
                current = packed_past.append(rotate_at(k, positions, position_bias), v, position_bias)
                result = current.attend(q, position_bias)
                result = result.transpose(1, 2).reshape(batch, length, num_heads*dim_head)
                return attention_out(result), packed_past if retrieving else current
            pk, pv, positions = packed_past.materialize()
            if not retrieving:
                past_key_value = RotaryCache(pk, pv, positions)
        if retrieving:
            if packed_past is None:
                pk, pv, ids, positions = past_key_value.snapshot.retrieve_layer(
                    past_key_value.layer, q, past_key_value.topk, past_key_value.profiler,
                    position_bias=position_bias, with_rope_positions=True)
                past_key_value.hits.update(ids)
        elif encoding:
            if not isinstance(past_key_value.past, RotaryCache):
                raise ValueError('Streaming encoding requires post-RoPE keys with explicit positions')
            pk, pv = past_key_value.past
            positions = past_key_value.past.positions
            if past_key_value.capture_scores:
                past_key_value.score_key = k[0].detach().clone()
        elif past_key_value is None:
            pk, pv = k[:, :, :0], v[:, :, :0]
            positions = torch.empty(0, dtype=torch.long, device=q.device)
        else:
            if not isinstance(past_key_value, RotaryCache):
                raise ValueError('Streaming decoding requires post-RoPE keys with explicit positions')
            pk, pv = past_key_value
            positions = past_key_value.positions
        total = pk.shape[-2] + k.shape[-2]
        local_start = max(0, total-length-window)
        new_positions = torch.arange(pk.shape[-2]-local_start, total-local_start, device=q.device)
        # Capture exactly the rotated K used by the local attention stage.
        k = rotate_at(k, new_positions, position_bias)
        if encoding:
            past_key_value.position_start = pk.shape[-2]-local_start
            past_key_value.captured = (k[0].detach().clone(), v[0].detach().clone())
        cached_positions = torch.cat((positions, new_positions))
        k, v = torch.cat([pk, k], -2), torch.cat([pv, v], -2)
        # Match ReKV's sliding local attention plus the complementary sink stage.
        lk, lv = k[:, :, local_start:], v[:, :, local_start:]
        target_positions = torch.arange(lk.shape[-2], device=q.device)
        # R(target - stored) R(stored) K = R(target) K. Never apply the
        # full target rotation again to an already rotated cached key.
        lk = rotate_at(lk, target_positions-cached_positions[local_start:], position_bias)
        lq = rotate_at(q, target_positions[-length:], position_bias)
        # The existing flash implementation only supports these head sizes.
        # Keep the packed backend's larger/non-power-of-two heads usable when
        # crossing back to the dense sliding-window path.
        attn_type = Attn if dim_head in (16, 32, 64, 128) else get_multi_stage_dot_production_attention(False)[0]
        attn = attn_type(lq.shape, lq.dtype, lq.device)
        attn.append(lq, lk, lv, sliding_window=window)
        if total > window:
            iq = rotate_at(q, torch.full((length,), window-1, device=q.device), position_bias)
            # ReKV's complementary sink stage uses zero-position keys.
            ik = rotate_at(k[:, :, :n_init], -cached_positions[:n_init], position_bias)
            iv = v[:, :, :n_init].contiguous()
        else:
            iq, ik, iv = q, k[:, :, :0], v[:, :, :0]
        attn.append(iq, ik, iv, end=True,
                    sliding_window=(total-length, window), complement_sliding_window=True)
        result, _ = attn.get_result()
        result = result.transpose(1, 2).reshape(batch, length, num_heads*dim_head)
        if encoding:
            updated = past_key_value  # Device-resident capture; the caller owns canonical storage.
        elif retrieving:
            updated = RotaryCache(pk, pv, positions)  # Retrieval query is not added to video memory.
        elif total <= n_init+window:
            updated = RotaryCache(k, v, cached_positions)
        else:
            updated = RotaryCache(torch.cat([k[:, :, :n_init], k[:, :, -window:]], -2),
                                  torch.cat([v[:, :, :n_init], v[:, :, -window:]], -2),
                                  torch.cat([cached_positions[:n_init], cached_positions[-window:]]))
        return attention_out(result), updated

    return forward
