"""ReKV attention with per-layer ragged retrieval and device-resident compressed KV."""
from dataclasses import dataclass
import torch
from model.attention.dot_production_attention import get_multi_stage_dot_production_attention


@dataclass
class EncodeLayer:
    past: tuple
    captured: object = None


@dataclass
class RetrieveLayer:
    snapshot: object
    layer: int
    topk: int
    hits: set


def streaming_attention_forward(n_local, n_init, fattn=True, **kwargs):
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
        if retrieving:
            pk, pv, ids, _ = past_key_value.snapshot.retrieve_layer(past_key_value.layer, q, past_key_value.topk)
            past_key_value.hits.update(ids)
        elif encoding:
            pk, pv = (x.to(q.device) for x in past_key_value.past)
            past_key_value.captured = (k[0].detach().clone(), v[0].detach().clone())
        elif past_key_value is None:
            pk, pv = k[:, :, :0], v[:, :, :0]
        else:
            pk, pv = past_key_value
        k, v = torch.cat([pk, k], -2), torch.cat([pv, v], -2)
        total = k.shape[-2]
        # Match ReKV's sliding local attention plus the complementary sink stage.
        local_start = max(0, total-length-window)
        lk, lv = k[:, :, local_start:], v[:, :, local_start:]
        lq, lk = position_bias(q, lk)
        attn = Attn(lq.shape, lq.dtype, lq.device)
        attn.append(lq, lk, lv, sliding_window=window)
        if total > window:
            iq = position_bias.apply_rotary_pos_emb_one_angle(q, window)
            ik, iv = k[:, :, :n_init].contiguous(), v[:, :, :n_init].contiguous()
        else:
            iq, ik, iv = q, k[:, :, :0], v[:, :, :0]
        attn.append(iq, ik, iv, end=True,
                    sliding_window=(total-length, window), complement_sliding_window=True)
        result, _ = attn.get_result()
        result = result.transpose(1, 2).reshape(batch, length, num_heads*dim_head)
        if encoding:
            updated = past_key_value  # Device-resident capture; the caller owns canonical storage.
        elif retrieving:
            updated = (pk, pv)  # The question used for retrieval is not added to video memory.
        elif total <= n_init+window:
            updated = (k, v)
        else:
            updated = (torch.cat([k[:, :, :n_init], k[:, :, -window:]], -2),
                       torch.cat([v[:, :, :n_init], v[:, :, -window:]], -2))
        return attention_out(result), updated

    return forward
