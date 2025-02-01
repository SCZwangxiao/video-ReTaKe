import math
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.utils import logging


logger = logging.get_logger(__name__)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1, reverse=False, attention_scaling=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    if reverse: # Rotate towards the opposite direction
        q_embed = ((q * cos) - (rotate_half(q) * sin)) / attention_scaling**2
        k_embed = ((k * cos) - (rotate_half(k) * sin)) / attention_scaling**2
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
        k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None

    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, reverse=False, attention_scaling=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if reverse: # Rotate towards the opposite direction
        q_embed = ((q * cos) - (rotate_half(q) * sin))  / attention_scaling**2
        k_embed = ((k * cos) - (rotate_half(k) * sin))  / attention_scaling**2
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
        k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None

    return q_embed, k_embed


class PivotKVCache(DynamicCache):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        if hasattr(config, 'text_config'):
            # LLaVA-OneVision
            llm_config = config.text_config
        else:
            # QWen2VL
            llm_config = config
        self.hidden_size = llm_config.hidden_size
        self.num_hidden_layers = llm_config.num_hidden_layers
        self.num_heads = llm_config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = llm_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Patch longvideo kwargs
        kv_compression_kwargs = config.longvideo_kwargs['kvcache_compression_kwargs']
        self.kvcache_compression = True
        self.kv_compression_kwargs = kv_compression_kwargs
        self.compression_ratio = kv_compression_kwargs['compression_ratio']
        self.compression_method = kv_compression_kwargs['compression_method']
        self.pos_embed_reforge = kv_compression_kwargs.get('pos_embed_reforge', False)
        self.position_cache: List[torch.Tensor] = []
        self.num_evicted_tokens: List[int] = []

    def before_forward(self, **kwargs):
        pass

    def after_forward(self, **kwargs):
        pass

    def update_num_evicted_tokens(
        self,
        num_tokens: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor]:
        """
        Updates the `num_evicted_tokens` with an increment `num_tokens` at layer `layer_idx`.
        If `num_tokens` = 0, this function get the number of evicted tokens in layer `layer_idx`.

        Parameters:
            num_tokens (`int`):
                The number of evicted tokens.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.num_evicted_tokens) <= layer_idx:
            # There may be skipped layers without eviction, fill them with 0
            for _ in range(len(self.num_evicted_tokens), layer_idx):
                self.num_evicted_tokens.append(0)
            self.num_evicted_tokens.append(num_tokens)
        else:
            self.num_evicted_tokens[layer_idx] += num_tokens

        return self.num_evicted_tokens[layer_idx]

    def update_position_ids(
        self,
        position_ids: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor]:
        """
        Updates the cache with the new `position_ids` for the layer `layer_idx`.

        Parameters:
            position_ids (`torch.Tensor`):
                The new key states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.position_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.position_cache), layer_idx):
                self.position_cache.append([])
            self.position_cache.append(position_ids)
        elif len(self.position_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.position_cache[layer_idx] = position_ids
        else:
            self.position_cache[layer_idx] = torch.cat([self.position_cache[layer_idx], position_ids], dim=-1)

        return self.position_cache[layer_idx]

    def get_prev_temporal_idx(self, layer_idx: int) -> torch.LongTensor:
        if len(self.position_cache) <= layer_idx:
            return -1
        cache_layer = self.position_cache[layer_idx]
        return cache_layer[0,0,-1] if cache_layer.ndim == 3 else cache_layer[0,-1]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input
            query_states: [bsz, num_heads, q_len, d]
            key_states: [bsz, num_key_value_heads, q_len, d]
            position_ids: [3, bsz, q_len] / [bsz, q_len]
        Output
            key_states_output: for calculating self attention
            value_states_output: for calculating self attention
        """
        logger.warning_once("Enable PivotKVCache compression: length after compression %.2f" % (self.compression_ratio))

        position_ids = cache_kwargs.pop('position_ids', None)

        # 1) Hidden states for the next layer remains uncompressed in current chunked prefill iter
        key_states_output, value_states_output = super().update(key_states, value_states, layer_idx, cache_kwargs)

        if self.kvcache_compression: # when prefilling visual tokens
            query_states = cache_kwargs.pop('query_states')
            rotary_emb_fn = cache_kwargs.pop('rotary_emb')
            mrope_section = cache_kwargs.pop('mrope_section', None) # For MRope only
            bsz, num_heads, q_len, head_dim = query_states.shape
            num_key_value_heads, k_len = key_states.shape[1:3]
            assert bsz == 1

            if self.pos_embed_reforge:
                cos, sin = rotary_emb_fn(value_states, position_ids)
                if mrope_section:
                    query_states, key_states = apply_multimodal_rotary_pos_emb(
                        query_states, key_states, cos, sin, mrope_section, 
                        reverse=True, attention_scaling=rotary_emb_fn.attention_scaling
                    )
                else:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin,
                        reverse=True, attention_scaling=rotary_emb_fn.attention_scaling
                    )
            key_states_repeated = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)

            # 2) Evit KV Cache based on query_states
            keep_len = max(1, int(self.compression_ratio * q_len)) # Evict new tokens only
            attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            ).detach() # [bsz, self.num_heads, q_len, q_len(k)]
            attn_weights = attn_weights[0].sum(1) # [self.num_heads, q_len(k)]
            attn_weights = attn_weights.reshape(self.num_key_value_heads, -1, q_len).mean(1) # [num_key_value_heads, q_len(k)]
            attn_weights = attn_weights.mean(0) # [q_len(k)]

            if getattr(self, "keypatches_mask_chunk", None) is not None:
                keypatches_mask_chunk = self.keypatches_mask_chunk
                attn_weights.masked_fill_(keypatches_mask_chunk, 1.) # Select key patches first

            _, keep_indices = attn_weights.topk(keep_len)
            keep_indices = keep_indices.sort().values # [keep_len]
            keep_indices_kv = keep_indices[None,None,:,None].repeat(bsz, self.num_key_value_heads, 1, self.head_dim) # [bsz, num_key_value_heads, keep_len, head_dim]
            compressed_key_states = torch.gather(input=key_states, dim=2, index=keep_indices_kv)
            compressed_value_states = torch.gather(input=value_states, dim=2, index=keep_indices_kv) # [bsz, num_k_heads, keep_len, head_dim]

            # Calculate new postional ids
            if mrope_section:
                keep_indices_ids = keep_indices[None,None,:].repeat(3, bsz, 1) # [3, bsz, keep_len]
                compressed_position_ids = torch.gather(input=position_ids, dim=2, index=keep_indices_ids) # [3, bsz, keep_len]
            else:
                keep_indices_ids = keep_indices[None,:].repeat(bsz, 1) # [bsz, keep_len]
                compressed_position_ids = torch.gather(input=position_ids, dim=1, index=keep_indices_ids) # [bsz, keep_len]

            if self.pos_embed_reforge:
                assert bsz == 1

                min_temp_id = compressed_position_ids[0].min()
                comp_ratio = keep_len / k_len # NOTE: avoid truncating issues when calculating keep_len
                compressed_position_ids[0] = min_temp_id + ((compressed_position_ids[0] - min_temp_id) * comp_ratio).long()

                # Add new rotary embedding
                cos, sin = rotary_emb_fn(compressed_value_states, compressed_position_ids)
                if mrope_section:
                    _, compressed_key_states = apply_multimodal_rotary_pos_emb(
                        None, compressed_key_states, cos, sin, mrope_section
                    )
                else:
                    _, compressed_key_states = apply_rotary_pos_emb(
                        None, compressed_key_states, cos, sin,
                    )

            if self.pos_embed_reforge:
                _ = self.update_position_ids(compressed_position_ids, layer_idx)
            _ = self.update_num_evicted_tokens(k_len - keep_len, layer_idx)

            # 3) Update KVCache
            self.key_cache[layer_idx] = torch.cat([
                key_states_output[...,:-q_len,:], compressed_key_states
            ], dim=2)
            self.value_cache[layer_idx] = torch.cat([
                value_states_output[...,:-q_len,:], compressed_value_states
            ], dim=2)
        else: # when prefilling textual tokens or decoding / kvcache compression disabled
            if self.pos_embed_reforge:
                _ = self.update_position_ids(position_ids, layer_idx)

        return key_states_output, value_states_output


def build_kvcache(config):
    if getattr(config, "longvideo_kwargs", None) is None or not config.longvideo_kwargs.get('kvcache_compression', False):
        return DynamicCache()
    else:
        compression_method = config.longvideo_kwargs['kvcache_compression_kwargs']['compression_method']
        if compression_method.lower() == 'pivotkv':
            return PivotKVCache(config)
        else:
            raise NotImplementedError
