import math
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
import numpy as np

from transformers.cache_utils import Cache
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)
from transformers import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLSdpaAttention,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    repeat_kv,
    apply_multimodal_rotary_pos_emb,
)

from .visual_compression import *
from .longvideo_cache import *

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func

    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None

DEBUG_MODE = False

logger = logging.get_logger(__name__)


def retake_Qwen2VLAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += cache_position[0] + 1

    # Update position_ids if positional embeddings are reforged
    if past_key_value is not None and getattr(past_key_value, "pos_embed_reforge", False):
        prev_tempo_idx = past_key_value.get_prev_temporal_idx(self.layer_idx)
        if prev_tempo_idx + 1 != position_ids[0,0,0]:
            assert bsz == 1
            # print("Discontinuous positional ids %d + 1 != %d at layer %d" % (prev_tempo_idx,  position_ids[0,0,0], self.layer_idx))
            position_ids[0,0,:] += prev_tempo_idx + 1 - position_ids[0,0,0]

    # NOTE: Compute position_ids internally to support positional id reforge from KV compression
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        # Specific to KVCache compression methods
        cache_kwargs.update({"query_states": query_states, "position_ids": position_ids, 
                             "rotary_emb": self.rotary_emb, "mrope_section": self.rope_scaling["mrope_section"]})
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Fix precision issues in Qwen2-VL float16 inference
    # Replace inf values with zeros in attention weights to prevent NaN propagation
    if query_states.dtype == torch.float16:
        attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def retake_Qwen2VLSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2VLModel is using Qwen2VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super(Qwen2VLSdpaAttention, self).forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Update position_ids if positional embeddings are reforged
    if past_key_value is not None and getattr(past_key_value, "pos_embed_reforge", False):
        prev_tempo_idx = past_key_value.get_prev_temporal_idx(self.layer_idx)
        if prev_tempo_idx + 1 != position_ids[0,0,0]:
            assert bsz == 1
            # print("Discontinuous positional ids %d + 1 != %d at layer %d" % (prev_tempo_idx,  position_ids[0,0,0], self.layer_idx))
            position_ids[0,0,:] += prev_tempo_idx + 1 - position_ids[0,0,0]

    # NOTE: Compute position_ids internally to support positional id reforge from KV compression
    cos, sin = self.rotary_emb(value_states, position_ids)

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        # Specific to KVCache compression methods
        cache_kwargs.update({"query_states": query_states, "position_ids": position_ids, 
                             "rotary_emb": self.rotary_emb, "mrope_section": self.rope_scaling["mrope_section"]})
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def retake_Qwen2VLFlashAttention2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Update position_ids if positional embeddings are reforged
    if past_key_value is not None and getattr(past_key_value, "pos_embed_reforge", False):
        prev_tempo_idx = past_key_value.get_prev_temporal_idx(self.layer_idx)
        if prev_tempo_idx + 1 != position_ids[0,0,0]:
            assert bsz == 1
            # print("Discontinuous positional ids %d + 1 != %d at layer %d" % (prev_tempo_idx,  position_ids[0,0,0], self.layer_idx))
            position_ids[0,0,:] += prev_tempo_idx + 1 - position_ids[0,0,0]

    # NOTE: Compute position_ids internally to support positional id reforge from KV compression
    cos, sin = self.rotary_emb(value_states, position_ids)

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        # Specific to KVCache compression methods
        cache_kwargs.update({"query_states": query_states, "position_ids": position_ids, 
                             "rotary_emb": self.rotary_emb, "mrope_section": self.rope_scaling["mrope_section"]})
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def retake_Qwen2VLForConditionalGeneration_compress_video_tokens(
    self, 
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor = None,
    video_embeds: torch.Tensor = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
    # Parse long video compression configs
    visual_compression = False
    if getattr(self.config, "longvideo_kwargs", None) is not None:
        visual_compression = self.config.longvideo_kwargs.get("visual_compression", False)
        if visual_compression:
            compression_kwargs = self.config.longvideo_kwargs['visual_compression_kwargs']
            compression_ratio = compression_kwargs.get("compression_ratio")
            compression_method = compression_kwargs.get("compression_method")
            patch_sync = compression_kwargs.get("patch_sync")
            return_keyframe_mask = compression_kwargs.get("return_keyframe_mask")

    if visual_compression:
        assert labels is None
        assert video_grid_thw.shape[0] <= 1, "Currently, interleaved videos are not supported"
        assert input_ids.shape[0] == 1, "Currently, only inference are supported"
        video_token_indices = torch.where(input_ids[0] == self.config.video_token_id)[0]
        s_index, e_index = video_token_indices[0], video_token_indices[-1]
        grid_t = video_grid_thw[0][0]
        grid_hw = video_embeds.shape[0] // grid_t
        ori_seq_len = input_ids.shape[1]

        tgt_mem_len = max(1, round(compression_ratio * grid_t.item()))
        num_frame_diff = grid_t - tgt_mem_len

        # Compress
        compressed_memory_bank = video_embeds.reshape(1, grid_t, grid_hw, -1)
        if compression_method == "MA-LLM":
            compression_size = torch.ones_like(compressed_memory_bank[:,:,:,0])
            while compressed_memory_bank.shape[1] > tgt_mem_len:
                compressed_memory_bank, compression_size = memory_bank_compress_MALLM(compressed_memory_bank, compression_size, sync=patch_sync)
            keypatches_mask = None
        elif compression_method == "MA-LLM-hard":
            while compressed_memory_bank.shape[1] > tgt_mem_len:
                compressed_memory_bank = memory_bank_compress_MALLM_hard(compressed_memory_bank, sync=patch_sync)
            keypatches_mask = None
        elif compression_method == "Keyframe":
            compressed_memory_bank, keypatches_mask = memory_bank_compress_keyframe(compressed_memory_bank, tgt_mem_len, 3, sync=patch_sync)
            keypatches_mask = keypatches_mask if return_keyframe_mask else None
        else:
            raise NotImplementedError
        video_embeds = compressed_memory_bank.flatten(1, 2)[0]
        tgt_seq_len = video_embeds.shape[0]

        # Reforge the input
        input_ids = torch.cat([
            input_ids[:, :s_index],
            input_ids[:, s_index:e_index+1][:,:tgt_seq_len],
            input_ids[:, e_index+1:]
            ],
            dim=1)
        num_token_diff = ori_seq_len - input_ids.shape[1]
        if num_token_diff and attention_mask is not None:
            attention_mask = attention_mask[:, :-num_token_diff]
        if num_token_diff and cache_position is not None:
            cache_position = cache_position[:-num_token_diff]
        if position_ids is not None:
            position_ids = torch.cat([
                position_ids[..., :s_index],
                position_ids[..., s_index:e_index+1][...,:tgt_seq_len],
                position_ids[..., e_index+1:]
                ],
            dim=2)
            position_ids[:, :, s_index+tgt_seq_len:] -= num_frame_diff
    else:
        keypatches_mask = None

    return input_ids, attention_mask, video_embeds, cache_position, position_ids, labels, keypatches_mask

def retake_Qwen2VLForConditionalGeneration_segment_input_ids(self, input_ids):
    """Split video and text segments in the input_ids
    return: list[(s, e, type)], indices of [s, e) are of `type`.
    """
    videomask = (input_ids[0] == self.config.video_token_id)
    # Find the difference between consecutive elements
    diff = torch.diff(videomask.long())
    diff_pos_indices = (torch.where(diff == 1)[0] + 1).cpu().numpy()
    diff_neg_indices = (torch.where(diff == -1)[0] + 1).cpu().numpy()

    # True mask
    start_indices_true = diff_pos_indices
    end_indices_true = diff_neg_indices
    if videomask[0] == True: # segment starts at the beginning
        start_indices_true = np.insert(start_indices_true, 0, 0)
    if videomask[-1] == True: # segment ends at the beginning
        end_indices_true = np.append(end_indices_true, len(videomask))

    # False mask
    start_indices_flase = diff_neg_indices
    end_indices_flase = diff_pos_indices
    if videomask[0] == False:
        start_indices_flase = np.insert(start_indices_flase, 0, 0)
    if videomask[-1] == False:
        end_indices_flase = np.append(end_indices_flase, len(videomask))

    segments = (
        list(zip(start_indices_true, end_indices_true, ['video']*len(end_indices_true))) + 
        list(zip(start_indices_flase, end_indices_flase, ['text']*len(end_indices_flase)))
    )
    segments = sorted(segments, key=lambda x: x[0])
    return segments

def retake_Qwen2VLForConditionalGeneration_get_chunk_size(self, config, video_grid_thw) -> int:
    # Calculate the number of tokens in each prefill chunk
    chunk_frames = (
        config.longvideo_kwargs.get('chunked_prefill_frames', None) if getattr(config, 'longvideo_kwargs', None) 
        else None
    )
    if chunk_frames is None:
        chunk_prefill_size = None
    else:
        T, H, W = video_grid_thw[0]
        t_factor = config.vision_config.spatial_merge_size**2 * config.vision_config.temporal_patch_size
        chunk_prefill_size = min(chunk_frames, T) * H * W // t_factor
        chunk_prefill_size = int(chunk_prefill_size.item())
        # Avoid machine error in ceil() when calculating `num_chunks`.
    return chunk_prefill_size

def retake_Qwen2VLForConditionalGeneration_forge_input_chunks(self, ss, ee, modality_segments, cache_position, position_ids, attention_mask, past_key_values, inputs_embeds):
    cache_position_chunk = cache_position[:ee]
    position_ids_chunk = position_ids[:,:,ss:ee]
    attention_mask_chunk = attention_mask[:,:ee] # NOTE: specially from 0 to ee
    inputs_embeds_chunk = inputs_embeds[:,ss:ee]
    prompt_length = None

    if getattr(self.config, 'longvideo_kwargs', None) and self.config.longvideo_kwargs.get('kvcache_compression', False):
        compression_kwargs = self.config.longvideo_kwargs['kvcache_compression_kwargs']
        if compression_kwargs.get('prompt_guided_compression', False) and compression_kwargs.get('compression_ratio', 1) < 1.0:
            # Prompt guided KV cache compression
            s_p, e_p, t_p = modality_segments[-1]
            # NOTE: make sure to include end-of-vision symbol like '<|vision_end|>'(151653)
            if isinstance(self, Qwen2VLForConditionalGeneration):
                # QWen2VL automatically satisfies, since only '<|vision_pad|>'(151656) is recognized as vision.
                pass
            else:
                raise NotImplementedError
            assert t_p == 'text'
            cache_position_chunk = cache_position[:ee+prompt_length]
            pos_offset = position_ids[0,0,s_p] - position_ids_chunk[0,0,-1] - 1 # (3, bs, seq_len)
            position_ids_chunk = torch.cat([position_ids_chunk, position_ids[:,:,s_p:e_p] - pos_offset], dim=2)
            attention_mask_chunk = torch.cat([attention_mask_chunk, attention_mask[:,s_p:e_p]], dim=1)
            inputs_embeds_chunk = torch.cat([inputs_embeds_chunk, inputs_embeds[:,s_p:e_p]], dim=1)
            prompt_length = e_p - s_p

    return cache_position_chunk, position_ids_chunk, attention_mask_chunk, inputs_embeds_chunk, prompt_length


def retake_Qwen2VLForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    assert input_ids.shape[0] == 1, "Batch inference of long video is not supported yet!"

    if (cache_position is not None and cache_position[0] == 0): # Prefill
        is_prefill = True
        # Calculate chunk size based on inputs
        chunk_size = self.get_chunk_size(self.config, video_grid_thw)
        # Configuring KV Cache compression kwargs
        if getattr(self.config, 'longvideo_kwargs', None) and self.config.longvideo_kwargs.get('kvcache_compression', False):
            compression_kwargs = self.config.longvideo_kwargs['kvcache_compression_kwargs']
            if compression_kwargs.get('dynamic_compression_ratio', False):
                # Dynamic compression ratio
                input_length = input_ids.shape[1]
                max_input_length = compression_kwargs['max_input_length']
                if input_length <= max_input_length:
                    compression_kwargs['compression_ratio'] = 1
                else:
                    compression_kwargs['compression_ratio'] = max_input_length / input_length
        if chunk_size is not None:
            modality_segments = self.segment_input_ids(input_ids)
            past_key_values = build_kvcache(self.config)
            use_cache = True
    else:
        is_prefill = False
        chunk_size = None

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length = input_ids.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    if inputs_embeds is None:
        # Extract visual features
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            grid_t, grid_h, grid_w = video_grid_thw[0]
            # NOTE: Split video into chunks to avoid OOM due to large activations during visual forward
            # chunk_size can be up to 128 or higher if you have flash attention
            frame_chunk_size = getattr(self.config, 'longvideo_kwargs', {}).get('frame_chunk_size', 1000000000)
            if grid_t < frame_chunk_size:
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            else:
                d = pixel_values_videos.shape[-1]
                pixel_values_videos = pixel_values_videos.reshape(grid_t, grid_h*grid_w, d)
                video_embeds = []
                for i in range(0, grid_t, frame_chunk_size):
                    pixel_values_videos_chunk = pixel_values_videos[i:i+frame_chunk_size]
                    grid_t_chunk = pixel_values_videos_chunk.shape[0]
                    video_grid_thw_chunk = video_grid_thw.clone()
                    video_grid_thw_chunk[0,0] = grid_t_chunk
                    video_embeds.append(
                        self.visual(pixel_values_videos_chunk.reshape(-1, d), grid_thw=video_grid_thw_chunk)
                    )
                video_embeds = torch.cat(video_embeds)
            # Compression video tokens
            input_ids, attention_mask, video_embeds, cache_position, position_ids, labels, keypatches_mask = self.compress_video_tokens(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                video_embeds=video_embeds, 
                cache_position=cache_position,
                position_ids=position_ids,
                labels=labels,
                video_grid_thw=video_grid_thw
            )

        # Concat visual and textual features
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            if keypatches_mask is not None:
                keypatches_mask = torch.zeros_like(input_ids).bool().masked_scatter(video_mask[:,:,0], keypatches_mask)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
        if position_ids is not None:
            position_ids = position_ids.to(inputs_embeds.device)

    if is_prefill and chunk_size is not None: # Chunked prefill stage
        assert past_key_values is not None
        kvcache_compression = getattr(past_key_values, 'kvcache_compression', False)
        for seg_id, (s, e, dtype) in enumerate(modality_segments):
            if dtype == 'text': # Prefill text without kvcache_compression
                past_key_values.kvcache_compression = False
                outputs = self.model(
                    input_ids=None,
                    position_ids=position_ids[:,:,s:e],
                    attention_mask=attention_mask[:,:e],
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds[:,s:e],
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position[:e],
                )
                past_key_values = outputs['past_key_values']
            elif dtype == 'video': # Prefill video may with kvcache_compression
                num_chunks = math.ceil((e - s) / chunk_size)
                past_key_values.kvcache_compression = kvcache_compression
                for idx in tqdm(range(num_chunks), total=num_chunks, desc='Prefilling chunk', disable=not DEBUG_MODE):
                    ss = s + idx * chunk_size
                    ee = min(s + (idx + 1) * chunk_size, e)
                    if keypatches_mask is not None:
                        past_key_values.keypatches_mask_chunk = keypatches_mask[0,ss:ee]
                    cache_position_chunk, position_ids_chunk, attention_mask_chunk, inputs_embeds_chunk, prompt_length = self.forge_input_chunks(
                        ss, ee, modality_segments, cache_position, position_ids, attention_mask, past_key_values, inputs_embeds
                    )
                    if hasattr(past_key_values, 'before_forward'):
                        past_key_values.before_forward(prompt_length=prompt_length)
                    outputs = self.model(
                        input_ids=None,
                        position_ids=position_ids_chunk,
                        attention_mask=attention_mask_chunk,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds_chunk,
                        use_cache=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position_chunk,
                    )
                    past_key_values = outputs['past_key_values']
                    if hasattr(past_key_values, 'after_forward'):
                        past_key_values.after_forward()
                past_key_values.keypatches_mask_chunk = None
                past_key_values.kvcache_compression = False # Turned off for decoding
            else:
                raise ValueError
    else: # Decode / Standard prefill stage
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )