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
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)
from transformers import Qwen2VLConfig
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionCausalLMOutputWithPast,
    image_size_to_num_patches,
)

from transformers.models.qwen2.modeling_qwen2 import (
    repeat_kv,
    apply_rotary_pos_emb,
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


def retake_Qwen2FlashAttention2_forward(
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

    # Update position_ids if positional embeddings are reforged
    if past_key_value is not None and getattr(past_key_value, "pos_embed_reforge", False):
        # This code reforge the `position_ids` of current chunk, 
        # the `position_ids` of previous chunks are reforged in KVCache.update()
        prev_tempo_idx = past_key_value.get_prev_temporal_idx(self.layer_idx)
        cur_tempo_idx = position_ids[0,0]
        if prev_tempo_idx + 1 != cur_tempo_idx:
            assert bsz == 1
            # print("Warning! Discontinuous positional ids %d (prev) + 1 != %d (cur) at layer %d. Fixed!" % (prev_tempo_idx,  cur_tempo_idx, self.layer_idx))
            # NOTE: clone `position_ids` to avoid influence of in-place ops in different layers
            position_ids = position_ids.clone()
            position_ids[0,:] += prev_tempo_idx + 1 - cur_tempo_idx
        position_embeddings = None # `position_embeddings` need to be re-calculated

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        kv_seq_len = key_states.shape[-2] + cache_position[0]
        if (
            getattr(self.config, "sliding_window", None) is not None
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
        cache_kwargs.update({"query_states": query_states, "position_ids": position_ids, "rotary_emb": self.rotary_emb})
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
        position_ids=position_ids,
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


def retake_LlavaOnevisionForConditionalGeneration_get_chunk_size(
    self, 
    config, 
    pixel_values_videos
) -> int:
    # Calculate the number of tokens in each prefill chunk
    chunk_frames = (
        config.longvideo_kwargs.get('chunked_prefill_frames', None) if getattr(config, 'longvideo_kwargs', None) 
        else None
    )
    if chunk_frames is None:
        chunk_prefill_size = None
    else:
        T, _, H, W = pixel_values_videos[0].shape
        H = math.ceil(H // self.config.vision_config.patch_size / self.pool_stride)
        W = math.ceil(W // self.config.vision_config.patch_size / self.pool_stride)
        chunk_prefill_size = min(chunk_frames, T) * H * W
    return chunk_prefill_size


def retake_LlavaOnevisionForConditionalGeneration_segment_input_ids(
    self,
    input_ids
):
    """Split video and text segments in the input_ids
    return: list[(s, e, type)], indices of [s, e) are of `type`.
    """
    videomask = (input_ids[0] == self.config.video_token_index)
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


def retake_LlavaOnevisionForConditionalGeneration_compress_video_tokens(
    self, 
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor = None,
    selected_video_feature: torch.Tensor = None,
    position_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
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

    grid_t, grid_hw = selected_video_feature.shape[:2]
    tgt_grid_t = grid_t
    if visual_compression:
        assert labels is None
        assert input_ids.shape[0] == 1, "Currently, only inference are supported"
        video_token_indices = torch.where(input_ids[0] == self.config.video_token_index)[0]
        s_index, e_index = video_token_indices[0], video_token_indices[-1]
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
        grid_hw_after_pool = math.ceil(height / self.pool_stride) * math.ceil(width / self.pool_stride)
        ori_seq_len = input_ids.shape[1]
        tgt_grid_t = max(1, round(compression_ratio * grid_t))

        # Compress
        compressed_memory_bank = selected_video_feature.reshape(1, grid_t, grid_hw, -1)
        if compression_method == "MA-LLM":
            compression_size = torch.ones_like(compressed_memory_bank[:,:,:,0])
            while compressed_memory_bank.shape[1] > tgt_grid_t:
                compressed_memory_bank, compression_size = memory_bank_compress_MALLM(compressed_memory_bank, compression_size, sync=patch_sync)
            keypatches_mask = None
        elif compression_method == "MA-LLM-hard":
            while compressed_memory_bank.shape[1] > tgt_grid_t:
                compressed_memory_bank = memory_bank_compress_MALLM_hard(compressed_memory_bank, sync=patch_sync)
            keypatches_mask = None
        elif compression_method == "Keyframe":
            compressed_memory_bank, keypatches_mask = memory_bank_compress_keyframe(compressed_memory_bank, tgt_grid_t, 3, sync=patch_sync)
            keypatches_mask = keypatches_mask if return_keyframe_mask else None
        else:
            raise NotImplementedError
        selected_video_feature = compressed_memory_bank[0]
        mem_len_after = tgt_grid_t * grid_hw_after_pool

        # Reforge the input
        input_ids = torch.cat([
            input_ids[:, :s_index],
            input_ids[:, s_index:e_index+1][:,:mem_len_after],
            input_ids[:, e_index+1:]
            ],
            dim=1)
        num_token_diff = ori_seq_len - input_ids.shape[1]
        if attention_mask is not None:
            attention_mask = attention_mask[:, num_token_diff:]
        if position_ids is not None:
            position_ids = position_ids[:,:-num_token_diff]
        if cache_position is not None:
            cache_position = cache_position[:-num_token_diff]
    else:
        keypatches_mask = None

    return input_ids, attention_mask, selected_video_feature, position_ids, cache_position, tgt_grid_t, keypatches_mask


def retake_LlavaOnevisionForConditionalGeneration_forge_input_chunks(
    self, 
    ss, 
    ee, 
    modality_segments, 
    position_ids, 
    cache_position, 
    attention_mask, 
    past_key_values, 
    inputs_embeds
):
    position_ids_chunk = position_ids[:,ss:ee]
    cache_position_chunk = cache_position[:ee]
    attention_mask_chunk = attention_mask[:,:ee] # NOTE: specially from 0 to ee
    inputs_embeds_chunk = inputs_embeds[:,ss:ee]
    prompt_length = None

    if getattr(self.config, 'longvideo_kwargs', None) and self.config.longvideo_kwargs.get('kvcache_compression', False):
        compression_kwargs = self.config.longvideo_kwargs['kvcache_compression_kwargs']
        if compression_kwargs.get('prompt_guided_compression', False) and compression_kwargs.get('compression_ratio', 1) < 1.0:
            # Prompt guided KV cache compression
            s_p, e_p, t_p = modality_segments[-1]
            assert t_p == 'text'
            pos_offset = position_ids[0,s_p] - position_ids_chunk[0,-1] - 1 # (bs, seq_len)
            # print('forge_input_chunks() temporal position_ids of prompts', position_ids[0,s_p:e_p])
            position_ids_chunk = torch.cat([position_ids_chunk, position_ids[:,s_p:e_p] - pos_offset], dim=1)
            cache_position_chunk = torch.cat([cache_position_chunk, cache_position[s_p:e_p] - pos_offset], dim=0)
            attention_mask_chunk = torch.cat([attention_mask_chunk, attention_mask[:,s_p:e_p]], dim=1)
            inputs_embeds_chunk = torch.cat([inputs_embeds_chunk, inputs_embeds[:,s_p:e_p]], dim=1)
            prompt_length = e_p - s_p

    return position_ids_chunk, cache_position_chunk, attention_mask_chunk, inputs_embeds_chunk, prompt_length


def retake_LlavaOnevisionForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    image_sizes: Optional[torch.LongTensor] = None,
    pixel_values_videos: torch.FloatTensor = None,
    image_sizes_videos: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    vision_aspect_ratio: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
) -> Union[Tuple, LlavaOnevisionCausalLMOutputWithPast]:
    assert input_ids.shape[0] == 1, "Batch inference of long video is not supported yet!"

    self.pool_stride = 2
    if input_ids.shape[1] > 1: # Prefill
        is_prefill = True
        # Calculate chunk size based on inputs
        chunk_size = self.get_chunk_size(self.config, pixel_values_videos)
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
            # print("modality_segments", modality_segments)
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
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )
    vision_aspect_ratio = (
        vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if (pixel_values is not None or pixel_values_videos is not None) and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values/pixel_values_videos and inputs_embeds at the same time, and must specify either one"
        )

    # Images are processed with Anyres
    if pixel_values is not None:
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        # unpad extra patches and concatenate them
        if pixel_values.dim() == 5:
            _pixel_values_list = [
                pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            # [batch_size*frames*num_patches, num_channels, height, width] where frames=1 for images
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)

        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, feature_lens = self.pack_image_features(
            image_features,
            image_sizes,
            image_newline=self.image_newline,
            vision_aspect_ratio=vision_aspect_ratio,
        )

    # Video are simply embedded and further pooled to decrease seq len
    if pixel_values_videos is not None:
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        # NOTE: Split video into chunks to avoid OOM due to large activations during visual forward
        # chunk_size can be up to 128 or higher if you have flash attention
        frame_chunk_size = getattr(self.config, 'longvideo_kwargs', {}).get('frame_chunk_size', 1000000000)
        if batch_size * frames < frame_chunk_size:
            video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
            selected_video_feature = video_features.hidden_states[vision_feature_layer]
        else:
            selected_video_feature = []
            for i in range(0, batch_size*frames, frame_chunk_size):
                pixel_values_videos_chunk = pixel_values_videos[i:i+frame_chunk_size]
                selected_video_feature.append(
                    self.vision_tower(pixel_values_videos_chunk, 
                                      output_hidden_states=True
                    ).hidden_states[vision_feature_layer]
                )
            selected_video_feature = torch.cat(selected_video_feature)

        # Compression video tokens
        input_ids, attention_mask, selected_video_feature, position_ids, cache_position, frames, keypatches_mask = self.compress_video_tokens(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            selected_video_feature=selected_video_feature, 
            position_ids=position_ids,
            cache_position=cache_position,
            labels=labels,
        )

        if vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)
        image_newline = self.image_newline[None, None, :].repeat(batch_size, 1, 1).to(video_features.device)
        video_features = torch.cat((video_features, image_newline), dim=1)
        video_features = video_features.flatten(0, 1)

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # Concatenate vision and language features
    if pixel_values is not None:
        special_image_mask = (
            (input_ids == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    if pixel_values_videos is not None:
        special_video_mask = (
            (input_ids == self.config.video_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

    if is_prefill and chunk_size is not None: # Chunked prefill stage
        assert past_key_values is not None
        kvcache_compression = getattr(past_key_values, 'kvcache_compression', False)
        for seg_id, (s, e, dtype) in enumerate(modality_segments):
            if dtype == 'text': # Prefill text without kvcache_compression
                past_key_values.kvcache_compression = False
                outputs = self.language_model(
                    attention_mask=attention_mask[:,:e],
                    position_ids=position_ids[:,s:e],
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds[:,s:e],
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position[:e],
                    num_logits_to_keep=num_logits_to_keep,
                )
                past_key_values = outputs['past_key_values']
            elif dtype == 'video': # Prefill video may with kvcache_compression
                num_chunks = math.ceil((e - s) / chunk_size)
                past_key_values.kvcache_compression = kvcache_compression
                for idx in tqdm(range(num_chunks), total=num_chunks, desc='Prefilling chunk', disable=not DEBUG_MODE):
                    ss = s + idx * chunk_size
                    ee = min(s + (idx + 1) * chunk_size, e)
                    if keypatches_mask is not None:
                        past_key_values.keypatches_mask_chunk = keypatches_mask[ss:ee]
                    position_ids_chunk, cache_position_chunk, attention_mask_chunk, inputs_embeds_chunk, prompt_length = self.forge_input_chunks(
                        ss, ee, modality_segments, position_ids, cache_position, attention_mask, past_key_values, inputs_embeds
                    )
                    if hasattr(past_key_values, 'before_forward'):
                        past_key_values.before_forward(prompt_length=prompt_length)
                    outputs = self.language_model(
                        attention_mask=attention_mask_chunk,
                        position_ids=position_ids_chunk,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds_chunk,
                        use_cache=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position_chunk,
                        num_logits_to_keep=num_logits_to_keep,
                    )
                    past_key_values = outputs['past_key_values']
                    if hasattr(past_key_values, 'after_forward'):
                        past_key_values.after_forward()
                past_key_values.keypatches_mask = None
                past_key_values.kvcache_compression = False # Turned off for decoding
            else:
                raise ValueError
    else: # Decode / Standard prefill stage
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaOnevisionCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
        video_hidden_states=video_features if pixel_values_videos is not None else None,
    )