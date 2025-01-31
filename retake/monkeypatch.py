import transformers

from retake.qwen2_vl import (
    retake_Qwen2VLAttention_init,
    retake_Qwen2VLAttention_forward,
    retake_Qwen2VLSdpaAttention_forward,
    retake_Qwen2VLFlashAttention2_forward,
    retake_Qwen2VLForConditionalGeneration_prepare_longvideo_prefill,
    retake_Qwen2VLForConditionalGeneration_prepare_longvideo_decode,
    retake_Qwen2VLForConditionalGeneration_generate,
    retake_Qwen2VLForConditionalGeneration_compress_video_tokens,
    retake_Qwen2VLForConditionalGeneration_forward,
)
from retake.llava_onevision import (
    retake_Qwen2FlashAttention2_forward,
    retake_LlavaOnevisionForConditionalGeneration_get_chunk_size,
    retake_LlavaOnevisionForConditionalGeneration_segment_input_ids,
    retake_LlavaOnevisionForConditionalGeneration_compress_video_tokens,
    retake_LlavaOnevisionForConditionalGeneration_forge_input_chunks,
    retake_LlavaOnevisionForConditionalGeneration_forward,
)


def patch_qwen2vl_config(config, exp_configs):
    # Rope Scaling
    if 'scaling_factor' in exp_configs:
        config.rope_scaling.pop('type')
        config.rope_scaling['rope_type'] = 'yarn'
        config.rope_scaling['factor'] = exp_configs['scaling_factor']
        config.rope_scaling['beta_fast'] = 32.0
        config.rope_scaling['beta_slow'] = 1.0
    # ReTaKe
    config.longvideo_kwargs = exp_configs.get('longvideo_kwargs', {})
    return config


def patch_llava_onevision_config(config, exp_configs):
    # Rope Scaling
    if 'scaling_factor' in exp_configs:
        config.text_config.rope_scaling = {
            'rope_type': 'yarn',
            'factor': exp_configs['scaling_factor'],
            'beta_fast': 32.0,
            'beta_slow': 1.0,
        }
    # ReTaKe
    config.longvideo_kwargs = exp_configs.get('longvideo_kwargs', {})
    return config


def patch_qwen2vl(method):

    if method == "retake":
        print("Using ReTaKe!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.__init__ = retake_Qwen2VLAttention_init
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = retake_Qwen2VLAttention_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLSdpaAttention.forward = retake_Qwen2VLSdpaAttention_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = retake_Qwen2VLFlashAttention2_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.prepare_longvideo_prefill = retake_Qwen2VLForConditionalGeneration_prepare_longvideo_prefill
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.prepare_longvideo_decode = retake_Qwen2VLForConditionalGeneration_prepare_longvideo_decode
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.generate = retake_Qwen2VLForConditionalGeneration_generate
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.compress_video_tokens = retake_Qwen2VLForConditionalGeneration_compress_video_tokens
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = retake_Qwen2VLForConditionalGeneration_forward
    else:
        raise NotImplementedError


def patch_llava_onevision(method):

    if method == "retake":
        print("Using ReTaKe!")
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = retake_Qwen2FlashAttention2_forward
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.get_chunk_size = retake_LlavaOnevisionForConditionalGeneration_get_chunk_size
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.segment_input_ids = retake_LlavaOnevisionForConditionalGeneration_segment_input_ids
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.compress_video_tokens = retake_LlavaOnevisionForConditionalGeneration_compress_video_tokens
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.forge_input_chunks = retake_LlavaOnevisionForConditionalGeneration_forge_input_chunks
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.forward = retake_LlavaOnevisionForConditionalGeneration_forward
    else:
        raise NotImplementedError
