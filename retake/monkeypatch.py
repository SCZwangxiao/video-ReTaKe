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


def patch_qwen2vl_config(config, exp_configs):
    # Rope Scaling
    if 'scaling_factor' in exp_configs:
        config.rope_scaling.pop('type')
        config.rope_scaling['rope_type'] = 'yarn'
        config.rope_scaling['factor'] = exp_configs['scaling_factor']
        config.rope_scaling['beta_fast'] = 32.0
        config.rope_scaling['beta_slow'] = 1.0
    # ReTaKe
    config.longvideo_kwargs = exp_configs.get('longvideo_kwargs', None)
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

