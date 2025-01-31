<h1 align="center"> <a href=https://arxiv.org/abs/2412.20504>ReTaKe: Reducing Temporal and Knowledge Redundancy for Long Video Understanding</a></h2>

## Contributions

- The training-free RETAKE is the first to jointly model temporal and knowledge redundancy for long video understanding, which enables 4x longer video sequence with less than 1% performance drop.

- We propose a novel keyframe selection method DPSelect to reduce low-level temporal redundancy, and a novel KV cache compression method PivotKV to reduce high-level knowledge redundancy in long videos.

<p align="center">
<img src="misc/overview.png" width=50%/>
</p>

## Prepare environment
```bash
# For GPU users
conda env create -f environment.yaml

# For NPU users
# conda env create -f environment_npu.yaml

apt-get install ffmpeg # NOTE: Quick demo does not require ffmpeg
```

## Quick demo
- Step 1:
Change `hf_qwen2vl7b_path` in `./demo.py` into your path to `Qwen2-VL-7B-Instruct`. Note that for NPU users, you need also change `config_path` into `'configs/retake_demo_npu.yaml'`.

- Step 2 (optional for LLaVA-Video):
```bash
# Convert LLaVA-Video model into huggingface format
# `Qwen2-7B-Instruct`, `siglip-so400m-patch14-384`, and `LLaVAVideoQwen2_7B` are downloadable models from huggingface.
python scripts/utils/convert_llava_video_weights_to_hf.py --text_model_id /path_to/Qwen2-7B-Instruct --vision_model_id /path_to/siglip-so400m-patch14-384 --output_hub_path /path_to/llava-video-qwen2-7b-hf --old_state_dict_id /path_to/LLaVAVideoQwen2_7B
```

- Step 3:
Run demo
```bash
python demo.py
```

## Reproduce ReTaKe
- Prepare the datasets following the docs.
  - Prepare [VideoMME](docs/prepare_videomme.md)
  - Prepare [MLVU](docs/prepare_mlvu.md)
  - Prepare [LVBench](docs/prepare_lvbench.md)
- Run script
```bash
bash scripts/infer_eval_retake.sh ${YOUR_PATH_TO_Qwen2-VL-7B-Instruct} configs/qwen2_vl/retake_qwen2-vl_videomme.yaml 8
bash scripts/infer_eval_retake.sh ${YOUR_PATH_TO_Qwen2-VL-7B-Instruct} configs/qwen2_vl/retake_qwen2-vl_mlvu.yaml 8
bash scripts/infer_eval_retake.sh ${YOUR_PATH_TO_Qwen2-VL-7B-Instruct} configs/qwen2_vl/retake_qwen2-vl_lvbench.yaml 8
```
The above script perform inference and evaluation all in one. Results can be found in `./results`
