import os
import cv2
import math
import yaml
import time
from PIL import Image
from typing import List, Union

import torch
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoProcessor

import retake


def get_frame_indices(total_frames, max_num_frames, sample_fps, extraction_fps):
    # Get number of sampled frames
    sample_frames = float(total_frames / extraction_fps) * sample_fps
    sample_frames = min(total_frames, max_num_frames, sample_frames)
    sample_frames = math.floor(sample_frames)
    sample_frames = int(sample_frames / 2) * 2
    # Get sampled frame indices
    frame_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
    return frame_indices


def load_specific_frames(cap, frame_indices):
    # List to store the frames
    frames = []
    # Read frames from the video
    for frame_index in frame_indices:
        # Set the video position to the desired frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Read the frame
        ret, frame = cap.read()
        # If the frame was read successfully, append it to the list
        if ret:
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create a PIL Image from the frame
            frame = Image.fromarray(frame_rgb)
            frames.append(frame)
        else:
            ValueError(f"Warning: Could not read frame at index {frame_index}. It may be out of range.")
    return frames


def load_video(video_path: str, max_num_frames: int, fps: Union[int, float]=None, frame_extraction_fps: Union[int, float]=None):
    """Load video frames at fps. If total frames larger than `max_num_frames`, do downsample.
       If 'fps' is `None`, load uniformly sample `max_num_frames` frames.

       video_path: Should either be a videofile or a directory of extracted frames.

       # NOTE: The extract frames must have name pattern of `%06d.(ext)`, or the loaded frame order will be wrong.
    """
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    if os.path.isdir(video_path): # directory extracted frames
        assert frame_extraction_fps is not None
        pass
    else: # filename of a video
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_extraction_fps = cap.get(cv2.CAP_PROP_FPS)
        # Get indices of sampled frame
        frame_indices = get_frame_indices(total_frames, max_num_frames, fps, frame_extraction_fps)
        # Get frames
        frames = load_specific_frames(cap, frame_indices)
        # Release the video capture object
        cap.release()

    # Convert into RGB format
    frames = [
        frame.convert("RGB") if frame.mode != "RGB" else frame
        for frame in frames
    ]

    return frames


def resize_image_longside(image, image_resolution):
    r"""
    Pre-processes a single image.
    """
    if max(image.width, image.height) > image_resolution:
        resize_factor = image_resolution / max(image.width, image.height)
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.NEAREST)

    return image


def resize_video_longside(frames: List, video_resolution):
    """
    frames: list of PIL images.
    """
    frames = [
        resize_image_longside(frame, video_resolution)
        for frame in frames
    ]
    return frames


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def fetch_video(video_info, max_num_frames, sample_fps, longsize_resolution):
    frames = load_video(video_info['video'], max_num_frames, sample_fps, video_info.get('frame_extraction_fps', None))
    frames = resize_video_longside(frames, longsize_resolution)
    frames = [pil_to_tensor(frame) for frame in frames]
    return frames


def load_and_patch_model(model_name, hf_model_path, exp_configs, device):
    model_name = model_name if model_name is not None else exp_configs['model_name']
    model_name = model_name.lower().replace('-', '').replace('_', '')
    if model_name == 'qwen2vl':
        from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
        from retake.monkeypatch import patch_qwen2vl, patch_qwen2vl_config
        retake.qwen2_vl.DEBUG_MODE = True
        patch_qwen2vl(exp_configs['method']) # Replace some functions of QWen2VL with those from ReTaKe
        qwen2vl_config = Qwen2VLConfig.from_pretrained(hf_model_path)
        qwen2vl_config = patch_qwen2vl_config(qwen2vl_config, exp_configs)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_model_path,
            config=qwen2vl_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=exp_configs.get('attn_implementation', None),
            device_map=device # "auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(hf_model_path)
    elif model_name == 'llavavideo':
        from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration
        from retake.monkeypatch import patch_llava_video_config
        llava_video_config = LlavaOnevisionConfig.from_pretrained(hf_model_path)
        llava_video_config = patch_llava_video_config(llava_video_config, exp_configs)
        processor = AutoProcessor.from_pretrained(hf_model_path) 
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            hf_model_path, 
            config=llava_video_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=exp_configs.get('attn_implementation', None),
            device_map=device # "auto"
        ) 
        # TODO: Patch LLava-Video with ReTaKe
    else:
        raise NotImplementedError
    return model, processor


DEMO_VIDEO = 'misc/Q8AZ16uBhr8_resized_fps2_mute.mp4'
DEMO_QUESTION = "Describe the video."


if __name__ == "__main__":
    #------------------- Modify the following configs ------------------#
    hf_model_path = 'Qwen/Qwen2-VL-7B-Instruct' # TODO: replace to local path if you have trouble downloading huggingface models
    model_name = 'qwen2_vl'
    # hf_model_path = '/path_to/llava-video-qwen2-7b-hf'
    # model_name = 'llava_video'
    # hf_model_path = 'llava-hf/llava-onevision-qwen2-7b-ov-hf'
    # model_name = 'llava_onevision'
    
    # NOTE: for Nvidia GPUs
    config_path = 'configs/retake_demo_tpot.yaml'
    device = 'cuda:0'

    # NOTE: for NPUs or GPUs without support for FlashAttention
    # config_path = 'configs/retake_demo_npu.yaml'
    # device = 'npu:0'

    #------------------------ No need to change ------------------------#
    exp_configs = load_yaml(config_path)

    model, processor = load_and_patch_model(model_name, hf_model_path, exp_configs, device)

    # Video
    video_info = {"type": "video", 
                  "video": DEMO_VIDEO, 
                  "fps": 2.0}
    video = fetch_video(video_info, 256, 2, exp_configs['longsize_resolution'])

    def cal_total_time(num_samples, max_new_tokens=512):
        total_times = []
        num_new_tokens = []
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": DEMO_QUESTION},
                ],
            }
        ]

        # Preprocess the inputs
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt")
        inputs = inputs.to(device)
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.bfloat16)

        # Inference: Generation of the output
        for idx in range(num_samples + 1):
            start_time = time.time()
            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            end_time = time.time()

            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output_text = output_text[0]

            if idx > 0: # Skip first sample, generally takes longer time
                total_times.append(end_time - start_time)
                num_new_tokens.append(generated_ids[0].shape[0])
        print('Output text:\n', output_text)
        return total_times, num_new_tokens

    total_times, num_new_tokens = cal_total_time(5, max_new_tokens=1)
    print('total_times', total_times)
    print('num_tokens', num_new_tokens)
    print('Avg. total time:', sum(total_times) / len(total_times))
    ttft = sum(total_times) / len(total_times)

    total_times, num_new_tokens = cal_total_time(5, max_new_tokens=512)
    print('total_times', total_times)
    print('num_tokens', num_new_tokens)
    print('Avg. total time:', sum(total_times) / len(total_times))
    tpot = (sum(total_times) / len(total_times) - ttft) / (sum(num_new_tokens) / len(num_new_tokens) - 1)

    print(f"Time To First Token (TTFT): {ttft:.6f} seconds")
    print(f"Time Per Output Token (TPOT): {tpot:.6f} seconds")
