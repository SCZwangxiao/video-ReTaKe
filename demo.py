import os
import cv2
import math
import yaml
from PIL import Image
from typing import List, Union

import torch
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration, AutoProcessor

from retake.monkeypatch import patch_qwen2vl, patch_qwen2vl_config


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


def load_model(hf_qwen2vl7b_path, exp_configs):
    qwen2vl_config = Qwen2VLConfig.from_pretrained(hf_qwen2vl7b_path)
    qwen2vl_config = patch_qwen2vl_config(qwen2vl_config, exp_configs)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        hf_qwen2vl7b_path,
        config=qwen2vl_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=exp_configs.get('attn_implementation', None),
        device_map=device # "auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(hf_qwen2vl7b_path)
    return model, processor


DEMO_VIDEO = 'misc/Q8AZ16uBQ8AZ16uBhr8_resized_fps2_mutehr8_resized_fps2.mp4'
DEMO_QUESTIONS = [
    "As depicted in the video, how is the relationship between the rabbit and human?\nOptions:\nA. Hostile.\nB. Friend.\nC. Cooperator.\nD. No one is correct above.\nAnswer with the option's letter from the given choices directly.",
    "What is the impression of the video?\nOptions:\nA. Sad.\nB. Funny.\nC. Horrible.\nD. Silent.\nAnswer with the option's letter from the given choices directly.",
    "What is the subject of the video?\nOptions:\nA. Rabbit likes to eat carrots.\nB. How to raise a rabbit.\nC. A rabbit gives people trouble.\nD. A rabbit performs for food.\nAnswer with the option's letter from the given choices directly.",
]
EXPECTED_ANSWERS = ['A', 'B', 'C']


if __name__ == "__main__":
    #------------------- Modify the following configs ------------------#
    hf_qwen2vl7b_path = 'Qwen/Qwen2-VL-7B-Instruct' # TODO: replace to local path if you have trouble downloading huggingface models
    config_path = 'configs/retake_demo.yaml' # NOTE: for Nvidia GPUs
    # config_path = 'configs/retake_demo_npu.yaml' # NOTE: for NPUs or GPUs without support for FlashAttention

    #------------------------ No need to change ------------------------#
    device = 'cuda:0'
    video_info = {"type": "video", 
                  "video": DEMO_VIDEO, 
                  "fps": 2.0}

    exp_configs = load_yaml(config_path)
    patch_qwen2vl(exp_configs['method']) # Replace some functions of QWen2VL with those from ReTaKe

    model, processor = load_model(hf_qwen2vl7b_path, exp_configs)

    # Video
    video = fetch_video(video_info, exp_configs['max_num_frames'], exp_configs['sample_fps'], exp_configs['longsize_resolution'])
    for question, expect_answer in zip(DEMO_QUESTIONS, EXPECTED_ANSWERS):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preprocess the inputs
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        print('Input prompt:\n', text_prompt)

        inputs = processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')

        # Inference: Generation of the output
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = output_text[0]
        print('Output text:\n', output_text)
        print('Expected answer:\n', expect_answer)