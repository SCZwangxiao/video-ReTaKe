import os
import os.path as osp
import json
import re
import yaml
import datetime
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration, AutoProcessor

from retake.monkeypatch import patch_qwen2vl, patch_qwen2vl_config
from retake.dataset_utils import get_dataset, get_eval_methods


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def trimm_results(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


class InferClient:
    def __init__(self, hf_qwen2vl7b_path, exp_configs, device) -> None:
        self.device = device
        self.do_sample = exp_configs.get('do_sample', False)
        self.load_model(hf_qwen2vl7b_path, exp_configs, device)

    def load_model(self, hf_qwen2vl7b_path, exp_configs, device):
        qwen2vl_config = Qwen2VLConfig.from_pretrained(hf_qwen2vl7b_path)
        qwen2vl_config = patch_qwen2vl_config(qwen2vl_config, exp_configs)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_qwen2vl7b_path,
            config=qwen2vl_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_qwen2vl7b_path)

    def infer(self, message):
        # Prepare inputs
        video = message['video']
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": message['question']},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Inference
        output_ids = self.model.generate(**inputs, do_sample=self.do_sample, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = output_text[0]

        return output_text


def parse_arguments():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="ReTaKe Evaluation")
    parser.add_argument('--hf_qwen2vl7b_path', 
                        type=str, 
                        default='Qwen/Qwen2-VL-7B-Instruct', 
                        help="Path to the experimental config")
    parser.add_argument('--config_path', 
                        type=str, 
                        default='configs/retake_videomme.yaml', 
                        help="Path to the experimental config")
    parser.add_argument('--video_frame_extraction_fps', 
                        type=int, 
                        default=25, 
                        help="Video frame extraction FPS")
    parser.add_argument('--n_gpus', 
                        type=int, 
                        default=8, 
                        help="Number of GPUs to use")
    parser.add_argument('--timeout', 
                        type=int, 
                        default=30, 
                        help="Timeout for waiting each GPU finish inference (in minutes).")
    args = parser.parse_args()
    return args


def setup(rank, world_size, timeout):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=timeout))
    torch.cuda.set_device(f'cuda:{rank}')


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    device = f'cuda:{rank}'
    setup(rank, world_size, args.timeout)

    exp_configs = load_yaml(args.config_path)
    patch_qwen2vl(exp_configs['method'])  # Replace some functions of QWen2VL with those from ReTaKe

    client = InferClient(args.hf_qwen2vl7b_path, exp_configs, device)

    dataset = get_dataset(
        dataset_name=exp_configs['dataset_name'],
        anno_file=exp_configs['anno_file'], 
        processor_kwargs=dict(
            video_fps=exp_configs['sample_fps'],
            video_maxlen=exp_configs['max_num_frames'],
            image_resolution=exp_configs['longsize_resolution'],
            video_frame_extraction_fps=args.video_frame_extraction_fps
        )
    )

    # Split dataset into shards
    # shard_size = len(dataset) / world_size
    # shard_start = rank * shard_size
    # shard_end = shard_start + shard_size if rank != world_size - 1 else len(dataset)
    # shard_dataset = Subset(dataset, range(round(shard_start), round(shard_end))) # round make sharding more even
    # Function to create a round-robin shard for a given rank
    indices = [i for i in range(len(dataset)) if i % world_size == rank]
    shard_dataset = Subset(dataset, indices)


    dataloader = DataLoader(shard_dataset, batch_size=None, num_workers=exp_configs['dataloader_num_workers'])

    # Inference
    anno_id2result = {}
    anno_id2meta = {}
    for sample in tqdm(dataloader, desc=f'rank {rank}'): # disable=rank!=0
        idx, message, meta = sample
        pred_answer = client.infer(message)
        pred_answer = trimm_results(pred_answer)
        anno_id2result[idx] = pred_answer
        anno_id2meta[idx] = meta

    # Gather results from all processes
    all_anno_id2result = [None] * world_size
    all_anno_id2meta = [None] * world_size
    dist.barrier()
    dist.all_gather_object(all_anno_id2result, anno_id2result)
    dist.all_gather_object(all_anno_id2meta, anno_id2meta)

    if rank == 0:
        # Merge results
        merged_anno_id2result = {k: v for d in all_anno_id2result for k, v in d.items()}
        merged_anno_id2meta = {k: v for d in all_anno_id2meta for k, v in d.items()}

        # Evaluate
        eval_func = get_eval_methods(exp_configs['dataset_name'])
        eval_result_df, infer_result_df = eval_func(merged_anno_id2result, merged_anno_id2meta)

        # Dump inference & evaluation results
        os.makedirs(exp_configs['output_dir'], exist_ok=True)
        answer_file = os.path.join(exp_configs['output_dir'], "anno_id2result.json")
        infer_res_file = os.path.join(exp_configs['output_dir'], "infer_results.csv")
        eval_res_file = os.path.join(exp_configs['output_dir'], "eval_results.csv")

        with open(answer_file, 'w') as F:
            json.dump(merged_anno_id2result, F)
        infer_result_df.to_csv(infer_res_file, index=False)
        eval_result_df.to_csv(eval_res_file, index=True)

    cleanup()


if __name__ == "__main__":
    args = parse_arguments()
    world_size = args.n_gpus
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
