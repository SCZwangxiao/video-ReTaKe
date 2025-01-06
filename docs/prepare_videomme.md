## Prepare VideoMME Dataset


### Step 1: download VideoMME dataset from [huggingface](https://huggingface.co/datasets/lmms-lab/Video-MME)
```bash
git clone https://huggingface.co/datasets/lmms-lab/Video-MME
```

Denote the root directory of download VideoMME dataset as `videomme_root`, it should has the following structure:
```
${videomme_root}/
├── videomme/
├── subtitle.zip
├── videos_chunked_01.zip
├── videos_chunked_02.zip
├── ...
└── videos_chunked_20.zip
```


### Step 2: Unzip everything
```bash
cd ${videomme_root}
unzip subtitle.zip
cat videos_chunked_*.zip > videos.zip
unzip videos.zip
```


### Step 3: Extract frames of all videos
```bash
cd ${retake_repo_root}
python scripts/utils/frame_extraction.py \
--videofile_tpl ${videomme_root}/data/'*.mp4' \
--results_dir ${videomme_root}/data_25fps \
--fps 25 \
--num_workers 32
```


### Step 4: Build VideoMME dataset
```bash
cd ${retake_repo_root}
python scripts/utils/build_videomme_dataset.py \
--hf_qwen2vl7b_path ${PATH_TO_Qwen2-VL-7B-Instruct} \
--hf_root ${videomme_root}
```
Note that you can NOT modify folder `${videomme_root}/data_25fps` after this step, since the absolute path of extracted frames are written into annotation files `video_mme.json` and `video_mme_subtitle.json`:
```
retake_repo_root/
├── dataset/
    ├── video_mme/
        ├── video_mme_subtitle.json
        ├── video_mme.json
├── ...
```