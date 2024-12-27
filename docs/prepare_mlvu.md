## Prepare MLVU Dataset


### Step 1: download MLVU dataset from [huggingface](https://huggingface.co/datasets/MLVU/MVLU)
```bash
git clone https://huggingface.co/datasets/MLVU/MVLU
```

Denote the root directory of download MLVU dataset as `mlvu_root`, it should has the following structure:
```
${mlvu_root}/
├── MLVU/
    ├── json
        ...
    ├── video
        ...
├── figs/
```


### Step 2: Extract frames of all videos
```bash
cd ${retake_repo_root}
python scripts/utils/frame_extraction.py \
--videofile_tpl ${mlvu_root}/MLVU/video/'*/*.mp4' \
--results_dir ${mlvu_root}/MLVU/video_25fps \
--fps 25 \
--num_workers 32
```


### Step 3: Build MLVU dataset
```bash
cd ${retake_repo_root}
python scripts/utils/build_mlvu_dataset.py --hf_root ${mlvu_root}
```
Note that you can NOT modify folder `${mlvu_root}/MLVU/video_25fps` after this step, since the absolute path of extracted frames are written into annotation files `mlvu.json`:
```
retake_repo_root/
├── dataset/
    ├── mlvu/
        ├── mlvu.json
├── ...
```