

# Downloads

Download the pretrained models and preprocessed data: [download](https://drive.google.com/file/d/14YLgo-HwXt7TRX4PD4r5sBIULZRfp1tm/view?usp=sharing)

Extract to base repo directory

# Prepare docker

```bash
docker build -t chalearn .
docker run -it --gpus all -v <REPO_PATH>:/ chalearn /bin/bash
cd /charlearn/
```

# Prepare lmdb format of data


```
python scripts/track2_lmdb_creator.py \
  --dest_dir PROCESSED/OSLWL_QUERY_TEST_SET/lmdb_videos/test_query \
  --dataset_dir <TEST_QUERY_SET_VIDEO_DIR> \
  --bbox_dir data/query_test_bboxes.json
```

```
python scripts/track2_lmdb_creator.py \
  --dest_dir PROCESSED/OSLWL_TEST_SET_VIDEOS/lmdb_videos/test_keys \
  --dataset_dir <TEST_SET_VIDEO__DIR>  \
  --bbox_dir data/oslwl_test_bboxes.json
```

# Extract Important Isolated Sign Frames

```
python scripts/good_query_frame_extractor.py \
  --query_lmdb_dir PROCESSED/OSLWL_QUERY_TEST_SET/lmdb_videos/test_query \
  --ckpt_path data/pretrained_models/model_bsl1k_wlasl.pth.tar \
  --save_dir data/dict_test_important_frames.json
```


# Run Similarity Extractor

Using the following parameters:
```
window_size = 8
num_repeats = 64
num_key_repeats = 64
transform_type = train
important_frames_dir = data/dict_test_important_frames.json
query_lmdb_dir = PROCESSED/OSLWL_QUERY_TEST_SET/lmdb_videos/test_query
key_lmdb_dir = PROCESSED/OSLWL_TEST_SET_VIDEOS/lmdb_videos/test_keys
save_dir = ckpts
seed = 42
```

Run the `similarity_extractor.py` script.
```
python similarity_extractor.py --query_item_name $(query_item_name) --window_size $(window_size) --ckpt_path $(ckpt_path) --num_repeats $(num_repeats) --num_key_repeats $(num_key_repeats) --transform_type $(transform_type) --important_frames_dir $(important_frames_dir) --query_lmdb_dir $(query_lmdb_dir) --key_lmdb_dir $(key_lmdb_dir) --save_dir $(save_dir) --seed $(seed)
```

Using either:
```
ckpt_path = data/pretrained_models/model_bsl1k_wlasl.pth.tar 
ckpt_path = data/pretrained_models/model_bsl1k_msasl.pth.tar
```

repeat for each query video item
```
query_item_name = p00_s20
query_item_name = p00_s21
...
query_item_name = p03_s39
```

This will create a `.pkl` file in the `save_dir` to be used for the ensemble.

# Run Ensemble Solution

```
python test_solution_creator_v2.py
```

Takes all the pickle files in `ckpts/test` and ensembles the results, saving in a `predictions.pkl` file.