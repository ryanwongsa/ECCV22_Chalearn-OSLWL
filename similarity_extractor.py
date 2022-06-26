from models.i3d_model import InceptionI3d

from augmentation.augment import train_transform, valid_transform

import pickle
from PIL import Image
import torch
import pandas as pd

import lmdb
import cv2
import numpy as np
from tqdm.auto import tqdm

from collections import OrderedDict
import copy
import json
import argparse
from reader_utils.reader import get_data_point, get_frames, transform_frames
from pathlib import Path
import random
import os


def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def query_feature_extractor(query_item_name, transform):
    query_item_data = get_data_point(query_lmdb_dir, query_item_name)
    query_item_frames = get_frames(query_item_data)
    important_frames = dict_important_frames[query_item_name]
    query_imp_frames = [
        x for i, x in enumerate(query_item_frames) if i in important_frames
    ]

    # --- REPEAT MULTIPLE TIMES ----
    list_of_query_features = []
    for i in range(num_repeats):
        query_item_norm_frames = transform_frames(query_imp_frames, transform)
        selected_frames = np.sort(
            np.random.choice(
                np.arange(0, query_item_norm_frames.shape[0]),
                window_size,
                replace=False
                if query_item_norm_frames.shape[0] > window_size
                else True,
            )
        )
        input_query_frames = query_item_norm_frames[selected_frames]
        with torch.no_grad():
            y = model(input_query_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).cuda())
            list_of_query_features.append(y[0].detach().cpu())
    list_of_query_features = torch.stack(list_of_query_features)
    return list_of_query_features


def get_query_key_similarity_extractor(
    key_data, list_of_query_features, query_id, transform
):
    dict_results = {}
    kq_data_list = [a for a in list(key_data.keys()) if query_id in a]

    for key_item_name in tqdm(kq_data_list):
        if query_id in key_item_name:
            key_item_data = get_data_point(key_lmdb_dir, key_item_name)
            key_item_frames = get_frames(key_item_data)
            list_kif = []
            for j in range(num_key_repeats):
                key_item_norm_frames = transform_frames(key_item_frames, transform)
                list_kif.append(key_item_norm_frames)
            key_item_norm_frames = torch.stack(list_kif)
            list_of_ss = []
            for i in range(key_item_norm_frames.shape[1] - window_size):
                strided_key_input = key_item_norm_frames[
                    :, i * stride : i * stride + window_size
                ]
                with torch.no_grad():
                    y = (
                        model(strided_key_input.permute(0, 2, 1, 3, 4).cuda())
                        .detach()
                        .cpu()
                    )
                    sim_scores = sim_matrix(list_of_query_features, y)
                    list_of_ss.append(sim_scores)

            ss = torch.stack(list_of_ss)
            dict_results[key_item_name] = ss
    return dict_results


parser = argparse.ArgumentParser()
parser.add_argument("--query_item_name", default="p00_s01")

parser.add_argument("--window_size", default=8, type=int)


parser.add_argument(
    "--ckpt_path",
    default="data/pretrained_models/model_bsl1k_wlasl.pth.tar",
)


parser.add_argument(
    "--num_repeats",
    type=int,
    default=32,
)

parser.add_argument(
    "--num_key_repeats",
    type=int,
    default=1,
)

parser.add_argument(
    "--transform_type",
    default="valid",
)


parser.add_argument(
    "--important_frames_dir",
    default="data/dict_test_important_frames.json",
)


parser.add_argument(
    "--query_lmdb_dir",
    default="OSLWL_QUERY_TEST_SET/lmdb_videos/test_query",
)

parser.add_argument(
    "--key_lmdb_dir",
    default="OSLWL_TEST_SET_VIDEOS/lmdb_videos/test_keys",
)

parser.add_argument(
    "--save_dir",
    default="ckpts",
)


parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

args = parser.parse_args()


window_size = args.window_size 
num_repeats = args.num_repeats
num_key_repeats = args.num_key_repeats
stride = 1


query_lmdb_dir = args.query_lmdb_dir
key_lmdb_dir = args.key_lmdb_dir

dict_important_frames = json.load(
    open(
        args.important_frames_dir,
        "r",
    )
)

key_data = get_data_point(key_lmdb_dir, "details")
query_data = get_data_point(query_lmdb_dir, "details")


setup_seed(args.seed)


model_params = {
    "ckpt_path": args.ckpt_path,
}
model = InceptionI3d(**model_params)
model.cuda()
model.eval()


if args.transform_type == "train":
    transform = train_transform
elif args.transform_type == "valid":
    transform = valid_transform

save_file = f"{args.save_dir}/{Path(query_lmdb_dir).stem}_{Path(args.ckpt_path).stem.split('.')[0]}_{num_repeats}_{num_key_repeats}_{args.transform_type}aug_{window_size}"
Path(save_file).mkdir(exist_ok=True, parents=True)


dict_q_all_results = {}
query_item_name = args.query_item_name
query_id = query_item_name.split("_")[1]
list_of_query_features = query_feature_extractor(query_item_name, transform)
dict_results = get_query_key_similarity_extractor(
    key_data, list_of_query_features, query_id, transform
)
dict_q_all_results[query_item_name] = dict_results


with open(
    f"{save_file}/{query_item_name}.pkl",
    "wb",
) as handle:
    pickle.dump(dict_q_all_results, handle, protocol=4)
