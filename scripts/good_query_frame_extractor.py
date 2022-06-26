import pickle
from PIL import Image
import torch
import pandas as pd
from augmentation.augment import train_transform, valid_transform
import lmdb
import cv2
import numpy as np
from tqdm.auto import tqdm

from collections import OrderedDict
import copy
import json
import argparse

from models.i3d_model import InceptionI3d
from reader_utils.reader import (
    get_data_point,
    get_frames,
    transform_frames,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_model(ckpt_path):
    model_params = {
        "ckpt_path": ckpt_path,
    }

    model = InceptionI3d(**model_params)
    model.cuda()
    model.eval()
    return model


def get_frame_features(model, norm_frames):
    all_features = []
    for norm_frame in norm_frames:
        with torch.no_grad():
            y = model(
                norm_frame.unsqueeze(0)
                .repeat(16, 1, 1, 1)
                .unsqueeze(0)
                .permute(0, 2, 1, 3, 4)
            )
            features = y[0]
            all_features.append(features)
    all_features = torch.stack(all_features)
    return all_features


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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--query_lmdb_dir",
    default="OSLWL_QUERY_TEST_SET/lmdb_videos/test_query",
)

parser.add_argument(
    "--ckpt_path",
    default="data/pretrained_models/model_bsl1k_wlasl.pth.tar",
)

parser.add_argument(
    "--save_dir",
    default="data/dict_test_important_frames.json",
)

args = parser.parse_args()

query_lmdb_dir = args.query_lmdb_dir
ckpt_path = args.ckpt_path
save_dir = args.save_dir

model = get_model(ckpt_path)
query_data = get_data_point(query_lmdb_dir, "details")


dict_query_items = {}
for query_item_name in tqdm(list(query_data.keys())):
    query_item_data = get_data_point(query_lmdb_dir, query_item_name)
    query_item_frames = get_frames(query_item_data)
    query_item_norm_frames = transform_frames(query_item_frames, valid_transform)

    query_features = get_frame_features(model, query_item_norm_frames.cuda())
    query_features = query_features.detach().cpu()
    dict_query_items[query_item_name] = query_features


dict_query_items = OrderedDict(sorted(dict_query_items.items()))


frame_counter = 0
dict_querys = {}
dict_important_frames = {}

# ----------------------------------------
query_stack = []
for q_name, query in dict_query_items.items():
    query_stack.append(query)
    q = sim_matrix(query, query)
query_stack = torch.vstack(query_stack)

query_sim_score = sim_matrix(query_stack, query_stack)
# ----------------------------------------

for q_name, query in dict_query_items.items():

    query_item_frames = get_frames(get_data_point(query_lmdb_dir, q_name))

    sim_score = sim_matrix(query, query)

    ind_query_sim = query_sim_score[frame_counter : frame_counter + query.shape[0]]
    ind_sim_score = query_sim_score.sum(axis=1)[
        frame_counter : frame_counter + query.shape[0]
    ]

    good_frames = torch.where(
        ind_sim_score < (ind_sim_score.max() - ind_sim_score.std() * 2)
    )[0]

    good_frames = torch.arange(good_frames[0], good_frames[-1] + 1)

    global_good_frames = [
        x for i, x in enumerate(query_item_frames) if i in good_frames
    ]
    print(q_name, len(global_good_frames))

    dict_important_frames[q_name] = np.array(good_frames)

    frame_counter += query.shape[0]

with open(save_dir, "w") as fp:
    json.dump(dict_important_frames, fp, cls=NumpyEncoder)
