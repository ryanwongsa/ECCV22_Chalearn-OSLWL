from pathlib import Path
import pickle
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
from metrics.ineval import compute_f1


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))




list_of_simplified = list(
    Path(
        "ckpts/test_query_model_bsl1k_wlasl_64_64_trainaug_8"
    ).glob("*.pkl")
)

dict_results = {}
for simp in list_of_simplified:
    name = simp.stem
    label_id = int(name.split("_s")[1])
    dict_res = pickle.load(open(simp, "rb"))
    print(name, label_id)
    for k, v in (list(dict_res.values())[0]).items():
        dict_results[k] = v


list_of_simplified2 = list(
    Path(
        "ckpts/test_query_model_bsl1k_msasl_64_64_trainaug_8"
    ).glob("*.pkl")
)

dict_results2 = {}
for simp in list_of_simplified2:
    name = simp.stem
    label_id = int(name.split("_s")[1])
    dict_res = pickle.load(open(simp, "rb"))
    print(name, label_id)
    for k, v in (list(dict_res.values())[0]).items():
        dict_results2[k] = v


dict_solution = defaultdict(list)
threshold = 0.9
window_size = 8
close_connect=10
threshold_origin = 0.36 #0.45
for k, v in dict_results.items():
    
    ss = v.mean(axis=1).mean(axis=1).unsqueeze(0)
    original_ss = ss

    original_ss_max = ss.max()
    
    ss =  ss/original_ss_max  

    v2 = dict_results2[k]
    ss2 = v2.mean(axis=1).mean(axis=1).unsqueeze(0)
    original_ss2_max = ss2.max()

    ss2 = ss2 / ss2.max()
    
    ss = (ss + ss2)/2
    
    good_indices = []
    if original_ss_max>threshold_origin and original_ss2_max>threshold_origin:
        for i in torch.where(ss[0] > threshold)[0]:
            good_indices.append(np.arange(i, i + window_size))


    if len(good_indices) > 0:
        good_indices = np.sort(np.array(list(set(np.concatenate(good_indices)))))
        additional_indices = []
        for r_i,r in enumerate(good_indices[:-1]):
            if  good_indices[r_i+1]-r>1 and good_indices[r_i+1]-r<=close_connect:
                additional_indices.append(np.arange(r,good_indices[r_i+1]+1))
            
        if len(additional_indices)>0:
            # print(additional_indices)
            good_indices = (list(good_indices)+list(np.concatenate(additional_indices)))
        else:
            good_indices = list(good_indices)
        good_indices = np.sort(np.array(list(set(good_indices))))
        
        
        for gi in ranges(good_indices):
            l = int(k.split("_s")[1].split("_")[0])
            dict_solution[k].append([l, int(gi[0] / 25 * 1000), int(gi[1] / 25 * 1000)])




with open('predictions.pkl', 'wb') as handle:
    pickle.dump(dict_solution, handle, protocol=4)
