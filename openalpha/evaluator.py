from google.cloud import storage
import random
import io
import numpy as np
import pandas as pd 
import time
from tqdm import tqdm

from openalpha.util import normalize_weight

class Evaluator():
    def __init__(self, universe:str):
        self.universe = universe
        bucket = storage.Client.create_anonymous_client().bucket("openalpha-public")
        blob_list = list(bucket.list_blobs(prefix=f"{self.universe}/feature/"))
        self.cache = []
        print("Downloading Data...")
        for blob in tqdm(blob_list):
            data = np.load(io.BytesIO(blob.download_as_bytes())) 
            self.cache.append(data)
        print("Done!")
        return None

    def eval_strategy(self, strategy)->pd.Series:
        ret = []
        alt_ret = []
        stime = time.time()
        for data in tqdm(self.cache):
            return_array = data["return_array"]
            universe_array = data["universe_array"]
            specific_feature_array = data["specific_feature_array"]
            common_feature_array = data["common_feature_array"]
            future_return_array = data["future_return_array"]

            weight_array = strategy(
                return_array = return_array,
                universe_array = universe_array,
                specific_feature_array = specific_feature_array,
                common_feature_array = common_feature_array,
                )
            weight_array = normalize_weight(
                weight_array = weight_array,
                return_array = return_array, 
                universe_array = universe_array,
                )
            ret.append(sum(future_return_array * weight_array))

            ########################

            idx_list = list(range(return_array.shape[1]))
            random.shuffle(idx_list)
            idx_list = idx_list[:len(idx_list)//2]

            return_array = return_array[:,idx_list]
            universe_array = universe_array[:,idx_list]
            specific_feature_array = specific_feature_array[:,idx_list,:]
            future_return_array = future_return_array[idx_list]

            weight_array = strategy(
                return_array = return_array,
                universe_array = universe_array,
                specific_feature_array = specific_feature_array,
                common_feature_array = common_feature_array,
                )
            weight_array = normalize_weight(
                weight_array = weight_array,
                return_array = return_array, 
                universe_array = universe_array,
                )
            alt_ret.append(sum(future_return_array * weight_array))

        ############################
        ret = pd.Series(ret)
        alt_ret = pd.Series(alt_ret)
        time_elapsed = time.time() - stime

        SR = ret.mean() / ret.std() * np.sqrt(52)
        MCC = 0.5
        SC = ret.corr(alt_ret)
        reward = max(0,SR) * (1-MCC) * max(0,SC) * 100
        reward = min(reward, 1000)

        info = {
            "estimated-return" : ret,
            "estimated-reward" : reward,
            "estimated-time" : time_elapsed / len(self.cache) * 1024 + 600,
            "estimated-SR" : SR,
            "estimated-MCC" : MCC,
            "estimated-SC" : SC,
        }
        return info

