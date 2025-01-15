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
        alt_num = 4

        return_df = {idx:[] for idx in range(1+alt_num)}
        stime = time.time()
        for data in tqdm(self.cache):
            return_array = data["return_array"].astype(float)
            universe_array = data["universe_array"].astype(bool)
            specific_feature_array = data["specific_feature_array"].astype(float)
            common_feature_array = data["common_feature_array"].astype(float)
            future_return_array = data["future_return_array"].astype(float)

            ########################

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
            return_df[0].append(sum(future_return_array * weight_array))

            ########################

            for i in range(alt_num):
                idx_list = list(range(return_array.shape[1]))
                random.shuffle(idx_list)
                idx_list = idx_list[:len(idx_list)//2]

                _return_array = return_array[:,idx_list]
                _universe_array = universe_array[:,idx_list]
                _specific_feature_array = specific_feature_array[:,idx_list,:]
                _future_return_array = future_return_array[idx_list]

                _weight_array = strategy(
                    return_array = _return_array,
                    universe_array = _universe_array,
                    specific_feature_array = _specific_feature_array,
                    common_feature_array = common_feature_array,
                    )
                _weight_array = normalize_weight(
                    weight_array = _weight_array,
                    return_array = _return_array, 
                    universe_array = _universe_array,
                    )
                return_df[i+1].append(sum(_future_return_array * _weight_array))

        ############################
        time_elapsed = time.time() - stime

        return_df = pd.DataFrame(return_df)
        ret = return_df[0]
        SR = ret.mean() / ret.std() * np.sqrt(52)
        MCC = 0.5
        MSC = return_df.corr().values[~np.eye(return_df.shape[1], dtype=bool)].min()
        reward = max(0,SR) * (1-MCC) * max(0,MSC) * 1_000

        info = {
            "estimated-return" : ret,
            "estimated-reward" : reward,
            "estimated-time" : time_elapsed / len(self.cache) * 1024 + 600,
            "estimated-SR" : SR,
            "estimated-MCC" : MCC,
            "estimated-MSC" : MSC,
        }
        return info

