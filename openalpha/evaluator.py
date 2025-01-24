from google.cloud import storage
import io
import numpy as np
import pandas as pd 
import time
from tqdm import tqdm

from openalpha.util import normalize_weight

def _get_return(strategy, f):
    return_array = f["return_array"]
    universe_array = f["universe_array"]
    common_feature_array = f["common_feature_array"]
    specific_feature_array = f["specific_feature_array"]
    future_return_array = f["future_return_array"]

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
    r = sum(future_return_array * weight_array)
    return r

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
        stime = time.time()
        for data in tqdm(self.cache):
            return_array = data["return_array"].astype(float)
            universe_array = data["universe_array"].astype(bool)
            specific_feature_array = data["specific_feature_array"].astype(float)
            common_feature_array = data["common_feature_array"].astype(float)
            future_return_array = data["future_return_array"].astype(float)

            f = {
                "return_array" : return_array,
                "universe_array" : universe_array,
                "specific_feature_array" : specific_feature_array,
                "common_feature_array" : common_feature_array,
                "future_return_array" : future_return_array,
            }
            r = _get_return(strategy, f)
            ret.append(r)
        ret = pd.Series(ret)

        ############################
        time_elapsed = time.time() - stime
        sharpe = ret.mean() / ret.std() * np.sqrt(52)
        info = {
            "estimated-return" : ret,
            "estimated-time" : time_elapsed / len(self.cache) * 1024 + 600,
            "estimated-performance" : sharpe,
        }
        return info

