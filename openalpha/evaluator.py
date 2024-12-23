from google.cloud import storage
import random
import io
import numpy as np
import pandas as pd 
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

    def eval_generator(self, generate)->pd.Series:
        r = []
        for data in tqdm(self.cache):
            feature_dict = {key:data[key] for key in ["return_array", "universe_array", "specific_feature_array", "common_feature_array"]}
            weight_array = generate(**feature_dict)
            weight_array = normalize_weight(
                weight_array = weight_array, 
                return_array = data["return_array"],
                universe_array = data["universe_array"],
                )
            future_return_array = np.nan_to_num(data["future_return_array"])
            r.append(sum(future_return_array * weight_array))
        return pd.Series(r)

