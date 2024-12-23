from google.cloud import storage
import random
import io
import numpy as np
from tqdm import tqdm

from openalpha.util import normalize_weight

class Downloader():
    def __init__(self, universe:str):
        self.universe = universe
        bucket = storage.Client.create_anonymous_client().bucket("openalpha-public")
        self.blob_list = list(bucket.list_blobs(prefix=f"{self.universe}/feature/"))
        self.cache = {}

    def get_feature_dict(self, ):
        idx = np.random.randint(len(self.blob_list))

        if idx in self.cache.keys():
            data = self.cache[idx]
        else:
            blob = self.blob_list[idx]
            data = np.load(io.BytesIO(blob.download_as_bytes())) 
            self.cache[idx] = data

        feature_dict = {key:data[key] for key in ["return_array", "universe_array", "specific_feature_array", "common_feature_array"]}
        return feature_dict

    def eval_generator(self, generate)->np.array:
        print("Downloading Data...")
        for idx,blob in tqdm(enumerate(self.blob_list)):
            if idx in self.cache.keys():
                pass 
            else:
                self.cache[idx] = np.load(io.BytesIO(blob.download_as_bytes())) 
        print("Downloading Done!")

        r = []
        for idx in tqdm(range(len(self.blob_list))):
            data = self.cache[idx]
            feature_dict = {key:data[key] for key in ["return_array", "universe_array", "specific_feature_array", "common_feature_array"]}
            weight_array = generate(**feature_dict)
            weight_array = normalize_weight(
                weight_array = weight_array, 
                return_array = data["return_array"],
                universe_array = data["universe_array"],
                )
            future_return_array = np.nan_to_num(data["future_return_array"])
            r.append(sum(future_return_array * weight_array))
        return r

