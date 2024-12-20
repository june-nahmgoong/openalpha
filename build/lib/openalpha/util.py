from google.cloud import storage
import random
import io
import numpy as np

def get_sample_feature(universe:str):
    bucket = storage.Client.create_anonymous_client().bucket("openalpha-public")
    blob_list = list(bucket.list_blobs(prefix=f"{universe}/feature-array/"))
    random.shuffle(blob_list)
    blob = blob_list[0]
    data = np.load(io.BytesIO(blob.download_as_bytes()))
    feature_dict = {k:v for k,v in data.items()}
    return feature_dict

