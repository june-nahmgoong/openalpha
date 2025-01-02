import numpy as np
import random
import io
from google.cloud import storage

def normalize_weight(weight_array:np.array, universe_array:np.array, return_array:np.array):

    T,N = return_array.shape
    assert weight_array.shape == (N,)
    assert universe_array.shape == (T,N)

    # universe filter
    filter = universe_array[-1,:].astype(float)
    filter[filter == 0] = np.nan    
    weight_array = weight_array * filter

    # market neutral
    weight_array = weight_array - np.nanmean(weight_array)
    weight_array = np.nan_to_num(weight_array)

    # leverage adjust
    cov = np.cov(return_array,rowvar=False)
    vol = np.matmul(np.matmul(cov,weight_array),weight_array)
    vol = np.sqrt(vol * 52)
    weight_array = weight_array / vol * 0.05 
    weight_array = np.nan_to_num(weight_array)

    return weight_array

def get_sample_feature_dict(universe:str)->dict:
    bucket = storage.Client.create_anonymous_client().bucket("openalpha-public")
    blob_list = list(bucket.list_blobs(prefix=f"{universe}/feature/"))
    random.shuffle(blob_list)
    blob = blob_list[0]
    data = np.load(io.BytesIO(blob.download_as_bytes())) 
    feature_dict = {key:data[key] for key in ["return_array", "universe_array", "specific_feature_array", "common_feature_array"]}
    return feature_dict 
