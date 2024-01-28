from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from inference import FeatureExtractor
from modules.config import cfg

import joblib
import numpy as np 
import torch
from PIL import Image

from sklearn.neighbors import KDTree
from lshashpy3 import LSHash
import faiss
import h5py

import time
from tqdm.auto import tqdm


class InputQuery(BaseModel):
    img_path:str
    attrs:list
    # lsis:str = 'faiss'
    

print("LOADING CONFIG", end=' ')
cfg.merge_from_file('./config/FashionAI/FashionAI.yaml')
cfg.merge_from_file('./config/FashionAI/s2.yaml')
cfg.freeze()
print("--- DONE!\n")

print("BUILDING MODEL", end=' ')
extractor = FeatureExtractor(cfg)
print("--- DONE!\n")

print("LOADING DATABASE", end=' ')
collection_id = joblib.load("collections/c_idxs.npy")
collections = []
with h5py.File("collections/data.h5", 'r') as data:
    for i in range(cfg.DATA.NUM_ATTRIBUTES):
        collection = data["attr" + str(i)][...]
        collections.append(collection)
print("--- DONE!\n")

print("CONSTRUCTING LSIS: KDTREE - LSH - FAISS")
kdtrees = []
lshs = []
index_flats = []

for i in tqdm(range(cfg.DATA.NUM_ATTRIBUTES), desc=str(i)):
    # kdtree = KDTree(collection)

    # lsh = LSHash(10, collection.shape[1], 3)
    # for i in range(len(collection)):
    #     lsh.index(collection[i], extra_data=i)

    index_flat = faiss.IndexFlatL2(collection.shape[1])
    # if faiss.get_num_gpus() > 0:
    #     res = faiss.StandardGpuResources()
    #     index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index_flat.train(collection) 
    index_flat.add(collection)
    
    # kdtrees.append(kdtree)
    # lshs.append(lsh)
    index_flats.append(index_flat)

print("CONSTRUCTING LSIS --- DONE!")

app = FastAPI()  


@app.get("/")
async def home():
    return "GROUP 1 - CS336.O11.KHTN"

@app.post("/submit")
async def submit(input_query: InputQuery, k=50):
    start_time = time.time()

    x = Image.open(input_query.img_path)
    attrs = torch.tensor(input_query.attrs)    
    
    multi_dists = []
    multi_ids = []
     
    for attr_idx, use_attr in enumerate(attrs):
        if use_attr == True: 
            # extract feature corresponding to given attribute
            feature = extractor(x, attr_idx).cpu().numpy()

            # calculating dists w.r.t current attribute
            dists, ids = index_flats[attr_idx].search(feature, k)
            multi_dists.append(dists)
            multi_ids.append(ids)
    
    print(len(multi_dists)) 
    print(len(multi_ids))
    
     
    finish_time = time.time() 
    print("execution time", finish_time - start_time)
    
    return finish_time - start_time


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
    