from fastapi import FastAPI
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
    attr:str
    lsis:str = 'faiss'
    

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
    kdtree = KDTree(collection)

    lsh = LSHash(10, collection.shape[1], 3)
    for i in range(len(collection)):
        lsh.index(collection[i], extra_data=i)

    index_flat = faiss.IndexFlatL2(collection.shape[1])
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index_flat.train(collection) 
    index_flat.add(collection)
    
    kdtrees.append(kdtree)
    lshs.append(lsh)
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
    a = torch.tensor(input_query.attr)    
    
    feature = extractor(x, a).cpu().numpy()
    
    if input_query.lsis == 'kdtree':
        dists, ids = kdtree.query(feature, k=k)
    elif input_query.lsis == 'lsh':
        neighbors = lsh.query(feature.flatten(), num_results=k, distance_func='euclidean') 
        ids = [neighbor[0][1] for neighbor in neighbors]
        dists = [neighbor[1] for neighbor in neighbors]
    elif input_query.lsis == 'faiss':
        dists, ids = index_flat.search(feature, k)
    else:
        dists = np.linalg.norm(collections[a] - feature, axis=1) 
        ids = np.argsort(dists)[:k]
        dists = dists[ids]
    
    finish_time = time.time() 
    print("execution time", finish_time - start_time)
    
    return {"ids": ids, "dists": dists}
