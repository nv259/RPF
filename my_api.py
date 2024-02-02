from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel
from inference import FeatureExtractor
from modules.config import cfg
from sentence_transformers  import SentenceTransformer, util
import torch

from helper import split_sentences

import joblib
from PIL import Image

# from sklearn.neighbors import KDTree
# from lshashpy3 import LSHash
import faiss
import h5py

import time
from tqdm.auto import tqdm
from collections import defaultdict


class InputQuery(BaseModel):
    img_path:str
    attrs:str
    # lsis:str = 'faiss'
    
    
class APIResponse(BaseModel):
    ids:list = [int]
    similarities:list = [float]
    retrieval_time:float
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
print("LOADING CONFIG", end=' ')
cfg.merge_from_file('./config/FashionAI/FashionAI.yaml')
cfg.merge_from_file('./config/FashionAI/s2.yaml')
cfg.freeze()
print("--- DONE!\n")

print("BUILDING MODEL", end=' ')
extractor = FeatureExtractor(cfg)
sentence_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
print("--- DONE!\n")

print("CALCULATING ATTRIBUTES EMBEDDINGS", end=' ')
attrs_gold = ['skirt', 'sleeve', 'coat', 'pant', 'collar', 'lapel', 'neckline', 'neck'] # cfg.DATA.ATTRIBUTES.NAME
attrs_gold_emb = sentence_model.encode(attrs_gold, convert_to_tensor=True, device=device, show_progress_bar=True)
print("--- DONE!\n")

print("LOADING DATABASE", end=' ')
collections = []
# collections_id = []
collection_id = joblib.load("collections/multi_attrs/c_idxs.npy")

with h5py.File("collections/multi_attrs/data.h5", 'r') as data:
    for i in range(cfg.DATA.NUM_ATTRIBUTES):
        collection = data["attr" + str(i)][...]
        collections.append(collection)
print("--- DONE!\n")

print("CONSTRUCTING LSIS: FAISS")
kdtrees = []
lshs = []
index_flats = []

for i in tqdm(range(cfg.DATA.NUM_ATTRIBUTES), desc=str(i)):
    # kdtree = KDTree(collection)

    # lsh = LSHash(10, collection.shape[1], 3)
    # for i in range(len(collection)):
    #     lsh.index(collection[i], extra_data=i)

    index_flat = faiss.IndexFlatL2(collections[i].shape[1])
    # if faiss.get_num_gpus() > 0:
    #     res = faiss.StandardGpuResources()
    #     index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index_flat.train(collections[i]) 
    index_flat.add(collections[i])
    
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
    
    k= int(k)
    x = Image.open(input_query.img_path) 
    
    multi_dists = []
    multi_ids = []
    
    attrs_list = split_sentences(input_query.attrs) 
    input_query.attrs = [0] * 8
    input_attrs_emb = sentence_model.encode(attrs_list, convert_to_tensor=True)
    
    # map attr to attrs_gold
    for attr_emb in input_attrs_emb:
        cos_sim = util.pytorch_cos_sim(attr_emb, attrs_gold_emb)
        # print(cos_sim.argmax().item())
        input_query.attrs[cos_sim.argmax().item()] = 1
    
    # retrieve items for each attribute
    for attr_idx, use_attr in enumerate(input_query.attrs):
        if use_attr == True: 
            # extract feature corresponding to given attribute
            feature = extractor(x, torch.tensor(attr_idx)).cpu().numpy()

            # calculating dists w.r.t current attribute
            dists, ids = index_flats[attr_idx].search(feature, k)
            multi_dists.append(dists.squeeze())
            multi_ids.append(collection_id[ids].squeeze())
    
    # rerank items by combining multiple attributes  
    final_ranked_list = combine_multi_attrs(multi_ids, multi_dists)
    
    finish_time = time.time() 
    retrieval_time = finish_time - start_time
    
    return APIResponse(ids=[int(id) for id in final_ranked_list.keys()], 
                       similarities=[float(similarity) for similarity in final_ranked_list.values()],
                       retrieval_time=retrieval_time)


def combine_multi_attrs(ids, dists):
    ranked_list = defaultdict(int)
    
    for i in range(len(ids)):
        for j in range(len(ids[i])):
            sim = (2 - dists[i][j] ** 2)/2 
            ranked_list[ids[i][j]] += sim

    ranked_list = sorted(ranked_list.items(), key=lambda x:x[1], reverse=True)
    ranked_list = dict(ranked_list)

    return ranked_list


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
    
