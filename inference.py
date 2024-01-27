import argparse
import time 

import torch
from torch import nn

from PIL import Image

# large scale image search
from sklearn.neighbors import KDTree
from lshashpy3 import LSHash
import faiss 
import h5py

import numpy as np
from modules.config import cfg
from modules.model import build_model
from modules.data.transforms import GlobalTransform, LocalTransform
import joblib

import random


class FeatureExtractor(nn.Module):
    def __init__(self, cfg, verbose=False):
        super().__init__()
        if verbose:
            print("BUILDING MODEL")
        self.model = build_model(cfg)
        if verbose: 
            print("DEVICE:", cfg.DEVICE)
        self.device = torch.device(cfg.DEVICE) 
        self.model.to(self.device)
        
        # load model state dict
        if verbose:
            print("LOADING CHECKPOINT:", cfg.MODEL.CHECKPOINT)
        checkpoint = torch.load(cfg.MODEL.CHECKPOINT, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        
        # load ViT pretrained
        if verbose: 
            print("LOADING ViT PRETRAINED TO MODEL:", cfg.MODEL.VIT_PRETRAINED) 
        vit_pretrained = np.load(cfg.MODEL.VIT_PRETRAINED)
        self.model.load_from(vit_pretrained)
       
        if verbose:
            print("BUILDING GLOBAL/LOCAL TRANSFORM")
        self.gt = GlobalTransform(cfg)
        self.lt = LocalTransform(cfg)
        
    def forward(self, x, a, beta=0.6):
        a = a.to(self.device)
        a = a.unsqueeze(dim=0)
        
        gx = self.gt(x)
        gx = gx.to(self.device)
        gx = gx.unsqueeze(dim=0)
        
        with torch.no_grad():
            g_feat, _, attmap = self.model(gx, a, level='global')
            
        attmap = attmap.cpu().numpy()
        
        lx = self.lt(x, attmap)
        lx = lx.to(self.device)
        lx = lx.unsqueeze(dim=0)
        
        with torch.no_grad():
            l_feat = self.model(lx, a, level='local')
            
        feature = torch.cat((torch.sqrt(torch.tensor(beta)) * nn.functional.normalize(g_feat, p=2, dim=1),
                             torch.sqrt(torch.tensor(1-beta)) * nn.functional.normalize(l_feat, p=2, dim=1)), dim=1)
        
        return feature


##### GLOBAL VARIABLES
LSIS = "faiss"
# extractor = FeatureExtractor(cfg)


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lsis", default='lsh', type=str, 
        choices=["kdtree", "lsh", "faiss"],
        help="large scale image search technique"
    ) 
    parser.add_argument(
        "--model-path", default="runs/FashionAI_s2/model_best.pth.tar",
        type=str, help="path to inference model"
    )
    parser.add_argument(
        "--pretrained", default="pretrained/resnet50-19c8e357.pth",
        type=str, help="path to pretrained ViT"
    )
    parser.add_argument(
        "--image", default="data/FashionAI/fashionAI_attributes_train1/Images/coat_length_labels/0a0a60fa935b5bc0186fe657bcc4a632.jpg",
        type=str, help="path to query image"
    )
    parser.add_argument(
        "--attr", default="neckline_design",
        type=str, choices=["skirt_length", "sleeve_length", "coat_length", "pant_lenght", "collar_design", "lapel_design", "neckline_design", "neck_design"],
        help="specific attribute"
    )
    parser.add_argument(
        '-v', "--verbose", action='store_true'
    )
    
    return parser.parse_args()


def main(args, cfg, k=50):
    extractor = FeatureExtractor(cfg, args.verbose)
    
    attr_idx = cfg.DATA.ATTRIBUTES.NAME.index(args.attr)
    x = Image.open(args.image).convert('RGB')
    a = torch.tensor(attr_idx)
    
    # extract feature of query image (with attribute information)
    feature = extractor(x, a).cpu().numpy()
    
    # load collection
    collection_id = joblib.load("collections/c_idxs.npy")
    with h5py.File("collections/data.h5", 'r') as file:
        collection = file["attr" + str(attr_idx)][...]
    
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
    
    # find `k` most similar images to query image
    start_time = time.time()
    
    if args.lsis is None:
        dists = np.linalg.norm(collection - feature, axis=1) 
        ids = np.argsort(dists)[:k]
    if args.lsis == 'kdtree':
        dists, ids = kdtree.query(feature, k=k)
    if args.lsis == 'lsh':
        neighbors = lsh.query(feature.flatten(), num_results=k, distance_func='euclidean') 
        ids = [neighbor[0][1] for neighbor in neighbors]
        dists = [neighbor[1] for neighbor in neighbors]
    if args.lsis == 'faiss':
        dists, ids = index_flat.search(feature, k) 
     
    finish_time = time.time()
    print("Retrieval time:", finish_time - start_time)
    print(collection_id[ids])
    
    
if __name__ == "__main__":
    torch.set_num_threads(1)
    args = parse_args()
    cfg.merge_from_file('./config/FashionAI/FashionAI.yaml')
    cfg.merge_from_file('./config/FashionAI/s2.yaml')
    cfg.freeze()
    set_seed()

    main(args, cfg)