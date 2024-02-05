import argparse
import time 
import os
import sys

import torch
from torch import nn

from PIL import Image

# large scale image search
from sklearn.neighbors import KDTree
from lshashpy3 import LSHash
import faiss 
import h5py

import numpy as np
from unittest import result
import math
import torch.nn as nn
from tqdm import tqdm
from modules.utils.logger import setup_logger
from modules.data import build_data
from modules.config import cfg
from modules.model import build_model
from modules.data.transforms import GlobalTransform, LocalTransform
from modules.utils.metric import APScorer, AverageMeter
import joblib

import random

def do_eval(model, query_loader, candidate_loader, gt, lt, attrs, device, logger, epoch=-1, beta=0.6):
    logger.info("Begin evaluation.")
    model.eval()

    logger.info("Forwarding query images...")
    q_feats,q_values = extract_features(model, query_loader, gt, lt, device, len(attrs), beta=beta)
    logger.info("Forwarding candidate images...")
    c_feats,c_values = extract_features(model, candidate_loader, gt, lt, device, len(attrs), beta=beta)


    mAP_None, mAP_KD, mAP_LSH, mAP_F = 0, 0, 0, 0
    Time_None, Time_KD, Time_LSH, Time_F = 0, 0, 0, 0
    for i, attr in enumerate(attrs):
        print(attr, ':')
        mN, tN, mK, tK, mL, tL, mF, tF = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
        mAP_None += mN; mAP_KD += mK; mAP_LSH += mL; mAP_F += mF
        Time_None += tN; Time_KD += tK; Time_LSH += tL; Time_F += tF

    print('Overall:')
    print('None:', mAP_None  * 100 / len(attrs))
    print('Time:', Time_None)
    print('KDTree:', mAP_KD  * 100 / len(attrs))
    print('Time:', Time_KD)
    print('LSH:', mAP_LSH  * 100 / len(attrs))
    print('Time:', Time_LSH)
    print('Faiss:', mAP_F * 100 / len(attrs))
    print('Time:', Time_F)

def extract_features(model, data_loader, gt, lt, device, n_attrs, beta=0.6):
    feats = []
    feats_g= []
    feats_l= []
    indices = [[] for _ in range(n_attrs)]
    values = []
    with tqdm(total=len(data_loader)) as bar:
        cnt = 0
        for idx, batch in enumerate(data_loader):
            x, bidxs, a, v = batch#x=index of 
            # print("eval a shape",a.shape)
            a = a.to(device)
            
            out= process_batch(model, x, a, gt, lt, device, beta=beta)
            
            feats.append(out.cpu().numpy())
            values.append(v.numpy())

            for i in range(a.size(0)):
                indices[a[i].cpu().item()].append(cnt)
                cnt += 1

            bar.update(1)
    feats = np.concatenate(feats)
    values = np.concatenate(values)
    feats = [feats[indices[i]] for i in range(n_attrs)]
    values = [values[indices[i]] for i in range(n_attrs)]
      
    return feats ,values

def process_batch(model, x, a, gt, lt, device, beta=0.6):
    gx = torch.stack([gt(i) for i in x], dim=0)
    gx = gx.to(device)
    with torch.no_grad():
        if lt is not None:
            g_feats, _, attmap = model(gx, a, level='global')
        else:
            g_feats, attmap = model(gx, a, level='global')
    if lt is None:
        return nn.functional.normalize(g_feats, p=2, dim=1)

    attmap = attmap.cpu().numpy()

    lx = torch.stack([lt(i, mask) for i, mask in zip(x, attmap)], dim=0)
    lx = lx.to(device)
    with torch.no_grad():
        l_feats = model(lx, a, level='local')
    
    out = torch.cat((torch.sqrt(torch.tensor(beta)) * nn.functional.normalize(g_feats, p=2, dim=1),
            torch.sqrt(torch.tensor(1-beta)) * nn.functional.normalize(l_feats, p=2, dim=1)), dim=1)

    return out

def mean_average_precision(queries, candidates, q_values, c_values, k = 50):
    '''
    calculate mAP of a conditional set. Samples in candidate and query set are of the same condition.
        cand_set: 
            type:   nparray
            shape:  c x feature dimension
        queries:
            type:   nparray
            shape:  q x feature dimension
        c_gdtruth:
            type:   nparray
            shape:  c
        q_gdtruth:
            type:   nparray
            shape:  q
    '''    

    if k is None:
        k = len(candidates)
    
    can = candidates
    scorer = APScorer(candidates.shape[0])

    start_time = time.time()
    ap_sum = 0
    for i in range(len(queries)):
        dists = np.linalg.norm(candidates - queries[i], axis=1) 
        ids = np.argsort(dists)[:k]
        sorted_labels = []
        for j in range(len(ids)):
            if c_values[ids[j]] == q_values[i]:
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)
        
        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP_None = ap_sum / len(queries)
    Time_None = time.time() - start_time

    print('None:', mAP_None * 100)
    print("Time:", Time_None)

    candidates = can
    kdtree = KDTree(candidates, leaf_size=300)

    candidates = can
    lsh = LSHash(14, candidates.shape[1], 3)
    for i in range(len(candidates)):
        lsh.index(candidates[i], extra_data=str(i))

    candidates = can
    index_flat = faiss.IndexFlatL2(candidates.shape[1])
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index_flat.train(candidates) 
    index_flat.add(candidates)

    start_time = time.time()
    ap_sum = 0
    for i in range(len(queries)):
        dists, ids = kdtree.query([queries[i]], k = k)
        ids = ids[0]
        sorted_labels = []
        for j in range(len(ids)):
            if c_values[ids[j]] == q_values[i]: 
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)
        
        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP_KD = ap_sum / len(queries)
    Time_KD = time.time() - start_time
    print('KDTree:', mAP_KD * 100)
    print("Time:", Time_KD)

    start_time = time.time()
    ap_sum = 0
    for i in range(len(queries)):
        neighbors = lsh.query(queries[i], num_results=k, distance_func='euclidean') 
        ids = [int(neighbor[0][1]) for neighbor in neighbors]
        dists = [neighbor[1] for neighbor in neighbors]
        sorted_labels = []
        for j in range(len(ids)):
            if c_values[ids[j]] == q_values[i]: 
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)
        
        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP_LSH = ap_sum / len(queries)
    Time_LSH = time.time() - start_time
    print('LSH:', mAP_LSH * 100)
    print("Time:", Time_LSH)

    start_time = time.time()
    ap_sum = 0
    for i in range(len(queries)):
        dists, ids = index_flat.search(np.array([queries[i]]), k)
        ids = ids[0]
        sorted_labels = []
        for j in range(len(ids)):
            if c_values[ids[j]] == q_values[i]: 
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)
        
        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP_F = ap_sum / len(queries)
    Time_F = time.time() - start_time
    print('Faiss:', mAP_F * 100)
    print("Time:", Time_F)

    return mAP_None, Time_None, mAP_KD, Time_KD, mAP_LSH, Time_LSH, mAP_F, Time_F

def main(cfg):
    logger = setup_logger(name=cfg.NAME, level=cfg.LOGGER.LEVEL, stream=cfg.LOGGER.STREAM)
    logger.info(cfg)
    device = torch.device(cfg.DEVICE)

    model = build_model(cfg)
    
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    logger.info(f"Number of parameters: {n_parameters}")
    
    model.to(device)

    start_epoch=0
    gt = GlobalTransform(cfg)#global stream data process

    test_query_loader, test_candidate_loader = build_data(cfg, 'TEST')

    path = 'runs/FashionAI_s2/model_best.pth.tar'
    if os.path.isfile(path):
        logger.info(f"Loading checkpoint '{path}'.")
        checkpoint = torch.load(path, map_location='cpu')
        logger.info(f"Best performance {checkpoint['mAP']} at epoch {checkpoint['epoch']}.")            
        logger.info(f"start at epoch {checkpoint['epoch']}.")
        model.load_state_dict(checkpoint['model'])
        logger.info(f"Loaded checkpoint '{path}'")
    else:
        logger.info(f"No checkpoint found at '{path}'.")
        sys.exit()

    lt = LocalTransform(cfg)
    print("vit")
    model.load_from(np.load("pretrained/imagenet21k_ViT-B_16.npz"))

    do_eval(
        model, 
        test_query_loader, 
        test_candidate_loader, 
        gt, 
        lt if cfg.MODEL.TRANSFORMER.ENABLE else None, 
        cfg.DATA.ATTRIBUTES.NAME, 
        device, 
        logger, 
        epoch=-1, 
        beta=cfg.SOLVER.BETA
    )

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    cfg.merge_from_file('./config/FashionAI/FashionAI.yaml')
    cfg.merge_from_file('./config/FashionAI/s2.yaml')
    cfg.freeze()
    set_seed()

    main(cfg)