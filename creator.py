import argparse
import os, sys
import time
import datetime

import torch

import numpy as np
from modules.config import cfg
from modules.utils.logger import setup_logger
from modules.data import build_data
from modules.model import build_model
from modules.data.transforms import GlobalTransform, LocalTransform
from modules.engine import create_collections, do_eval
import random


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(cfg):
    logger = setup_logger(name=cfg.NAME, level=cfg.LOGGER.LEVEL, stream=cfg.LOGGER.STREAM)
    device = torch.device(cfg.DEVICE)

    model = build_model(cfg)
    
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    
    model.to(device)

    start_epoch=0
    gt = GlobalTransform(cfg)#global stream data process

    test_candidate_loader = build_data(cfg, 'TEST')

    path = './runs/FashionAI_s2/model_best.pth.tar'
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        logger.info(f"No checkpoint found at '{path}'.")
        sys.exit()

    lt = LocalTransform(cfg)
    x = np.load("pretrained/imagenet21k_ViT-B_16.npz")
    model.load_from(x)

    idxs, feats = create_collections(
        model,  
        test_candidate_loader, 
        gt, 
        lt, 
        cfg.DATA.ATTRIBUTES.NAME, 
        device, 
        logger, 
        epoch=-1, 
        beta=cfg.SOLVER.BETA
    )

if __name__ == "__main__":
    torch.set_num_threads(1)
    cfg.merge_from_file('./config/FashionAI/FashionAI.yaml')
    cfg.merge_from_file('./config/FashionAI/s2.yaml')
    cfg.freeze()
    set_seed()
    main(cfg)