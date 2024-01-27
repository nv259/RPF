import argparse
import os, sys

import torch

import numpy as np
from modules.config import cfg
from modules.model import build_model
from modules.engine.inference import extract_features
from modules.data.transforms import GlobalTransform, LocalTransform

import random


class FeatureExtractor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = build_model(cfg)
        self.device = torch.device(cfg.DEVICE)
        self.model.to(self.device)
        
        # load model state dict
        checkpoint = torch.load(cfg.MODEL.CHECKPOINT, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        
        # load ViT pretrained
        vit_pretrained = np.load(cfg.MODEL.VIT_PRETRAINED)
        self.model.load_from(vit_pretrained)
        
        self.gt = GlobalTransform(cfg)
        self.lt = LocalTransform(cfg)
        
    def forward(self, x, a):
        x = x.to(self.device)
        a = a.to(self.device)
        
        gx = self.gt(x)



##### GLOBAL VARIABLES
LSIS = "faiss"
MODEL_PATH = "runs/FashionAI_s2/model_best.pth.tar"
VIT_PRETRAINED = "pretrained/resnet50-19c8e357.pth"



def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lsis", default=None, type=str, 
        choices=["kde", "lsh", "faiss"],
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
    
    return parser


def main(cfg):
    pass
     
    
if __name__ == "__main__":
    torch.set_num_threads(1)
    args = parse_args()
    cfg.merge_from_file('./config/FashionAI/FashionAI.yaml')
    cfg.merge_from_file('./config/FashionAI/s2.yaml')
    cfg.freeze()
    set_seed()
    
    main(cfg)