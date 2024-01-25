from unittest import result
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from modules.utils.metric import APScorer, AverageMeter


def do_infer(
    model,
    candidate_loader,
    gt,
    lt,
    attrs,
    device,
    logger,
    epoch=-1,
    beta=0.6
):
    model.eval()

    c_idxs, c_feats = extract_features(model, candidate_loader, gt, lt, device, len(attrs), beta=beta)
    for i, attr in enumerate(attrs):
        print(c_idxs[i].shape)
        print(c_feats[i].shape)

    return c_idxs, c_feats

def extract_features(model, data_loader, gt, lt, device, n_attrs, beta=0.6):
    feats = []
    indices = [[] for _ in range(n_attrs)]
    idxs = []

    with tqdm(total=len(data_loader)) as bar:
        cnt = 0
        for idx, batch in enumerate(data_loader):
            x, bidxs, a, v = batch
            a = a.to(device)
            
            out= process_batch(model, x, a, gt, lt, device, beta=beta)
            
            feats.append(out.cpu().numpy())

            for i in range(a.size(0)):
                indices[a[i].cpu().item()].append(cnt)
                cnt += 1

            idxs += bidxs

            bar.update(1)

    print(feats)
    feats = np.concatenate(feats)
    idxs = np.concatenate([idxs])

    feats = [feats[indices[i]] for i in range(n_attrs)]
    idxs = [idxs[indices[i]] for i in range(n_attrs)]

    return idxs, feats

def process_batch(model, x, a, gt, lt, device, beta=0.6):
    gx = torch.stack([gt(i) for i in x], dim=0)
    gx = gx.to(device)
    with torch.no_grad():
        g_feats, _, attmap = model(gx, a, level='global')

    attmap = attmap.cpu().numpy()

    lx = torch.stack([lt(i, mask) for i, mask in zip(x, attmap)], dim=0)
    lx = lx.to(device)
    with torch.no_grad():
        l_feats = model(lx, a, level='local')
    
    out = torch.cat((torch.sqrt(torch.tensor(beta)) * nn.functional.normalize(g_feats, p=2, dim=1),
            torch.sqrt(torch.tensor(1-beta)) * nn.functional.normalize(l_feats, p=2, dim=1)), dim=1)

    return out