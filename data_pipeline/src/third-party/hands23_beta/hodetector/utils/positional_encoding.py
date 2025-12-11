import numpy as np
import torch

def calculate_center(bb):
    return torch.tensor([(bb[0] + bb[2])/2, (bb[1] + bb[3])/2])

def get_diag(bb):
    v = torch.tensor([(bb[2]-bb[0]), (bb[3]-bb[1])])
    return torch.linalg.norm(v)
  

def get_position_encoding(bb1, bb2, L=10):
    c1 = calculate_center(bb1)
    c2 = calculate_center(bb2)
    diag = get_diag(bb1)
    v = (c2-c1)/diag
    pos = []
    for x in v:
        for j in range(L):
            pos.append(torch.sin((2**j)*np.pi*x))
            pos.append(torch.cos((2**j)*np.pi*x))
    pos.append(torch.linalg.norm(v))
    return torch.stack(pos).unsqueeze(0).cuda()