import torch
from torch import nn
from torch.nn import Parameter
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

import numpy as np

def batch_to_one_hot(batch_cat_id, num_cats):
    """
    Arguments
    ---------
    batch_cat_id : torch.tensor [bs, seq_len, 1]
    
    Returns
    -------
    batch_cat_OH : torch.tensor [bs, seq_len, num_cats]
    
    """
    cat_samples = batch_cat_id.chunk(len(batch_cat_id), dim = 0)
    batch_cat_OH = list()
    for cat_sample in cat_samples:
        cat_id = cat_sample.squeeze()
        cat_OH = torch.zeros(len(cat_id), num_cats)
        cat_OH[torch.arange(len(cat_id)), cat_id] = 1
        batch_cat_OH.append(cat_OH)

    return torch.stack(batch_cat_OH, dim = 0)


def CESequenceLoss(p_y_x, y):
    """Cross Entropy Loss for Sequential Data
    
    Arguments
    ---------
    p_y_x : torch.tensor [bs, t, d]
        Probability vector for all categories with values [0, 1]
        
    y : torch.tensor [bs, t, d]
        One Hot Encoded ground truth labels
    
    Returns
    -------
    E : torch.tensor []
    """
    log_p_y_x = nn.LogSoftmax(-1)(log_p_y_x)
    log_p_y_x = torch.log(p_y_x).clamp(min = -100)
    E_i_t = - (y * log_p_y_x).sum(dim = 2)
    E_i = E_i_t.sum(dim = 1)
    E = E_i.mean(dim = 0)
    return E