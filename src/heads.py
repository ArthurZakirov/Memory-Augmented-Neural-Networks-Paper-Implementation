import torch
from torch import nn
from torch.nn import Parameter
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

import numpy as np


class Head(nn.Module):
    """ Head : Superclass of ReadingHead and WritingHead
    
    Attributes
    ----------
    project_to_key : nn.Linear
    project_to_temperature : nn.Linear
    project_to_gate : nn.Linear
    project_to_shift : nn.Linear
    project_to_gamma : nn.Linear
    
    Methods
    -------
    project_to_variables(h) -> (k, ß, g, s, y)
    adjust_variables_for_attention(k, ß, g, s, y) -> (k, ß, g, s, y)
    
    """

    def __init__(self):
        super(Head, self).__init__()
        
        self.project_to_key = nn.Linear(d_hidden, d_hidden)
        self.project_to_temperature = nn.Linear(d_hidden, 1)
        self.project_to_gate = nn.Linear(d_hidden, 1)
        self.project_to_shift = nn.Linear(d_hidden, 3)
        self.project_to_gamma = nn.Linear(d_hidden, 1)
        
         
    def project_to_variables(self, h):
        """Create Parameters for Attention
        
        Arguments
        ---------
        h : torch.tensor [bs, d_hidden]
        
        Returns
        -------
        k : torch.tensor [bs, d_hidden]
        ß : torch.tensor [bs, 1]
        g : torch.tensor [bs, 1]
        s : torch.tensor [bs, 3]
        y : torch.tensor [bs, 1]
        
        """
        
        k = self.project_to_key(h)
        ß = self.project_to_temperature(h)
        g = self.project_to_gate(h)
        s = self.project_to_shift(h)
        y = self.project_to_gamma(h)
        
        return k, ß, g, s, y
    
    def adjust_variables_for_attention(self, k, ß, g, s, y):
        """Adjust variables to correct value ranges

        Arguments
        -------
        k : torch.tensor [bs, d_hidden]
        ß : torch.tensor [bs, 1]
        g : torch.tensor [bs, 1]
        s : torch.tensor [bs, 3]
        y : torch.tensor [bs, 1]
        
        Returns
        -------
        k : torch.tensor [bs, d_hidden]
        ß : torch.tensor [bs, 1]
        g : torch.tensor [bs, 1]
        s : torch.tensor [bs, 3]
        y : torch.tensor [bs, 1]
        
        """
        
        k = k.clone()
        ß = nn.ReLU()(ß)
        g = nn.Sigmoid()(g)
        y = torch.ones(1) + nn.ReLU()(y) 
        s = nn.Softmax(dim = 1)(s)
        return k, ß, g, s, y
        
    
    def forward(self, h):
        k, ß, g, s, y = self.project_to_variables(h)
        k, ß, g, s, y = self.adjust_variables_for_attention(k, ß, g, s, y)
        return k, ß, g, s, y


class ReadingHead(Head):
    """ReadingHead : Subclass of Head
    
    Attributes
    ----------
    project_to_key : nn.Linear
    project_to_temperature : nn.Linear
    project_to_gate : nn.Linear
    project_to_shift : nn.Linear
    project_to_gamma : nn.Linear
    
    Methods
    -------
    project_to_variables(h) -> (k, ß, g, s, y)
    adjust_variables_for_attention(k, ß, g, s, y) -> (k, ß, g, s, y)
    
    """
    
    def __init__(self):
        super(ReadingHead, self).__init__()


class WritingHead(Head):
    """WritingHead : Subclass of Head
    
    Attributes
    ----------
    project_to_key : nn.Linear
    project_to_temperature : nn.Linear
    project_to_gate : nn.Linear
    project_to_shift : nn.Linear
    project_to_gamma : nn.Linear
    
    project_to_add : nn.Linear
    project_to_erase : nn.Linear
    
    Methods
    -------
    project_to_variables(h) -> (k, ß, g, s, y)
    adjust_variables_for_attention(k, ß, g, s, y) -> (k, ß, g, s, y)
    variables_for_memory(h) -> (e,a)
    """
    
    
    def __init__(self):
        super(WritingHead, self).__init__()
        
        self.project_to_add = nn.Linear(d_hidden, d_hidden)
        self.project_to_erase = nn.Linear(d_hidden, d_hidden)
        
    def variables_for_memory(self, h):
        """Erase and Add for Memory writing
        Arguments
        ---------
        h : torch.tensor [bs, d_hidden]
        
        Returns
        -------
        a : torch.tensor [bs, d_hidden]
        e : torch.tensor [bs, d_hidden]
        """
        a = self.project_to_add(h)
        e = self.project_to_erase(h)
        return e,a