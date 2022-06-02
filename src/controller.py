import torch
from torch import nn

class Controller(nn.Module):
    
    def __init__(self):
        """Controller : Recurrent Net Cell
        
        Attributes
        ----------
        cell : nn.GRUCell
        output_embedder : nn.Linear
        
        Methods
        -------
        next_state(x_t, h, r) -> (h)
        output(h, r) -> (y)
        
        """
        super(Controller, self).__init__()
        
        self.cell = nn.GRUCell(2 * d_hidden, d_hidden)
        self.output_embedder = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.Softmax(-1))
            
        
    def next_state(self, x_t, h, r):
        """Concat input und read, return hidden state
        
        Arguments
        ---------
        x_t : torch.tensor [bs, d_hidden]
            Input timesteps t of sequence
        
        h : torch.tensor [bs, d_hidden]
            Hidden State
            
        r : torch.tensor [bs, d_hidden]
            Read 
            
        Returns
        -------
        h : torch.tensor [bs, d_hidden]
            Hidden State  
        """
        
        xr = torch.cat([x_t, r], dim = -1)
        h = self.cell(xr, h)
        return h
        
    def output(self, h, r):
        """Concat input und read, return hidden state
        
        Arguments
        ---------
        h : torch.tensor [bs, d_hidden]
            Hidden State
            
        r : torch.tensor [bs, d_hidden]
            Read 
            
        Returns
        -------
        y : torch.tensor [bs, d_output]
            Output, passed trough Activation
        """
        
        hr = torch.cat([h,r], dim = -1)
        y = self.output_embedder(hr)
        return y