import torch
from torch import nn

from src.controller import Controller
from src.heads import WritingHead, ReadingHead
from src.memory import Memory


class NeuralTuringMachine(nn.Module):
    """NTM : Total Model Infrastucture
    
    Attributes
    ----------
    controller
    writing_head
    reading_head
    memory
    
    Methods
    -------
    initialize_state_and_read(bs) -> (h, r)
    forward(x) -> (output)
    """
    
    def __init__(self, ft=10, d_hidden=16):
        super(NeuralTuringMachine, self).__init__()
        
        self.controller = Controller()
        self.writing_head = WritingHead()
        self.reading_head = ReadingHead()
        self.memory = Memory()

        self.ft = ft
        self.d_hidden = d_hidden
        
    def initialize_state_and_read(self, bs):
        """Initialize h and r for first timestep
        
        Arguments
        ---------
        bs : int
        
        Returns
        -------
        h : torch.tensor [bs, d_hidden]
            initial hidden state
            
        r : torch.tensor [bs, d_hidden]
            initial read
        """
        h_0 = torch.zeros(bs, self.d_hidden)
        r_0 = torch.zeros(bs, self.d_hidden)
        return h_0, r_0
      
    def forward(self, x):
        """Read in Sequence of length ht, output Sequence of length ft
        
        Arguments
        ---------
        ht : int
            History sequence timesteps
            
        ft : int
            Future sequence timesteps
            
        x : torch.tensor [bs, ht, d_input]
            Input sequence
            
        Returns
        -------
        output : torch.tensor [bs, ft, d_output]
            Output  sequence
        """
        bs = x.shape[0]
        r,h = self.initialize_state_and_read(bs)
        self.memory.init_w_previous(bs)
        self.memory.init_memory(bs)
        
        output = list()
        
        for t in range(ft):
            h = self.controller.next_state(x[:,t,:], h, r)
            
            k_r, ß_r, g_r, s_r, y_r = self.reading_head(h)
            k_w, ß_w, g_w, s_w, y_w = self.writing_head(h)
            e,a = self.writing_head.variables_for_memory(h)
            
            attention_r = self.memory.attention(k_r, ß_r, g_r, s_r, y_r)
            attention_w = self.memory.attention(k_w, ß_w, g_w, s_w, y_w)

            r = self.memory.read(attention_r)
            self.memory.write(attention_w, e, a)
            
            y = self.controller.output(h,r)
            output.append(y)
            
        output = torch.stack(output, dim = 1)    
        return output