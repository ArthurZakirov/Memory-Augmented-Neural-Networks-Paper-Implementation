import torch
from torch import nn


class Memory(nn.Module):
    """
    Attributes
    ----------
    N : int []
        Number of Memory states
        
    M : torch.tensor [bs, N, d_hidden]
        Memory Tensor
        
    w_previous : torch.tensor [bs, N]
        Attention weights of the previous timestep
        
    Methods
    -------
    attention(self, k, ß, g, s, y) -> (w)        
    read(w) -> (r)
    write(w,e,a) -> void
    
    """
    
    def __init__(self):
        super(Memory, self).__init__()
        
        self.N = N
        
        self.w_previous = nn.Softmax(-1)(torch.randn((bs, N)))
        
        self.init_w_previous_buffer()
        self.init_memory_buffer()
        
    def init_w_previous_buffer(self):
        """Create Attention buffer
        
        Arguments
        ---------
        N : int
            Number of Memory states
            
        Returns
        -------
        w_previous_buffer : torch.tensor [N]
        
        """
        self.register_buffer("w_previous_buffer", nn.Softmax(-1)(torch.Tensor(self.N)))
        
    def init_w_previous(self, bs):
        """Create Attention buffer
        
        Arguments
        ---------
        bs : int
        w_previous_buffer : torch.tensor [N]
            
        Returns
        -------
        w_previous : torch.tensor [bs, N]
        
        """
        self.w_previous = self.w_previous_buffer.clone().repeat(bs, 1)
        
    def init_memory_buffer(self):
        """Create Memory buffer
        
        Arguments
        ---------
        N : int
            Number of Memory states
            
        d_hidden : int
            Dimension of Memory states
            
        Returns
        -------
        M_init : torch.tensor [N, d_hidden]
        
        """
        self.register_buffer('M_init', torch.Tensor(N, d_hidden))
        nn.init.uniform_(self.M_init, 0, 1)
        
    def init_memory(self, bs):
        """Expand Memory Buffer to batchsize for first timestep in Training
        
        Arguments
        ---------
        bs : int
        M_init : torch.tensor [N, d_hidden]
        
        Returns
        -------
        M : torch.tensor [bs, N, d_hidden]
        """
        self.M = self.M_init.clone().repeat(bs, 1, 1)
    
    def attention_content_focus(self, k, ß):
        """
        Arguments
        ---------
        k : torch.tensor [bs, d_hidden]
            Key, (technically it acts more as the query in this case)

        M : torch.tensor [bs, N, d_hidden]
            Memory, (technically it is the keys and values)

        Returns
        -------
        w : torch.tensor [bs, N]


                       < k * M[i] >
        w_i = Softmax(-------------- * ß ) 
                       |k| * |M[i]|

        """

        dot_product = torch.bmm(self.M, k.unsqueeze(2)).squeeze(-1)
        M_norm = torch.linalg.norm(self.M, dim = -1)
        k_norm = torch.linalg.norm(k, dim = - 1).unsqueeze(-1)
        mul_norms = M_norm * k_norm

        alignment = dot_product / mul_norms
        w = nn.Softmax(dim = 1)(alignment * ß)

        return w

    def attention_location_focus(self, w_current, g):
        """
        Arguments
        ---------
        w : torch.tensor [bs, N]
        w_previous : torch.tensor [bs, N]
        g : torch.tensor [bs, 1]
        
        Returns
        -------
        w_g : torch.tensor [bs, N]
            
        """
        
        w_g = g * w_current + (torch.tensor(1.) - g) * self.w_previous
        return w_g

    def attention_convolution(self, w, s):
        """
        Arguments
        ---------
        w : torch.tensor [bs, N]
        s : torch.tensor [bs, 3]
            the indices of s are [-1, 0, 1]

        Returns
        -------
        w_shifted : torch.tensor [bs, N]

        """
        w_d = w[:,-1].unsqueeze(1)
        w_0 = w[:,0].unsqueeze(1)
        w_cycle = torch.cat([w_d, w, w_0], dim = -1).unsqueeze(1) # [bs, 1, N+2]
        s = s.flip(dims = (1,)).unsqueeze(2) # [bs, 3, 1]

        max_first_idx = w_cycle.shape[2] - 3
        w_shifted = torch.cat([torch.bmm(w_cycle[:,:,i:i+3],s) 
                               for i in range(max_first_idx + 1)], 
                              dim = -1).squeeze(1)

        return w_shifted


    def attention_sharpen(self, w, y):
        """Apply Temperature to shifted weights
        
        Arguments
        ---------
        w : torch.tensor [bs, N]
        
        Returns
        -------
        w : torch.tensor [bs, N]
    
        """
        nominator = (w ** y)
        denominator = nominator.sum(dim = 1).unsqueeze(1)
        
        w = nominator / denominator
        return w

    def attention(self, k, ß, g, s, y):
        """
        Arguments
        ---------
        k : torch.tensor [bs, d_hidden]
            Key, (technically it acts more as the query in this case)

        M : torch.tensor [N, d_hidden]
            Memory, (technically it is the keys and values)

        Returns
        -------
        w : torch.tensor [bs, N]

        """
        w = self.attention_content_focus(k,ß)
        w = self.attention_location_focus(w, g)
        w = self.attention_convolution(w, s)
        w = self.attention_sharpen(w, y)
        
        self.w_previous = w
        return w
    
    def read(self, w):
        """
        Arguments
        ---------
        w : torch.tensor [bs, N]

        M : torch.tensor [bs, N, d_hidden]
            Memory, (technically it is the keys and values)


        Returns
        -------
        r : torch.tensor [bs, d_hidden]

        """

        r = torch.bmm(w.unsqueeze(1), self.M).squeeze(1)
        return r
    
    
    def write(self, w, e, a):
        """Create Memory Matrix for next timestep

        Arguments
        ---------
        M : torch.tensor [bs, N, d_hidden]
        w : torch.tensor [bs, N]
        e : torch.tensor [bs, d_hidden]
        a : torch.tensor [bs, d_hidden]

        Returns
        -------
        M : torch.tensor [bs, N, d_hidden]

            Mt[i] = Mt-1[i] * (I - w[i] * diag(e)) + w[i] * a

        """
        bs = w.shape[0]
        M_next = list()
        
        for i in range(N):
            w_i = w[:,i].reshape(bs, 1, 1)         # [bs, 1, 1]
            I = torch.eye(d_hidden).repeat(bs,1,1) # [bs, d, d]
            e_diag = torch.diag_embed(e)           # [bs, d, d]
            M_i = self.M[:,i,:].unsqueeze(1)       # [bs, 1, d]
            
            M_i = torch.bmm(M_i, I - w_i * e_diag) + w_i * a.unsqueeze(1) 
            M_next.append(M_i)
            
        M_next = torch.cat(M_next, dim = 1)
        self.M = M_next