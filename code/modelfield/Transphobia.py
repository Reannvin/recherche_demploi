import numpy as np
import torch
from torch import nn
import math


# OK NOW WE HAVE TO CONSTRUCT OUR ATTENTION MODEL.
#
#            _______________
#           |               |                       
#     ______|    Output     |                           
#    |      | Probabilities |                               
#    |      |_______________|
#    |       ___     ___     ____
#    |      |   |   |   |   |    |                       S : Softmax
#    |______| S |___| L |___| AN |                       L : Linear
#           |___|   |___|   |____|                      AN : Add & Norm
#                             |                         FF : Feed Forward
#                            _|___
#        ___________        |_FF_|      
#      _|__        |         _|___
#     |_AN_|       |        |_AN_|
#      _|__        |        __|___
#     |_FF_|       |_______|_MHA__|
#      _|__                  _|__
#     |_AN_|                |_AN_|
#     __|___             ______|_____
#    |_MHA__|           |_Masked_MHA_|
#       |                     |
#  _____|_________      ______|_______
# |               |    |              |
# |  Positional   |    | Positional   |
# |  Encoding     |    | Encoding     |
# |_______________|    |______________| 
#      |                       |
#     _|_                     _|_
#    | I |                   | O |
#    |_E_|                   |_E_|
#    |_m_|                   |_m_|
#    |_b_|                   |_b_|
#    |_e_|                   |_e_|
#    |_d_|                   |_d_|
                          
#  Inputs                  Outputs
#                     (shifted right)

#-------------------------------1-------------------------------------------------
#                   
#   PE(pos,2i) = sin(pos/10000**(2i/dmodel))
# PE(pos,2i+1) = cos(pos/10000**(2i/dmodel))
#   
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        
        # Same size with input matrix
        self.encoding = torch.zeros(max_len, d_model, device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1) # 1D->2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, sted=2, device=device)
        # 'i' means index of d_model (e.g. embedding size = 50, i = [0,50])
        # step = 2 means 2*i

        self.encoding[:, 0::2] = torch.sin(pos/(10000)**(_2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos/(10000)**(_2i / d_model))

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 120, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it gonna add with tok_emb : [128, 30, 512]

#-------------------------------2-------------------------------------------------
#
#    ___                      ______________________
#   |_V_| -----> [Linear]--->|                      |
#    ___                     |   Scale Dot-Product  |
#   |_K_| -----> [Linear]--->|                      | -----> [Concat] ---> [Linear] --> Multi-head Attention
#    ___                     |      Attention       |
#   |_Q_| -----> [Linear]--->|______________________| x h
#
#   Multihead(Q,K,V) = Concat(head_1, head_2...)W_o
            # head_i = Attention(Q*Wi_q, K*Wi_k, V*Wi_v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_concat = nn.Linear(d_model,d_model)
    
    def spilt(self, tensor):
        # Spilt tensor by number of head
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1,2)
        return tensor

    def concat(self, tensor):
        # inverse function of self.split
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1,2).contiguous().view(batch_size, length, d_model)
        return tensor

    def forward(self, q, k, v, mask = None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v,mask=mask)

        # 4.concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

#-----------------------------------3-------------------------------------------------
#            ________     
#   Q ----> |        |                                                      _________
#           | MatMul | ---> [scale] ---> [Mask(opt).] ---> [SoftMax] ----> |         |
#   K ----> |________|                                                     | MatMul  |
#   V -------------------------------------------------------------------> |_________|
#
#
#                                    Q * K.transpose
#       Attention(Q,K,V) = softmax(-------------------) V
#                                       dk.sqrt()
#
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, q, k, v, mask = None, e = 1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2,3)
        score = ( q @ k_t ) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)
        v = score @ v
        return v, score