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

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
        
class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

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
    
class LayerNorm(nn.Module):
    #            x - E(x)
    #   y = -------------------  *  gemma + beta
    #       [Var(s) + eps].sqrt
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gemma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbias = False, keepdim = True)
        # -1 means last dimension
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gemma * out + self.beta
        return out
#   
#   FF - Layer
#
#   FFN(s) = max(0,xW1 + b1)W2 + b2
#   
#   input is [batchsize, m, 512]
#   Output is [batchsize, m, 512]
#
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear1 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p = drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model = d_model,
                                        max_len = max_len,
                                        vocab_size = enc_voc_size,
                                        drop_prob = drop_prob,
                                        device = device)
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model,
                                                  ffn_hidden = ffn_hidden,
                                                  n_head = n_head,
                                                  drop_prob = drop_prob)
                                     for _ in range(n_layers)])
    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x

class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):    
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output
    

        