import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(self, r:int, lora_alpha: int, lora_dropout: float, merge_weights:bool) -> None:
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional Dropout
        if lora_alpha > 0:
            self.lora_dropout = nn.Dropout(p = lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Marked the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(self, num_embeddings:int, embedding_dim:int, r:int = 0, lora_alpha:int = 1,merge_weights:bool = True,**kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((num_embeddings, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained wight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)
    
    def train(self, mode:bool=True):
        nn.Embedding.train(self,mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure the weight are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0,1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0,1) * self.scaling
                self.merged = True
                
