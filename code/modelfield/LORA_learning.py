import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
# LoRA(Low-Rank Adaptation)是一种高效的模型微调方法,主要用于大型预训练模型的特定领域或任务适应。
# 它的基本思想是在预训练模型的参数基础上,引入一组低秩的可训练矩阵,以捕获特定任务或领域的知识,从而避免了完全重新训练模型的巨大计算开销。
# LoRA的核心公式如下:

# 1. 对于线性层(如全连接层):
# y = xW + b
# 其中,x是输入,W是权重矩阵,b是偏置项。
# LoRA通过引入两个低秩矩阵A和B,对权重矩阵W进行修正:
# W = W + BA
# 其中,A和B是可训练的低秩矩阵,它们的秩远小于W的秩。

# 2. 对于其他层(如卷积层、Self-Attention等):
# W = W dot (I + BA)
# 其中,dot表示元素级相乘,I是单位矩阵。
# 通过这种方式,LoRA只需要训练低秩矩阵A和B,从而大大降低了微调所需的内存和计算量。同时,由于A和B的秩较低,它们所引入的参数量也很小,因此不会对原始模型产生较大扰动。
# 在推理阶段,LoRA直接利用修正后的权重矩阵$\tilde{W}$进行计算,无需保存A和B,从而也避免了额外的存储开销。
# 总的来说,LoRA通过在原始模型基础上引入少量的低秩修正项,以高效、内存友好的方式实现了模型的微调和适应。
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
                
