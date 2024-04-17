# LORA技术笔记

##  0 x 00 Intro

LoRA (Low-Rank Adaptation) 指的是冻结预训练模型权重，并在每个Transformer块中注入可训练层（秩分解矩阵），通过在模型的Linear Layer旁边加两个模块，A和B。A将数据从d维降到r维，r是LoRA的秩。B将数据从r维升到d维，B部分的参数初始为0。

![img](https://img-blog.csdnimg.cn/direct/7badde72bcf84a619ecb279249253e12.png)

## 0 x 01 Pourquoi avons - nous besoin de Lora?

大模型的微调成本和部署成本极高，如果根据不同下游任务微调多个模型就会需要针对不同的用户微调不同的模型，成本高。

## 0 x 02 À propos de la résolution de Lora

- 重参数
  - 也就是结构重参数化，先构造原始网络结构，在将其权重参数等价转换为另一组参数。
- 本征维度
  - 指的是一个数据集的有效维度或者说高效率维度的数量，即可以用最少的维度来表达数据集最多的信息。确定一个数据集的本征维度一般使用主成成分分析，独立成分分析，多维分析