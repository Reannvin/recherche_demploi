# 多模态大模型 

## 基本

​	首先是基本梳理，这里的“多模态”更多指的是“图文多模态”，需要有一定的视觉表征和多模态融合的相关知识。对于图文多模态，最重要的还是得从 **视觉表征** 与 **视觉与自然语言对齐** 着手。

- 视觉表征：
  - 如何建模视觉输入特征。
  - 如何通过预训练进行充分学习表征。
- 视觉与自然语言对齐：
  - 将视觉和自然语言建模在同一表征空间并进行融合，实现自然语言和视觉语义的互通。这点也是多模态大模型的前提。Alignment! Alignment! Alignment!

视觉表征可以从CNN与VIT两大脉络着手，两者分别有自己的表征，与训练以及多模态对齐的发展路线。

因此全文分为两大部分。

1. 以CNN为基础的视觉表征和预训练手段，多模态对齐的方法。
2. VIT视觉表征的预训练探索工作，多模态对齐的预训练工作。

- CNN
  - 视觉表征预训练
    - 基于ResNet进行下游业务微调
  - 多模态对齐预训练
    - ViLBERT, UNITER等（Region Feature）
    - Pixel-BERT（Grid Feature）
- VIT
  - 视觉表征预训练
    - BEIT
    - MAE
  - 多模态对齐预训练
    - CLIP
    - VILT
    - VL-BEIT
      - VLMO - BEIT-3
    - ALBEF
    - COCA
  - 多模态与大模型
    - Flamingo
    - BLIP
    - LLaVA
    - MiniGPT - 4
    - Qwen-VL
    - VILA
    - Gemini
    - LWM

## 正文 1 - CNN

### 1.1 卷积视觉信息的表征和预训练

​	简单来说就是将图像信息转换成深度学习需要的特征向量（序列），CNN凭借 **局部链接区域** 、 **权重共享** 和 **平移不变形** 等特点，天然的符合图像信息的建模归纳假设，成为早期最适合视觉表征的模型。从LeNet-5, AlexNet, VGG 和 ResNet等模型的演进可以一探究竟。

​	LeNet-5的结构是[CONV - POOL - CONV -POOL - FC -FC]。卷积层使用5 * 5的卷积核，步长为1；池化层使用2 * 2的卷积核，步长为2；后面接两个全连接层。AlexNet调整了参数加了层数，VGG用了更小的卷积核。

​	当神经网络的层数爆发性增长后，梯度消失和梯度爆炸的问题由为严重，这使得模型训练难度更上一层楼。ResNet使用Residual Network残差网络解决了这一问题。

### 1.2 卷积视觉预训练

​	早期CNN视觉表征下的预训练更多被叫为迁移学习，早在BERT预训练+微调前就以及已经流行。在CNN负责具体任务之前现在ImageNet上做预训练，再用预训练的CNN权重初始化Backbone再增加一些任务定制网络模块，完成下游任务的微调。

​	“是的，用Backbone加上全连接层就是分类任务。”

### 2 早期多模态融合与预训练

​	CNN体系下的多模态融合和预训练，视觉与语言的跨模态对齐与融合有两种表现形式。

- 双塔结构：多模态分别表征，通过对比学习机制实现视觉与文本在同一空间中的距离度量。
- 表征融合：视觉表征与文本表征通过网络结构融合成多模态表征，进而完成下游任务应用。

​	“某种程度来说双塔结构是表征融合的特例。”

​	也就是先使用CNN Backbone提取视觉特征，再和语言特征融合。

​	2019年之后，BERT开始流行，多模态更多的开始借鉴BERT的成功，使用Transformer Encoder来作为Modality Interaction模块来融合视觉和语言特征，进而通过大规模与训练来学习多模态表征。

​	“显而易见，这个方案把以前的堆叠全连接网络替代了。”

​	**如何对视觉特征进行有效编码，得到和文本一样的Token Embedding序列作为模型输入？**

- CNN时期
  - Region Feature Base : 先通过基于CNN的目标检测模型(Fast-RCNN之类的)识别图像中的ROI，再提取ROI中的表征向量作为**视觉输入Embedding序列**。好处是能让每个人ROI都有明确的语义表达，方便后续和文本特征对齐。比如LXMERT, VL-BERT和UNITER。
  - Grid Feature Base: 上述方法尽管合理，但还是很依赖前置的目标检测模型。难道你不觉得整个链路过于繁重了吗？不经过区域检测，直接用CNN网络提取深层的像素特征作为交互模型输入也是一种方法。Pixel-Bert。
  - 这两个方法的区别就是一个先用CNN Backbone提取ROI Feature，一个直接用CNN Backbone。
    - LXMERT的网络结构如下，使用**两路深层表征输入结构**。在视觉上，图像经过目标检测得到区域块的特征序列，又经过Transformer做进一步编码区域块之间的关系（Object-Relationship Encoder）。文本侧则是通过BERT结构得到文本特征序列。二者使用Transformer 做交叉Attention来进行多任务的预训练。LXMERT的预训练任务包括Masked图像特征预测，Label预测，VQA，图文匹配程度。![img](https://pic2.zhimg.com/v2-a4649f0b1e2aaa1757b6b146ebf547b1_r.jpg)
    - VL-BERT跟LXMERT最大的区别就是VL-BERT是单路输入模式，视觉Region特征被提取后直接和文本Embedding一起拼接输入Transformer进行多模态的交叉Attention。![img](https://pic1.zhimg.com/80/v2-07d298dcc2f4447162b5b41e76d72ac4_720w.webp)VL-BERT的预训练任务包括两个，带视觉特征的**掩码**语言学习模型 和 带文本特征的视觉**Region**分类
