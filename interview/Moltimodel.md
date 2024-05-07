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