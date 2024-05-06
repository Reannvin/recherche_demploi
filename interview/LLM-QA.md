# LLM - Q & A

- 什么是transformer
  - transformer的重点在Q,K,V结构上，相对于CNN用卷积表达不同位置数值关系，RNN用反馈表达前后关系，transformer用Query - 问题，Value - 答案和Key - 辅助来表达。


- 为什么现在大模型都是Decoder-only结构
  - 与encoder-only相比，既能理解也能生成，预训练数据量和参数量上去后decoder-only的zero-shot泛化能力很好，而BERT这样的encoder-only模型一般还需要少量下游标注数据来fine-tune。
  - 但是，上述回答并不全面，即T5作为encoder-decoder架构，也没提及GLM和XLNET。
  - 首先需要明白集中主要架构，BERT是encoder-only，T5和BART是encoder-decoder,而GPT则是decoder-only，甚至以UNILM（更改了attention mask的GPT）。
  - BERT作为encoder-only使用的masked language modeling预训练，不擅长做生成任务，做NLU一般需要有监督的下游数据微调。而decoder-only的模型使用next token prediction预训练兼顾了理解和生成，在zero-shot和few-shot中泛化性极佳。
  - 实验结果上说，decoder-only的泛化性更好。
  - 注意力满秩问题：双向attention的注意力矩阵容易退化为低秩状态，而causal的注意力矩阵是下三角矩阵，必然满秩且具有更强的建模能力。
  - 预训练任务难度问题：纯粹的decoder-only架构+next token prediction与训练每个位置能接触的信息比其他架构少，预测下一个token难度更高。当模型足够大，数据足够多时decoder-only模型通用表征的上限更高。
  - 上下文学习为decoder-only架构带来的更好的few-shot性能，prompt和demonstration的信息可以视为对模型的隐式微调，decoder-only的架构相比encoder-decoder在in-context learning上会更有优势，因为prompt可以更加直接作用于decoder每一层的参数，微调的信号更强。
  - causal attention（decoder-only的单向attention）具有隐式的位置编码功能，打破了transformer的位置不变性，带有双向attention的模型如果没有位置编码，双向attention的部分token可以对换不改变表示，对语序的区分能力天生较弱。
  - decoder-only支持复用KV-Cache，对多轮对话更友好，每个token的表示只和它之前的输入有关，而encoder-decoder和PrefixLM就难以做到。
  - OpenAI作为开拓者，以decoder-only架构为基础摸索出一套行之有效的训练方法和Scaling Law，形成先发优势后Megatron和flash attention对causal attention的支持更好。

