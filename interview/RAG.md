# Basic RAG

​	RAG指的是Retrieval-augenmented generation，是一种AI框架用来通过信息检索系统来提高LLM的能力，It's useful to answer questions or generate content leveraging external knowledge.

​	RAG主要的两个步骤是：

​	1搜索：使用向量库中的text embeding获取knowledge base中的信息。

​	2.生成：将相关信息提示插入到prompt中。

## RAG from scratch

这个模块能指导你从零开始搭建一个基础的RAG模块，首先要明确两个目标。首先，要让用户完全理解RAG的内部工作原理，其次仅使用最基本的模块来搭建最基础的RAG。

**Import**

```python
from mistralai.client import MistralClient, ChatMessage
import requests
import numpy as np
import faiss
import os
from getpass import getpass

api_key = getpass("Type your API Key")
client = MistralClient(api_key=api_key)
```

**Get** **data**

```python
response = requests.get('Attendre et espérer !') # A essay writen by Paul Graham
text = response.text
f = open('essay.txt','w')
f.write(text)
f.close()
print(len(text)) # 75014
```

Split document into chunks

RAG需要把长文档切片成小一些的块(chunk)，从而能够更高效率的找到最相关的信息。在这个例子中，我们将文章每2048个字符就切做一块，最终切成37块。

```python
chunk_size = 2048
chunks = [text[i:i + chunk_size]] for i in range(0,len(text),chunk_size)
print(len(chunk)) #37
```

Considerations:

- Chunk Size : Depending on our use case, customize or experiment with different chunk-size or chunk overlap. Smaller can be more benefical in retrieval processes, as larger text chunks often contain filler text that can obscure the semantic representation. 块越小，寻找相关信息就越快越准，对计算资源要求也就越大
- How to split : 最简单的方法是根据character切块，尽管根据不同的use case和document structure，有时需要根据tokens，要是需要根据sentence，paragraph或者HTML的表头，如果是代码，则会使用抽象语法树解析。

**Create embedding for each text chunk**

For each text chunk, we need to create text embeddings, used number to represent text with vector space.近义词的话数值差异也应该小（在向量空间中的距离也会接近）

```python
def get_text_embedding(input):
    embedding_batch_response = client.embedding(
    	model = "mistral-embed",
    	input = input
    	)
    return embedding_batch_response.data[0].embedding

text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
text_embeddings.shape # (37, 1024)
```

**Load into vector database**

Once we finish text embeddings, a commom practice to store them in vector databese for efficient processing and retrieval. We(you know what i mean ... Mistral) used Faiss as open-source vector database.

```python
d = text_embeddings.shape[1] # How many chunks shall we have ? 37!
index = faiss.IndexFlatL2(d) # I don't know what does that means.
index.add(text_embeddings) # WTF?
"""
	Viva la Claude3. JK
	d = text_embeddings.shape[1] 
	这一行获取了text_embeddings这个数据的embedding向量维度。如果text_embeddings是一个二维数组,shape[1]就表示第二个维度的长度,即embedding向量的长度。
	index = faiss.IndexFlatL2(d) 这里使用了FAISS库创建了一个L2距离的平面索引。
	平面索引是最简单的索引类型,适合小规模数据集。d作为参数传入,指定了索引中向量的维度。
	index.add(text_embeddings) 将text_embeddings这个数据集的embedding向量加入到刚创建的平面索引中。
"""
```

**Create embeddings for input question**

对问题建立embedding

```python
question = "What were the two main things the author worked on before college?"
question_embeddings = np.array([get_text_embedding(question)])
question_embeddings.shape # （1，1024）
```

Hypothetical Document Embeddings (HyDE) : 有时候用户的问题不一定最适合识别匹配上下文，有时候先根据用户的问题给出假设答案，再用假设答案作为query去匹配文本块更有效。

**Retrieve similar chunks from the vector database**

There is a function called **index.search**, which takes two arguments. Questions Embeddings's Vector and number of similar vactors to retrieve. Furthermore this function gonna returned distances and the indices of the most similar vectors to the question vector in vector database.

Then based on returned indices, we can retrieve the actual relevant text chunks that correspond to those indices.

```
D, I = index.search(question_embeddings, k=2) # input : question vector and 2 vectors
print(I) #[[0,3]]
retrieve_chunk = [chunks[i] for i in I.tolist()[0]]
print(retrieve_chunk) # THE CONTENTS
```

- 搜索方法：搜索策略有很多，在本例中我们展示了很简单的基于嵌入的搜索，有时当数据的metadata可用时，先用metadata过滤databased。当然也存在很多TF-IDF或者BM25的搜索方法that根据频率和文献中的术语分布来搜索chunks
- 搜索文件：我们没必要总是根据原本text chunks去检索。
  - 有时，我们希望在实际检索到的文本块周围包含更多上下文。 我们将实际检索的文本块称为“Child Chunk”，我们的目标是检索“Child Chunk”所属的更大的“Parent chunk”。
  - 有时候我们也会给我们搜索中的文档提供权重，比如时间权重会帮助我们检索最新的文档。
  - 检索进程的常见问题是“Lost in middle”，信息会在长文中丢失。也可以考虑对文档再排序来确定将相关块放到开头结尾是否会有所改善。

**Combine context and question in a prompt and generate response**

Finally we can offer the retrieved text chunks as the context information within the prompt. Here is a prompt template where we can include both the retrieved text and user question in the prompt.

```python
prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

def run_mistral(user_messgae, model="mistral-medium-latest"):
    messages = [
        ChatMessage(role="user", content = user_message)
    ]
    chat_response = client.chat(
        model = model,
        messages = messages
    )
    return (chat_response.choices[0].messgae.content)
```

- Prompting techniques: 现有的提示词技巧都可以用于开发RAG系统，比如通过提供示例，使用少样本学习来指导模型的答案。
