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

**Split document into chunks**

RAG需要把长文档切片成小一些的块(chunk)，从而能够更高效率的找到最相关的信息。在这个例子中，我们将文章每2048个字符就切做一块，最终切成37块。

```python
chunk_size = 2048
chunks = [text[i:i + chunk_size]] for i in range(0,len(text),chunk_size)
print(len(chunk)) #37
```

