# LangChain 核心组件速查

## LangChain 是什么

LangChain 是开发大语言模型应用的开源框架,提供文档加载、文本切分、向量检索、提示词模板、模型调用、链式编排等一整套组件。主要语言是 Python 和 JavaScript。

## 核心组件

### 文档加载器(Document Loader)

把各种格式的文件读成统一的 Document 对象(包含 page_content 文本和 metadata 元数据)。常用:PyPDFLoader(PDF)、TextLoader(纯文本)、WebBaseLoader(网页)。

### 文本切分器(Text Splitter)

RecursiveCharacterTextSplitter 是最常用的切分器:按分隔符列表从大到小递归切分(先按段落、再按句子、最后按字符),尽量保持语义完整。

### 向量模型(Embeddings)

把文本转成向量。LangChain 中统一接口是 embed_documents(批量文本)和 embed_query(单个查询)。智谱提供 embedding-2 和 embedding-3 两个模型。

### 向量数据库(Vector Store)

存储向量并支持相似度检索。LangChain 支持几十种,轻量本地场景常用 Chroma;生产大规模场景常用 Milvus、Pinecone。

### 检索器(Retriever)

向量库的统一检索接口,输入字符串、输出相关 Document 列表,支持配置 search_kwargs 里的 k(返回条数)和 filter(元数据过滤)。

### 提示词模板(Prompt Template)

ChatPromptTemplate 用消息列表组织提示词,支持 system / human / ai 三种角色,以及 MessagesPlaceholder 动态插入对话历史。

### 模型(Chat Model)

ChatZhipuAI 对接智谱 GLM 系列:glm-4-flash 免费、速度快;glm-4-plus 能力更强。temperature 控制随机性,问答场景建议 0~0.3。

### 输出解析器(Output Parser)

StrOutputParser 把模型返回的消息对象转成纯字符串,是最简单的解析器。

## LCEL(LangChain Expression Language)

LCEL 用管道符 | 把组件串成链:`chain = prompt | llm | parser`。支持 invoke(单次调用)、stream(流式输出)、batch(批量调用),并且天然支持异步。数据流是多步链:上一步的输出作为下一步的输入;用 RunnableParallel 可以同时执行多个分支。

## 本项目用到的编排方式

本项目(基于智谱 GLM 的知识库问答)的链路:
1. retriever 检索相关 chunk
2. RunnableLambda 把 chunk 格式化成带来源的参考资料文本
3. ChatPromptTemplate 组装系统提示 + 问题
4. ChatZhipuAI 生成回答
5. StrOutputParser 输出字符串

多轮对话时,app.py 维护 chat_history 列表,每轮把"对话历史 + 当前问题"合并后送入检索,解决指代不清的问题。
