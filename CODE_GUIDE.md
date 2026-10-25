# 代码工程详解(CODE_GUIDE)

> 本文档面向**想读懂、维护和改造这份代码**的读者。README 是使用手册(怎么跑起来),本文是工程文档(为什么这么写、每一行在做什么、踩过哪些坑)。建议对照源码逐节阅读。
>
> 全部核心代码 513 行(含详细注释),另有 tests/ 下 7 个单元测试。阅读顺序建议:config → loaders → vectorstore → chain → cli。

---

## 目录

- [0. 先读我:LangChain 到底扮演了什么角色](#0-先读我langchain-到底扮演了什么角色)
- [1. 总体架构](#1-总体架构)
- [2. 数据结构一览](#2-数据结构一览)
- [3. 逐模块详解](#3-逐模块详解)
  - [3.1 rag/config.py — 配置中心](#31-ragconfigpy--配置中心)
  - [3.2 rag/loaders.py — 加载与切分](#32-ragloaderspy--加载与切分)
  - [3.3 rag/vectorstore.py — 向量库封装](#33-ragvectorstorepy--向量库封装)
  - [3.4 rag/chain.py — RAG 链路(核心)](#34-ragchainpy--rag-链路核心)
  - [3.5 rag/cli.py — 命令行入口](#35-ragclipy--命令行入口)
  - [3.6 rag/cli.py 的 chat() — 问答 REPL](#36-ragclipy-的-chat--问答-repl)
- [4. 设计决策记录](#4-设计决策记录)
- [5. 踩坑复盘(真实 Bug 档案)](#5-踩坑复盘真实-bug-档案)
- [6. 一次提问的完整调用时序](#6-一次提问的完整调用时序)
- [7. 常见改造点速查](#7-常见改造点速查)

---

## 0. 先读我:LangChain 到底扮演了什么角色

初学 RAG 时最容易困惑的问题:"检索是 Chroma 做的、向量是智谱算的、回答是 GLM 生成的——**那 LangChain 到底干了什么?**"

一句话回答:**LangChain 什么"脏活"都不干,它是介于你的代码和厂商 SDK 之间的"适配 + 编排"胶水层。** 看清分层就明白了:

```
┌─────────────────────────────────────────────┐
│ 你的代码(rag/ 包)                           │  ← 业务逻辑:你的规则、你的链路设计
├─────────────────────────────────────────────┤
│ LangChain(适配层 + 编排层)                  │  ← 本节主角:统一接口、统一数据结构、LCEL 编排
├─────────────────────────────────────────────┤
│ 厂商 SDK:zhipuai、chromadb、pypdf           │  ← 真正的"工人":调 HTTP API、存数据、算相似度
├─────────────────────────────────────────────┤
│ 外部服务:智谱 API(embedding / chat)         │  ← 真正的"算力":向量化、生成回答
└─────────────────────────────────────────────┘
```

**真正干活的从来不是 LangChain**:向量化是智谱服务器算的,数据是 Chroma 存的,回答是 GLM 生成的。LangChain 提供的是三样东西:

### ① 统一数据结构——让各环节"说同一种语言"

`Document`、`Message`、chunk 在整个链路里格式不变,所以加载器换格式、切分器换算法、向量库换产品,上下游代码都不用动。没有这层,你自己就得发明一个 dict 约定并在每个环节手动转换(对照实验里 `metadatas=[{"source": ...}] * len(chunks)` 就是手动维护的缩影)。

### ② 统一接口——"换厂商不改代码"的来源

`ChatZhipuAI` 和 `ChatOpenAI` 都是 `.invoke()`,`ZhipuAIEmbeddings` 和 `HuggingFaceEmbeddings` 都是 `embed_documents()`。这是**适配器模式**的批量应用:每个厂商一个薄包装,把各家五花八门的 SDK 参数翻译成同一套方法签名。项目里"换模型只改 chain.py 两行"的承诺,兑现靠的就是这层。

### ③ LCEL 编排——把"过程式步骤"变成"声明式管道"

`itemgetter("question") | retriever | prompt | llm | parser` 本质是把一串函数调用写成管道。它送你三样手写很难白拿的东西:`stream()`(流式)、`batch()`(并发批量)、`ainvoke()`(异步)——**同一个链,零改动获得全部能力**。手写版本想加流式,得把每一步都改造成生成器。

### 诚实的成本核算:哪里值,哪里只是薄包装

| 项目里的组件 | LangChain 提供了什么 | 诚实评价 |
|--------------|---------------------|----------|
| RecursiveCharacterTextSplitter | 现成的递归切分算法 | **真赚**:自己写要处理边界、overlap、递归降级 |
| ChatPromptTemplate / MessagesPlaceholder | 模板渲染、变量校验、历史槽位 | **真赚**:手拼 f-string 没有校验,变量拼错静默出错 |
| Document / LCEL / Retriever | 数据标准、编排、流式 | **真赚**:生态里的教程、组件全都基于这套词汇 |
| ZhipuAIEmbeddings / ChatZhipuAI | 厂商 SDK 的薄包装 | **很薄**:约等于直接调 zhipuai,价值主要在"接口统一" |
| langchain-chroma 的 Chroma | chromadb 的薄包装 | **很薄**:同上 |

这也是社区对 LangChain 的经典争议:如果项目**永远**只用一家模型一个向量库、永远不需要流式和批量,那胶水层的收益确实有限。但反过来,只要"可能换组件、想要流式、想接入生态",这层抽象就开始回本——而学习它的接口模型,能让你看懂市面上几乎所有的 RAG 教程和开源项目。

### 最好的理解方式:看"没有它"的版本

`examples/without_langchain.py` 是一个**一行 LangChain 都不用**的等价实现(裸 zhipuai SDK + 裸 chromadb 客户端,约 90 行),功能与 app.py 单轮问答相同。对照着读,你会发现:

- 它**完全可行**——LangChain 没有做任何"不做就做不到"的事,这就是"胶水层"的含义;
- 切分算法是手写的滑窗(没有递归边界感知)、提示词是手拼字符串(没有变量校验)、取回答要剥三层响应结构、每换一家厂商所有调用代码重写;
- 而 LangChain 版把这些各自收敛到一个组件里,你的代码只表达"业务决策"(用什么参数、什么规则、什么链路)。

跑对照实验:`uv run python examples/without_langchain.py "你的问题"`(它写入独立的 raw_demo 集合,不污染正式向量库)。

---

## 1. 总体架构

### 模块依赖关系(从上到下依赖,箭头指向被依赖方)

```
app.py ────────────┐
ingest.py ─────────┤  (薄壳:只 import 并调用)
                   ▼
              rag/cli.py
               ╱          ╲
              ╱            ╲
   rag/chain.py       rag/loaders.py
        │                   │
        ▼                   │
  rag/vectorstore.py        │
        ╲                  ╱
         └────▶ rag/config.py ◀────┘(loaders 也依赖 config)
                     │
                     ▼
               .env(密钥与参数)
```

依赖单向、无环。根目录的 app.py / ingest.py 是**薄壳**(几行 import + 调用),命令行逻辑集中在 rag/cli.py(它同时被 pyproject.toml 的 [project.scripts] 注册为 rag-qa / rag-ingest 命令):chat 路径走 chain,ingest 路径直接用 loaders + vectorstore;chain 依赖 vectorstore,vectorstore/loaders 依赖 config。**config 是唯一允许读 .env 的模块**,其他模块只 import 具体配置值——这样换配置方式(环境变量注入、云配置中心)只改一个文件。

### 分层原则

| 层 | 文件 | 职责 | 明确不做的事 |
|----|------|------|--------------|
| 入口薄壳 | app.py / ingest.py | 仅 import rag.cli 并调用 | 不含任何逻辑 |
| 命令行层 | rag/cli.py | 命令行交互、启动检查、历史管理、入库流程编排 | 不出现检索/提示词/模型的具体逻辑 |
| 编排层 | rag/chain.py | 组装检索、提示词、模型 | 不管理文件、不碰向量库目录 |
| 数据层 | rag/loaders.py、rag/vectorstore.py | 文档 I/O、向量库 I/O | 不读 .env(用 config 的值) |
| 配置层 | rag/config.py | 常量与配置 | 不做任何 I/O 之外的逻辑 |
| 测试层 | tests/ | 纯函数与守卫逻辑的单元测试(pytest) | 不做网络调用、不依赖 API Key |

这个分层的直接收益:进阶练习里"给问答套 Streamlit 网页"只需新写一个入口文件替换 app.py,rag/ 一行不改;"换向量库"只改 vectorstore.py;"换模型"只改 chain.py 里两行。

## 2. 数据结构一览

整个工程里只有三个跨模块流转的数据结构,全部来自 LangChain 标准库,没有自定义类型:

**Document(langchain_core.documents.Document)** —— 文档加载后的统一形态,贯穿加载、切分、入库、检索四个环节:

```python
Document(
    page_content="RAG 的标准流程……",       # str:正文文本
    metadata={"source": "D:\\...\\rag-基础概念.md"}  # dict:元数据
)
```

`metadata["source"]` 由各 Loader 自动填入文件路径,是回答中"注明来源"和将来做元数据过滤的数据基础。PDF 按页加载,每个 Document 是一页。

**Message(langchain_core.messages)** —— 对话历史的最小单元,app.py 每轮追加 `HumanMessage` + `AIMessage` 各一条:

```python
chat_history = [HumanMessage("RAG 的标准流程是什么"), AIMessage("RAG(检索增强生成)……")]
```

**chunk(切分后的 Document)** —— 结构和 Document 完全相同,只是"文档"变成了"片段",`split_documents` 不会丢失 metadata,所以每个 chunk 都知道自己来自哪个文件。

其余的字典结构(`{"question": ..., "chat_history": ...}`)是 LCEL 链的输入/输出约定,在 3.4 节详述。

## 3. 逐模块详解

### 3.1 rag/config.py — 配置中心(30 行)

唯一的职责:把 `.env` 和代码默认值收拢成一组**模块级常量**。要点:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
```

`PROJECT_ROOT` 通过 `__file__` 向上推导,保证**无论从哪个工作目录启动脚本,knowledge/ 和 data/ 都定位到项目根**——否则在别的目录执行 `python D:\...\ingest.py` 会在错误的位置找文档、建向量库。

`load_dotenv(...)` 的机制:把 .env 中的键值对注入 `os.environ`(**已存在的环境变量不会被覆盖**),之后 `os.getenv("ZHIPUAI_API_KEY")` 才能读到。这是行业惯例而非绕路:本地开发用 .env 文件,部署到服务器/CI 时用真正的环境变量注入密钥,代码不用改。

数值型配置(CHUNK_SIZE 等)包了 `int(...)` 做**类型守卫**——环境变量读出来永远是字符串,不转类型后面做比较就会静默出错。`TOP_K`、`CHAT_MODEL` 等都有默认值,.env 少配甚至不配也能跑(只有 API Key 是硬依赖,在入口处检查)。

### 3.2 rag/loaders.py — 加载与切分(43 行)

两个纯函数,不碰网络、不碰向量库,是最适合单元测试的模块。

**`load_documents()`** —— 目录扫描 + 按扩展名分发:

```python
LOADERS = {".pdf": PyPDFLoader, ".md": TextLoader, ".txt": TextLoader}
```

用字典做"注册表"模式:加新格式(如 Word)就是加一行映射,不用改控制流。`rglob("*")` 递归遍历子目录,不认识的扩展名静默跳过。

关键的一处(踩过坑,见 5.1):

```python
loader_kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
documents.extend(loader_cls(str(path), **loader_kwargs).load())
```

**`split_documents()`** —— 把 Document 列表切成 chunk 列表。参数全部来自 config,语义如下:

| 参数 | 值 | 作用 |
|------|-----|------|
| `chunk_size` | 500 | 每片段最大字符数。太大→检索不精准;太小→语义破碎 |
| `chunk_overlap` | 80 | 相邻片段重叠字符数(约 15%),保证跨片段的句子至少在一个片段里完整 |
| `separators` | 见下 | 递归切分的边界符,从大到小 |

```python
separators=["\n\n", "\n", "。", "!", "?", ";", " ", ""]
```

这是**为本项目专门改过的**:`RecursiveCharacterTextSplitter` 默认分隔符是 `["\n\n", "\n", " ", ""]`,对中文几乎失效——中文正文没有空格,前两级切完还是超长,最终全部退化成 `""` 硬切。把中文句末标点(`。!?;`)插入后,切分会优先落在句子边界,chunk 的语义完整度明显更好。切分时 metadata 原样保留,所以每个 chunk 仍带 source。

> 若要更进一步,可引入 `MarkdownHeaderTextSplitter` 按 `#` 标题切,让 chunk 天然携带章节标题,属于进阶练习。

### 3.3 rag/vectorstore.py — 向量库封装(42 行)

四个函数,各一个职责:

```python
get_embeddings()      # 构造 ZhipuAIEmbeddings(检查 API Key + 显式传 api_key)
get_vectorstore()     # 打开(不存在则创建)data/chroma 下的持久化 Chroma
add_documents(chunks) # 批量向量化写入,返回条数
count_documents()     # 当前库内条数(cli 启动检查用)
as_retriever()        # 包装成 Retriever,检索 top_k=config.TOP_K
```

设计要点:

- **`get_embeddings()` 是全工程唯一的向量模型构造点**。入库和查询必须用同一个模型(不同模型的向量空间不互通),单点构造从结构上杜绝了"两边模型名不一致"这类事故。
- `Chroma(persist_directory=...)` 指定落盘目录后,数据存在 `data/chroma/`,进程重启不丢;`collection_name="knowledge_base"` 相当于库里的表名。
- `count_documents()` 里的注释解释了为什么访问 `_collection.count()`:LangChain 的 Chroma 封装没有公开 count 方法(`len()` 也不支持——这是踩过的坑,见 5.4),访问私有属性封装在本文件内,避免散落在 app.py。
- `as_retriever(search_kwargs={"k": config.TOP_K})`:Retriever 是 LangChain 对"检索"的统一抽象,`search_kwargs` 会透传给底层相似度搜索。将来加元数据过滤(如只搜某个文件)就是往这个字典里加 `filter`。

### 3.4 rag/chain.py — RAG 链路(核心,82 行)

这是全工程最核心的文件,值得逐段精读。

**① 系统提示词(SYSTEM_TEMPLATE)**

三条规则是有意设计的"行为宪法":

1. *只使用参考资料回答* —— 压制模型自由发挥,防幻觉的前提;
2. *没有就明说* —— 给模型一个体面的拒答出口,否则它会在规则 1 和"有问必答"的倾向之间挣扎,反而更容易编造;
3. *注明参考了哪些文档* —— 可溯源,而溯源信息来自 ② 的格式化。

**② `_format_docs()`** —— 检索结果的展示层:

```python
parts.append(f"[{i}] 来源:{source}\n{doc.page_content}")
```

把 top_k 个 chunk 编号、带来源拼接。编号让模型回答时能引用"[2]",来源让引用可定位。空检索时返回"(未检索到相关内容)"而不是空串——空串会让提示词的参考资料区块看起来像"正常但无内容",模型容易自行脑补。

**③ `build_rag_chain()`** —— 链路组装,分四步:

第一步,模型与检索器:

```python
llm = ChatZhipuAI(model=config.CHAT_MODEL, temperature=0.1, api_key=config.ZHIPUAI_API_KEY)
```

`temperature=0.1` 是刻意压低的:知识问答要忠实于资料,不要创造性。`api_key` 显式传入而非依赖环境变量推断,因为 ChatZhipuAI 在**构造时**就会校验 Key(踩过的坑,见 5.4 的关联问题)。

第二步,提示词模板:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder("chat_history"),   # 可伸缩槽位:有历史插历史,无历史为空
    ("human", "{question}"),
])
```

`MessagesPlaceholder` 是关键:它声明"这里插入一组消息",渲染时 `chat_history` 是空列表就什么都不插,是两条消息就插两条。**模板里的每个变量(context、chat_history、question)都必须由上游分支提供**,少一个就报 `missing variables`(踩过的坑,见 5.3)。

第三步,LCEL 链本体:

```python
rag_chain = (
    {
        "context": itemgetter("question") | retriever | RunnableLambda(_format_docs),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt | llm | StrOutputParser()
)
```

逐层拆解数据流(输入是 `{"question": str, "chat_history": list}`):

1. **第一个 dict 是 RunnableParallel**:它的每个分支都拿**同一份完整输入 dict**,各自执行后合并成一个新 dict。所以:
   - `context` 分支:`itemgetter("question")` 先取出问题字符串 → 喂给 `retriever` 检索 → `RunnableLambda(_format_docs)` 格式化。**`itemgetter` 不能省**:直接接 `retriever` 的话,整个 dict 会被送去向量化,API 报 1210(踩过的坑,见 5.2);
   - `question` / `chat_history` 分支:纯透传。**这两个分支也不能省**——并行 dict 的输出就是这三个键的合并,提示词要用的变量必须在这里全部备齐,这是 5.3 坑的修复方式;
2. **prompt**:把三个键填入模板,产出一条渲染好的消息列表;
3. **llm**:调用智谱生成 AIMessage;
4. **StrOutputParser**:剥出 `.content` 纯字符串。

第四步,包一层历史改写(见 ④),返回最终链:

```python
return RunnablePassthrough() | RunnableLambda(rewrite_with_history) | rag_chain
```

**④ `rewrite_with_history()`** —— 多轮对话的检索预处理:

```python
history_text = "\n".join(...)
return {"question": f"对话历史:\n{history_text}\n\n当前问题:{question}", "chat_history": []}
```

两段式设计的意图:检索器和生成模型对上下文的需求不同——

- **检索器**需要"完整独立的问题"("它是什么"检索不到任何东西),所以把最近 3 轮历史(`history[-6:]`,6 条消息)以文本形式并入问题;
- **生成模型**需要原始对话以保持语气连贯,通过 `chat_history` 槽位传入原始 Message 列表。

注意改写后返回的 `"chat_history": []`——历史已经并入问题文本,槽位置空即可,不会丢信息。

`history[-6:]` 的截断是**成本控制**:历史越长 token 越贵、越干扰检索。3 轮对大多数知识问答够用;更严格的做法是用 LLM 生成一个"独立完整问题"替代文本拼接(README 进阶练习第 2 项)。

### 3.5 rag/cli.py — 命令行入口(chat / ingest)

根目录的 app.py / ingest.py 只是几行的薄壳,真正的命令行逻辑在 `rag/cli.py`,拆成两个函数:chat()(问答 REPL)与 ingest()(入库流程)。它同时被注册为可安装命令,所以"加 Web 界面"时直接复用同一套启动检查。

**ingest()** 的流程:`检查 Key → [--rebuild 清库] → 加载 → 切分 → 向量化写入`。三个工程细节:

- `sys.exit("错误:...")` 把缺 Key、空目录这类**用户可自行修复的问题**变成带指引的友好退出,而不是半屏 traceback;
- `--rebuild` 用 `delete_collection()` 清空重建。为什么需要它:入库是**纯增量**的,改了切分参数后旧 chunk 仍留在库里,新旧混检会污染检索结果;
- 每步打印数量(加载 N 个文档 → 切成 M 个片段 → 写入 M 条),让入库过程可观察——M 远大于或小于预期时,第一现场就是切分参数有问题。

### 3.6 rag/cli.py 的 chat() — 问答 REPL

命令行 REPL,职责边界刻意收窄:启动检查 → 构建链 → 循环(读输入 → invoke → 打印 → 追加历史)。要点:

- **启动双检查**:Key 存在 + `count_documents() > 0`。第二个检查是真实教训:目录存在 ≠ 库里有数据(上次入库可能失败到一半),空库检索永远返回空,报错却发生在提问时,离根因很远;
- `chat_history` 在循环外初始化,每轮成功回答后 `extend` 两个 Message——**失败/中断的轮次不进历史**,避免污染下一轮检索;
- `clear` 把列表重新赋值为 `[]`(而非 `clear()` 方法),配合 `q`/Ctrl+C/EOF 多出口,交互状态始终显式;
- 输出用 `flush=True` 逐段打印,为将来换 `chain.stream()` 流式输出预留了同样的交互模式。

## 4. 设计决策记录

摘录几个"这样做而不那样做"的理由,方便评判和演进:

| 决策 | 替代方案 | 选择理由 |
|------|----------|----------|
| 历史改写用文本拼接 | 调一次 LLM 生成独立问题(condense) | 零额外延迟和费用;教程可读性优先。文档量大、多轮深时再升级 |
| chat_history 只带最近 3 轮 | 全量历史 / 固定 token 预算 | 简单有效;token 预算法需引入 tokenizer,超出教学项目复杂度 |
| Chroma 单 collection | 按文件分 collection | Chroma 支持元数据过滤,单库 + filter 的扩展性足够,管理也更简单 |
| 显式 `api_key=` 传参 | 依赖环境变量推断 | 智谱类在构造时校验,显式传参把"缺 Key"提前到启动时刻,报错点离根因近 |
| loaders 用字典注册表 | if/elif 分支 | 加格式零改动控制流;LOADERS 表本身就是文档 |
| 入口逻辑收进 rag/cli.py 而非写在 app.py | 逻辑写在根目录脚本里 | app.py 保持几行薄壳;逻辑进包才能注册 [project.scripts] 命令入口、被测试 import;换 Web 壳时 rag/ 零改动 |

## 5. 踩坑复盘(真实 Bug 档案)

以下是本项目**实际遇到过的四个 Bug**的完整记录,按发现顺序排列。它们不是假设性问题,每个都真实挡住过运行,复盘价值高于任何教科书案例。

### 5.1 GBK 解码错误(Windows 文本编码)

**现象**:入库时 `UnicodeDecodeError: 'gbk' codec can't decode byte 0xaf...`,指向 TextLoader 读 `.md` 文件。

**根因**:`TextLoader` 不传 encoding 时用**操作系统默认编码**,中文 Windows 是 GBK;而文档是 UTF-8。GBK 读 UTF-8 的中文字节流必然撞上非法序列。

**修复**:loaders.py 显式 `TextLoader(path, encoding="utf-8")`,全工程文本统一按 UTF-8 处理。

**教训**:所有跨平台的文本 I/O 都要显式声明编码,不能依赖系统默认值。同类高危点还有:`open()` 读配置、日志文件写入。

### 5.2 检索报 1210「API 调用参数有误」(LCEL 分支拿到 dict)

**现象**:入库成功,提问时报 `APIRequestFailedError: 400, code 1210`。原生 SDK 直接调 embedding 却正常——参数错误但参数本身没问题?

**根因**:LCEL 的 `RunnableParallel` 中,**每个分支拿到的都是完整的输入 dict**。写成 `{"context": retriever, ...}` 时,整个 `{"question":..., "chat_history":...}` 被当成"查询文本"送去向量化,智谱 API 收到 `[dict]` 返回 1210。入库之所以正常,是因为入库传的是纯字符串列表。

**修复**:分支内先取字符串再检索:`"context": itemgetter("question") | retriever | ...`。

**教训**:LCEL 管道里**类型是隐式流动的**,`RunnableParallel` 的"分支共享完整输入"语义和直觉("每条管道只流自己那条")相反。排查这类问题的杀手锏是**在分支中间插一个 `RunnableLambda(print)` 观察流过的数据**,一次就能看清。

### 5.3 提示词 missing variables(LCEL 输出键必须显式备齐)

**现象**:修好 5.2 后,报 `Input to ChatPromptTemplate is missing variables {'chat_history'}. Expected: [...] Received: ['context', 'question']`。

**根因**:`RunnableParallel` 的输出**就是分支字典的键**,没有透传的键会消失。提示词模板声明了三个变量,分支只提供了两个。

**修复**:分支里显式补 `"chat_history": itemgetter("chat_history")`。

**教训**:这个报错信息其实非常友好(Expected vs Received 都列出来了),看到它第一反应应该是"检查并行分支少了哪个键",而不是去怀疑模板写错。

### 5.4 `Chroma` 没有 `len()`(封装抽象的边界)

**现象**:启动检查写 `len(get_vectorstore()) == 0`,报 `TypeError: object of type 'Chroma' has no len()`。

**根因**:LangChain 的 Chroma 封装类没实现 `__len__`,统计条数要走底层 `._collection.count()`。

**修复**:在 vectorstore.py 提供 `count_documents()` 封装,私有属性访问收敛在数据层单点。

**教训**:对第三方封装类的接口做假设前,先 `hasattr` 验证或查源码;必须用私有 API 时,包一层并写注释,把"债务"关进一个文件。

### 5.5 附:构造期校验(次生问题)

`ChatZhipuAI(model=...)` 在**构造时**就要求 Key 存在(pydantic 校验),导致"没配 Key"的错误以 pydantic ValidationError 的形式在链构建时爆出,而不是入口处的友好提示。修复是显式传 `api_key=config.ZHIPUAI_API_KEY`,并让入口脚本在构建链之前检查 Key——**把失败尽量提前、把报错尽量翻译成人话**。

## 6. 一次提问的完整调用时序

```
用户输入问题
 │ app.py → rag/cli.py · chat()
 │ ① question.strip() 检查控制命令(clear/q)
 ▼
chain.invoke({"question": q, "chat_history": [上一轮 HumanMessage/AIMessage, ...]})
 │ chain.py · RunnablePassthrough
 ▼
rewrite_with_history                    无历史:原样通过
 │                                      有历史:历史[-6:] 并入问题文本
 ▼
RunnableParallel ──┬─ itemgetter("question") → retriever
 │                 │    │ vectorstore.py · Chroma 相似度检索
 │                 │    │    └─ embed_query(question) ──HTTP──▶ 智谱 embedding-3
 │                 │    │    └─ cosine 检索 top_k=4 ◀── 4 个 Document
 │                 │    └─ _format_docs → "[1] 来源:...\n...\n\n[2] ..."
 │                 ├─ itemgetter("question") ──────────────▶ str 原样
 │                 └─ itemgetter("chat_history") ───────────▶ list 原样
 ▼
ChatPromptTemplate 渲染
 │  [system] 宪法三条 + 参考资料:{context}
 │  (chat_history 非空则插在中间)
 │  [human] {question}
 ▼
ChatZhipuAI(glm-4-flash) ──HTTP──▶ 智谱 ──▶ AIMessage
 ▼
StrOutputParser → str
 ▼
app.py 打印回答,chat_history.extend([HumanMessage, AIMessage])
```

两处外部 I/O(两次 HTTP 调用:embedding、chat)都发生在 invoke 内部,其余全是本地计算。

## 7. 常见改造点速查

| 想做什么 | 改哪里 | 预估改动 |
|----------|--------|----------|
| 支持新文档格式(如 Word) | `uv add 包名` 装包 + `loaders.py` 的 LOADERS 加一行 | 2 行 |
| 换对话模型(DeepSeek/OpenAI 兼容) | `chain.py` 换 `ChatOpenAI` + config 加 base_url | ~5 行 |
| 换本地向量模型(免 API 费) | `vectorstore.py` 的 `get_embeddings()` 换 `HuggingFaceEmbeddings` | ~3 行 |
| 换向量库(Milvus/FAISS) | 重写 `vectorstore.py` 的五个函数,签名不变 | 单文件 |
| 调检索条数/切分粒度 | `.env` 改 `TOP_K`/`CHUNK_SIZE`,然后 `ingest.py --rebuild` | 0 行代码 |
| 流式输出 | `app.py` 的 `invoke` 换 `stream`,逐 chunk 打印 | ~5 行 |
| 元数据过滤(只搜某文件) | `vectorstore.py` 的 `search_kwargs` 加 `filter` | 1 行 |
| Web 界面 | 新写 `web.py`(Streamlit),复用 `build_rag_chain()` | 新文件 |
| 问题改写升级为 LLM 生成 | 替换 `rewrite_with_history` 的拼接逻辑 | ~15 行 |
| 混合检索(BM25 + 向量) | `vectorstore.py` 加一路 BM25 检索,`chain.py` 合并两路结果 | ~40 行 |

改造时的回归验证清单:`python ingest.py --rebuild` 全绿 → `app.py` 单轮问答有来源 → `它和微调相比有什么优势` 二连问指代正确 → 不在知识库里的问题能拒答。
