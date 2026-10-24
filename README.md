# LangChain 实战:基于智谱 GLM 的 RAG 知识库问答

一个完整的 **RAG(Retrieval-Augmented Generation,检索增强生成)** 实战项目,适合作为 LangChain 的第一个系统性练手项目。

把你自己的文档(PDF / Markdown / 纯文本)向量化存入本地向量库,然后通过命令行进行**带引用来源、支持多轮对话**的智能问答。代码总量不到 300 行,但覆盖了 RAG 的全部核心环节,每一行都能在本文档中找到对应的原理讲解。

---

## 目录

- [效果演示](#效果演示)
- [背景知识:什么是 RAG](#背景知识什么是-rag)
- [技术栈与版本说明](#技术栈与版本说明)
- [环境准备与安装](#环境准备与安装)
- [配置说明](#配置说明)
- [使用步骤](#使用步骤)
- [项目结构](#项目结构)
- [核心原理与代码详解](#核心原理与代码详解)
- [参数调优指南](#参数调优指南)
- [常见问题 FAQ](#常见问题-faq)
- [进阶改造练习](#进阶改造练习)

---

## 效果演示

```
============================================================
知识库问答就绪(模型:glm-4-flash,输入 q 退出,输入 clear 清空对话)
============================================================

你 > 什么是 RAG?
助手 > RAG(检索增强生成)是一种把"外部知识检索"和大语言模型生成结合起来的技术。
用户提问时,先从知识库中检索出相关片段,把这些片段塞进提示词,再让大模型基于
这些资料回答。它主要解决三个问题:大模型不知道私有/最新信息、会产生幻觉、
以及微调成本高。(参考:rag-基础概念.md)

你 > 它能解决幻觉问题吗?
助手 > 能。上一轮提到的 RAG 让回答"有据可查"——每个回答都基于检索到的真实
文档片段,可以标注引用来源,而不是依赖模型记忆自由发挥……
```

注意第二问的"它":得益于对话历史改写机制,检索时问题已被补全为"RAG 能解决幻觉问题吗",这就是多轮 RAG 与单轮 RAG 的关键区别(详见[多轮对话机制](#5-多轮对话机制))。

## 背景知识:什么是 RAG

大语言模型有三个天然缺陷:

1. **知识有截止日期**,训练之后新发生的事情一概不知;
2. **不知道私有数据**,公司内部文档、个人笔记从没进过训练集;
3. **会幻觉**,对不知道的东西一本正经地编造。

RAG 的思路很直接:**与其让模型"回忆",不如先帮它"查资料"**。提问时先从文档库里检索出最相关的几个片段,作为参考资料放进提示词,让模型基于资料作答。

对比另外两种让模型掌握私有知识的方式:

| 方式 | 成本 | 更新知识 | 可溯源 | 适用场景 |
|------|------|----------|--------|----------|
| 重新训练 | 极高 | 需重训 | 否 | 改变模型基础能力 |
| 微调 Fine-tuning | 高 | 需重训 | 否 | 改变回答风格/格式 |
| **RAG** | **低** | **重新入库即可** | **是** | **知识型问答** |

RAG 的完整流程分两个阶段:

```
【离线:入库阶段】                        【在线:问答阶段】

 docs/ 目录                              用户提问
    │                                        │
    ▼                                        ▼
 ① 加载 Load                           ① 问题向量化
    │                                        │
    ▼                                        ▼
 ② 切分 Split                          ② 向量库相似度检索(取 top_k 个片段)
    │                                        │
    ▼                                        ▼
 ③ 向量化 Embed        ──────写入─────▶ ③ 组装提示词(资料+规则+问题)
    │                                        │
    ▼                                        ▼
 ④ 存储 Index(Chroma)                ④ 大模型生成回答
```

每个环节更细的原理见 `docs/rag-基础概念.md`(它本身也是本项目知识库里的文档,可以入库后直接问它)。

## 技术栈与版本说明

| 组件 | 版本 | 作用 |
|------|------|------|
| uv | 0.13+ | Python 环境与依赖管理(pyproject.toml + uv.lock 锁定版本) |
| Python | 3.12 | 运行环境(LangChain 1.x 要求 ≥ 3.9,由 uv 按 `.python-version` 自动准备) |
| langchain | 1.x | 主框架:链式编排(LCEL) |
| langchain-community | 0.4.x | 智谱模型集成:ChatZhipuAI / ZhipuAIEmbeddings / 文档加载器 |
| langchain-chroma | 1.x | Chroma 向量库的官方 LangChain 集成 |
| chromadb | 1.x | 轻量级本地向量数据库,零外部服务 |
| zhipuai | 2.x | 智谱官方 Python SDK(被上面两个类依赖) |
| pypdf | - | PDF 解析(PyPDFLoader 依赖) |
| python-dotenv | - | 从 .env 文件读取环境变量 |

> **版本说明**:安装时会看到 `langchain-community is being sunset` 的弃用警告。LangChain 官方正在把各模型厂商的集成拆分成独立包,但智谱目前没有官方独立集成包(`langchain-zhipu` 是第三方维护、`langchain-zhipuai` 几乎无内容),社区版 `ChatZhipuAI` 仍是最稳妥的选择,与国内大多数教程一致。将来若出现官方包,迁移成本只是换一个 import。

> **为什么选 Chroma**:单机零部署、支持持久化、API 简单,是学习 RAG 的标准起步选择。生产环境数据量大时再换 Milvus / Elasticsearch 等,代码只需改动 `rag/vectorstore.py` 一个文件。

## 环境准备与安装

本项目用 **uv** 管理环境:依赖声明在 `pyproject.toml`,精确版本锁定在 `uv.lock`,任何机器上 `uv sync` 都能还原出完全一致的环境。

### 方式一:一键脚本(推荐)

双击项目根目录的 `install.bat`(或命令行执行它),脚本会自动:

1. 检测并安装 uv(通过 winget,已装则跳过);
2. 按 `uv.lock` 创建 `.venv` 并安装全部依赖(走 pyproject.toml 里配置的清华镜像);
3. 打印接下来的操作指引。

> 如果脚本提示"PATH 未刷新",关掉窗口重开终端再运行一次即可。

### 方式二:手动命令

```powershell
cd D:\gd\git\zcode\practice\rea\langchain

# 1. 安装 uv(已装可跳过)
winget install -e --id astral-sh.uv

# 2. 同步环境:自动准备 Python 3.12(必要时下载)+ 创建 .venv + 按 uv.lock 装依赖
uv sync
```

常用命令速查:

| 命令 | 作用 |
|------|------|
| `uv sync` | 按 uv.lock 创建/更新虚拟环境(拉新机器后第一步) |
| `uv run xxx.py` | 在项目虚拟环境中运行脚本(自动保证环境就绪) |
| `uv add 包名` | 新增依赖(自动更新 pyproject.toml 和 uv.lock) |
| `uv remove 包名` | 移除依赖 |
| `uv lock --upgrade` | 升级所有依赖到允许范围内的最新版 |

### 3. 获取智谱 API Key

1. 打开 [智谱开放平台](https://open.bigmodel.cn),用手机号注册登录;
2. 进入控制台 → 「API 密钥」→ 创建 API Key,复制保存;
3. 新用户注册会赠送免费额度;对话模型 `glm-4-flash` 本身免费,向量模型 `embedding-3` 按量计费(入库两篇示例文档的成本可以忽略不计)。

## 配置说明

复制配置模板并根据下表修改:

```powershell
copy .env.example .env
```

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `ZHIPUAI_API_KEY` | ✅ | 无 | 智谱 API 密钥,程序启动时会检查 |
| `CHAT_MODEL` | | `glm-4-flash` | 对话模型。`glm-4-flash` 免费;追求质量可换 `glm-4-plus`(收费) |
| `EMBEDDING_MODEL` | | `embedding-3` | 向量模型,智谱新一代 2048 维向量模型 |
| `CHUNK_SIZE` | | `500` | 每个文本片段的最大字符数,详见[调优指南](#参数调优指南) |
| `CHUNK_OVERLAP` | | `80` | 相邻片段重叠的字符数(约为 chunk_size 的 15%) |
| `TOP_K` | | `4` | 每次提问检索返回的片段数 |

> `.env` 已加入 `.gitignore`,**永远不会被提交到 git**,密钥只留在本机。

## 使用步骤

### 第 1 步:文档入库

```powershell
uv run ingest.py        # 或双击 ingest.bat
```

正常输出:

```
加载文档:D:\gd\git\zcode\practice\rea\langchain\docs
  加载了 2 个文档
  切分为 7 个片段(chunk_size=500, overlap=80)
正在向量化入库(调用智谱 embedding API)...
完成!共写入 7 个片段到 D:\gd\git\zcode\practice\rea\langchain\data\chroma
```

做了什么:遍历 `docs/` 下所有 `.md / .txt / .pdf` → 切分成片段 → 逐片段调用智谱向量 API → 向量连同原文、来源文件名存入本地 Chroma(持久化在 `data/chroma/`)。

两个实用变体:

```powershell
uv run ingest.py --rebuild   # 清空向量库重建。修改了 CHUNK_SIZE 等参数后必须用这个
```

想换成自己的知识库:把你的文档放进 `docs/`(支持子目录),重新跑一遍即可。删除某篇文档后用 `--rebuild` 重建,因为当前版本是纯增量入库。

### 第 2 步:开始问答

```powershell
uv run app.py           # 或双击 start.bat
```

| 输入 | 行为 |
|------|------|
| 任意问题 | 检索知识库并回答,回答中注明参考文档 |
| `clear` | 清空对话历史(之后的提问不再参考前文) |
| `q` / Ctrl+C | 退出 |

值得试的三个问题(检验 RAG 是否真正工作):

- `RAG 的标准流程是什么?` —— 基础检索问答;
- `它和微调相比有什么优势?` —— 测试多轮指代消解("它"= 上一轮的 RAG);
- `鲁迅的原名是什么?` —— 知识库里没有的内容,正确行为是回答"知识库中没有找到相关信息",如果模型开始编造,说明提示词约束需要加强。

## 项目结构

```
langchain/
├── app.py                  # 问答入口:命令行多轮对话循环
├── ingest.py               # 入库脚本:加载 → 切分 → 向量化 → 写入 Chroma
├── rag/                    # 核心逻辑包(与入口分离,便于复用和测试)
│   ├── __init__.py
│   ├── config.py           # 唯一读 .env 的地方,其他模块只 import 配置值
│   ├── loaders.py          # 文档加载与切分(纯函数,不碰网络)
│   ├── vectorstore.py      # 向量库封装:打开库、写入库、取检索器
│   └── chain.py            # RAG 链路组装:检索 → 提示词 → 模型 → 解析
├── docs/                   # 知识库文档目录(自带 2 篇示例,可整体替换)
│   ├── rag-基础概念.md
│   └── langchain-核心组件.md
├── data/chroma/            # Chroma 向量库持久化目录(ingest 后生成,git 忽略)
├── pyproject.toml          # 依赖与项目元数据(uv 的唯一配置入口)
├── uv.lock                 # 依赖精确版本锁文件(提交到 git,保证环境可复现)
├── install.bat             # 一键安装:装 uv + uv sync
├── ingest.bat              # 入库快捷脚本(参数透传,如 ingest.bat --rebuild)
├── start.bat               # 问答启动快捷脚本
├── .python-version         # 锁定 Python 版本 3.12(uv 据此自动准备解释器)
├── .env.example            # 配置模板
├── .gitignore              # 忽略 .env / data/ / .venv 等
├── CODE_GUIDE.md           # 代码工程详解(逐模块精读 + 踩坑复盘)
└── README.md               # 本文档
```

分层原则:**入口脚本只做 I/O(命令行交互、报错退出),`rag/` 包只做核心逻辑**。想给项目加 Web 界面,只需把 `app.py` 换成 Streamlit 页面,`rag/` 一行不用动。

## 核心原理与代码详解

按数据流顺序讲解每个环节,建议对照源码阅读。

### 1. 配置加载(rag/config.py)

```python
load_dotenv(PROJECT_ROOT / ".env")          # 把 .env 的键值注入进程环境变量
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")   # 再从环境变量读出
```

为什么绕一道环境变量而不是直接读文件?这是行业惯例:`load_dotenv` 只在本地开发时生效,部署到服务器/容器时改用真正的环境变量注入密钥,代码零改动。

### 2. 文档加载与切分(rag/loaders.py)

**加载**:LangChain 把各种格式统一抽象为 `Document` 对象——就两个字段:`page_content`(文本)和 `metadata`(字典,本项目中是 `{"source": "文件路径"}`)。`load_documents()` 按扩展名分发到对应 Loader(PDF 用 `PyPDFLoader`,文本用 `TextLoader`),PDF 会自动按页加载,每页一个 Document。

**切分**是 RAG 效果的第一决定因素。核心矛盾:片段太大 → 检索不精准(一个 2000 字的片段里只有 100 字和问题相关,剩下的都是噪音,还烧 token);片段太小 → 语义被拦腰截断,检索到了也读不懂。

`RecursiveCharacterTextSplitter` 的策略是**按分隔符从大到小递归切**:先尝试按空行(段落)切,切出来还超过 `CHUNK_SIZE` 就降级按换行、句号、分号继续切,实在不行才硬切字符。本项目把中文标点(`。! ?;`)加进了分隔符列表,因为中文没有空格分词,默认分隔符对中文几乎失效。

```python
separators=["\n\n", "\n", "。", "!", "?", ";", " ", ""]
```

`CHUNK_OVERLAP`(重叠)解决切分边界问题:一个关键句刚好跨在两个片段的分界处时,重叠区能保证它至少在其中一个片段里是完整的。

### 3. 向量化与存储(rag/vectorstore.py)

**向量化(Embedding)**:把一段文本映射成一个 2048 维浮点向量。关键是这个映射学过语义——"检索增强生成"和"RAG 技术"用词完全不同,但向量距离很近;"检索增强生成"和"今天天气不错"则距离很远。检索的本质就是**在向量空间里找距离最近的片段**。

`embedding-3` 对中文支持好。注意入库和查询用的是**同一个**向量模型(项目里 `get_embeddings()` 全局唯一保证这一点),不同模型的向量空间不互通,混用等于拿中文拼音查英文词典。

**存储**:Chroma 是嵌入式数据库(类似 SQLite 之于 MySQL),一个本地目录就是一套库。每个片段写入三样东西:向量(用于算距离)、原文(命中后送进提示词)、元数据(用于溯源和过滤)。`persist_directory` 让数据落盘,重启进程不用重新入库。

### 4. RAG 链路组装(rag/chain.py)—— 本项目最核心的文件

链路用 **LCEL(LangChain Expression Language)** 编排,用管道符把组件串起来:

```python
rag_chain = (
    {"context": retriever | _format_docs, "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)
```

从右往左读,数据流是:

1. **第一个 dict** 是一个并行分支:输入进来自动分两路——`retriever` 拿输入里的 `question` 去向量库检索,`_format_docs` 把命中的片段格式化成带编号和来源的参考资料文本;`itemgetter` 则直接透传原问题;
2. **prompt**(`ChatPromptTemplate`)把这些填进模板。`{context}` 占位符吃参考资料,`MessagesPlaceholder("chat_history")` 是一个可伸缩的槽位,有历史就插入历史消息,没有就是空;
3. **llm**(`ChatZhipuAI`)生成回答。`temperature=0.1` 刻意调低——知识问答要的是忠实于资料,不是文采飞扬;
4. **StrOutputParser** 把模型返回的消息对象剥成纯字符串。

系统提示词里的三条规则值得细看:它们是控制模型行为的"宪法"——**只准用资料回答、资料没有就明说、注明来源**。去掉第一条试试,你会亲眼看到幻觉出现,这是理解提示词工程最直观的实验。

### 5. 多轮对话机制

朴素的做法是把整个对话历史都塞给模型——但历史越长 token 越贵,而且历史里的旧问题会干扰检索。本项目用**两段式**处理:

```python
def rewrite_with_history(inputs):
    # 无历史:原样通过
    # 有历史:拼成 "对话历史:…\n当前问题:…" 再去检索
    # history[-6:] —— 只带最近 3 轮(6 条消息),控制 token
```

工作流程:用户问"它能解决幻觉吗?"时,**检索前**先把它补全为"RAG 能解决幻觉问题吗"(上下文在 `rewrite_with_history` 里以文本形式给了模型可见的线索,同时 `chat_history` 槽位让生成模型也能看到原始对话),检索命中,回答连贯。这套"历史改写 + 截断"是生产级 RAG 的标准做法,更严格的实现是再调一次 LLM 做问题改写(condense question),属于[进阶练习](#进阶改造练习)第 2 项。

对话历史本身由 `app.py` 维护:

```python
chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
```

每轮把问答对追加进列表,下一轮 `invoke` 时传入。用 `clear` 清空后,模型立刻"失忆"——可以自己验证。

### 6. 完整数据流(一次提问的全过程)

```
你: "它能解决幻觉问题吗?"
 │ app.py
 ▼
{"question": "它能解决幻觉问题吗?", "chat_history": [上一轮问答]}
 │ chain.py · rewrite_with_history
 ▼
{"question": "对话历史:\n用户: 什么是RAG?…\n当前问题: 它能解决幻觉问题吗?", ...}
 │ vectorstore.py · retriever        ──▶ Chroma 相似度检索 top_k=4
 │                                   ◀── 4 个片段(含来源)
 ▼
context: "[1] 来源:…\docs\rag-基础概念.md\nRAG 让回答有据可查……"
 │ ChatPromptTemplate
 ▼
[system] 你是知识库问答助手…参考资料:{context} → [human] 当前问题
 │ ChatZhipuAI(glm-4-flash)
 ▼
"能。RAG 让回答有据可查,每个回答都基于检索到的真实文档片段…"
 │ StrOutputParser
 ▼
打印回答 + 本轮问答追加进 chat_history
```

## 参数调优指南

改参数后必须 `python ingest.py --rebuild` 重建向量库(chunk_size/overlap 改动只影响入库数据)。

| 症状 | 可能原因 | 调整 |
|------|----------|------|
| 回答说"没有找到相关信息",但你确定文档里有 | 切分把答案截碎了;或检索没命中 | 增大 `CHUNK_SIZE` / 增大 `TOP_K`;确认关键词原文确实存在 |
| 回答啰嗦、跑题,夹带资料外内容 | 片段太大,噪音多;返回片段太多 | 减小 `CHUNK_SIZE` / 减小 `TOP_K` |
| 片段读起来上下文断裂 | 重叠不够 | 增大 `CHUNK_OVERLAP`(到 chunk_size 的 20% 左右) |
| 回答开始编造事实 | 模型自由发挥 | 保持 temperature 低;强化 system 提示词的约束条款 |
| 回答偏慢/偏贵 | 模型太重 | `CHAT_MODEL` 保持 `glm-4-flash`;文档量大时减小 `TOP_K` |

经验起点:中文文档 `CHUNK_SIZE=500`、`CHUNK_OVERLAP=80`、`TOP_K=4` 是一个稳健的默认组合,先跑通再调。

## 常见问题 FAQ

**Q:提问时报 `APIRequestFailedError: 400, code 1210「API 调用参数有误」`**
先看入库是否成功——入库成功而提问报错,几乎可以断定是检索分支把**非字符串**(典型是整个 dict)送去做向量化。LCEL 并行分支 `{"context": retriever, ...}` 里每个分支拿到的都是完整输入 dict,必须写 `itemgetter("question") | retriever` 先取出问题字符串;同理,提示词用到的每个变量(含 `chat_history`)都要在并行分支里显式透传,否则报 `missing variables`。本项目 `rag/chain.py` 的最终写法就是修复后的样子。

**Q:入库报 `UnicodeDecodeError: 'gbk' codec can't decode ...`(Windows 常见)**
文本文件的实际编码和读取编码不一致导致的,项目已修复(加载文本一律按 UTF-8)。如果你自己的 txt 文件是 GBK 编码(记事本另存为 ANSI 的旧文件),用记事本/VS Code 把它另存为 UTF-8 编码即可;也可以在 `rag/loaders.py` 里按文件后缀区分编码。

**Q:启动报 `未配置 ZHIPUAI_API_KEY`**
`.env` 必须在项目根目录(和 `app.py` 同级),密钥值两端不要带引号和空格。可用 `type .env` 检查内容。

**Q:入库时报 embedding 接口鉴权错误(`APIAuthenticationError`)**
API Key 无效或已过期,回开放平台重新生成。注意 Key 格式形如 `xxxx.yyyy`(两段以点号连接)。

**Q:入库很慢或超时**
每个片段都要调一次向量 API。文档多时属正常现象(几秒到几分钟);持续超时检查网络,必要时确认镜像/代理设置。

**Q:向量库数据存在哪?能删吗?**
`data/chroma/`,整个目录可以随时删除,删掉后 `ingest.py --rebuild` 重建即可,无任何不可逆损失。

**Q:能加 Word/Excel 文档吗?**
能。执行 `uv add docx2txt openpyxl`(自动更新 pyproject.toml 和 uv.lock),然后在 `rag/loaders.py` 的 `LOADERS` 字典里加一行映射即可,这就是练手第一个小改造。

**Q:换 OpenAI / DeepSeek 行不行?**
行。DeepSeek 兼容 OpenAI 接口,把 `chain.py` 里的模型类换成 `langchain-openai` 的 `ChatOpenAI` 并指定 `base_url`,向量模型仍用智谱(或换本地 BGE 模型,连 API 费都省了)。

**Q:`langchain-community` 弃用警告要紧吗?**
不影响功能,详见[技术栈与版本说明](#技术栈与版本说明)。

## 进阶改造练习

按难度递增排序,每项都基于现有代码做增量修改:

1. **⭐ 流式输出**:把 `app.py` 里的 `chain.invoke(...)` 换成 `chain.stream(...)`,逐字打印。LCEL 链天然支持流式,改动只有几行,但能直观理解 LangChain 的流式协议。
2. **⭐⭐ LLM 问题改写**:现在的问题改写是文本拼接(够用但粗糙),改成调用 `glm-4-flash` 生成一个"独立完整的问题"再检索(即 Condense Question 模式),对比检索命中率差异。
3. **⭐⭐ 混合检索**:加一路 BM25 关键词检索(`rank_bm25` 包),向量检索 + 关键词检索结果按排名融合(RRF),解决"专有名词精确匹配"场景下纯向量检索的短板。
4. **⭐⭐ 重排序(Rerank)**:先检索 `top 10`,再调用智谱 rerank 模型精选 `top 4` 送入提示词。这是生产 RAG 中性价比最高的单项优化。
5. **⭐⭐ Web 界面**:用 Streamlit(约 50 行)或 Gradio 给问答套上网页,加聊天框和文档上传入口,上传后自动入库。
6. **⭐⭐⭐ 评估体系**:构造 20 条「问题-标准答案」测试集,引入 RAGAS 框架量化 faithfulness(忠实度)和 answer relevancy(相关性),让每次调参有数据依据而不是凭感觉。

## 延伸阅读

- [CODE_GUIDE.md](CODE_GUIDE.md) —— 代码工程详解:逐模块精读、设计决策、踩坑复盘(想读懂/改造代码必看)
- 项目内置知识库文档:`docs/rag-基础概念.md`、`docs/langchain-核心组件.md`
- [LangChain 官方文档](https://python.langchain.com)(RAG 教程与 LCEL 指南)
- [智谱开放平台文档](https://docs.bigmodel.cn)(模型列表与 API 参考)
- [Chroma 官方文档](https://docs.trychroma.com)
