"""向量库封装:向量化、持久化存储、检索(RAG 的存储层)。

对应入库与检索两条链路:
    入库:chunks --add_documents()--> 向量 + 原文 + 元数据 写入 Chroma
    检索:问题 --as_retriever()--> 语义相似度 top_k 个 Document

关键设计:get_embeddings() 是全工程唯一的向量模型构造点。
入库和查询必须用同一个向量模型(不同模型的向量空间不互通,
拿 A 模型的向量去 B 模型的空间里找近邻,等于拿拼音查英文词典),
单点构造从结构上杜绝了"两边模型名不一致"这类事故。
"""
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document

from rag import config


def get_embeddings() -> ZhipuAIEmbeddings:
    """构造智谱向量模型客户端(embedding-3,2048 维)。

    显式检查并传入 api_key 的原因:ZhipuAIEmbeddings 内部虽然也能从
    环境变量推断,但显式传参能把"没配 Key"的失败提前到构造时刻,
    并且报错信息由我们掌控(而不是 SDK 的 pydantic 校验堆栈)。
    """
    if not config.ZHIPUAI_API_KEY:
        raise SystemExit("错误:未配置 ZHIPUAI_API_KEY,请复制 .env.example 为 .env 并填入密钥。")
    return ZhipuAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.ZHIPUAI_API_KEY)


def get_vectorstore() -> Chroma:
    """打开本地持久化的 Chroma 向量库(不存在则自动创建)。

    Chroma 是嵌入式数据库(类比:SQLite 之于 MySQL),一个本地目录
    就是一套完整的库,零部署、零服务。三个参数的含义:
    - collection_name: 库里的"表"名,一个 Chroma 实例可存多个集合;
    - embedding_function: 查询时自动用它把问题文本转向量;
    - persist_directory: 落盘目录,数据在 data/chroma/ 下,
      进程重启不丢;删掉整个目录 = 清空知识库。
    """
    return Chroma(
        collection_name="knowledge_base",
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def add_documents(chunks: list[Document]) -> int:
    """向量化并写入向量库,返回写入条数。

    每个片段写入三样东西:
    - 向量:embedding-3 生成的 2048 维浮点数组,用于算相似度;
    - 原文:page_content,检索命中后送进提示词的内容;
    - 元数据:metadata(含 source 文件路径),用于溯源和过滤。

    add_documents 内部按批调用智谱 embedding API,文档多时耗时主要在这里。
    """
    if not chunks:
        return 0
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def count_documents() -> int:
    """返回向量库中当前存储的片段数(用于启动前检查是否已入库)。

    存在的原因:LangChain 的 Chroma 封装没有公开 count 方法,
    len() 也不支持(TypeError: object of type 'Chroma' has no len(),
    真实踩过的坑,详见 CODE_GUIDE.md 5.4),只能访问底层 collection 的
    count()。把私有属性访问封装在数据层单点,不散落到入口脚本。
    """
    return get_vectorstore()._collection.count()


def as_retriever():
    """返回检索器:按语义相似度取 TOP_K 个最相关的 chunk。

    Retriever 是 LangChain 对"检索"的统一抽象:输入字符串、
    输出 Document 列表,可像普通组件一样接进 LCEL 链。
    search_kwargs 会透传给底层相似度搜索:
    - k: 返回条数;
    - 将来加元数据过滤(如只搜某个文件)就是在这里加 filter。
    """
    return get_vectorstore().as_retriever(search_kwargs={"k": config.TOP_K})
