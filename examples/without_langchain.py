"""不用 LangChain 的最小 RAG 问答(教学对照版)。

和 app.py 的问答功能等价(单轮),但只用智谱官方 SDK(zhipuai)和
chromadb 原生客户端,一行 LangChain 代码都没有。

跑法(先完成过一次正常入库):
    uv run python examples/without_langchain.py "什么是RAG"

注意:它用独立的 raw_demo 集合存储,每次运行全量重建,
不会污染 LangChain 版本的 knowledge_base 集合。

读这份代码时思考两个问题:
1. 哪些地方"也能做,但要自己动手"?(切分算法、提示词拼接、取回答字段)
2. 哪些地方"想换就很难"?(换模型厂商 = 重写调用代码)
"""

import sys

import chromadb
from zhipuai import ZhipuAI

# 只复用密钥与参数配置(dotenv 读取,与 LangChain 无关)
import rag.config as config

# ── 直连厂商 SDK:没有 LangChain 的统一接口,就是裸的客户端 ──────────────
zhipu = ZhipuAI(api_key=config.ZHIPUAI_API_KEY)
chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
collection = chroma.get_or_create_collection("raw_demo")


def embed(texts: list[str]) -> list[list[float]]:
    """文本转向量:直接调智谱 SDK,和 ZhipuAIEmbeddings 干的事一模一样。"""
    resp = zhipu.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def split_text(text: str) -> list[str]:
    """手写的简化版切分:固定步长滑窗,没有递归边界、没有句号感知。

    对比 rag/loaders.py 里 RecursiveCharacterTextSplitter 干的事——
    这就是 LangChain "送你的" 那部分算法。
    """
    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return [text[start : start + config.CHUNK_SIZE] for start in range(0, len(text), step)]


def ingest_demo() -> int:
    """把 knowledge/ 下的 md 文档入库(每次运行先清空重建)。"""
    chroma.delete_collection("raw_demo")
    collection = chroma.get_or_create_collection("raw_demo")
    total = 0
    for path in sorted(config.KNOWLEDGE_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = split_text(text)
        collection.add(
            ids=[f"{path.name}#{i}" for i in range(len(chunks))],
            documents=chunks,                                   # 原文
            embeddings=embed(chunks),                           # 向量(自己调 API)
            metadatas=[{"source": str(path)}] * len(chunks),    # 元数据(自己定义)
        )
        total += len(chunks)
    return total


def query(question: str) -> str:
    """检索 + 生成:每一步都是手动的。"""
    # 1. 检索:自己把问题转向量,再调 Chroma 原生 query
    hits = collection.query(query_embeddings=embed([question]), n_results=config.TOP_K)

    # 2. 拼参考资料:和 _format_docs 等价,但是手动的 f-string
    context = "\n\n".join(
        f"[{i}] 来源:{src}\n{doc}"
        for i, (src, doc) in enumerate(zip(hits["metadatas"][0], hits["documents"][0]), 1)
    )

    # 3. 提示词:没有 ChatPromptTemplate,系统规则和参考资料全靠手拼字符串
    prompt = (
        "你是一个严谨的知识库问答助手。请仅根据参考资料回答,"
        "资料里没有就明说。\n\n"
        f"参考资料:\n{context}\n\n问题:{question}"
    )

    # 4. 调模型:没有 ChatPromptTemplate 的消息结构,消息列表手工构造
    resp = zhipu.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )

    # 5. 取回答:没有 StrOutputParser,自己按 SDK 的响应结构一层层剥
    return resp.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('用法:uv run python examples/without_langchain.py "你的问题"')
    n = ingest_demo()
    print(f"[对照实验] 无 LangChain 版:入库 {n} 个片段,开始问答\n")
    print(query(sys.argv[1]))
