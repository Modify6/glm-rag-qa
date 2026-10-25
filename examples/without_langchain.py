"""不用 LangChain 的最小 RAG 问答(教学对照版)。

和 app.py 的问答功能等价(单轮),只依赖:fastembed(本地向量模型)、
chromadb 原生客户端、智谱官方 SDK(仅对话)——embedding 零 API 成本。

跑法(无需先入库,脚本自己重建 raw_demo 集合):
    uv run python examples/without_langchain.py "什么是rag"

注意:它用独立的 raw_demo 集合存储,每次运行全量重建,
不会污染 LangChain 版本的 knowledge_base 集合。

读这份代码时思考两个问题:
1. 哪些地方"也能做,但要自己动手"?(切分算法、提示词拼接、取回答字段)
2. 哪些地方"想换就很难"?(换模型厂商 = 重写调用代码)
"""

import os
import sys

import chromadb
from fastembed import TextEmbedding
from zhipuai import ZhipuAI

# 只复用密钥与参数配置(dotenv 读取,与 LangChain 无关)
import rag.config as config

# ── 直连厂商 SDK:没有 LangChain 的统一接口,就是裸的客户端 ──────────────
zhipu = ZhipuAI(api_key=config.ZHIPUAI_API_KEY)
chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

# 本地向量模型(fastembed,零 API 成本):没有 LangChain 的 Embeddings
# 统一接口,向量能力由你自己选择和接入——这里选开源 BGE 中文模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像
embedding_model = TextEmbedding(model_name=config.LOCAL_EMBEDDING_MODEL)

# 向量集合:与 LangChain 版的 knowledge_base 相互独立,互不污染
collection = chroma.get_or_create_collection("raw_demo")


def embed(texts: list[str]) -> list[list[float]]:
    """文本转向量:本地模型推理,不花一分钱、不需要联网。"""
    return [vec.tolist() for vec in embedding_model.embed(texts)]


def split_text(text: str) -> list[str]:
    """手写的简化版切分:固定步长滑窗,没有递归边界、没有句号感知。

    对比 rag/loaders.py 里 RecursiveCharacterTextSplitter 干的事——
    这就是 LangChain "送你的" 那部分算法。
    """
    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return [text[start : start + config.CHUNK_SIZE] for start in range(0, len(text), step)]


def ingest_demo() -> int:
    """把 knowledge/ 下的 md 文档入库(每次运行先清空重建)。"""
    global collection
    chroma.delete_collection("raw_demo")
    collection = chroma.get_or_create_collection("raw_demo")  # 重建后必须回写全局引用
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
