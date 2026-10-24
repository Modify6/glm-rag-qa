"""入库脚本:RAG 离线阶段的完整入口。

把 docs/ 目录下的文档变成可检索的知识库:
    加载(load_documents)→ 切分(split_documents)→ 向量化写入(add_documents)

用法:
    python ingest.py              # 增量入库 docs/ 下所有文档
    python ingest.py --rebuild    # 清空向量库后重新入库

为什么需要 --rebuild:入库是【纯增量】的,修改了切分参数(CHUNK_SIZE 等)
后,旧 chunk 仍留在库里,新旧混检会污染检索结果,必须清空重建。

入口脚本的职责边界:只做命令行 I/O、启动前检查和进程退出,
核心逻辑全部在 rag/ 包里(便于复用和测试)。
"""
import argparse
import sys

import rag.config as config
from rag.loaders import load_documents, split_documents
from rag.vectorstore import add_documents, get_vectorstore


def main() -> None:
    # --rebuild 是个开关参数(python ingest.py --rebuild 时为 True)
    parser = argparse.ArgumentParser(description="知识库文档入库")
    parser.add_argument("--rebuild", action="store_true", help="清空向量库后重建")
    args = parser.parse_args()

    # 启动检查 1:密钥。用 sys.exit 输出"用户能自己修复"的指引,
    # 而不是等调用 API 时爆出半屏 traceback。
    if not config.ZHIPUAI_API_KEY:
        sys.exit("错误:未配置 ZHIPUAI_API_KEY,请复制 .env.example 为 .env 并填入密钥。")

    # 可选:清空重建。delete_collection 会连数据和表结构一起删除,
    # 下一步 get_vectorstore() 时会自动重建空表。
    if args.rebuild:
        print(f"清空向量库:{config.CHROMA_DIR}")
        get_vectorstore().delete_collection()

    # 第 1 步:加载。把每个文件变成带 source 元数据的 Document。
    print(f"加载文档:{config.DOCS_DIR}")
    documents = load_documents()
    if not documents:
        sys.exit(f"错误:{config.DOCS_DIR} 下没有可加载的文档(支持 .md / .txt / .pdf)。")
    print(f"  加载了 {len(documents)} 个文档")

    # 第 2 步:切分。长文档 → 大小合适的 chunk(chunk_size/overlap 见 config)。
    # 打印数量是为了可观察:M 远大于或小于预期时,第一现场就是切分参数问题。
    chunks = split_documents(documents)
    print(f"  切分为 {len(chunks)} 个片段(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    # 第 3 步:向量化写入。每条 chunk 调一次智谱 embedding API,
    # 这是入库的主要耗时点(纯网络开销,文档多时属正常现象)。
    print("正在向量化入库(调用智谱 embedding API)...")
    count = add_documents(chunks)
    print(f"完成!共写入 {count} 个片段到 {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
