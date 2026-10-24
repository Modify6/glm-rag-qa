"""命令行入口:问答 REPL 与文档入库。

为什么单独一个 cli 模块:pyproject.toml 的 [project.scripts] 把这里注册成
可安装的命令(rag-qa / rag-ingest),而根目录的 app.py / ingest.py 只作
为直接运行的薄壳。逻辑放包内、壳放包外,两边都可用且不重复代码。

职责边界:只做命令行 I/O、启动检查、历史管理;
检索、提示词、模型调用的核心逻辑在 rag/chain.py。
"""
import argparse
import sys

from langchain_core.messages import AIMessage, HumanMessage

import rag.config as config
from rag.chain import build_rag_chain
from rag.loaders import load_documents, split_documents
from rag.vectorstore import add_documents, count_documents, get_vectorstore


# =====================================================================
# 问答 REPL(在线阶段)
# =====================================================================
def chat() -> None:
    """知识库问答主循环:读输入 → RAG 链(检索+生成)→ 打印 → 追加历史。

    交互命令:任意问题(检索回答)/ clear(清空历史)/ q 或 Ctrl+C(退出)。
    """
    # ===== 启动检查(把所有可预见的失败拦在交互开始之前)=====
    # 检查 1:密钥。ChatZhipuAI 构造时就会校验,但提前检查能给出友好提示。
    if not config.ZHIPUAI_API_KEY:
        sys.exit("错误:未配置 ZHIPUAI_API_KEY,请复制 .env.example 为 .env 并填入密钥。")

    # 检查 2:向量库存在且非空。两个检查缺一不可:
    # - 目录不存在:完全没入库;
    # - 目录存在但为空:上次入库可能失败到一半(目录会被提前创建),
    #   空库检索永远返回空,报错却发生在提问时,离根因很远。
    if not config.CHROMA_DIR.exists():
        sys.exit("错误:向量库不存在,请先运行 `python ingest.py` 完成文档入库。")

    if count_documents() == 0:
        sys.exit("错误:向量库是空的(上次入库可能失败了),请重新运行 `python ingest.py`。")

    # 构建一次链,循环内反复使用(链是无状态的,历史由本函数管理)
    chain = build_rag_chain()
    # 对话历史:每轮追加 [HumanMessage, AIMessage],传给链的 chat_history 槽位。
    # 放在循环外,让历史跨轮次保留;clear 时的重新赋值 [] 即"失忆"。
    chat_history: list = []

    print("=" * 60)
    print(f"知识库问答就绪(模型:{config.CHAT_MODEL},输入 q 退出,输入 clear 清空对话)")
    print("=" * 60)

    while True:
        try:
            question = input("\n你 > ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C / Ctrl+D 也要体面退出
            print("\n再见!")
            break

        # 空输入:重新提示
        if not question:
            continue
        # 控制命令:q 退出 / clear 清空历史
        if question.lower() == "q":
            print("再见!")
            break
        if question.lower() == "clear":
            chat_history = []  # 重新赋值空列表,模型立即"失忆"
            print("已清空对话历史。")
            continue

        print("助手 > ", end="", flush=True)
        # 调用链:invoke 内部完成 历史改写 → 检索 → 提示词 → 生成 全流程。
        # 将来换流式输出:把 invoke 换成 stream,逐 chunk 打印(进阶练习 1)。
        answer = chain.invoke({"question": question, "chat_history": chat_history})
        print(answer)

        # 关键:本轮问答对追加进历史,供下一轮改写检索问题使用。
        # 只有成功拿到回答才追加——失败/中断的轮次不进历史,避免污染下一轮检索。
        chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])


# =====================================================================
# 文档入库(离线阶段)
# =====================================================================
def ingest() -> None:
    """RAG 离线入库主流程:加载 → 切分 → 向量化写入。

    为什么需要 --rebuild:入库是【纯增量】的,修改了切分参数(CHUNK_SIZE 等)
    后,旧 chunk 仍留在库里,新旧混检会污染检索结果,必须清空重建。
    """
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
    print(f"加载文档:{config.KNOWLEDGE_DIR}")
    documents = load_documents()
    if not documents:
        sys.exit(f"错误:{config.KNOWLEDGE_DIR} 下没有可加载的文档(支持 .md / .txt / .pdf)。")
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
