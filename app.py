"""知识库问答入口:命令行多轮对话 REPL。

在线阶段:读输入 → RAG 链(检索 + 生成)→ 打印回答 → 追加历史。

交互命令:
    任意问题   检索知识库并回答(带来源引用)
    clear      清空对话历史(之后的提问不再参考前文)
    q / Ctrl+C 退出

入口脚本职责:命令行 I/O、启动检查、历史管理。
检索、提示词、模型调用的逻辑全部在 rag/chain.py,这里只调用。
"""
import sys

from langchain_core.messages import AIMessage, HumanMessage

import rag.config as config
from rag.chain import build_rag_chain


def main() -> None:
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

    from rag.vectorstore import count_documents

    if count_documents() == 0:
        sys.exit("错误:向量库是空的(上次入库可能失败了),请重新运行 `python ingest.py`。")

    # 构建一次链,循环内反复使用(链是无状态的,历史由本文件管理)
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


if __name__ == "__main__":
    main()
