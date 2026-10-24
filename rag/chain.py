"""RAG 链路组装:检索 → 组装提示词 → 大模型回答(全工程核心模块)。

链路结构(LCEL):
    输入 {"question", "chat_history"}
      → 改写(把历史并入问题,便于检索)
      → 并行分支{context: 检索+格式化, question, chat_history}
      → prompt → ChatZhipuAI → 字符串输出
对话历史列表由 app.py 维护,每轮追加一轮问答对。

两个易踩的 LCEL 语义(真实踩过的坑,详见 CODE_GUIDE.md 5.2/5.3):
1. RunnableParallel 的每个分支拿到的都是【完整的输入 dict】,
   不是"自己那条管道的数据"——所以分支内必须先 itemgetter 取字段;
2. 并行 dict 的输出【只包含分支里显式列出的键】,提示词模板用到的
   每个变量都必须在这里备齐,少一个就报 missing variables。
"""
from operator import itemgetter

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag import config
from rag.vectorstore import as_retriever

# 系统提示词:三条规则是有意设计的"行为宪法"——
# 1. 只准用资料回答:压制模型自由发挥,防幻觉的前提;
# 2. 没有就明说:给模型一个体面的拒答出口,否则它会在"规则 1"和
#    "有问必答"的倾向之间挣扎,反而更容易编造;
# 3. 注明来源:可溯源。来源信息来自 _format_docs 的编号格式。
SYSTEM_TEMPLATE = """你是一个严谨的知识库问答助手。请仅根据下面提供的参考资料回答用户的问题。

规则:
1. 只使用参考资料中的信息回答,不要编造。
2. 如果资料里没有相关内容,直接说明"知识库中没有找到相关信息"。
3. 回答用中文,尽量给出要点,并注明参考了哪些文档。

参考资料:
{context}"""


def _format_docs(docs) -> str:
    """把检索到的 chunk 拼成一段带编号、带来源的参考资料文本。

    检索器返回的是 Document 列表,直接拼进提示词模型分不清哪段来自
    哪个文件;编号 + 来源让模型能在回答里引用"[2]",引用可定位。
    空检索时返回占位文本而不是空串:空串会让"参考资料"区块看起来
    像正常但无内容,模型容易自行脑补;明确的占位文本配合系统规则 2,
    能稳定引导模型拒答。
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[{i}] 来源:{source}\n{doc.page_content}")
    return "\n\n".join(parts) if parts else "(未检索到相关内容)"


def build_rag_chain():
    """组装完整 RAG 链,返回的 Runnable 接口:
        输入: {"question": str, "chat_history": [HumanMessage/AIMessage, ...]}
        输出: str(模型回答文本)
    """
    # 启动前拦截缺 Key:比让 pydantic 在模型构造时抛 ValidationError 友好得多
    if not config.ZHIPUAI_API_KEY:
        raise SystemExit("错误:未配置 ZHIPUAI_API_KEY,请复制 .env.example 为 .env 并填入密钥。")

    # temperature=0.1 刻意调低:知识问答要的是忠实于资料,不是文采飞扬;
    # 显式传 api_key 而非依赖环境变量推断,把失败提前到构造时刻。
    llm = ChatZhipuAI(model=config.CHAT_MODEL, temperature=0.1, api_key=config.ZHIPUAI_API_KEY)
    retriever = as_retriever()

    # 提示词模板:三条消息的组合。
    # MessagesPlaceholder("chat_history") 是一个"可伸缩槽位":
    # chat_history 是空列表就什么都不插入,是 4 条消息就插 4 条。
    # 多轮对话时它让生成模型看到原始对话,保持语气连贯。
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    # RAG 链本体(LCEL 从左到右读,数据从左流向右):
    #
    # 第一步 RunnableParallel:每个分支都拿到同一份输入 dict,
    # 各自执行后合并成一个新 dict。三个分支:
    # - context 分支:itemgetter("question") 先取出问题字符串
    #   【不能省!直接接 retriever 的话整个 dict 会被送去向量化,
    #   智谱 API 报 1210 参数错误】→ 检索 → _format_docs 格式化;
    # - question / chat_history 分支:纯透传
    #   【也不能省!并行 dict 的输出只有这里列出的键,
    #   模板要用的变量必须全部备齐,否则报 missing variables】。
    rag_chain = (
        {
            "context": itemgetter("question")
            | retriever
            | RunnableLambda(_format_docs),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt          # 把三个键填入模板,产出渲染好的消息列表
        | llm             # 调用智谱生成 AIMessage
        | StrOutputParser()  # 剥出 .content,转成纯字符串
    )

    def rewrite_with_history(inputs: dict) -> dict:
        """多轮对话的检索预处理(两段式历史设计)。

        检索器和生成模型对上下文的需求不同:
        - 检索器需要"完整独立的问题":用户第二轮说"它是什么",
          拿"它是什么"去检索什么都搜不到,所以要把历史并入问题文本;
        - 生成模型需要原始对话保持连贯:原始 Message 列表经
          chat_history 槽位传入,不受改写影响。
        """
        question = inputs["question"]
        history = inputs.get("chat_history", [])
        # 无历史(第一轮):原样通过,不做任何加工
        if not history:
            return {"question": question, "chat_history": []}
        # 有历史:把最近 3 轮(history[-6:],每轮 2 条消息)拼成文本
        # 并入问题。截断是成本控制——历史越长 token 越贵、越干扰检索。
        history_text = "\n".join(
            f"{'用户' if getattr(m, 'type', '') == 'human' else '助手'}: {m.content}"
            for m in history[-6:]
        )
        # 改写后历史已并入问题文本,槽位置空即可,不丢信息
        return {
            "question": f"对话历史:\n{history_text}\n\n当前问题:{question}",
            "chat_history": [],
        }

    # 对外暴露的最终链:
    # RunnablePassthrough 透传输入 → 历史改写 → RAG 链本体。
    # 更严格的实现是把拼接改写换成"调一次 LLM 生成独立问题"
    # (condense question 模式),属于 README 进阶练习第 2 项。
    return RunnablePassthrough() | RunnableLambda(rewrite_with_history) | rag_chain
