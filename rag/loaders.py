"""文档加载与切分:RAG 离线入库阶段的前两步。

流程:knowledge/ 下的文件 --load_documents()--> Document 列表 --split_documents()--> chunk 列表

本模块是纯函数(不碰网络、不碰向量库),只做文本 I/O 与切分,
是最容易写单元测试的部分。
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config

# 各扩展名对应的加载器(注册表模式:加新格式只需在此加一行,不改控制流)
# .md 和 .txt 都是纯文本,共用 TextLoader;PDF 由 PyPDFLoader 按页加载,
# 每页产生一个 Document(自动带页码元数据)。
LOADERS = {
    ".pdf": PyPDFLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}


def load_documents(knowledge_dir: Path | None = None) -> list[Document]:
    """加载目录下所有支持的文档,返回 Document 列表(每个文件/页一个)。

    Document 是 LangChain 对"一段带元数据的文本"的统一抽象:
        page_content: 正文文本(str)
        metadata:     元数据(dict),Loader 会自动填入 {"source": 文件路径}
    source 后续有两个用途:回答中注明来源;按文件过滤检索结果。

    不认识的扩展名静默跳过,保证目录里混入无关文件(如 .env.example)不报错。
    """
    knowledge_dir = knowledge_dir or config.KNOWLEDGE_DIR
    documents: list[Document] = []
    # rglob 递归遍历所有子目录;sorted 保证入库顺序稳定(便于复现和排查)
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file():
            continue
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        # 关键:TextLoader 不传 encoding 时会用操作系统默认编码,
        # 中文 Windows 是 GBK,读 UTF-8 的中文文档必然报 UnicodeDecodeError。
        # 因此必须显式指定 UTF-8(真实踩过的坑,详见 CODE_GUIDE.md 5.1)。
        # PyPDFLoader 自己解析二进制 PDF,不需要 encoding 参数。
        loader_kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
        documents.extend(loader_cls(str(path), **loader_kwargs).load())
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """把长文档切成 chunk,检索的最小单位就是 chunk。

    返回值结构和输入相同(仍是 Document 列表),metadata 原样保留,
    所以每个 chunk 都知道自己来自哪个文件。

    为什么必须切分:
    - 不切:整篇文档作为检索单位,一段 2000 字里只有 100 字相关,
      其余全是噪音,还烧 token;
    - 切太碎:句子被拦腰截断,检索到了也读不懂。
    chunk_size=500 + overlap=80 是中文文档的稳健起点,详见 README 调优指南。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # 递归切分策略:按分隔符列表"从大到小"逐级降级——
        # 先尝试按空行(段落)切,切出来还超过 chunk_size 就降级按换行、
        # 句号、分号继续切,最后一个 "" 表示实在不行才硬切字符。
        #
        # 注意:默认分隔符是 ["\n\n", "\n", " ", ""] ,对中文几乎失效——
        # 中文正文没有空格,前两级切完仍超长,最终全部退化成硬切。
        # 这里特意加入中文句末标点,让切分尽量落在句子边界上,
        # chunk 的语义完整度明显更好。
        separators=["\n\n", "\n", "。", "!", "?", ";", " ", ""],
    )
    return splitter.split_documents(documents)
