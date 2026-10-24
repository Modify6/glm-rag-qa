"""rag.loaders 的单元测试:加载与切分是纯本地逻辑,不需要网络和 API Key。"""
from rag import config
from rag.loaders import load_documents, split_documents


def test_load_documents_returns_docs_with_source():
    """加载内置知识库:至少 1 个文档,且每个 Document 都带 source 元数据。"""
    documents = load_documents()
    assert len(documents) >= 1
    for doc in documents:
        assert doc.page_content.strip(), "正文不应为空"
        assert doc.metadata.get("source"), "Loader 应自动填入 source 文件路径"


def test_split_documents_respects_chunk_size():
    """切分后每个 chunk 不超过 CHUNK_SIZE(硬切上限),且 metadata 保留。"""
    chunks = split_documents(load_documents())
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.page_content) <= config.CHUNK_SIZE
        assert chunk.metadata.get("source"), "切分不应丢失 source 元数据"


def test_split_documents_produces_overlap():
    """overlap 生效:相邻 chunk 之间存在内容重叠。

    注意语义:chunk_overlap 是"每次切分边界最多保留 N 字符"的上限,
    实际重叠量取决于边界处最后一个切分单元的长度(如最后一个标题行
    只有 26 字符,重叠就是 26),不是恒等于 CHUNK_OVERLAP。
    因此断言用严格的定义:存在 k>0 使 chunk1 的后缀 == chunk2 的前缀。
    """
    chunks = split_documents(load_documents())
    if len(chunks) < 2:
        return  # 语料太短切不出相邻片段时跳过
    first, second = chunks[0], chunks[1]
    assert first.metadata["source"] == second.metadata["source"], (
        "前两个 chunk 应来自同一文档,否则重叠断言无意义"
    )
    overlap = 0
    max_check = min(len(first.page_content), len(second.page_content), config.CHUNK_OVERLAP + 50)
    for k in range(1, max_check):
        if first.page_content.endswith(second.page_content[:k]):
            overlap = k
    assert overlap > 0, "相邻 chunk 之间应存在内容重叠(chunk_overlap > 0)"
