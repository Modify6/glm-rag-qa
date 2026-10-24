"""rag.chain 的单元测试:纯函数与启动守卫,不触发任何网络调用。"""
import pytest

from rag import config
from rag.chain import _format_docs, build_rag_chain


def test_format_docs_with_documents():
    """有检索结果时:输出带编号和来源,顺序与输入一致。"""
    from langchain_core.documents import Document

    docs = [
        Document(page_content="内容甲", metadata={"source": "a.md"}),
        Document(page_content="内容乙", metadata={"source": "b.md"}),
    ]
    text = _format_docs(docs)
    assert "[1]" in text and "[2]" in text
    assert "a.md" in text and "b.md" in text
    assert text.index("内容甲") < text.index("内容乙"), "保持检索排序"


def test_format_docs_empty_returns_placeholder():
    """空检索结果:返回占位文本而不是空串,配合系统规则引导模型拒答。"""
    assert _format_docs([]) == "(未检索到相关内容)"


def test_format_docs_missing_source_falls_back():
    """元数据缺 source 时不崩溃,回退为"未知来源"。"""
    from langchain_core.documents import Document

    text = _format_docs([Document(page_content="x", metadata={})])
    assert "未知来源" in text


def test_build_rag_chain_requires_api_key(monkeypatch):
    """没配 API Key 时,构建链应立刻友好退出,而不是深处抛 pydantic 错误。"""
    monkeypatch.setattr(config, "ZHIPUAI_API_KEY", "")
    with pytest.raises(SystemExit, match="ZHIPUAI_API_KEY"):
        build_rag_chain()
