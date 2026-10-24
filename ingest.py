"""文档入库入口(薄壳)。

真实逻辑在 rag/cli.py 的 ingest()——它同时被 pyproject.toml 的
[project.scripts] 注册为 rag-ingest 命令。保留这个薄壳是为了
`uv run ingest.py --rebuild` / `python ingest.py` 也能直接运行。

用法:
    python ingest.py              # 增量入库 knowledge/ 下所有文档
    python ingest.py --rebuild    # 清空向量库后重新入库
"""
from rag.cli import ingest

if __name__ == "__main__":
    ingest()
