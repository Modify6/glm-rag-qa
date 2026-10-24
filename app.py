"""知识库问答入口(薄壳)。

真实逻辑在 rag/cli.py 的 chat()——它同时被 pyproject.toml 的
[project.scripts] 注册为 rag-qa 命令。保留这个薄壳是为了
`uv run app.py` / `python app.py` 也能直接运行。
"""
from rag.cli import chat

if __name__ == "__main__":
    chat()
