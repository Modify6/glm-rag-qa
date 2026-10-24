"""全局配置模块:整个项目唯一的 .env 读取点。

设计意图:
- 其他模块只 import 这里的常量(如 config.CHUNK_SIZE),绝不自己读环境变量;
- 这样将来更换配置方式(真实环境变量、云配置中心)时,只需改这一个文件。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录:由当前文件位置向上推导(rag/config.py → rag/ → 项目根)。
# 好处:无论从哪个工作目录启动脚本,docs/ 和 data/ 都能正确定位,
# 避免"在别的目录执行 python D:\...\ingest.py 时找不到文档"的问题。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 把 .env 文件中的"键=值"注入进程环境变量(已存在的环境变量不会被覆盖)。
# 之后所有 os.getenv() 才能读到。.env 已被 .gitignore 排除,密钥不会进 git。
load_dotenv(PROJECT_ROOT / ".env")

# ========== 智谱 API 密钥(硬依赖) ==========
# 没有它无法调用任何模型接口。入口脚本(ingest.py / app.py)启动时会检查,
# 为空则给出友好提示并退出。到 https://open.bigmodel.cn 控制台获取。
ZHIPUAI_API_KEY: str = os.getenv("ZHIPUAI_API_KEY", "")

# ========== 对话模型(生成回答用) ==========
# glm-4-flash:免费、速度快,知识问答场景够用;
# 追求更强推理可换 glm-4-plus(按量计费)。
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "glm-4-flash")

# ========== 向量模型(文本转向量用) ==========
# embedding-3:智谱新一代向量模型,输出 2048 维向量,中文效果好。
# 注意:入库和查询必须用同一个模型(不同模型的向量空间不互通,
# 本项目通过"只在 vectorstore.py 的 get_embeddings() 构造一次"来保证)。
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embedding-3")

# ========== 路径配置 ==========
# 知识库文档目录:ingest.py 递归扫描这里的 .md / .txt / .pdf
DOCS_DIR: Path = PROJECT_ROOT / "docs"
# 向量库持久化目录:Chroma 会在这里落盘(sqlite + 集合数据),
# 删掉整个目录 = 清空知识库,重新 ingest.py --rebuild 即可重建。
CHROMA_DIR: Path = PROJECT_ROOT / "data" / "chroma"

# ========== 文本切分参数(改完必须 ingest.py --rebuild 才生效) ==========
# 每个片段(chunk)的最大字符数。权衡:
#   太大 → 一个片段里只有一小部分和问题相关,检索不精准、浪费 token;
#   太小 → 语义被拦腰截断,检索到了也读不懂。
# 500 对中文文档是稳健起点。
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
# 相邻片段重叠的字符数(约取 chunk_size 的 15%)。
# 作用:关键句刚好跨在切分边界时,重叠区保证它至少在一个片段里是完整的。
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

# ========== 检索参数 ==========
# 每次提问从向量库取回的相似片段数。
# 太少 → 可能漏掉含答案的片段;太多 → 噪音多、token 费用高。
TOP_K: int = int(os.getenv("TOP_K", "4"))

# 补充说明:os.getenv 的返回值永远是字符串,所以数值型配置必须 int() 转换,
# 否则后续做数值比较时会静默出错。
