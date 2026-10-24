@echo off
REM ============================================================
REM  知识库问答启动脚本(命令行多轮对话)
REM  前置条件:已运行 install.bat 装好环境,ingest.bat 完成入库,
REM            且 .env 里已填 ZHIPUAI_API_KEY
REM  注意:本文件必须保存为 ANSI/GBK 编码
REM ============================================================
uv run app.py
pause
