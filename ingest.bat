@echo off
REM ============================================================
REM  文档入库脚本:把 docs/ 下的文档向量化写入本地向量库
REM  用法:
REM    ingest.bat              增量入库
REM    ingest.bat --rebuild    清空向量库后重建(改了切分参数后用)
REM  参数会原样传给 ingest.py(%*)
REM  注意:本文件必须保存为 ANSI/GBK 编码
REM ============================================================
uv run ingest.py %*
pause
