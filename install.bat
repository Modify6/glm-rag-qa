@echo off
setlocal
REM ============================================================
REM  一键安装脚本:检查/安装 uv -> 按锁文件创建虚拟环境并装依赖
REM  用法:双击运行,或在项目目录执行 install.bat
REM  注意:本文件必须保存为 ANSI/GBK 编码(cmd 默认代码页才能正确解析中文)
REM ============================================================

echo ===== LangChain RAG 环境安装 =====

REM ---- 第 1 步:确认 uv 可用 ----
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [1/2] 未检测到 uv,正在通过 winget 安装 Astral uv ...
    winget install -e --id astral-sh.uv --accept-source-agreements --accept-package-agreements
)

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [提示] uv 刚刚安装完成,但当前终端的 PATH 还没有刷新。
    echo        请关闭本窗口,重新打开终端后【再次运行本脚本】即可继续。
    echo.
    pause
    exit /b 1
)
echo [1/2] uv 已就绪:
for /f "tokens=*" %%i in ('uv --version') do echo        %%i

REM ---- 第 2 步:同步依赖 ----
REM uv sync 会:
REM   1. 读取 .python-version,准备 Python 3.12(必要时自动下载)
REM   2. 在项目下创建/更新 .venv
REM   3. 按 uv.lock 锁定的精确版本安装依赖(走 pyproject.toml 里配置的清华镜像)
echo [2/2] 同步依赖(uv sync)...
uv sync
if %errorlevel% neq 0 (
    echo [错误] 依赖同步失败,请检查网络后重试。
    pause
    exit /b 1
)

echo.
echo [完成] 环境就绪!接下来的三步:
echo   1. 复制 .env.example 为 .env,填入你的 ZHIPUAI_API_KEY
echo   2. 运行 ingest.bat 完成文档入库(换文档后用 ingest.bat --rebuild 重建)
echo   3. 运行 start.bat 开始知识库问答
echo.
pause
