@echo off
chcp 65001 >nul
title MAVEA
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo [错误] 在当前目录找不到 .venv 虚拟环境。
  echo 请把本文件放到项目根目录（和 .venv 文件夹同一层）。
  echo 当前目录：%cd%
  pause
  exit /b
)

call .venv\Scripts\activate.bat
start "" cmd /c "timeout /t 6 >nul & start http://127.0.0.1:7860"

echo ============================================
echo   MAVEA 正在启动，约6秒后自动打开浏览器...
echo   用完直接关闭本窗口即可停止程序。
echo ============================================

mavea-web
echo.
echo 程序已退出（如上方有红字报错请截图）。
pause
