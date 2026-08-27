"""MAVEA — Multi-Agent Video Editing Assistant
多Agent智能视频剪辑助手：用户上传商品图片/视频片段+输入营销文案，
系统通过多个AI Agent协作，自动完成素材分析→脚本规划→剪辑执行→质量评估全流程。
"""
import os as _os

# 必须在任何 huggingface 库导入前设置国内镜像
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

__version__ = "0.1.0"
