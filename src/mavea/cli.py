"""MAVEA 命令行入口。

用法：
    mavea --video input.mp4 --prompt "做一个30秒带货视频"
    mavea --video clip1.mp4 clip2.mp4 --image product.jpg --prompt "..." --output result.mp4
    mavea-web                    # 启动 Gradio WebUI
    mavea-mcp                    # 启动 MCP Server (stdio)
    mavea-mcp --transport sse    # 启动 MCP Server (HTTP)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="mavea",
        description="MAVEA - Multi-Agent Video Editing Assistant",
    )
    parser.add_argument(
        "--video", "-v",
        nargs="+",
        default=[],
        help="视频素材路径（可多个）",
    )
    parser.add_argument(
        "--image", "-i",
        nargs="+",
        default=[],
        help="图片素材路径（可多个）",
    )
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="剪辑需求描述",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出文件路径（默认自动生成）",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="最大迭代轮数（默认3）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出结果",
    )

    args = parser.parse_args()

    # 收集素材
    material_paths = [str(Path(p).resolve()) for p in args.video + args.image]
    if not material_paths:
        parser.error("请提供至少一个素材文件（--video 或 --image）")

    # 检查文件存在
    for p in material_paths:
        if not Path(p).exists():
            parser.error(f"文件不存在: {p}")

    # 检查 FFmpeg
    import shutil
    if not shutil.which("ffmpeg"):
        print("错误：未找到 FFmpeg，请先安装 FFmpeg 并加入 PATH", file=sys.stderr)
        sys.exit(1)

    # 运行流水线
    from mavea.agents.graph import run_pipeline

    print(f"🎬 MAVEA 开始处理 {len(material_paths)} 个素材")
    print(f"📝 需求: {args.prompt}")
    print("-" * 60)

    final_state = asyncio.run(run_pipeline(
        material_paths=material_paths,
        user_prompt=args.prompt,
        max_iterations=args.max_iterations,
    ))

    def _get(key, default=None):
        if isinstance(final_state, dict):
            return final_state.get(key, default)
        return getattr(final_state, key, default)

    exec_result = _get("execution_result")
    eval_result = _get("evaluation_result")

    if exec_result and exec_result.success:
        print("\n✅ 剪辑完成！")
        print(f"📁 输出: {exec_result.output_path}")
        print(f"🔧 工具调用: {exec_result.tool_call_count} 次")
        print(f"⏱️ 耗时: {exec_result.total_duration_ms / 1000:.1f}s")
        if eval_result:
            print(f"⭐ 质量评分: {eval_result.overall}/5.0")
            if eval_result.issues:
                print(f"⚠️ 问题: {'; '.join(eval_result.issues[:3])}")
    else:
        errors = _get("errors", ["未知错误"])
        print(f"\n❌ 剪辑失败: {'; '.join(str(e) for e in errors)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
