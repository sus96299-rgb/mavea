# MAVEA - Multi-Agent Video Editing Assistant

<div align="center">

**多 Agent 智能视频剪辑助手**

上传商品图片/视频片段 + 输入营销文案，AI Agent 自动完成素材分析→脚本规划→剪辑执行→质量评估全流程，输出可发布的带货短视频。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-2.1.0-purple)](https://modelcontextprotocol.io)

</div>

## 核心特性

- **4 阶段 Agent 流水线**：素材分析 → 剪辑规划 → 工具执行 → 质量评估，LangGraph 状态图编排；评估不通过自动带着反馈返工（最多3轮），并根据历轮分数收敛情况自适应提前停止
- **Planner 内嵌「增强导演」**：在卡点/美颜/AI 配音的白名单工具池内做受限自主决策（等价单次 function-calling），任何失败都回退默认，不阻断主流程
- **MCP v2 工具服务**：11 个视频处理工具通过 MCP Server 暴露，支持 stdio（Claude Desktop/Cursor）和 streamable-http 双传输；Agent 内部走 in-process transport 复用完整协议校验，无跨进程开销
- **RAG 剪辑模板检索**：Chroma 向量 + BM25 关键词混合检索 + BGE Reranker 重排序，根据需求匹配最佳剪辑模板
- **视觉理解**：Qwen-VL 逐镜头生成画面描述，faster-whisper 音频转写
- **无参考质量评估**：模糊度/黑帧/响度爆音/时长偏差等硬指标程序化检测成片 + LLM-as-Judge 评审「方案-需求」匹配度
- **LLM 可靠性护栏**：素材强制全覆盖、目标时长锁定、BGM 兜底等确定性后处理，防止模型漏素材、时长漂移或成片无声
- **三种使用方式**：Gradio WebUI / FastAPI / 命令行

## 架构

```
用户输入（素材+需求）
       │
       ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  Analyzer   │───▶│   Planner    │───▶│  Executor   │───▶│  Evaluator  │
│  场景检测    │    │  RAG检索      │    │ 一站式渲染   │    │ 画质/响度   │
│  Qwen-VL    │    │  LLM规划      │    │ 失败降级逐步 │    │ 黑帧/时长   │
│  Whisper    │    │ +增强导演*    │    │ MCP工具调用  │    │ LLM方案评审 │
└─────────────┘    └──────────────┘    └─────────────┘    └──────┬──────┘
                                                           │
                                              passed? ─────┤
                                                ├─ Yes ──▶ END
                                                └─ No ──▶ Planner（最多3轮，分数收敛则提前停止）

* 增强导演：Planner 内的受限子决策，在卡点/美颜/AI 配音白名单内自主选择，非独立 Agent 节点
```

## 快速开始

### 1. 环境要求

- Python 3.10+（推荐 3.12/3.13）
- FFmpeg（`ffmpeg` 和 `ffprobe` 在 PATH 中）
- API Key：DeepSeek（主 LLM）+ Qwen（视觉理解，可用 OpenAI 替代）

### 2. 安装

```bash
git clone <repo-url> && cd mavea
pip install -e .

cp .env.example .env
# 编辑 .env，填入 API Key
```

### 3. 使用

**WebUI（推荐）**：
```bash
mavea-web
# 浏览器打开 http://127.0.0.1:7860
```
Windows 用户也可双击项目根目录的 `start.bat` 一键启动（自动激活 .venv 并打开浏览器）。

**命令行**：
```bash
mavea --video product.mp4 --prompt "做一个30秒运动鞋带货短视频，前3秒展示上脚效果"
mavea --image product1.jpg product2.jpg --prompt "30秒产品展示，突出材质细节"
```

**MCP Server（接入 Claude Desktop / Cursor）**：
```bash
mavea-mcp                      # stdio 模式
mavea-mcp --transport sse      # HTTP 模式，端口 8765
```

Claude Desktop 配置示例（`claude_desktop_config.json`）：
```json
{
  "mcpServers": {
    "mavea": {
      "command": "mavea-mcp",
      "args": []
    }
  }
}
```

## MCP 工具清单

| 工具 | 功能 |
|------|------|
| `get_video_info` | 获取视频元信息（分辨率/帧率/时长/编码/音轨） |
| `cut_clip` | 裁剪视频片段 |
| `concat_videos` | 拼接视频（支持 fade/dissolve/wipe/zoom/slide 转场） |
| `add_subtitle` | 烧录 SRT 字幕 |
| `add_text_overlay` | 添加文字贴纸（7种位置） |
| `add_bgm` | 添加背景音乐（混音保留人声/替换） |
| `image_to_video` | 图片转视频（Ken Burns 推拉效果） |
| `extract_audio` | 提取音频（wav/mp3/aac） |
| `transcribe_audio` | Whisper 语音转文字 |
| `detect_scenes` | PySceneDetect 镜头边界检测 |
| `create_video_from_timeline` | 一站式时间轴渲染 |

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 编排 | LangGraph 1.x |
| MCP 协议 | MCP Python SDK 2.1（2026-07-28 stateless 规范） |
| LLM | DeepSeek / Qwen-VL / OpenAI（OpenAI 兼容接口统一抽象） |
| 向量检索 | Chroma + BGE-large-zh + BM25 + BGE-Reranker-v2-m3 |
| 视频处理 | FFmpeg + PySceneDetect + OpenCV |
| 语音 | faster-whisper + edge-tts |
| 接口 | FastAPI + Gradio 6.x |
| 部署 | Docker + HuggingFace Space |

## 项目结构

```
mavea/
├── src/mavea/
│   ├── agents/          # 4个Agent节点 + director增强导演子决策 + graph.py状态图
│   ├── mcp/             # MCP Server/Client + 11个工具
│   ├── rag/             # 模板JSON + 向量库 + 混合检索 + 素材索引
│   ├── video/           # FFmpeg封装/场景检测/帧提取/质量指标
│   ├── llm/             # LLM抽象层（DeepSeek/Qwen/OpenAI）
│   ├── api/             # FastAPI 路由
│   ├── web/             # Gradio WebUI
│   ├── models.py        # 共享Pydantic模型（接口契约）
│   ├── config.py        # pydantic-settings 配置
│   └── cli.py           # 命令行入口
├── docs/
│   └── interfaces.md    # 接口契约（项目宪法）
├── tests/
├── examples/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 安全设计

- FFmpeg 命令全部参数数组构建（`subprocess.run` list 形式），禁止 `shell=True`
- MCP 工具输入全部 Pydantic 模型校验
- API Key 通过环境变量读取，禁止硬编码
- 输出路径限制在工作目录内，防止路径穿越

## License

MIT
