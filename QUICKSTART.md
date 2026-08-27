# MAVEA 快速部署指南（Windows + Python 3.13）

## 前置要求

- **Python 3.10+**（推荐 3.12 或 3.13）
- **FFmpeg**：必须安装并加入系统 PATH
- **Git**（用于推送到 GitHub）
- **API Key**：通义千问（免费额度）或 DeepSeek

---

## 第一步：安装 FFmpeg

如果还没装 FFmpeg：

1. 打开 https://www.gyan.dev/ffmpeg/builds/ 下载 `ffmpeg-release-essentials.zip`
2. 解压到 `C:\ffmpeg`
3. 把 `C:\ffmpeg\bin` 添加到系统环境变量 PATH
4. 重新打开 PowerShell，验证：
```powershell
ffmpeg -version
```

---

## 第二步：解压项目

把 `mavea-complete.zip` 解压到：
```
D:\pycharm-exercise\14_Multi-Agent-Video-Editing-Assistant\
```

解压后目录结构应该是：
```
D:\pycharm-exercise\14_Multi-Agent-Video-Editing-Assistant\
├── src\mavea\
├── docs\
├── tests\
├── examples\
├── pyproject.toml
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

---

## 第三步：创建虚拟环境并安装

```powershell
cd "D:\pycharm-exercise\14_Multi-Agent-Video-Editing-Assistant"

# 创建虚拟环境（指定 Python 3.13，如果装了多个版本用 py -3.13）
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate

# 升级 pip
python -m pip install --upgrade pip

# 安装项目（开发模式，含所有依赖）
pip install -e ".[dev]"
```

> 安装过程会下载 torch、transformers、chromadb 等大包，约 3-5GB，需要 10-20 分钟。

---

## 第四步：验证安装

```powershell
# 验证包能导入
python -c "import mavea; print(mavea.__version__)"
# 预期输出：0.1.0

# 验证 CLI 入口
mavea --help
# 预期：显示帮助信息

# 验证图构建
python -c "from mavea.agents.graph import build_graph; g = build_graph(); print('节点:', list(g.nodes))"
# 预期：节点: ['__start__', 'analyzer', 'planner', 'executor', 'evaluator', 'increment']

# 验证 FFmpeg
ffmpeg -version
```

---

## 第五步：配置 API Key

```powershell
# 复制环境变量模板
copy .env.example .env

# 用记事本打开编辑
notepad .env
```

在 `.env` 中至少配置以下内容：

```env
# 使用通义千问（推荐，有免费额度）
MAVEA_LLM__PROVIDER=qwen
MAVEA_LLM__QWEN_API_KEY=sk-你的key填这里

# 测试阶段建议跳过语音转写（避免下载 Whisper 模型）
MAVEA_VIDEO__SKIP_TRANSCRIBE=true
```

通义千问 API Key 获取：https://dashscope.console.aliyun.com/apiKey

---

## 第六步：启动 WebUI

```powershell
mavea-web
```

看到以下输出说明启动成功：
```
Running on local URL: http://127.0.0.1:7860
```

浏览器打开 http://127.0.0.1:7860

---

## 第七步：测试剪辑

1. 上传一个 **5-15秒的短视频**（MP4格式）
2. 输入需求，例如：`帮我做一个10秒的视频，加个标题"测试"`
3. 点击"开始剪辑"
4. **观察 PowerShell 终端**，会有实时日志：
   ```
   [Analyzer] 探测视频信息: xxx.mp4
   [Analyzer] 视频信息: 1920x1080, 10.0s, h264, 音频=无
   [Analyzer] 检测场景边界...
   [Analyzer] 调用视觉模型描述画面（Qwen-VL）...
   [Planner] 视频类型: vlog_highlight
   [Planner] 调用 LLM 生成剪辑方案...
   [Executor] 一站式渲染时间轴...
   [Evaluator] 评估完成: 总分 3.5/5.0, 通过
   ```
5. 网页上的进度条和"实时进度"框会同步更新

> **首次运行注意**：会下载 BGE 向量模型（~1.3GB），已自动配置国内镜像 hf-mirror.com，耐心等待。

---

## 第八步：推送到 GitHub

确认本地能跑通后：

```powershell
# 在项目根目录初始化 Git
git init
git add .
git commit -m "feat: MAVEA v0.1.0 - multi-agent video editing assistant"

# 在 GitHub 上创建仓库 mavea（public），然后：
git branch -M main
git remote add origin https://github.com/你的用户名/mavea.git
git push -u origin main
```

> `.env` 文件已在 `.gitignore` 中，不会被提交（API Key 不会泄露）。

---

## 常见问题

### Q: 启动时报 `NameError: APISettings`
A: 用最新的 zip 覆盖 `src/mavea/config.py`。

### Q: 卡在"正在分析素材"不动
A: 看 PowerShell 终端最后一行日志。如果是模型下载慢，等几分钟；如果是 API 报错，检查 Key 是否正确。

### Q: `ModuleNotFoundError: No module named 'mavea'`
A: 确保虚拟环境已激活（命令行前面有 `(.venv)`），并执行了 `pip install -e .`。

### Q: FFmpeg 报错
A: 确保 `ffmpeg -version` 能正常输出。视频编码建议用 H.264 MP4。

### Q: Qwen API 报错
A: 检查 `.env` 中 Key 是否正确，通义千问需要开通 qwen-plus 和 qwen-vl-max 模型（新用户免费额度自动覆盖）。

### Q: 想用 DeepSeek 代替 Qwen
A: 在 `.env` 中设置 `MAVEA_LLM__PROVIDER=deepseek` 和 `MAVEA_LLM__DEEPSEEK_API_KEY=sk-xxx`。但视觉理解仍需 Qwen 或 OpenAI 的 Key。

---

## 三种使用方式

```powershell
# 1. WebUI（推荐演示用）
mavea-web

# 2. 命令行
mavea --video test.mp4 --prompt "做一个15秒产品展示"

# 3. MCP Server（接入 Claude Desktop）
mavea-mcp
```
