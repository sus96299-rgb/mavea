"""MCP Client 封装：供工具执行 Agent 通过 MCP 协议调用工具。
使用 MCP v2 的 Client API，支持进程内直连（in-process）模式。
Agent 在同一进程内调用工具，无需子进程或网络开销。
"""
from __future__ import annotations

import time
from typing import Any

import structlog
from mcp import Client

from mavea.mcp.server import mcp as mavea_mcp_server
from mavea.progress import report_tool

logger = structlog.get_logger(__name__)


class MCPClient:
    """MCP 客户端封装。

    使用 MCP v2 的进程内 Client（直接传入 MCPServer 对象），
    走完整的 MCP 协议栈（参数校验、序列化），但无网络/子进程开销。
    """

    def __init__(self):
        self._server = mavea_mcp_server
        self._client: Client | None = None
        self._tools_cache: list[str] = []

    async def __aenter__(self) -> MCPClient:
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def initialize(self) -> None:
        """初始化：连接 Server 并发现工具。"""
        self._client = Client(self._server)
        await self._client.__aenter__()
        tools_result = await self._client.list_tools()
        self._tools_cache = [t.name for t in tools_result.tools]
        logger.info("mcp.client.initialized", tools=self._tools_cache)

    def list_tools(self) -> list[str]:
        """返回所有可用工具名。"""
        return list(self._tools_cache)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用工具，并实时上报工具调用状态（供 WebUI 工具日志滚动显示）。

        Args:
            name: 工具名
            arguments: 工具参数字典

        Returns:
            工具返回的 dict（success/data/error 结构）
        """
        if self._client is None:
            await self.initialize()
        if name not in self._tools_cache:
            report_tool(name, "error", 0, f"未知工具: {name}")
            return {
                "success": False,
                "error": f"未知工具: {name}，可用: {self._tools_cache}",
            }
        start = time.time()
        report_tool(name, "start")
        logger.info("mcp.client.call_tool", tool=name, args=list(arguments.keys()))
        try:
            result = await self._client.call_tool(name, arguments)
            duration_ms = int((time.time() - start) * 1000)
            # MCP v2 返回 CallToolResult，structured_content 是工具返回的结构化数据
            if result.structured_content:
                parsed = result.structured_content
            elif result.content:
                # 降级：从 text content 解析 JSON
                parsed = {"success": False, "error": "工具返回空结果"}
                for block in result.content:
                    if hasattr(block, "text"):
                        import json
                        try:
                            parsed = json.loads(block.text)
                        except json.JSONDecodeError:
                            parsed = {"success": True, "data": {"text": block.text}}
                        break
            else:
                parsed = {"success": False, "error": "工具返回空结果"}

            ok = bool(parsed.get("success", False))
            if ok:
                report_tool(name, "success", duration_ms)
            else:
                report_tool(
                    name, "error", duration_ms,
                    str(parsed.get("error", "工具执行失败"))[:120],
                )
            return parsed
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            logger.error("mcp.client.tool_error", tool=name, error=str(e))
            report_tool(name, "error", duration_ms, str(e)[:120])
            return {"success": False, "error": str(e)}

    async def close(self) -> None:
        """关闭连接。"""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None


# 全局单例
_client: MCPClient | None = None


async def get_mcp_client() -> MCPClient:
    """获取 MCP 客户端单例。"""
    global _client
    if _client is None:
        _client = MCPClient()
        await _client.initialize()
    return _client
