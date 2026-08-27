"""FastAPI 应用入口。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mavea import __version__
from mavea.api.routes import router

app = FastAPI(
    title="MAVEA API",
    description="Multi-Agent Video Editing Assistant API",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def main():
    """启动 API 服务器。"""
    import uvicorn

    from mavea.config import get_settings
    settings = get_settings()
    uvicorn.run(
        "mavea.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
