"""FastAPI 应用创建 + 生命周期"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import load_config
from app.core.partitioner import RoomPartitioner
from app.api import routes

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("room_partitioner")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化，关闭时清理"""
    config = load_config()
    partitioner = RoomPartitioner(config)
    # 使用 app.state 保存运行时依赖，避免模块级全局变量带来的并发风险
    app.state.partitioner = partitioner
    app.state.config = config
    logger.info("RoomPartitioner 服务启动完成")
    logger.info("Triton: %s", config.get("triton_url") or "未配置 (fallback 模式)")
    yield
    logger.info("RoomPartitioner 服务关闭")


app = FastAPI(
    title="RoomPartitioner",
    version="0.2.0",
    description="房间划分服务 - 自动/手动分区 API",
    lifespan=lifespan,
)

app.include_router(routes.router)
