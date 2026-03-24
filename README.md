# RoomPartitioner

房间划分服务 — 将栅格地图自动划分为独立房间区域，支持手动调整。

## 工程结构

```
RoomPartitioner/
├── cli.py                           # 命令行入口（自动分区并导出图片）
├── app/
│   ├── main.py                      # FastAPI（uvicorn app.main:app）
│   ├── handler.py                   # Lambda handler（app.handler.handler）
│   ├── api/                         # 路由层
│   ├── core/                        # 核心编排/配置
│   ├── services/                   # 分区/合并实现
│   ├── pipeline/                   # 图像预处理/推理/后处理
│   ├── schemas/                    # 请求模型（Pydantic）
│   └── utils/                      # 坐标/图结构/序列化/S3 等
├── config/
│   └── default.yaml
├── tests/
├── dataset/                        # 可选数据目录
├── Dockerfile
├── Dockerfile.lambda
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

## 处理流程

```
输入地图 → 前处理 → Triton推理 → 后处理 → 自动分区 → 扩展分区 → [手动调整]
                                                                    ├── 手动划分
                                                                    └── 手动合并
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查 |
| POST | `/auto_partition` | 自动分区（返回JSON） |
| POST | `/auto_partition/image` | 自动分区（返回图片） |
| POST | `/extend_partition` | 扩展分区 |
| POST | `/manual/split_line` | 画线分割 |
| POST | `/manual/split_polyline` | 折线分割 |
| POST | `/manual/assign_polygon` | 多边形划分 |
| POST | `/manual/merge` | 按ID合并房间 |
| POST | `/manual/merge_by_point` | 点选合并 |
| GET  | `/current/image` | 当前结果图片 |
| GET  | `/current/info` | 当前房间信息 |

## 部署

```bash
docker compose up -d
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRITON_URL` | (空) | Triton 服务地址，如 `triton:8001` |
| `MODEL_NAME` | `room_seg` | Triton 模型名 |
| `TRITON_PROTOCOL` | `grpc` | `grpc` 或 `http` |
| `MIN_ROOM_AREA` | `1.0` | 最小房间面积 m² |
| `WALL_THRESHOLD` | `128` | 墙壁像素阈值 |
| `RESOLUTION` | `0.05` | 地图分辨率 m/pixel |
| `TARGET_SIZE` | `512,512` | 模型输入尺寸 |
| `DOOR_WIDTH` | `20` | 门口宽度阈值(像素) |
