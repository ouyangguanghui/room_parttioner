# RoomPartitioner

房间划分服务 — 将栅格地图自动划分为独立房间区域，支持手动调整。

## 工程结构

```
RoomPartitioner/
├── cli.py                           # 命令行入口（自动分区并导出图片）
├── app/
│   ├── main.py                      # FastAPI 入口（uvicorn app.main:app）
│   ├── handler.py                   # Lambda 入口（app.handler.handler）
│   ├── core/
│   │   ├── config.py                # 配置加载（yaml + 环境变量）
│   │   └── errors.py                # 结构化业务异常
│   ├── services/
│   │   ├── services.py              # 总接口 RoomService（路由 + 预处理）
│   │   ├── auto_partition.py        # 自动分区（Triton 推理 / 连通域 fallback）
│   │   ├── extended_partition.py    # 扩展分区（增量检测新区域 + 归属判定）
│   │   ├── manual_partition.py      # 手动划分（画线分割房间）
│   │   └── manual_merge.py          # 手动合并（Shapely union）
│   ├── pipeline/
│   │   ├── preprocessor.py          # 地图前处理（去噪、补墙、平滑）
│   │   ├── inferencer.py            # Triton 推理 + OBB 解码 + NMS
│   │   ├── postprocessor.py         # 推理后处理（OBB → label_map）
│   │   └── triton_client.py         # Triton gRPC/HTTP 客户端
│   └── utils/
│       ├── coordinate.py            # 像素 ↔ 世界坐标变换
│       ├── geometry_ops.py          # Shapely 几何工具（分割、合并）
│       ├── graph.py                 # 邻接图 + 五色着色 + DFS 排序
│       ├── landmark.py              # 平台标记点生成
│       ├── s3_loader.py             # S3 / 本地数据加载
│       ├── contour_expander.py      # 轮廓外扩
│       ├── beautifier.py            # 轮廓美化（bbox + 门槛线）
│       └── common.py                # 通用验证工具
├── config/
│   └── default.yaml                 # 默认配置
├── tests/                           # 单元测试（217+ tests）
├── dataset/                         # 可选数据目录
├── Dockerfile                       # HTTP 部署镜像
├── Dockerfile.lambda                # Lambda 部署镜像
├── docker-compose.yml               # 编排配置
├── requirements.txt                 # 生产依赖
├── requirements-lambda.txt          # Lambda 精简依赖
└── requirements-dev.txt             # 开发依赖
```

## 处理流程

```
Lambda event / HTTP request
    ↓
RoomService.room_edit(operation)
    ├── 预处理: Preprocessor (去噪 → 补墙 → 平滑)
    ├── 公共工具: CoordinateTransformer, RoomGraph, LandmarkManager
    └── 按 operation 路由:
        ├── split (无labels)   → AutoPartitioner   (Triton推理 / 连通域)
        ├── split (有labels)   → ExtendedPartitioner (增量检测新区域)
        ├── repartition        → AutoPartitioner   (强制重新分区)
        ├── division           → ManualPartitioner  (画线分割)
        └── merge              → ManualMerger       (Shapely合并)
```

## 部署

### Lambda 部署（生产）

```bash
docker build -f Dockerfile.lambda -t room-partitioner-lambda .
```

入口: `app.handler.handler`

### HTTP 部署（本地开发）

```bash
# 方式1: 直接启动
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 方式2: Docker
docker compose up --build

# 接口: POST /room_edit, GET /health
```

### CLI（快速测试）

```bash
python cli.py -i map.png -o result.png -r 0.05
```

## Lambda 接口

```json
// 请求
{
    "operation": "split | repartition | merge | division",
    "bucket": "my-bucket",
    "key": "path/to/map",
    "roomMergeList": ["ROOM_001", "ROOM_002"],
    "divisionCroodsDict": {"id": "ROOM_001", "A": [x,y], "B": [x,y]}
}

// 响应
{
    "statusCode": 200,
    "body": "<labels_json>"
}
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
| `ROOM_PARTITIONER_DEBUG` | `0` | 调试模式开关 |
| `ROOM_PARTITIONER_LOCAL_DIR` | (空) | 调试模式本地数据目录 |

## 本地调试模式（跳过 S3）

```bash
export ROOM_PARTITIONER_DEBUG=1
export ROOM_PARTITIONER_LOCAL_DIR=/path/to/case_dir
```

本地目录应至少包含：`saved_map.png`、`saved_map.json`

可选文件：`labels.json`、`mapinfo.json`、`markers.json`（K20 机型）
