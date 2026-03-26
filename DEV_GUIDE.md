# RoomPartitioner 开发流程与测试指南

## 一、环境准备

### 1.1 本地开发环境

```bash
cd /home/ouyang/project/RoomPartitioner

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖（开发模式，含测试工具）
pip install -r requirements-dev.txt
```

### 1.2 目录结构说明

```
RoomPartitioner/
├── app/
│   ├── main.py                 # FastAPI 入口 (uvicorn app.main:app)
│   ├── handler.py              # Lambda 入口 (app.handler.handler)
│   ├── core/
│   │   ├── config.py           # 配置加载（yaml + 环境变量）
│   │   └── errors.py           # 结构化业务异常
│   ├── services/
│   │   ├── services.py         # 总接口 RoomService（路由 + 预处理）
│   │   ├── auto_partition.py   # 自动分区
│   │   ├── extended_partition.py  # 扩展分区（增量新区域检测）
│   │   ├── manual_partition.py # 手动划分（画线分割）
│   │   └── manual_merge.py     # 手动合并
│   ├── pipeline/
│   │   ├── preprocessor.py     # 地图前处理（去噪、补墙、平滑）
│   │   ├── inferencer.py       # Triton 推理 + OBB 解码 + NMS
│   │   ├── postprocessor.py    # 推理后处理
│   │   └── triton_client.py    # Triton gRPC/HTTP 客户端
│   └── utils/
│       ├── coordinate.py       # 像素 ↔ 世界坐标变换
│       ├── geometry_ops.py     # Shapely 几何工具
│       ├── graph.py            # 邻接图 + 五色着色 + DFS 排序
│       ├── landmark.py         # 平台标记点生成
│       ├── s3_loader.py        # S3 / 本地数据加载
│       ├── contour_expander.py # 轮廓外扩
│       ├── beautifier.py       # 轮廓美化
│       └── common.py           # 通用验证工具
├── config/default.yaml         # 默认配置
├── tests/                      # 单元测试
├── Dockerfile                  # HTTP 部署镜像
├── Dockerfile.lambda           # Lambda 部署镜像
└── cli.py                      # CLI 工具（可选）
```

---

## 二、架构说明

### 2.1 调用链路

```
入口层              总接口层              服务层
─────────           ─────────           ─────────
handler.py    →     RoomService         AutoPartitioner
main.py       →     .room_edit()   →    ExtendedPartitioner
cli.py                                  ManualPartitioner
                                        ManualMerger
```

### 2.2 RoomService 路由逻辑

| operation | 条件 | 调用服务 |
|-----------|------|----------|
| `split` | 无 labels | `AutoPartitioner.process()` |
| `split` | 有 labels | `ExtendedPartitioner.process()` |
| `repartition` | 任何 | `AutoPartitioner.process(repartition=True)` |
| `division` | 需要 labels | `ManualPartitioner.process()` |
| `merge` | 需要 labels | `ManualMerger.process()` |

### 2.3 公共工具

每次请求由 `RoomService` 创建并传入各服务：

- `CoordinateTransformer` — 像素 ↔ 世界坐标
- `RoomGraph` — 邻接图构建 + 五色着色 + DFS 排序
- `LandmarkManager` — 平台标记点生成

---

## 三、开发流程

### Step 1：修改业务逻辑

| 要改什么 | 改哪个文件 |
|---------|-----------|
| 地图去噪/补墙 | `pipeline/preprocessor.py` |
| 模型推理 | `pipeline/inferencer.py` + `pipeline/triton_client.py` |
| 推理后处理 | `pipeline/postprocessor.py` |
| 自动分区逻辑 | `services/auto_partition.py` |
| 增量区域检测 | `services/extended_partition.py` |
| 手动划线分割 | `services/manual_partition.py` |
| 手动合并 | `services/manual_merge.py` |
| 路由/预处理 | `services/services.py` |

### Step 2：新增配置项

- `app/core/config.py` — 添加默认值和环境变量映射
- `config/default.yaml` — 添加 yaml 配置

### Step 3：运行单元测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行指定模块
pytest tests/test_services.py -v
pytest tests/test_auto_partition.py -v
pytest tests/test_extended_partition.py -v

# 带覆盖率报告
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Step 4：本地启动服务调试

```bash
# HTTP 模式（开发）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker compose up --build

# 接口文档
# http://localhost:8000/docs
```

### Step 5：接口验证

```bash
# 健康检查
curl http://localhost:8000/health

# 房间编辑（与 Lambda event 格式一致）
curl -X POST http://localhost:8000/room_edit \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "split",
    "bucket": "my-bucket",
    "key": "path/to/map"
  }'
```

### Step 6：打包部署

```bash
# Lambda 镜像
docker build -f Dockerfile.lambda -t room-partitioner-lambda .

# HTTP 镜像
docker build -t room-partitioner .
```

---

## 四、单元测试说明

### 4.1 测试文件

```
tests/
├── conftest.py                 # 公共 fixtures
├── test_config.py              # 配置加载
├── test_services.py            # 总接口路由
├── test_auto_partition.py      # 自动分区
├── test_extended_partition.py  # 扩展分区
├── test_manual_partition.py    # 手动划分
├── test_manual_merge.py        # 手动合并
├── test_inferencer.py          # Triton 推理
├── test_triton_client.py       # Triton 客户端
├── test_preprocessor.py        # 前处理
├── test_postpross.py           # 后处理
├── test_geometry_ops.py        # 几何工具
└── test_s3_loader.py           # S3 加载
```

### 4.2 Mock 策略

- 外部服务（Triton、S3）全部 mock
- 各服务的 `process()` 方法在集成测试中用 `patch.object` mock
- 像素级算法（label_map 操作）直接用真实 numpy 数组测试
