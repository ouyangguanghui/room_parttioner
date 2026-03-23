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
├── app/                        # 主应用包
│   ├── main.py                 # FastAPI 入口 + 生命周期
│   ├── api/routes.py           # API 路由定义
│   ├── schemas/requests.py     # Pydantic 请求/响应模型
│   ├── core/
│   │   ├── config.py           # 配置加载（yaml + 环境变量）
│   │   └── partitioner.py      # 核心编排器
│   └── services/               # 业务逻辑
│       ├── preprocessor.py     # 地图前处理（去噪、补墙、平滑）
│       ├── inferencer.py       # Triton 模型推理
│       ├── postprocessor.py    # 推理后处理
│       ├── triton_client.py    # Triton 客户端封装
│       ├── auto_partition.py   # 自动分区
│       ├── extended_partition.py  # 扩展分区（门口检测+区域生长）
│       ├── manual_partition.py # 手动划分（画线/多边形）
│       └── manual_merge.py     # 手动合并
├── config/default.yaml         # 默认配置
├── tests/                      # 单元测试
├── Dockerfile                  # 生产镜像
├── docker-compose.yml          # 编排配置
├── requirements.txt            # 生产依赖
├── requirements-dev.txt        # 开发依赖
└── cli.py                      # CLI 工具（可选）
```

---

## 二、开发流程（分步骤）

### Step 1：修改业务逻辑

所有算法逻辑在 `app/services/` 下，按模块修改：

| 要改什么 | 改哪个文件 |
|---------|-----------|
| 地图去噪/补墙 | `services/preprocessor.py` |
| 模型推理 | `services/inferencer.py` + `services/triton_client.py` |
| 推理后处理 | `services/postprocessor.py` |
| 自动分区逻辑 | `services/auto_partition.py` |
| 门口检测/区域生长 | `services/extended_partition.py` |
| 手动划线分割 | `services/manual_partition.py` |
| 手动合并 | `services/manual_merge.py` |

### Step 2：修改 API 接口

- 新增/修改路由 → `app/api/routes.py`
- 新增/修改请求体 → `app/schemas/requests.py`
- 新增配置项 → `app/core/config.py` + `config/default.yaml`

### Step 3：运行单元测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行指定模块
pytest tests/test_partitioner.py -v
pytest tests/test_preprocessor.py -v
pytest tests/test_api.py -v

# 运行指定类/方法
pytest tests/test_partitioner.py::TestAutoPartition::test_basic -v

# 带覆盖率报告
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Step 4：本地启动服务调试

```bash
# 方式1：直接启动（开发模式，热重载）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 方式2：Docker 启动
docker compose up --build

# 访问 API 文档
# http://localhost:8000/docs        (Swagger UI)
# http://localhost:8000/redoc       (ReDoc)
```

### Step 5：接口验证

```bash
# 健康检查
curl http://localhost:8000/health

# 自动分区（返回 JSON）
curl -X POST http://localhost:8000/auto_partition \
  -F "file=@test_map.png" \
  -F "res=0.05"

# 自动分区（返回可视化图片）
curl -X POST http://localhost:8000/auto_partition/image \
  -F "file=@test_map.png" \
  -o result.png

# 手动画线分割
curl -X POST http://localhost:8000/manual/split_line \
  -H "Content-Type: application/json" \
  -d '{"pt1": [50, 0], "pt2": [50, 100]}'

# 手动合并
curl -X POST http://localhost:8000/manual/merge \
  -H "Content-Type: application/json" \
  -d '{"room_ids": [1, 2]}'

# 查看当前状态
curl http://localhost:8000/current/info
```

### Step 6：打包镜像交付

```bash
# 构建镜像
docker build -t room-partitioner:latest .

# 测试镜像
docker run --rm -p 8000:8000 room-partitioner:latest
curl http://localhost:8000/health

# 导出镜像（给后端同事）
docker save room-partitioner:latest | gzip > room-partitioner.tar.gz

# 或推送到仓库
docker tag room-partitioner:latest your-registry.com/room-partitioner:v0.2.0
docker push your-registry.com/room-partitioner:v0.2.0
```

---

## 三、单元测试详解

### 3.1 测试文件规划

```
tests/
├── __init__.py
├── conftest.py              # 公共 fixtures
├── test_config.py           # 配置加载测试
├── test_preprocessor.py     # 前处理测试
├── test_postprocessor.py    # 后处理测试
├── test_partitioner.py      # 核心编排测试（已有）
├── test_auto_partition.py   # 自动分区测试
├── test_manual.py           # 手动划分 + 合并测试
└── test_api.py              # API 接口测试
```

### 3.2 各测试文件内容

以下是完整的测试代码，直接创建使用。
