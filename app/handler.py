"""
AWS Lambda Handler

入参:
    event = {
        "operation": "split" | "repartition" | "merge" | "division",
        "bucket": "my-bucket",
        "key": "path/to/map",
        "roomMergeList": ["ROOM_001", "ROOM_002"],       # merge 时必传
        "divisionCroodsDict": {"id": "ROOM_001", "A": [x,y], "B": [x,y]}  # division 时必传
    }

出参:
    {
        "statusCode": 200 | 190 | 171,
        "body": "<labels_json_string>" | "<error_code>"
    }
"""

import json
import logging

from botocore.exceptions import ClientError

from app.core.config import load_config
from app.core.errors import RoomPartitionerError
from app.services.services import RoomService
from app.utils.s3_loader import S3DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("room_partitioner")


def handler(event, _context):
    """Lambda 入口 —— 与旧 app.py 完全兼容"""
    logger.info(json.dumps(event))

    operation = event.get('operation')
    bucket = event.get('bucket')
    key = event.get('key')

    # 参数校验
    if operation not in ['split', 'repartition', 'merge', 'division']:
        return {
            'statusCode': 171,
            'body': 'no such operation'
        }

    if operation == 'merge' and 'roomMergeList' not in event:
        return {
            'statusCode': 171,
            'body': 'need room merge list'
        }

    if operation == 'division' and 'divisionCroodsDict' not in event:
        return {
            'statusCode': 171,
            'body': 'need division croods list'
        }

    try:
        # 加载数据
        loader = S3DataLoader(bucket, key)
        map_data = loader.load()

        # 执行操作
        config = load_config()
        service = RoomService(config)
        labels_json = service.room_edit(
            map_data=map_data,
            operation=operation,
            division_croods_dict=event.get('divisionCroodsDict'),
            room_merge_list=event.get('roomMergeList'),
        )

        return {
            'statusCode': 200,
            'body': json.dumps(labels_json, ensure_ascii=False),
        }

    except RoomPartitionerError as e:
        logger.error(f"业务错误 (code={e.code}): {e}")
        return {
            'statusCode': 190,
            'body': str(e.code),
        }
    except ClientError as e:
        logger.error(f"S3 错误: {e}")
        return {
            'statusCode': 190,
            'body': "11",
        }
    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        return {
            'statusCode': 190,
            'body': "0",
        }
