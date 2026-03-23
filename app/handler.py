"""
AWS Lambda Handler —— 与旧 app.py handler 签名完全一致

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

from app.core.editor import RoomEditor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("room_partitioner")


def handler(event, context):
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

    # 执行
    editor = RoomEditor(bucket, key)

    try:
        room_merge_list = event.get('roomMergeList', None)
        division_croods_dict = event.get('divisionCroodsDict', None)

        seg_state, labels_json = editor.room_edit(
            detection=operation in ['split', 'repartition'],
            repartition=operation == 'repartition',
            division_croods_dict=division_croods_dict,
            room_merge_list=room_merge_list,
        )

        # 格式化返回值 (与旧代码一致)
        if not seg_state:
            if isinstance(labels_json, int) and labels_json in range(1, 12):
                body = str(labels_json)
            else:
                body = "0"
        else:
            body = json.dumps(labels_json, ensure_ascii=False)

        return {
            'statusCode': 200 if seg_state else 190,
            'body': body,
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
