"""结构化业务异常 —— 保留数字 code 以兼容旧 Lambda 返回格式"""


class RoomPartitionerError(Exception):
    """房间划分器基础异常"""

    code: int = 0

    def __init__(self, message: str = ""):
        super().__init__(message or self.__class__.__doc__)


class DataLoadError(RoomPartitionerError):
    """S3 数据加载失败"""
    code = 1


class InvalidResolutionError(RoomPartitionerError):
    """分辨率为 0"""
    code = 2


class NoLabelsError(RoomPartitionerError):
    """labels 为空"""
    code = 3


class InvalidParameterError(RoomPartitionerError):
    """参数无效"""
    code = 4


class InsufficientIntersectionsError(RoomPartitionerError):
    """分割交点不足"""
    code = 5


class RoomIndexOutOfRangeError(RoomPartitionerError):
    """房间索引越界"""
    code = 7


class RoomsNotConnectedError(RoomPartitionerError):
    """待合并房间不连通"""
    code = 8


class OperationFailedError(RoomPartitionerError):
    """操作执行失败"""
    code = 9


class RoomTooSmallError(RoomPartitionerError):
    """分割后房间面积过小"""
    code = 10


class S3Error(RoomPartitionerError):
    """S3 访问错误"""
    code = 11


class SplitNotConnectedError(RoomPartitionerError):
    """分割后区域不连通"""
    code = 12
