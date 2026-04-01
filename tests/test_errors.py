"""core.errors 异常体系单测。"""

from app.core.errors import (
    RoomPartitionerError,
    DataLoadError,
    InvalidResolutionError,
    NoLabelsError,
    InvalidParameterError,
    InsufficientIntersectionsError,
    RoomIndexOutOfRangeError,
    RoomsNotConnectedError,
    OperationFailedError,
    RoomTooSmallError,
    S3Error,
    SplitNotConnectedError,
)


class TestErrorHierarchy:
    def test_all_subclass_base(self):
        """所有业务异常应继承 RoomPartitionerError"""
        for cls in [
            DataLoadError, InvalidResolutionError, NoLabelsError,
            InvalidParameterError, InsufficientIntersectionsError,
            RoomIndexOutOfRangeError, RoomsNotConnectedError,
            OperationFailedError, RoomTooSmallError, S3Error,
            SplitNotConnectedError,
        ]:
            assert issubclass(cls, RoomPartitionerError)

    def test_unique_codes(self):
        """每个异常的 code 应唯一"""
        classes = [
            DataLoadError, InvalidResolutionError, NoLabelsError,
            InvalidParameterError, InsufficientIntersectionsError,
            RoomIndexOutOfRangeError, RoomsNotConnectedError,
            OperationFailedError, RoomTooSmallError, S3Error,
            SplitNotConnectedError,
        ]
        codes = [cls.code for cls in classes]
        assert len(codes) == len(set(codes))


class TestErrorMessages:
    def test_default_message_from_docstring(self):
        """无参数时应使用 docstring 作为 message"""
        err = DataLoadError()
        assert str(err) == "S3 数据加载失败"

    def test_custom_message(self):
        """传入自定义 message 应覆盖默认"""
        err = InvalidParameterError("bad value")
        assert str(err) == "bad value"

    def test_code_attribute(self):
        assert DataLoadError.code == 1
        assert RoomIndexOutOfRangeError.code == 7
        assert S3Error.code == 11

    def test_catchable_as_base(self):
        """应可用基类捕获"""
        try:
            raise NoLabelsError()
        except RoomPartitionerError as e:
            assert e.code == 3
