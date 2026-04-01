"""handler.py Lambda 入口单测。"""

import json
from unittest.mock import patch, MagicMock

import pytest

from app.handler import handler


class TestParameterValidation:
    def test_invalid_operation(self):
        result = handler({"operation": "invalid"}, None)
        assert result["statusCode"] == 171
        assert "no such operation" in result["body"]

    def test_merge_without_list(self):
        result = handler({"operation": "merge", "bucket": "b", "key": "k"}, None)
        assert result["statusCode"] == 171
        assert "room merge list" in result["body"]

    def test_division_without_coords(self):
        result = handler({"operation": "division", "bucket": "b", "key": "k"}, None)
        assert result["statusCode"] == 171
        assert "division croods" in result["body"]


class TestSuccessPath:
    @patch("app.handler.RoomService")
    @patch("app.handler.S3DataLoader")
    @patch("app.handler.load_config")
    def test_split_success(self, mock_config, mock_loader_cls, mock_service_cls):
        mock_config.return_value = {}
        mock_loader = MagicMock()
        mock_loader.load.return_value = {"cleaned_img": None}
        mock_loader_cls.return_value = mock_loader

        mock_service = MagicMock()
        mock_service.room_edit.return_value = {"data": []}
        mock_service_cls.return_value = mock_service

        result = handler({
            "operation": "split",
            "bucket": "test-bucket",
            "key": "test-key",
        }, None)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert "data" in body
        mock_service.room_edit.assert_called_once()


class TestErrorHandling:
    @patch("app.handler.S3DataLoader")
    def test_business_error(self, mock_loader_cls):
        from app.core.errors import NoLabelsError
        mock_loader = MagicMock()
        mock_loader.load.side_effect = NoLabelsError()
        mock_loader_cls.return_value = mock_loader

        result = handler({
            "operation": "split",
            "bucket": "b",
            "key": "k",
        }, None)

        assert result["statusCode"] == 190
        assert result["body"] == "3"  # NoLabelsError.code

    @patch("app.handler.S3DataLoader")
    def test_s3_client_error(self, mock_loader_cls):
        from botocore.exceptions import ClientError
        mock_loader = MagicMock()
        mock_loader.load.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "not found"}},
            "GetObject",
        )
        mock_loader_cls.return_value = mock_loader

        result = handler({
            "operation": "split",
            "bucket": "b",
            "key": "k",
        }, None)

        assert result["statusCode"] == 190
        assert result["body"] == "11"

    @patch("app.handler.S3DataLoader")
    def test_unknown_error(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader.load.side_effect = RuntimeError("boom")
        mock_loader_cls.return_value = mock_loader

        result = handler({
            "operation": "split",
            "bucket": "b",
            "key": "k",
        }, None)

        assert result["statusCode"] == 190
        assert result["body"] == "0"
