"""配置加载测试"""

import os
import pytest
from app.core.config import load_config


class TestConfigDefault:
    def test_default_values(self):
        cfg = load_config()
        assert cfg["min_room_area"] == 1.0
        assert cfg["wall_threshold"] == 128
        assert cfg["resolution"] == 0.05
        assert cfg["triton_url"] == "192.168.2.55:8001"
        assert cfg["model_name"] == "room_seg"

    def test_target_size_default(self):
        cfg = load_config()
        assert cfg["target_size"] == [1280, 1280]


class TestConfigYaml:
    def test_load_yaml(self):
        cfg = load_config("config/default.yaml")
        assert "resolution" in cfg
        assert "morph_kernel_size" in cfg

    def test_yaml_not_found(self):
        """yaml 文件不存在时用默认值"""
        cfg = load_config("nonexistent.yaml")
        assert cfg["min_room_area"] == 1.0


class TestConfigOverrides:
    def test_code_override(self):
        cfg = load_config(overrides={"min_room_area": 5.0})
        assert cfg["min_room_area"] == 5.0

    def test_override_multiple(self):
        cfg = load_config(overrides={"min_room_area": 5.0, "resolution": 0.1})
        assert cfg["min_room_area"] == 5.0
        assert cfg["resolution"] == 0.1


class TestConfigEnvOverride:
    def test_env_override_float(self, monkeypatch):
        monkeypatch.setenv("MIN_ROOM_AREA", "3.5")
        cfg = load_config()
        assert cfg["min_room_area"] == 3.5

    def test_env_override_int(self, monkeypatch):
        monkeypatch.setenv("WALL_THRESHOLD", "200")
        cfg = load_config()
        assert cfg["wall_threshold"] == 200

    def test_env_override_str(self, monkeypatch):
        monkeypatch.setenv("TRITON_URL", "my-triton:8001")
        cfg = load_config()
        assert cfg["triton_url"] == "my-triton:8001"

    def test_env_beats_yaml_and_override(self, monkeypatch):
        """环境变量优先级最高"""
        monkeypatch.setenv("MIN_ROOM_AREA", "99.0")
        cfg = load_config("config/default.yaml", overrides={"min_room_area": 5.0})
        assert cfg["min_room_area"] == 99.0

    def test_target_size_env(self, monkeypatch):
        monkeypatch.setenv("TARGET_SIZE", "256,256")
        cfg = load_config()
        assert cfg["target_size"] == [256, 256]
