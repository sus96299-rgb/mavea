"""配置和 FFmpeg 基础测试。"""


from mavea.config import get_settings


def test_settings_singleton():
    """验证配置单例。"""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_paths():
    """验证工作目录路径自动创建。"""
    settings = get_settings()
    assert settings.workspace_path.exists()
    assert settings.output_path.exists()
    assert settings.temp_path.exists()


def test_validate_path():
    """验证路径穿越防护。"""
    settings = get_settings()
    # 工作目录内的路径应该通过
    safe = settings.workspace_path / "test.txt"
    resolved = settings.validate_path(safe)
    assert resolved == safe.resolve()
