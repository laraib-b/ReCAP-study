from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "integration: 需要真实数据或完整主流程的集成测试")
    config.addinivalue_line("markers", "effectiveness: 需要论文时间范围和近论文参数的有效性测试")
