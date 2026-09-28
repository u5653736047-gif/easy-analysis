"""Jev 决策工具包。

- client.py         Jev API 共享客户端（配置解析 / 重试客户端）
- question_type.py  题型分类工具
- error_cause.py    错因分析工具（多因标签）

对外只暴露工具本体与测试需要的实现体，client 细节按需从子模块导入。
"""

from .client import JEV_MODEL, MAX_RETRIES, REQUEST_TIMEOUT, make_client, resolve_config
from .error_cause import (
    ERROR_CAUSES,
    MULTI_CAUSE_THRESHOLD,
    analyze_error_cause,
    analyze_error_cause_impl,
    analyze_error_cause_raw,
    build_questions,
    build_state,
)
from .question_type import (
    QUESTION_TYPES,
    classify_question_type,
    classify_question_type_impl,
    classify_question_type_raw,
)

__all__ = [
    # client
    "JEV_MODEL", "MAX_RETRIES", "REQUEST_TIMEOUT", "make_client", "resolve_config",
    # question_type
    "QUESTION_TYPES", "classify_question_type", "classify_question_type_impl",
    "classify_question_type_raw",
    # error_cause
    "ERROR_CAUSES", "MULTI_CAUSE_THRESHOLD", "analyze_error_cause",
    "analyze_error_cause_impl", "analyze_error_cause_raw", "build_questions", "build_state",
]
