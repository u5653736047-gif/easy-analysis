"""Jev（TypeSafe System One）API 共享客户端。

本模块只负责「怎么调 Jev」，不含任何业务 taxonomy / 阈值逻辑：
- 配置解析：两套 env 命名（Command Code 网关 / 官方）与 base_url 后缀处理；
- 客户端工厂：带重试策略的 TypeSafeClient（实测 Command Code 网络偶发重置，
  重试后成功率明显提升）。

question_type / error_cause 两个业务工具都通过 make_client() 复用这里。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from typesafe_sdk import RetryPolicy, TypeSafeClient

JEV_MODEL = "jev-latest"
REQUEST_TIMEOUT = 30.0   # 单次请求超时（秒）
MAX_RETRIES = 4          # SDK 内置退避重试次数

_API_KEY_ENVS = ("COMMAND_CODE_API_KEY", "TYPESAFE_API_KEY")
_BASE_URL_ENVS = ("COMMAND_CODE_BASE_URL", "TYPESAFE_BASE_URL")

_dotenv_loaded = False


def _load_dotenv_once() -> None:
    """CLI 已加载过 .env 时跳过；独立运行时从当前目录读一次。"""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    if Path(".env").exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(Path(".env"))
        except Exception:  # noqa: BLE001 - dotenv 不可用时静默降级
            pass


def resolve_config() -> tuple[str | None, str | None]:
    """返回 (api_key, base_url)。

    base_url 已剥掉尾部 /v1/systemone 后缀——SDK 会自行拼接该路径，
    COMMAND_CODE_BASE_URL 这类已含全路径的配置需要先处理。
    """
    _load_dotenv_once()
    api_key = next((v for e in _API_KEY_ENVS if (v := os.getenv(e))), None)
    base_url = next((v for e in _BASE_URL_ENVS if (v := os.getenv(e))), None)
    if base_url:
        base_url = re.sub(r"/v1/systemone/?$", "", base_url.rstrip("/"))
    return api_key, base_url


def make_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> TypeSafeClient:
    """构造带重试策略的 Jev 客户端（返回未进入的上下文管理器）。

    api_key/base_url 缺省时从 env 解析；base_url 允许调用方显式指定
    （测试中用于指向本地假服务）。
    """
    if api_key is None:
        api_key, base_url = resolve_config()
    if not api_key:
        raise RuntimeError(
            "未配置 Jev API Key（COMMAND_CODE_API_KEY 或 TYPESAFE_API_KEY）"
        )
    client_kwargs: dict = {
        "api_key": api_key,
        "timeout": timeout,
        "retry": RetryPolicy(max_retries=MAX_RETRIES, backoff_max=8.0),
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    return TypeSafeClient(**client_kwargs)
