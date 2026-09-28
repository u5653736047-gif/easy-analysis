"""Jev（TypeSafe System One）题型分类工具。

把「题干文本 -> 题型」这一封闭集判断交给 Jev 完成：
- Choice 原语 + 固定 taxonomy（选项内容与顺序均固定，缓解选项顺序敏感性）；
- 返回选项、置信度、完整概率分布、前两名概率差 margin；
- needs_review 判定：置信度 < CONFIDENCE_THRESHOLD 或 margin < MARGIN_THRESHOLD；
- 任何异常（未配置 Key / 网络失败 / 上游 5xx）都返回结构化错误 JSON，
  由 agent 降级为自行判断，工具不抛异常打断主流程。

配置（.env，两套命名均可，Command Code 网关优先）：
    COMMAND_CODE_API_KEY   + COMMAND_CODE_BASE_URL
    TYPESAFE_API_KEY       + TYPESAFE_BASE_URL       （官方地址兜底）
COMMAND_CODE_BASE_URL 形如 https://api.commandcode.ai/provider/v1/systemone，
SDK 会自动拼接 /v1/systemone，这里需要剥掉该后缀。

重试：SDK 内置 RetryPolicy 对连接错误/超时/429/5xx 退避重试（实测
Command Code 网络偶发重置，重试后成功率明显提升）。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from agents import function_tool
from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

JEV_MODEL = "jev-latest"
CONFIDENCE_THRESHOLD = 0.70   # 置信度低于该值 -> needs_review
MARGIN_THRESHOLD = 0.15       # 前两名概率差低于该值 -> needs_review（抗选项顺序敏感）
REQUEST_TIMEOUT = 30.0        # 单次请求超时（秒）
MAX_RETRIES = 4               # SDK 内置重试次数

# 固定 taxonomy：内容与顺序都是代码常量，不允许运行时自由发挥
QUESTION_TYPES: dict[str, str] = {
    "选择题": "提供多个备选项，需要从中选出一个或多个答案",
    "判断题": "只需求判断对错（√/× 或 是/否）",
    "填空题": "题干中留有待填空格，答案通常简短",
    "解答题": "需要写出完整解题过程或步骤的大题",
    "证明题": "需要证明某个结论或命题成立",
    "其他": "以上类型均不符合",
}

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


def resolve_jev_config() -> tuple[str | None, str | None]:
    """返回 (api_key, base_url)。base_url 已剥掉尾部 /v1/systemone 后缀。"""
    _load_dotenv_once()
    api_key = next((v for e in _API_KEY_ENVS if (v := os.getenv(e))), None)
    base_url = next((v for e in _BASE_URL_ENVS if (v := os.getenv(e))), None)
    if base_url:
        base_url = re.sub(r"/v1/systemone/?$", "", base_url.rstrip("/"))
    return api_key, base_url


def classify_question_type_raw(
    question_text: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> dict:
    """实际调用 Jev，返回 {type, confidence, probabilities}，异常向上抛。"""
    if api_key is None:
        api_key, base_url = resolve_jev_config()
    if not api_key:
        raise RuntimeError("未配置 Jev API Key（COMMAND_CODE_API_KEY 或 TYPESAFE_API_KEY）")

    client_kwargs: dict = {
        "api_key": api_key,
        "timeout": timeout,
        "retry": RetryPolicy(max_retries=MAX_RETRIES, backoff_max=8.0),
    }
    if base_url:
        client_kwargs["base_url"] = base_url

    with TypeSafeClient(**client_kwargs) as client:
        resp = client.system_one(
            state=question_text,
            questions={
                "题型": Choice(
                    instructions="这道题属于哪种题型？",
                    criteria=QUESTION_TYPES,
                ),
            },
            model=JEV_MODEL,
        )
        ans = resp.answers["题型"]
        return {
            "type": ans.choice,
            "confidence": round(float(ans.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in ans.probabilities.items()},
        }


def classify_question_type_impl(question_text: str) -> dict:
    """工具实现体：无论成功失败都返回结构化 dict，绝不抛异常打断 agent。"""
    try:
        result = classify_question_type_raw(question_text)
    except Exception as exc:  # noqa: BLE001 - 统一兜底：降级提示交给 agent
        return {
            "source": "error",
            "needs_review": True,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "Jev 分类服务不可用，请根据题干自行判断题型，结果标记 source=llm。",
        }

    probs = sorted(result["probabilities"].values(), reverse=True)
    margin = round(probs[0] - (probs[1] if len(probs) > 1 else 0.0), 4)
    result["margin"] = margin
    result["source"] = "jev"
    result["needs_review"] = (
        result["confidence"] < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD
    )
    if result["needs_review"]:
        result["hint"] = (
            f"置信度 {result['confidence']} 或 margin {margin} 偏低，"
            "请结合题干复核后再写入蓝图。"
        )
    return result


@function_tool
def classify_question_type(question_text: str) -> str:
    """调用 Jev 判定一道题的题型。

    当需要确定某道题的题型（选择/判断/填空/解答/证明）时调用本工具，
    传入完整题干文本，返回题型、置信度与各选项概率分布；
    低置信度时结果会标记 needs_review=true 并附复核提示。

    Args:
        question_text: 这道题的完整题干文本。
    """
    return json.dumps(classify_question_type_impl(question_text), ensure_ascii=False)
