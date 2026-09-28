"""Jev 题型分类工具。

把「题干文本 -> 题型」这一封闭集判断交给 Jev 完成：
- Choice 原语 + 固定 taxonomy（选项内容与顺序均固定，缓解选项顺序敏感性）；
- 返回选项、置信度、完整概率分布、前两名概率差 margin；
- needs_review 判定：置信度 < CONFIDENCE_THRESHOLD 或 margin < MARGIN_THRESHOLD；
- 任何异常（未配置 Key / 网络失败 / 上游 5xx）都返回结构化错误 JSON，
  由 agent 降级为自行判断，工具不抛异常打断主流程。

结构约定（与 jev/error_cause.py 一致）：
    *_raw()  只走成功路径，异常上抛；
    *_impl() 负责阈值加工与异常兜底，返回结构化 dict；
    @function_tool 只做序列化，交付给 SDK。
"""

from __future__ import annotations

import json

from agents import function_tool

from .client import JEV_MODEL, REQUEST_TIMEOUT, make_client
from typesafe_sdk import Choice

CONFIDENCE_THRESHOLD = 0.70   # 置信度低于该值 -> needs_review
MARGIN_THRESHOLD = 0.15       # 前两名概率差低于该值 -> needs_review（抗选项顺序敏感）

# 固定 taxonomy：内容与顺序都是代码常量，不允许运行时自由发挥
QUESTION_TYPES: dict[str, str] = {
    "选择题": "提供多个备选项，需要从中选出一个或多个答案",
    "判断题": "只需求判断对错（√/× 或 是/否）",
    "填空题": "题干中留有待填空格，答案通常简短",
    "解答题": "需要写出完整解题过程或步骤的大题",
    "证明题": "需要证明某个结论或命题成立",
    "其他": "以上类型均不符合",
}


def classify_question_type_raw(
    question_text: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> dict:
    """实际调用 Jev，返回 {type, confidence, probabilities}，异常向上抛。"""
    with make_client(api_key=api_key, base_url=base_url, timeout=timeout) as client:
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
