"""Jev 错因分析工具。

一道错题往往是多因的（既有概念不清又有计算失误），因此一次 system_one 调用里
组合两类问题并行求解（Jev 不按输出计费，多问题几乎零边际成本）：

1. 一个 Choice 问题：「primary」—— 主错因（单标签，带概率分布/置信度/margin）；
2. 每个错因一个 Noul 问题：「multi::<错因>」—— 该错因是否存在（概率），
   按 MULTI_CAUSE_THRESHOLD 阈值汇总为 detected_causes 多因标签，支撑 idea 里
   「学生易错标签可累加」的需求。

输入 state 由固定模板拼装：题目 / 参考答案 / 学生作答。

降级策略与 jev/question_type.py 一致：任何异常返回 source=error 的结构化 JSON，
由 agent 自行分析；低置信/margin 过小/多因为空时标记 needs_review。
"""

from __future__ import annotations

import json

from agents import function_tool

from .client import JEV_MODEL, REQUEST_TIMEOUT, make_client
from typesafe_sdk import Choice, Noul

CONFIDENCE_THRESHOLD = 0.70   # 主错因置信度低于该值 -> needs_review
MARGIN_THRESHOLD = 0.15       # 前两名概率差低于该值 -> needs_review
MULTI_CAUSE_THRESHOLD = 0.75  # Noul 概率高于该值才计入「多因」列表

# 固定错因 taxonomy：内容与顺序都是代码常量（与 my-idea.md 的错因示例对齐）
ERROR_CAUSES: dict[str, str] = {
    "概念不清": "对概念、公式、定理、运算法则理解错误",
    "计算失误": "算术运算、符号处理或变形过程出错",
    "审题不清": "误解题意、漏看条件、答非所问、看错数字",
    "方法错误": "解题思路或方法选择错误（如用错定理、模型）",
    "粗心失误": "抄写、笔误、漏写单位、正负号遗漏等低级错误",
    "知识遗忘": "对应知识未掌握或基本空白",
    "表述不完整": "思路正确但步骤缺失、跳步、书写不规范",
    "其他": "以上类型均不符合",
}

QUERY_PREFIX = "multi"


def build_state(question_text: str, reference_answer: str, student_answer: str) -> str:
    """按固定模板拼装 Jev 的输入 state。"""
    return (
        f"题目：{question_text.strip()}\n"
        f"参考答案：{reference_answer.strip() or '(未提供)'}\n"
        f"学生作答：{student_answer.strip() or '(空白，学生未作答)'}"
    )


def build_questions() -> dict:
    """构造 primary Choice + 每错因一个 Noul 的问题字典。"""
    questions: dict = {
        "primary": Choice(
            instructions="这道错题的主要错因是什么？",
            criteria=ERROR_CAUSES,
        ),
    }
    for cause, desc in ERROR_CAUSES.items():
        questions[f"{QUERY_PREFIX}::{cause}"] = Noul(
            instructions=f"学生的错误中是否存在「{cause}」？（{desc}）",
            criteria={
                "true": f"作答明确表现出{cause}",
                "false": f"作答未表现出{cause}",
            },
        )
    return questions


def analyze_error_cause_raw(
    question_text: str,
    reference_answer: str,
    student_answer: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> dict:
    """实际调用 Jev，返回 {primary_cause, confidence, probabilities, causes}，异常向上抛。"""
    with make_client(api_key=api_key, base_url=base_url, timeout=timeout) as client:
        resp = client.system_one(
            state=build_state(question_text, reference_answer, student_answer),
            questions=build_questions(),
            model=JEV_MODEL,
        )

    primary = resp.answers["primary"]
    multi_causes = [
        {"cause": cause, "probability": round(float(resp.answers[f"{QUERY_PREFIX}::{cause}"].noul), 4)}
        for cause in ERROR_CAUSES
    ]
    multi_causes.sort(key=lambda c: c["probability"], reverse=True)
    return {
        "primary_cause": primary.choice,
        "confidence": round(float(primary.confidence), 4),
        "probabilities": {k: round(float(v), 4) for k, v in primary.probabilities.items()},
        "causes": multi_causes,
    }


def analyze_error_cause_impl(
    question_text: str,
    reference_answer: str,
    student_answer: str,
) -> dict:
    """工具实现体：无论成功失败都返回结构化 dict，绝不抛异常打断 agent。"""
    try:
        result = analyze_error_cause_raw(question_text, reference_answer, student_answer)
    except Exception as exc:  # noqa: BLE001 - 统一兜底：降级提示交给 agent
        return {
            "source": "error",
            "needs_review": True,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "Jev 错因分析不可用，请根据题目/参考答案/学生作答自行分析错因，"
                    "结果标记 source=llm。",
        }

    probs = sorted(result["probabilities"].values(), reverse=True)
    margin = round(probs[0] - (probs[1] if len(probs) > 1 else 0.0), 4)
    detected = [c for c in result["causes"] if c["probability"] >= MULTI_CAUSE_THRESHOLD]

    out = {
        "source": "jev",
        "primary_cause": result["primary_cause"],
        "confidence": result["confidence"],
        "margin": margin,
        "probabilities": result["probabilities"],
        "detected_causes": detected,
        "needs_review": (
            result["confidence"] < CONFIDENCE_THRESHOLD
            or margin < MARGIN_THRESHOLD
            or not detected
        ),
    }
    if out["needs_review"]:
        out["hint"] = (
            f"主错因置信度 {result['confidence']} / margin {margin} / "
            f"多因命中 {len(detected)} 项偏低，请结合本题分析后再给学生打标签。"
        )
    return out


@function_tool
def analyze_error_cause(question_text: str, reference_answer: str, student_answer: str) -> str:
    """调用 Jev 分析一道学生错题的错因（支持多因并返回置信度）。

    当需要判断某道错题「错在哪」时调用本工具：传入题目、参考答案和学生作答，
    返回主错因、各错因概率分布、以及多因标签列表；
    低置信度时结果会标记 needs_review=true 并附复核提示。

    Args:
        question_text: 这道题的题干。
        reference_answer: 该题的参考答案或正确答案。
        student_answer: 这名学生的作答内容（空白则传空字符串）。
    """
    return json.dumps(
        analyze_error_cause_impl(question_text, reference_answer, student_answer),
        ensure_ascii=False,
    )
