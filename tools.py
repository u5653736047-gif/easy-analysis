"""「小析易」Agent 示例工具集。

每个函数通过 @function_tool 装饰器注册为 Agent 可调用的工具：
- 函数签名（参数名 + 类型标注）会自动生成 JSON Schema，供模型理解入参；
- docstring 会作为工具描述进入模型上下文，请写清楚「什么时候该用这个工具」。

新增工具的步骤：
1. 在下方写一个带类型标注和 docstring 的函数；
2. 在 agent_core.py 的 build_agent() 中把它加入 tools 列表。
"""
from __future__ import annotations

import json
from datetime import datetime
from statistics import fmean, median

from agents import function_tool


@function_tool
def get_current_time() -> str:
    """获取当前的日期和时间（本地时区，格式 YYYY-MM-DD HH:MM:SS）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@function_tool
def classroom_score_stats(scores: list[float]) -> str:
    """统计一个班级某次考试的成绩分布。

    当老师提供了一组学生分数（如 88, 92, 59, ...）时调用本工具，
    返回人数、平均分、中位数、最高/最低分、及格率、优秀率等汇总指标（JSON）。

    Args:
        scores: 全班学生的分数列表，例如 [88, 92, 59, 45, 76]。
    """
    if not scores:
        return json.dumps({"error": "成绩列表为空，无法统计"}, ensure_ascii=False)
    n = len(scores)
    stats = {
        "人数": n,
        "平均分": round(fmean(scores), 2),
        "中位数": round(median(scores), 2),
        "最高分": max(scores),
        "最低分": min(scores),
        "及格率(>=60)": f"{sum(s >= 60 for s in scores) / n:.1%}",
        "优秀率(>=85)": f"{sum(s >= 85 for s in scores) / n:.1%}",
    }
    return json.dumps(stats, ensure_ascii=False)


@function_tool
def summarize_error_types(error_types: list[str]) -> str:
    """统计学生错因类型的分布，找出班级的高发易错点。

    当老师提供了一组错因描述（如 ["计算失误", "概念不清", "计算失误"]）时调用本工具，
    返回每个错因出现的次数、占比，并按占比降序排列（JSON）。

    Args:
        error_types: 错因描述列表，例如 ["计算失误", "概念不清", "审题不清"]。
    """
    if not error_types:
        return json.dumps({"error": "错因列表为空，无法统计"}, ensure_ascii=False)
    total = len(error_types)
    counts: dict[str, int] = {}
    for e in error_types:
        counts[e] = counts.get(e, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    result = {
        "total_errors": total,
        "summary": [
            {"错因": k, "次数": v, "占比": f"{v / total:.1%}"} for k, v in ranked
        ],
    }
    return json.dumps(result, ensure_ascii=False)
