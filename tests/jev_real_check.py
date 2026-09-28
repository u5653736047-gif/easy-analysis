"""真实 Jev 可靠性评测：用模拟题目数据测试 classify_question_type 工具。

运行：
    .venv/bin/python tests/jev_real_check.py

前提：.env 已配置 COMMAND_CODE_API_KEY / COMMAND_CODE_BASE_URL（或官方 Key）。
注意：Command Code 网络偶发重置，每次调用带 3 次外层重试。

输出：每题 [期望 -> 实际] 判定、置信度、margin、needs_review、耗时；
统计准确率（以加权标注的“标准答案”为参照）与 needs_review 命中情况。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_tool import classify_question_type_impl

# (题干, 期望题型, 说明)：含 3 个易混淆用例
QUESTIONS: list[tuple[str, str, str]] = [
    ("下列选项中，说法正确的是（ ）A. 1是质数  B. 2是合数  C. 3是偶数  D. 4是奇数",
     "选择题", "典型选择题（带备选项）"),
    ("请判断：三角形的内角和等于180°。这句话对吗？（ ）",
     "判断题", "典型判断题（√/×）"),
    ("已知 a=2，b=-1，则 a² - 2ab + b² 的值为 ______。",
     "填空题", "典型填空题（空格）"),
    ("已知函数 f(x)=x²-2x，求 f(x) 在区间 [0,3] 上的最小值，并说明取得最值时的 x。",
     "解答题", "典型解答题（求参数+说明）"),
    ("求证：不论 m 取何值，关于 x 的一元二次方程 x²+(m+2)x+2m=0 总有两个不相等的实数根。",
     "证明题", "典型证明题（“求证”）"),
    ("默写李白的《静夜思》。",
     "填空题", "易混淆：语文默写（答案不唯一，倾向填空）"),
    ("下列四个选项中，能作为“该数列是等差数列”的充分条件的是（ ）",
     "选择题", "易混淆：需推理但以选项形式出现"),
    ("计算：(-2)³ + |-5| - 3×(1/3)，要求写出完整的计算过程。",
     "解答题", "易混淆：含“计算”但明确要求过程"),
    ("This sentence is grammatically ___ (correct). 请用所给词的适当形式填空。",
     "填空题", "中英混合填空题"),
    ("请结合所学知识，论述工业革命对现代社会结构的影响。",
     "解答题", "论述大题（语文/文综风格）"),
    ("本题共12分，请同学们注意审题。",
     "其他", "噪声文本（非题目），考察是否低置信"),
    ("若 x² + y² = 1，且 x>0, y>0，求 x+y 的最大值。",
     "解答题", "标准条件极值大题"),
]


def call_with_retry(question: str, attempts: int = 3) -> dict:
    """外层重试：Command Code 网络偶发重置，SDK 内部已重试 4 次。"""
    last: dict = {}
    for i in range(attempts):
        r = classify_question_type_impl(question)
        if r.get("source") == "jev":
            return r
        last = r
        print(f"    ↻ 第 {i + 1} 次失败：{(r.get('error') or '')[:60]}，重试…")
    return last


def main() -> int:
    print("=" * 78)
    print("Jev 题型分类 · 模拟数据可靠性评测")
    print("=" * 78)

    passed = reach = 0
    review_correct = 0
    flagged_errors = 0
    errors = 0
    latencies: list[float] = []
    rows: list[str] = []
    for i, (q, expected, note) in enumerate(QUESTIONS, 1):
        t0 = time.time()
        r = call_with_retry(q)
        elapsed = time.time() - t0

        if r.get("source") != "jev":
            rows.append(f"{i:>2}. ✗ 网络失败 | {(r.get('error') or '')[:50]}")
            print(f"  {rows[-1]}  ({note})")
            continue
        reach += 1

        actual = r["type"]
        ok = actual == expected
        passed += ok
        latencies.append(elapsed)
        if not ok:
            errors += 1
            if r.get("needs_review"):
                flagged_errors += 1  # 误判但被置信度机制捕获 -> 会降级，不污染蓝图
        elif r.get("needs_review"):
            review_correct += 1  # 判对但被标记待复核（工具偏保守，可接受）

        mark = "✓" if ok else ("△" if r.get("needs_review") else "✗")
        rows.append(
            f"{i:>2}. {mark} 期望={expected:<3} 实际={actual:<3} "
            f"置信度={r['confidence']:.2f} margin={r['margin']:.2f} "
            f"复核={('是' if r.get('needs_review') else '否')} {elapsed:.1f}s | {note}"
        )
        print(f"  {rows[-1]}")

    total = len(QUESTIONS)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    print("\n" + "=" * 78)
    print(f"网络可达: {reach}/{total}   判定准确: {passed}/{total}"
          f"   误判: {errors} 题（其中 {flagged_errors} 题被 needs_review 捕获，自动降级不污染蓝图）")
    print(f"平均耗时: {avg_lat:.1f}s/题（Command Code 网关转发，含 SDK 重试）")
    if reach == 0:
        print("✗ 全部调用失败：网络不可达，请检查代理或稍后重试")
        return 1
    print("✓ 评测完成（准确率为当前阈值与 taxonomy 下的初测结果，"
          "上线前建议按置信度分布人工复核 30~50 题）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
