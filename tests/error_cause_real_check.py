"""真实 Jev 错因分析评测：用模拟学生错题数据测试 analyze_error_cause 工具。

运行：
    .venv/bin/python tests/error_cause_real_check.py

前提：.env 已配置 COMMAND_CODE_API_KEY / COMMAND_CODE_BASE_URL。
每次调用带 3 次外层重试（Command Code 网络偶发重置）。

数据集 12 题覆盖 8 类错因，含 4 个易混淆/多因用例；
每题标注期望主错因（可接受备选）。输出：判定对照、置信度、margin、
detected_causes 多因命中、needs_review 捕获率、耗时。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from error_cause_tool import analyze_error_cause_impl

# (题目, 参考答案, 学生作答, 期望主错因集合, 说明)
CASES: list[tuple[str, str, str, set[str], str]] = [
    (
        "解方程 2(x+1) = x - 1，求 x。",
        "x = -3",
        "去括号得 2x + 1 = x - 1，移项合并得 x = 0。",
        {"概念不清"},
        "去括号漏乘 2（分配律）→概念不清，兼有计算参与",
    ),
    (
        "计算 3/4 + 1/2，结果化为最简分数。",
        "5/4",
        "3/4 + 1/2 = 4/6 = 2/3。",
        {"概念不清"},
        "分母直接相加 → 分数加法概念错误",
    ),
    (
        "下列说法中不正确的是（ ）A. 平行四边形对角线互相平分 B. 矩形的对角线相等 C. 正方形的对角线互相垂直 D. 等腰三角形是等边三角形",
        "D",
        "选 C。",
        {"审题不清"},
        "题目问「不正确」却选了正确项 → 审题不清",
    ),
    (
        "在△ABC中，AB=AC=5，BC=6，求BC边上的高。",
        "4（勾股定理：h=√(5²-3²)）",
        "直接用勾股定理：5² + 6² = h²，解得 h = √61。",
        {"方法错误"},
        "误把两边直接当直角边 → 方法错误",
    ),
    (
        "若 x²-5x+6=0，求 x 的值。",
        "x = 2 或 x = 3",
        "因式分解得 (x-2)(x-3)=0，所以 x=-2 或 x=-3。",
        {"粗心失误"},
        "过程完全正确但符号抄反 → 粗心失误",
    ),
    (
        "默写唐代诗人王维《使至塞上》的颔联。",
        "大漠孤烟直，长河落日圆。",
        "",
        {"知识遗忘"},
        "完全空白 → 知识遗忘",
    ),
    (
        "证明：等腰三角形两个底角相等。",
        "作顶角平分线，证明左右两个三角形全等，得底角相等。",
        "因为 AB=AC，所以∠B=∠C。",
        {"表述不完整"},
        "结论正确但省略证明过程 → 表述不完整",
    ),
    (
        "用所给动词的适当形式填空：He ___ (go) to school by bike every day.",
        "goes",
        "going",
        {"概念不清"},
        "动词形式受主谓一致/时态概念影响 → 概念不清",
    ),
    (
        "一个物体质量为 2kg，受到 10N 的水平拉力，求加速度。",
        "a = 5 m/s²",
        "a = 5。",
        {"粗心失误", "表述不完整"},
        "漏写单位 → 粗心/表述不完整（二者均可接受）",
    ),
    (
        "阅读《背影》，回答：文中「我」对父亲的情感发生了怎样的变化？",
        "从最初的不解、暗自嫌麻烦，到后来的理解、感念父爱。",
        "父亲是一个很关心儿子的人，作者很喜欢父亲。",
        {"审题不清"},
        "答非所问 → 审题不清",
    ),
    (
        "计算：(2/3)² × 3/4，结果化为最简分数。",
        "1/3",
        "(2/3) × 2 × (3/4) = 1。",
        {"审题不清"},
        "把平方看成乘2 → 审题不清（看错符号）",
    ),
    (
        "计算：∫₀¹ 2x dx。",
        "1",
        "得 2x²……后面不会了。",
        {"知识遗忘"},
        "只写出幂函数原形后放弃 → 知识遗忘",
    ),
]


def call_with_retry(q: str, ref: str, ans: str, attempts: int = 3) -> dict:
    last: dict = {}
    for i in range(attempts):
        r = analyze_error_cause_impl(q, ref, ans)
        if r.get("source") == "jev":
            return r
        last = r
        print(f"    ↻ 第 {i + 1} 次失败：{(r.get('error') or '')[:60]}，重试…")
    return last


def main() -> int:
    print("=" * 80)
    print("Jev 错因分析 · 模拟学生错题评测（12 题）")
    print("=" * 80)

    passed = reach = errors = flagged = 0
    multi_hit = 0
    latencies: list[float] = []

    for i, (q, ref, ans, expected, note) in enumerate(CASES, 1):
        t0 = time.time()
        r = call_with_retry(q, ref, ans)
        elapsed = time.time() - t0
        latencies.append(elapsed)

        if r.get("source") != "jev":
            errors += 1
            print(f"  {i:>2}. ✗ 网络失败 | {(r.get('error') or '')[:50]} | {note}")
            continue
        reach += 1

        actual = r["primary_cause"]
        ok = actual in expected
        detected = [c["cause"] for c in r.get("detected_causes", [])]
        covered = bool(expected & set(detected))  # 期望错因是否出现在多因标签中
        passed += ok
        multi_hit += covered
        flagged += r.get("needs_review", False)

        mark = "✓" if ok else ("△" if r.get("needs_review") else "✗")
        exp_str = "/".join(sorted(expected))
        print(
            f"  {i:>2}. {mark} 期望={exp_str:<6} 实际={actual:<6} "
            f"置信={r['confidence']:.2f} margin={r['margin']:.2f} "
            f"多因=[{','.join(detected) or '空'}] 复核={'是' if r.get('needs_review') else '否'} "
            f"{elapsed:.1f}s | {note}"
        )

    total = len(CASES)
    avg = sum(latencies) / len(latencies) if latencies else 0
    print("\n" + "=" * 80)
    print(f"网络可达: {reach}/{total}   主错因判定: {passed}/{total}   "
          f"多因覆盖(期望错因被标出): {multi_hit}/{total}")
    print(f"needs_review 触发: {flagged}/{total} 题   平均耗时: {avg:.1f}s/题")
    print("说明：✓ 判定正确；△ 判定与标注不一致但已被 needs_review 捕获（自动降级，不污染标签）；"
          "✗ 静默误判。")
    return 0 if reach > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
