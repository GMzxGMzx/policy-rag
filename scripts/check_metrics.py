"""评测指标门槛校验：CI 门禁 + 本地回归检查。

为什么单独一个脚本：指标"看起来还行"和"达标"是两回事。
把门槛写成可执行的断言，才能真正挡住"改了提示词导致引用有效率下降却没人发现"这类回归。

用法：
    python -m scripts.check_metrics --dir data/samples/runs
    python -m scripts.check_metrics --dir data/eval/runs --min-recall 0.9 --min-citation 0.9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def latest_metrics(run_dir: Path) -> tuple[Path | None, dict | None]:
    files = sorted(run_dir.glob("*.metrics.json"))
    if not files:
        return None, None
    path = files[-1]  # 文件名带时间戳，按名称排序即按时间排序
    return path, json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description="校验评测指标是否达到门槛")
    parser.add_argument("--dir", default="data/eval/runs", help="评测输出目录")
    parser.add_argument("--min-recall", type=float, default=0.9, help="条款召回率下限")
    parser.add_argument("--min-citation", type=float, default=0.9, help="引用有效率下限")
    args = parser.parse_args()

    run_dir = Path(args.dir)
    if not run_dir.exists():
        print(f"目录不存在：{run_dir}")
        return 1

    path, metrics = latest_metrics(run_dir)
    if metrics is None:
        print(f"没有找到 metrics 文件：{run_dir}/*.metrics.json")
        return 1

    print(f"读取：{path.name}")
    print(f"样本数：{metrics.get('样本数')}（可答 {metrics.get('可答样本数')} / 应拒答 {metrics.get('应拒答样本数')}）")

    checks = [
        ("Recall@k_条款", metrics.get("Recall@k_条款"), args.min_recall),
        ("引用有效率", metrics.get("引用有效率"), args.min_citation),
    ]

    failed = 0
    for name, value, floor in checks:
        if value is None:
            # 全部样本都被拒答时这两项不产出，属正常，跳过而非判失败
            print(f"  [跳过] {name}：本次未产出该指标")
            continue
        passed = float(value) >= floor
        print(f"  [{'通过' if passed else '未通过'}] {name} = {value}（门槛 {floor}）")
        if not passed:
            failed += 1

    if failed:
        print(f"\n{failed} 项指标未达门槛")
        return 1
    print("\n全部达标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
