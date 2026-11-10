"""API 冒烟测试：验证服务能起来、问答链路能通。

为什么单独写：用手敲 curl/PowerShell 发中文请求，极容易在客户端编码上翻车
（实测 PowerShell 会把中文 body 按 Latin-1 发出去，服务端收到乱码后检索为空，
看起来像服务端 bug，其实是客户端问题）。用 httpx 发 JSON 可以排除这个干扰。

用法：
    python -m scripts.smoke_api
    python -m scripts.smoke_api --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import sys

import httpx

QUESTIONS = [
    "动态核查多久更新一次？",
    "劳务动态核查的评分由哪几部分组成？",
    "建筑工人职业技能等级评价遵循什么原则？",
    "2021年江苏建筑业总产值是多少？",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="API 冒烟测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    with httpx.Client(timeout=60.0) as client:
        try:
            health = client.get(f"{base}/api/health")
            health.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"健康检查失败：{exc}")
            return 1

        data = health.json()
        print("健康检查：", data)
        if data.get("status") != "ok":
            print("索引未就绪，请先执行 python -m scripts.ingest")
            return 1

        failed = 0
        for question in QUESTIONS:
            try:
                resp = client.post(f"{base}/api/chat", json={"question": question, "top_k": 3})
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:  # noqa: BLE001
                print(f"  [失败] {question}: {exc}")
                failed += 1
                continue

            cites = " / ".join(c["citation"] for c in body.get("citations", [])) or "（无）"
            mode = (body.get("debug") or {}).get("mode", "")
            refused = "拒答" if body.get("refused") else "作答"
            print(f"\n问题：{question}")
            print(f"  {refused}｜模式={mode}｜耗时={body.get('latency_ms')}ms")
            print(f"  引用：{cites}")
            print(f"  答案：{body.get('answer', '')[:120]}")
            if body.get("refused"):
                failed += 1

        print(f"\n{len(QUESTIONS) - failed}/{len(QUESTIONS)} 个问题正常作答")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
