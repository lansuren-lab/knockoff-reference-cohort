"""
audit_claims.py -- 反向审核：claims.csv 登记的值，手稿是否真的用到

`verify_claims.py` 的第 1 层核对「claim ↔ 结果 CSV」，**不读手稿**。因此它
有一个方向的盲区：手稿里的句子被改写或删除之后，对应的 claim 照样通过。
本脚本补上反方向——claim 声称「手稿里有这个数」，手稿里到底有没有。

这个方向也不是万能的：一条 claim 可能是在为**转述**做溯源（手稿写
"almost every fit fails"，claim 登记 0.983），也可能对应表格单元格或被画进
图里的点。所以本脚本报的是「手稿逐字找不到」，而不是「这条 claim 错了」，
需要人工判断属于哪一类。

判定分三档，刻意从严以避免把碰巧相同的短数字当成命中：
  EXACT   手稿里逐字出现
  ROUNDED 手稿里以较短形式出现
  UNUSED  手稿里找不到——需人工确认是转述、表格单元格、还是已废弃

Run:  python scripts/audit_claims.py     （从仓库根目录）
"""
from __future__ import annotations

import csv
import os
import re
import sys

TEX = os.path.join("paper", "manuscript_v3.tex")
CLAIMS = os.path.join("scripts", "claims.csv")


def has(body: str, s: str) -> bool:
    return re.search(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", body) is not None


def main() -> None:
    if not (os.path.exists(TEX) and os.path.exists(CLAIMS)):
        raise SystemExit(f"run from the repo root ({TEX} / {CLAIMS} not found)")
    text = open(TEX, encoding="utf-8").read()
    body = text[text.find("\\begin{document}"):]
    rows = list(csv.DictReader(open(CLAIMS, newline="", encoding="utf-8")))

    exact, rounded, unused = 0, [], []
    for c in rows:
        exp = (c.get("expected") or "").strip()
        if not exp:
            continue
        try:
            v = float(exp)
        except ValueError:
            continue
        if has(body, exp) or has(body, "%g" % v):
            exact += 1
            continue

        hit = None
        if float(v).is_integer() and abs(v) >= 10:
            if has(body, f"{v:.0f}"):
                hit = f"{v:.0f}"
        else:
            for sig in (4, 3):
                cand = f"{v:.{sig}g}"
                if has(body, cand):
                    hit = cand
                    break
            if hit is None and 0 < abs(v) < 0.1:
                for nd in (1, 2):
                    cand = f"{v * 100:.{nd}f}"
                    if has(body, cand):
                        hit = cand + "%"
                        break
        (rounded if hit else unused).append((c, hit))

    print(f"claim 总数 {len(rows)}    EXACT {exact}    "
          f"ROUNDED {len(rounded)}    UNUSED {len(unused)}\n")
    if rounded:
        print("—— 手稿中以较短形式出现（映射成立）——")
        for c, hit in rounded:
            print(f"  {c['id']:>4}  {c['expected']:>10}  ->  手稿 {hit}")
        print()
    if unused:
        print("—— 手稿逐字找不到，需人工确认类别 ——")
        byart: dict[str, list[str]] = {}
        for c, _ in unused:
            byart.setdefault(c["artifact"], []).append(c["id"])
        for art, ids in sorted(byart.items()):
            print(f"  {art}")
            print(f"      {', '.join(ids)}")
        print("\n  这三类都不算错，但含义不同：")
        print("    转述     手稿用文字表达（'almost every fit fails' 对应 0.983）")
        print("    表格/图  值是表格单元格或被画进图里的点")
        print("    已废弃   该分析已不在手稿中——这条 claim 应当退休")


if __name__ == "__main__":
    main()
