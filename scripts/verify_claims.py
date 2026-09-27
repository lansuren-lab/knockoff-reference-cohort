"""
verify_claims.py -- 把手稿里的每个数字追溯到产物

存在理由（这是本项目最重要的一段工具代码）：

手稿里曾出现一个"42%"，它不来自任何脚本、任何日志、任何 CSV。旧脚本
R=30、输出步长为 3.3%，42% 在数学上根本无法产生。这个数字进了摘要，
并在正文出现 5 次。它不是计算错误，是**没有任何机制阻止手稿写下一个
无法追溯的数字**。

本脚本提供这个机制，分两层：

  第 1 层  【正向核对】claims.csv 逐条列出：
             id, location(手稿里的锚点文字), artifact(产物文件),
             column, selector(选哪一行), expected, tol
           逐条到产物里取值并比对，不一致就报错并以非零码退出。

  第 2 层  【反向覆盖】扫描 manuscript.tex 里的所有数值字面量，
           检查每一个是否被至少一条 claim 覆盖。未覆盖的会被列出来。
           这是防止"新写入一个数字但没有产物"的关键一步 ——
           单靠正向核对只能保证已登记的数字正确，"漏登记"仍需这一层。

用法：
    python scripts/verify_claims.py                     # 在仓库根运行
    python scripts/verify_claims.py --root . --strict    # 有任何未覆盖数字即失败

claims.csv 的 selector 语法（够用且不易写错）：
    "*"                     唯一一行
    "col=value"             该列等于 value（字符串相等，或在数值容差内）
    "col=value;col2=value2" 多条件与
留空则要求产物只有一行。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select(rows, selector):
    if not selector or selector.strip() == "*":
        if len(rows) != 1:
            raise ValueError(f"selector 为空但产物有 {len(rows)} 行")
        return rows[0]
    out = rows
    for cond in selector.split(";"):
        cond = cond.strip()
        if not cond:
            continue
        if "=" not in cond:
            raise ValueError(f"selector 片段无法解析: {cond!r}")
        col, val = cond.split("=", 1)
        col, val = col.strip(), val.strip()
        kept = []
        for r in out:
            rv = (r.get(col) or "").strip()
            if rv == val:
                kept.append(r)
                continue
            try:                                  # 数值容差比较
                if abs(float(rv) - float(val)) <= 1e-9:
                    kept.append(r)
            except (TypeError, ValueError):
                pass
        out = kept
    if not out:
        raise ValueError(f"没有行匹配 selector {selector!r}")
    return out[0]


def verify_claims(root, claims_path):
    if not os.path.exists(claims_path):
        print(f"[warn] 找不到 {claims_path}，跳过正向核对")
        return [], []
    results, problems = [], []
    for c in load_csv_rows(claims_path):
        cid = c.get("id") or "?"
        art = c.get("artifact") or ""
        p = os.path.join(root, art)
        try:
            if not os.path.exists(p):
                raise FileNotFoundError(art)
            row = select(load_csv_rows(p), c.get("selector") or "")
            col = (c.get("column") or "").strip()
            if col not in row:
                raise KeyError(f"{art} 无列 {col!r}")
            got = float(row[col])
            want = float(c["expected"])
            tol = float(c.get("tol") or 0.005)
            ok = abs(got - want) <= tol
            results.append((cid, art, col, want, got, ok))
            if not ok:
                problems.append(f"[{cid}] {c.get('location','')}  "
                                f"手稿 {want} vs {art}:{col} = {got:.6g}")
        except Exception as e:
            problems.append(f"[{cid}] {c.get('location','')}  无法核对: {e}")
    return results, problems


def scan_uncovered(root, claims_path):
    """反向覆盖：手稿里出现的数值字面量，有多少条 claim 覆盖它。"""
    tex = os.path.join(root, "paper", "manuscript_v3.tex")
    if not os.path.exists(tex):
        print(f"[warn] 找不到 {tex}")
        return []
    text = open(tex, encoding="utf-8").read()
    # 只看正文：preamble 里的数字是 LaTeX 参数（字号、间距），不是论断
    body_start = text.find("\\begin{document}")
    offset = body_start if body_start > 0 else 0
    body = text[offset:]
    declared = set()
    if os.path.exists(claims_path):
        for c in load_csv_rows(claims_path):
            for k in ("expected", "location"):
                v = (c.get(k) or "").strip()
                if v:
                    declared.add(v)
                    try:
                        declared.add(f"{float(v):g}")
                    except ValueError:
                        pass
    uncovered = []
    for m in NUMBER_RE.finditer(body):
        s = m.group(0)
        if s in declared or s.lstrip("-") in declared:
            continue
        line = text.count("\n", 0, offset + m.start()) + 1
        ctx = body[max(0, m.start() - 12):m.end() + 12]
        if any(t in ctx for t in ("@", "http", "htt", ".cn", ".edu", ".org",
                                  "texttt", "url{", "includegraphics",
                                  "paperwidth", "textwidth", "linewidth")):
            continue          # 邮箱/网址/图片尺寸，不是论断
        if re.fullmatch(r"(19|20)\d\d", s):
            continue          # 年份
        uncovered.append((line, s))
    return uncovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--claims", default="scripts/claims.csv")
    ap.add_argument("--strict", action="store_true",
                    help="有任何未覆盖的数值即以非零码退出")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    claims_path = os.path.join(root, args.claims)

    print("=" * 78)
    print("第 1 层：正向核对（手稿数字 vs 产物）")
    print("=" * 78)
    results, problems = verify_claims(root, claims_path)
    for cid, art, col, want, got, ok in results:
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] {cid:<12s} {art}:{col:<22s} 手稿={want:<10.4g} 产物={got:<10.6g}")
    print(f"\n  核对了 {len(results)} 条，失败 {len(problems)} 条")

    print()
    print("=" * 78)
    print("第 2 层：反向覆盖（手稿里未被任何 claim 覆盖的数值）")
    print("=" * 78)
    unc = scan_uncovered(root, claims_path)
    print(f"  未覆盖数值 {len(unc)} 个")
    for line, s in unc[:60]:
        print(f"    manuscript_v3.tex:{line}  {s}")
    if len(unc) > 60:
        print(f"    ... 另有 {len(unc)-60} 个")

    if problems:
        print("\n失败条目：")
        for p in problems:
            print(f"  - {p}")
        return 1
    if args.strict and unc:
        print(f"\n--strict：仍有 {len(unc)} 个数值无产物支撑")
        return 1
    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
