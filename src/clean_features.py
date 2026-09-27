"""
clean_features.py -- 把特征矩阵清洗成 knockoff 分析可直接用的完整矩阵

背景：
    ABIDE 部分受试者扫描野未覆盖小脑下部/眶额区，导致那些 ROI 时间序列是常量，
    相关系数无法计算。build_features.py 已把这些格子标为 NaN（而不是伪造成 0）。
    但 knockoffs 需要完整矩阵，所以这里处理缺失。

清洗规则（默认，可用命令行参数改）：
    第一步：删掉缺失超过 --max-roi-miss 人的 ROI（列）
    第二步：删掉在保留 ROI 上仍不完整的受试者（行）

    默认阈值 20 的来历：缺失高度集中（75 个 ROI 有缺失，但只有 3 个缺超过 20 人，
    全是小脑下部）。阈值提到 20 只损失 3 个 ROI，却把受试者损失从 68 人压到 33 人。

Run:
    python clean_features.py                    # 用默认阈值 20
    python clean_features.py --max-roi-miss 0   # 最保守：谁缺都不留
    python clean_features.py --max-roi-miss 10  # 更严
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

FEATDIR = "features"


def report_shape(tag, X, man):
    n, p = X.shape
    line = f"{tag:12s} {n:4d} x {p:3d}"
    for coh in ("main", "reference"):
        idx = [i for i, r in enumerate(man) if r["cohort"] == coh]
        a = sum(1 for i in idx if man[i]["dx_group"] == "ASD")
        c = sum(1 for i in idx if man[i]["dx_group"] == "control")
        line += f"   {coh}: n={len(idx):4d} (ASD {a}, 对照 {c})"
    print(line)


def load_matrix(path):
    with open(path) as f:
        names = f.readline().strip().split(",")
    return np.loadtxt(path, delimiter=",", skiprows=1), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-roi-miss", type=int, default=20,
                    help="一个 ROI 最多允许有多少受试者缺失（默认 20）")
    ap.add_argument("--outdir", default="features_clean")
    args = ap.parse_args()

    man = list(csv.DictReader(open(os.path.join(FEATDIR, "subject_manifest.csv"))))
    X, names = load_matrix(os.path.join(FEATDIR, "X_meanconn.csv"))
    Xm, namesm = load_matrix(os.path.join(FEATDIR, "X_multifeat.csv"))
    N, P = X.shape
    print(f"读入 {N} x {P}；缺失 {int(np.isnan(X).sum())} 格 "
          f"({np.isnan(X).mean():.3%})\n")

    print("清洗前：")
    report_shape("原始", X, man)
    print()

    # ---- 第一步：砍列（缺失过多的 ROI）----
    roi_miss = np.isnan(X).sum(axis=0)
    keep_roi = roi_miss <= args.max_roi_miss
    dropped_roi = [(names[j], int(roi_miss[j])) for j in range(P) if not keep_roi[j]]

    # ---- 第二步：砍行（在保留 ROI 上仍不完整的受试者）----
    keep_sub = (~np.isnan(X[:, keep_roi])).all(axis=1)

    Xc = X[np.ix_(keep_sub, keep_roi)]
    names_c = [names[j] for j in range(P) if keep_roi[j]]

    # multifeat 按 "前缀_ROI名" 同步筛列
    keep_cols_m = [k for k, nm in enumerate(namesm)
                   if nm.split("_", 1)[1] in set(names_c)]
    Xmc = Xm[np.ix_(keep_sub, keep_cols_m)]
    names_mc = [namesm[k] for k in keep_cols_m]

    man_c = [man[i] for i in range(N) if keep_sub[i]]

    os.makedirs(args.outdir, exist_ok=True)
    np.savetxt(os.path.join(args.outdir, "X_meanconn.csv"), Xc, delimiter=",",
               fmt="%.6f", header=",".join(names_c), comments="")
    np.savetxt(os.path.join(args.outdir, "X_multifeat.csv"), Xmc, delimiter=",",
               fmt="%.6f", header=",".join(names_mc), comments="")
    with open(os.path.join(args.outdir, "subject_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man_c[0].keys()))
        w.writeheader()
        w.writerows(man_c)

    y = np.array([int(r["y"]) for r in man_c])
    is_ref = np.array([0 if r["cohort"] == "main" else 1 for r in man_c])
    np.savetxt(os.path.join(args.outdir, "y.csv"), y, fmt="%d")
    np.savetxt(os.path.join(args.outdir, "cohort.csv"), is_ref, fmt="%d")

    print(f"ROI 缺失上限 = {args.max_roi_miss}")
    print(f"  删 ROI  {len(dropped_roi)} 个："
          + (", ".join(f"{n}(缺{m})" for n, m in dropped_roi) if dropped_roi else "无"))
    print(f"  删受试者 {N - int(keep_sub.sum())} 人")
    print()
    print("清洗后：")
    report_shape("可用矩阵", Xc, man_c)
    print()
    print(f"  多特征版 {Xmc.shape[0]} x {Xmc.shape[1]}")
    print(f"  诊断 y 分布: ASD={int(y.sum())}  对照={int((y==0).sum())}")
    n_ref, n_main = int(is_ref.sum()), int((1 - is_ref).sum())
    need = 2 * len(names_c)
    print(f"  主队列 n={n_main}   参考队列 n={n_ref}"
          f"   要求 > 2p = {need}  ->  {'满足' if n_ref > need else '不满足!'}")
    print()
    print(f"已写入 {args.outdir}/：X_meanconn.csv, X_multifeat.csv, "
          f"subject_manifest.csv, y.csv, cohort.csv")


if __name__ == "__main__":
    main()
