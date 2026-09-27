"""
exp_diagnostics.py -- E1: 把「谱诊断」做成论文的第一个结果

论文旧版把 meanconn 造成的秩塌缩当成 ABIDE 数据集的固有性质。实测显示它是
**特征定义的算术后果**：对相关矩阵取行均值，行均值必然被第一主成分主导。

本脚本对 5 个数据集 x 4 种特征定义，输出论文推荐用户自查的那一组量：
    lambda_min / lambda_max / 条件数 / PC1 占比 / effective rank

同时输出各定义之间的 per-ROI 一致性，用来量化"换定义到底改了什么"。

Run:  python exp_diagnostics.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (DEF_DATASETS_FOR_DIAG, ledoit_sigma, pd_floor,   # noqa: E402
                           corr_of, prepare, provenance, spectral_report,
                           split_xy, write_csv)

DEFS = ("meanconn", "gsr", "pcresid", "partial")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows, agree = [], []
    mats = {}
    for ds, d in DEF_DATASETS_FOR_DIAG.items():
        for deff in DEFS:
            try:
                X, names, man, ism, ms = prepare(d, fname=f"X_{deff}.csv")
            except Exception as e:                          # pragma: no cover
                print(f"  [skip] {ds}/{deff}: {e}")
                continue
            Xm, ym, Xr = split_xy(X, man, ism)
            if Xr.shape[0] < 20 or X.shape[1] < 5:
                continue
            S_emp = pd_floor(corr_of(Xr))
            S_lw = ledoit_sigma(Xr)
            ra, rb = spectral_report(S_emp), spectral_report(S_lw)
            rows.append({
                "dataset": ds, "feature": deff, "p": X.shape[1],
                "n_main": int(ism.sum()), "n_ref": int((~ism).sum()),
                "lam_min_emp": ra["lam_min"], "lam_min_lw": rb["lam_min"],
                "shrink_gain": rb["lam_min"] / max(ra["lam_min"], 1e-300),
                "s_emp": 2 * ra["lam_min"], "s_lw": 2 * rb["lam_min"],
                "lam_max": ra["lam_max"], "cond": ra["cond"],
                "pc1_share": ra["pc1_share"], "eff_rank": ra["eff_rank"],
            })
            mats[(ds, deff)] = X
            print(f"  {ds:18s} {deff:9s} p={X.shape[1]:3d} "
                  f"lam_min={ra['lam_min']:.5f} effrank={ra['eff_rank']:6.2f} "
                  f"PC1={100*ra['pc1_share']:5.1f}%")

        # 各定义之间的 per-ROI 一致性（同一批受试者、同一批 ROI）
        for a in DEFS:
            for b in DEFS:
                if a >= b or (ds, a) not in mats or (ds, b) not in mats:
                    continue
                A, B = mats[(ds, a)], mats[(ds, b)]
                if A.shape != B.shape:
                    continue
                ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
                if ok.sum() < 20:
                    continue
                r = [np.corrcoef(A[ok, j], B[ok, j])[0, 1]
                     for j in range(A.shape[1])]
                agree.append({"dataset": ds, "a": a, "b": b,
                              "median_r": float(np.nanmedian(r)),
                              "min_r": float(np.nanmin(r)),
                              "max_r": float(np.nanmax(r)),
                              "n_below_0.5": int(np.nansum(np.array(r) < 0.5)),
                              "p": A.shape[1]})

    p1 = write_csv(os.path.join(args.outdir, "diagnostics.csv"), rows)
    p2 = write_csv(os.path.join(args.outdir, "feature_agreement.csv"), agree)
    prov = provenance(args.outdir, "exp_diagnostics", args)
    print(f"\n-> {p1}\n-> {p2}\n-> {prov}")

    print("\n=== 换特征定义带来的秩变化（参考队列经验相关）===")
    print(f"{'dataset':20s}" + "".join(f"{d:>13s}" for d in DEFS))
    for ds in DEF_DATASETS_FOR_DIAG:
        cells = []
        for deff in DEFS:
            m = [r for r in rows if r["dataset"] == ds and r["feature"] == deff]
            cells.append(f"{m[0]['eff_rank']:13.2f}" if m else f"{'-':>13s}")
        print(f"{ds:20s}" + "".join(cells))


if __name__ == "__main__":
    main()
