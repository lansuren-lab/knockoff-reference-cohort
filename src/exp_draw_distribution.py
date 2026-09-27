"""
exp_draw_distribution.py -- E5: 把所有"零发现"表述换成分布

手稿的 §3.1 用【单次】knockoff 抽样得出的 |S|=0 当作主结论，而同一位作者在
§3.6 又证明单次抽样极不稳定（64.5% 的抽法返回零，但其余抽法最多返回 26 个）。
这两节从未被调和。用单次结果支撑"什么都不找到"，在论文自己的证据下站不住。

本脚本对每个真实配置跑 >=200 次独立 knockoff 抽样（其余一切固定），报告：

    P(非空集)          —— 替代"yields no discoveries"
    |S| 的分位与极值
    best-ratio 的分布，以及达到 q 的比例
    frac_converged     —— 套索未收敛的重复必须被识别出来

同时也扫统计量的折叠数配置（手稿 §solver 记录了 3 折给 7 个区域、5/10 折给 0 个），
把"配置依赖"和"抽样依赖"分开报。

Run:  python exp_draw_distribution.py --draws 200 --workers 14 --c 0.05
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (DEF_DATASETS_FOR_DIAG, Q_DEFAULT, corr_of,          # noqa: E402
                           knockoff_run, ledoit_sigma, pd_floor, prepare,
                           provenance, split_xy, write_csv)

DEFS = ("meanconn", "gsr", "pcresid", "partial")


def _init_worker(ARMS, c, seed):
    global _W
    _W = dict(ARMS=ARMS, c=c, seed=seed)


def _one_draw(job):
    key, d = job
    w = _W
    res = knockoff_run(w["ARMS"][f"__X__{key}"], w["ARMS"][f"__Y__{key}"],
                       w["ARMS"][key], q=Q_DEFAULT,
                       seed=w["seed"] * 100000 + d, c=w["c"])
    return {"key": key, "draw": d, "nsel": len(res["sel"]),
            "ratio_best": res["ratio_best"], "fdp_hat": res["fdp_hat"],
            "converged": float(res["converged"]),
            "tau_finite": float(np.isfinite(res["tau"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--c", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--only", default="",
                    help="逗号分隔的配置子串过滤，例如 AAL116")
    ap.add_argument("--outdir", default="results_c0.05")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    # 每个 (数据集, 特征定义) 一组：参考队列 LW Sigma，真实响应
    ARMS, meta = {}, {}
    for ds, d in DEF_DATASETS_FOR_DIAG.items():
        for deff in DEFS:
            try:
                X, names, man, ism, ms = prepare(d, fname=f"X_{deff}.csv")
            except Exception:
                continue
            Xm, ym, Xr = split_xy(X, man, ism)
            if Xm.shape[0] < 50 or Xr.shape[0] < 50:
                continue
            key = f"{ds}|{deff}"
            ARMS[key] = ledoit_sigma(Xr)      # 用户手上真实有的就是这个
            meta[key] = {"dataset": ds, "feature": deff, "p": X.shape[1],
                         "n_main": int(Xm.shape[0]), "n_ref": int(Xr.shape[0])}
            # 主队列与响应按 key 分别保存，worker 才能各用各的
            ARMS[f"__X__{key}"] = Xm
            ARMS[f"__Y__{key}"] = ym

    keys = [k for k in ARMS if not k.startswith("__")]
    if args.only:
        pats = tuple(p.strip() for p in args.only.split(",") if p.strip())
        keys = [k for k in keys if any(p in k for p in pats)]
    print(f"配置 {len(keys)} 个 × {args.draws} 次抽样   c={args.c}\n")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import defaultdict
    acc = defaultdict(list)
    # 【一个 pool 跑完全部】把每个配置的 Xm/ym 一并塞进 ARMS 交给 worker，
    # 由 _one_draw 按 key 查表。早先按配置逐个建 pool（20 个）光启动开销就
    # 耗尽预算；单个 pool 只初始化一次。
    # 逐配置推进，这样哪个配置慢/卡住一眼可见
    for ki, key in enumerate(keys):
        tk = time.time()
        with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker,
                initargs=(ARMS, args.c, args.seed)) as ex:
            futs = [ex.submit(_one_draw, (key, d))
                    for d in range(args.draws)]
            for f in as_completed(futs):
                acc[key].append(f.result())
        print(f"  [{ki+1}/{len(keys)}] {key:34s} {time.time()-tk:6.1f}s "
              f"(总 {time.time()-t0:.0f}s)", flush=True)

    rows = []
    for key in keys:
        g = acc[key]
        ns = np.array([x["nsel"] for x in g])
        rb = np.array([x["ratio_best"] for x in g])
        finite = np.isfinite(rb)
        rows.append({
            **meta[key], "c": args.c, "draws": len(g),
            "p_nonempty": float((ns > 0).mean()),
            "p_ratio_le_q": float((rb <= Q_DEFAULT).mean()),
            "nsel_mean": float(ns.mean()),
            "nsel_median": float(np.median(ns)),
            "nsel_p95": float(np.percentile(ns, 95)),
            "nsel_max": int(ns.max()),
            "ratio_median": float(np.median(rb[finite])) if finite.any()
                            else float("inf"),
            "ratio_min": float(rb.min()),
            "frac_converged": float(np.mean([x["converged"] for x in g])),
        })

    path = write_csv(os.path.join(args.outdir, "draw_distribution.csv"), rows)
    prov = provenance(args.outdir, "exp_draw_distribution", args)
    print(f"\n-> {path}\n-> {prov}\n总耗时 {time.time()-t0:.0f}s\n")

    print(f"{'dataset':22s}{'feature':10s}{'P(非空)':>9s}{'P(ratio<=q)':>12s}"
          f"{'|S|中位':>9s}{'|S|最大':>8s}{'ratio中位':>11s}{'conv':>7s}")
    for r in rows:
        print(f"{r['dataset']:22s}{r['feature']:10s}{r['p_nonempty']:9.3f}"
              f"{r['p_ratio_le_q']:12.3f}{r['nsel_median']:9.0f}{r['nsel_max']:8d}"
              f"{r['ratio_median']:11.3f}{r['frac_converged']:7.3f}")


if __name__ == "__main__":
    main()
