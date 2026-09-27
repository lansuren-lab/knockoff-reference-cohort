"""
exp_multienv_positive_control.py -- 多环境滤波器的【正对照】

动机（审稿意见）：`exp_multi_environment.py` 只在真实响应上报告了"什么都不选"，
另有一个打乱响应的阴性对照，但**没有正对照**。因此那 0 个选中无法与
"该配置下这个方法本来就没有功效"区分开——零结果不是阴性证据。

本脚本补上正对照：注入一个人为构造的、**真正跨环境一致**的信号，检验滤波器
能否把它找回来。若能，则真实数据上的 0 个选中才构成"没有跨站点可复制关联"
的证据；若不能，则那 0 个选中什么也不说明。

注入定义（务必与 `exp_positive_control_v3.py` 同一套标定，否则两处不可比）：
  * 每个环境内先把特征标准化（与 knockoff_W 内部一致）；
  * 支撑集在所有环境【完全相同】（同 12 个坐标）；
  * 符号相同、幅度相同（a > 0），幅度用条件 |t| 标定到 target（默认 6），
    标定在【合并后的标准化特征】上做，使"标准化效应量"跨环境一致；
  * 噪声为独立标准正态，与主实验一致。

产出 results_multienv/positive_control.csv：
  每个 r 下的 mean_power（12 个注入坐标中被选中的平均比例）、
  n_selected_mean（总选中数，含假发现）、以及打乱对照的同名量。

Run:  python exp_multienv_positive_control.py --reps 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (C_FIXED, Q_DEFAULT, conditional_t_all, prepare,   # noqa: E402
                           provenance, write_csv)
from exp_positive_control_v3 import calibrate_amplitude                      # noqa: E402
from exp_multi_environment import multienv                                   # noqa: E402

K_INJECT = 12
TARGET_COND_T = 6.0


def build_environments(min_n, mode="sites", n_env=4, seed=0):
    """构造环境。

    mode="sites"：与 exp_multi_environment.py 完全相同的按站点切分，保证可比。
    mode="split"：把全部受试者随机等分成 n_env 组（忽略站点）。
        用来区分"滤波器本身无功效"与"环境太小导致无功效"——
        若 split 能恢复注入信号而 sites 不能，则失效归因于环境规模。
    """
    X, names, man, ism, ms = prepare("feat2_abide1_aal")
    sites = np.array([r["site"] for r in man])
    y = np.array([float(r["y"]) if str(r["y"]).strip() != "" else np.nan
                  for r in man])
    ok = np.isfinite(y)
    X, y, sites = X[ok], y[ok], sites[ok]

    def standardize(Xe):
        mu, sd = Xe.mean(0), Xe.std(0, ddof=1)
        sd[sd == 0] = 1.0
        return (Xe - mu) / sd

    envs = []
    if mode == "sites":
        for s_ in sorted(set(sites)):
            m = sites == s_
            if m.sum() >= min_n and len(set(y[m])) == 2:
                envs.append((s_, standardize(X[m])))
    elif mode == "split":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        for k, chunk in enumerate(np.array_split(idx, n_env)):
            Xe, ye = X[chunk], y[chunk]
            if len(set(ye)) == 2:
                envs.append((f"split{k}", standardize(Xe)))
    else:
        raise SystemExit(f"unknown --env-mode {mode}")
    return envs, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--q", type=float, default=Q_DEFAULT)
    ap.add_argument("--c", type=float, default=C_FIXED)
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--target-cond-t", type=float, default=TARGET_COND_T)
    ap.add_argument("--env-mode", default="sites",
                    choices=("sites", "split"),
                    help="sites = 按站点（与主实验一致）；split = 随机等分")
    ap.add_argument("--n-env", type=int, default=4, help="split 模式的环境数")
    ap.add_argument("--outcsv", default="positive_control.csv")
    ap.add_argument("--outdir", default="results_multienv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    envs, names = build_environments(args.min_n, mode=args.env_mode,
                                     n_env=args.n_env, seed=args.seed)
    E = len(envs)
    p = envs[0][1].shape[1]
    n_tot = sum(e[1].shape[0] for e in envs)
    print(f"环境 {E} 个（ABIDE~I 站点，n>={args.min_n} 且两类俱全），"
          f"合计 n={n_tot}，p={p}")
    for s_, Xe in envs:
        print(f"    {s_:12s} n={Xe.shape[0]:4d}")

    r_values = [r for r in (1, 2, 3, 4, 5) if r <= E]

    # 标定用的合并标准化特征：条件 |t| 在这里算，使标准化效应量跨环境一致
    Xpool = np.vstack([e[1] for e in envs])
    _, design = conditional_t_all(Xpool, np.zeros(len(Xpool)))

    X_list = [e[1] for e in envs]
    acc = {r: [] for r in r_values}          # 每个 r 的每信号功效
    nsel = {r: [] for r in r_values}         # 每个 r 的总选中数
    acc0 = {r: [] for r in r_values}         # 打乱对照
    nsel0 = {r: [] for r in r_values}
    amps = []

    print(f"\n注入 {K_INJECT} 个坐标，条件 |t| 标定到 {args.target_cond_t}，"
          f"重复 {args.reps} 次")
    print("=" * 72)
    for rep in range(args.reps):
        rr = np.random.default_rng(args.seed * 100003 + rep)
        supp = tuple(sorted(rr.choice(p, K_INJECT, replace=False).tolist()))
        noise_pool = rr.standard_normal(len(Xpool))
        a = calibrate_amplitude(Xpool, design, supp, args.target_cond_t,
                                noise_pool)
        amps.append(a)
        beta = np.zeros(p)
        beta[list(supp)] = a
        supp_set = set(supp)

        y_list, y_list0 = [], []
        for Xe in X_list:
            ne = rr.standard_normal(Xe.shape[0])
            y_list.append(Xe @ beta + ne)
            y_list0.append(rr.permutation(y_list[-1]))

        res = multienv(X_list, y_list, args.q, r_values, seed=args.seed + rep,
                       c=args.c)
        res0 = multienv(X_list, y_list0, args.q, r_values,
                        seed=args.seed + rep + 500000, c=args.c)
        for r in r_values:
            sel = set(res[r]["sel"].tolist())
            acc[r].append(len(sel & supp_set) / K_INJECT)
            nsel[r].append(len(sel))
            sel0 = set(res0[r]["sel"].tolist())
            acc0[r].append(len(sel0 & supp_set) / K_INJECT)
            nsel0[r].append(len(sel0))
        if (rep + 1) % 10 == 0:
            print(f"  完成 {rep+1}/{args.reps}  ({time.time()-t0:.0f}s)")

    print(f"\n注入幅度 a: 中位 {np.median(amps):.3f}  "
          f"范围 {min(amps):.3f}~{max(amps):.3f}")
    print("-" * 72)
    print(f"{'r':>2} {'power(注入信号)':>16} {'选中数':>16} "
          f"{'power(打乱)':>14} {'选中数(打乱)':>16}")
    rows = []
    for r in r_values:
        m_pow, s_pow = float(np.mean(acc[r])), float(np.std(acc[r]))
        m_n, s_n = float(np.mean(nsel[r])), float(np.std(nsel[r]))
        m_p0 = float(np.mean(acc0[r]))
        m_n0 = float(np.mean(nsel0[r]))
        print(f"{r:>2} {m_pow:>9.3f}±{s_pow:.3f} {m_n:>9.2f}±{s_n:.2f} "
              f"{m_p0:>14.3f} {m_n0:>16.2f}")
        rows.append({"r": r, "E": E, "q": args.q, "c": args.c,
                     "reps": args.reps, "k_injected": K_INJECT,
                     "target_cond_t": args.target_cond_t,
                     "mean_amplitude": float(np.median(amps)),
                     "power_mean": m_pow, "power_sd": s_pow,
                     "n_selected_mean": m_n, "n_selected_sd": s_n,
                     "power_shuffled_mean": m_p0,
                     "n_selected_shuffled_mean": m_n0})

    path = write_csv(os.path.join(args.outdir, args.outcsv), rows)
    provenance(args.outdir, "exp_multienv_positive_control", args)
    print(f"\n-> {path}\n总耗时 {time.time()-t0:.0f}s")
    print("\n读数：power>0 且打乱对照 ~0 时，真实数据上的 0 个选中才构成阴性证据。")


if __name__ == "__main__":
    main()
