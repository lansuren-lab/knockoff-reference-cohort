"""
mechanism_replication.py -- 机制复现：ABIDE I (AAL116) vs ABIDE II (Power2011)

论文的核心论断是「机制稳、输出不稳」。§3.8 已在同一队列内用跨管线验证过机制稳。
现在用【真正独立的第二个队列】ABIDE II 再验一次。

两个队列的差异是全方位的：
    不同受试者、不同扫描仪、不同站点、不同图谱(AAL116 vs Power2011)、
    不同预处理管线(CPAC vs DCAN)、甚至不同数据格式(文本 vs CIFTI)。

要检验的是：以下机制是否在 ABIDE II 上仍然成立
    (1) 参考队列经验 Sigma 的 lam_min 偏小（近奇异）
    (2) 收缩抬高 lam_min，进而抬高 s = 2*lam_min
    (3) 收缩把阈值比 (1+F)/R 压低
    (4) 上述方向在两个队列上一致

统计量统一用固定-lambda 套索（完全确定），避免把 §3.9 的配置依赖混进来。

Run:  python mechanism_replication.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, "knockoff_adni")
sys.path.insert(0, ".")
from knockoffs import solve_s_diag, sample_knockoffs, knockoff_threshold  # noqa: E402
from sklearn.linear_model import Lasso  # noqa: E402

MAX_ROI_MISS = 20
C_FIXED = 0.15
Q = 0.20
N_MAIN_SITES = 6


def corr_of(A):
    C = np.corrcoef(A, rowvar=False); C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0); return C


def pd_floor(C, eps=1e-6):
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C); w = np.clip(w, eps, None)
    C = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C)); return C / np.outer(d, d)


def ledoit_sigma(Xr):
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(Xr).covariance_
    d = np.sqrt(np.clip(np.diag(lw), 1e-300, None))
    return pd_floor(lw / np.outer(d, d))


def lcd_fixed(Xa, y, c=C_FIXED):
    Xa = Xa - Xa.mean(0); y = y - y.mean()
    lam = c * np.abs(Xa.T @ y).max() / len(y)
    co = Lasso(alpha=lam, max_iter=100000, tol=1e-6).fit(Xa, y).coef_
    p = Xa.shape[1] // 2
    return np.abs(co[:p]) - np.abs(co[p:])


def prepare(d, main_sites=None, n_main=N_MAIN_SITES):
    f = os.path.join(d, "X_meanconn.csv")
    X = np.loadtxt(f, delimiter=",", skiprows=1)
    names = open(f).readline().strip().split(",")
    man = list(csv.DictReader(open(os.path.join(d, "subject_manifest.csv"))))
    keep_roi = np.isnan(X).sum(0) <= MAX_ROI_MISS
    keep_sub = (~np.isnan(X[:, keep_roi])).all(1)
    X = X[np.ix_(keep_sub, keep_roi)]
    names = [names[j] for j in range(len(names)) if keep_roi[j]]
    man = [man[i] for i in range(len(man)) if keep_sub[i]]
    # 主/参考：按站点人数取最大的 n_main 个
    cnt = {}
    for r in man:
        cnt[r["site"]] = cnt.get(r["site"], 0) + 1
    if main_sites is None:
        main_sites = sorted(cnt, key=lambda s: -cnt[s])[:n_main]
    ism = np.array([r["site"] in main_sites for r in man])
    return X, names, man, ism, main_sites


def analyse(X, names, man, ism, label):
    y = np.array([int(r["y"]) for r in man], float)
    Xm, ym, Xr = X[ism], y [ism], X[~ism]
    p = X.shape[1]
    res = {}
    for tag, S in (("empirical", pd_floor(corr_of(Xr))),
                   ("LW", ledoit_sigma(Xr))):
        mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
        sd[sd == 0] = 1.0
        Xs = (Xm - mu) / sd
        s = solve_s_diag(S, method="equicorrelated")
        Xt = sample_knockoffs(Xs, S, s, rng=np.random.default_rng(0))
        W = lcd_fixed(np.hstack([Xs, Xt]), ym)
        tau = knockoff_threshold(W, Q, plus_one=True)
        sel = np.where(W >= tau)[0] if np.isfinite(tau) else np.array([], int)
        if np.isfinite(tau) and len(sel):
            R = int((W >= tau).sum()); F = int((W <= -tau).sum())
            ratio = (1 + F) / R
        else:
            ratio = np.inf
        res[tag] = dict(lam=float(np.linalg.eigvalsh(S).min()),
                        s=float(np.median(s)), ratio=ratio,
                        nsel=int(len(sel)), sel=set(sel.tolist()), W=W)
    print(f"\n{'='*78}\n[{label}]  主队列 n={Xm.shape[0]}  参考队列 n={Xr.shape[0]}  "
          f"p={p}  站点 {len(set(r['site'] for r in man))}")
    print(f"  {'Sigma':<12s}{'lam_min':>10s}{'s中位':>9s}{'阈值比':>10s}{'|S|':>6s}")
    for t in ("empirical", "LW"):
        v = res[t]
        rs = f"{v['ratio']:.3f}" if np.isfinite(v["ratio"]) else "inf"
        print(f"  {t:<12s}{v['lam']:10.5f}{v['s']:9.4f}{rs:>10s}{v['nsel']:6d}")
    print(f"  收缩把 lam_min 抬高 {res['LW']['lam']/res['empirical']['lam']:.2f} 倍；"
          f"s 抬高 {res['LW']['s']/res['empirical']['s']:.2f} 倍")
    return res, ism, y


def main():
    print("=" * 78)
    print("机制复现检验：ABIDE I (AAL116, CPAC) vs ABIDE II (Power2011, DCAN)")
    print("=" * 78)

    Xi, ni, mi, ismi, sites_i = prepare("features")
    X2, n2, m2, ism2, sites2 = prepare("features_abide2")
    print(f"\nABIDE I  主队列站点: {sites_i}")
    print(f"ABIDE II 主队列站点: {sites2}")

    ri, _, _ = analyse(Xi, ni, mi, ismi, "ABIDE I  (AAL116, CPAC)")
    r2, _, _ = analyse(X2, n2, m2, ism2, "ABIDE II (Power2011, DCAN)")

    print()
    print("=" * 78)
    print("判读")
    print("=" * 78)
    checks = [
        ("lam_min(经验) 是否偏小（近奇异）",
         ri["empirical"]["lam"] < 0.1 and r2["empirical"]["lam"] < 0.1),
        ("收缩是否抬高 lam_min",
         ri["LW"]["lam"] > ri["empirical"]["lam"] and
         r2["LW"]["lam"] > r2["empirical"]["lam"]),
        ("收缩是否抬高 s",
         ri["LW"]["s"] > ri["empirical"]["s"] and
         r2["LW"]["s"] > r2["empirical"]["s"]),
        ("收缩是否压低阈值比",
         ri["LW"]["ratio"] < ri["empirical"]["ratio"] and
         r2["LW"]["ratio"] < r2["empirical"]["ratio"]),
    ]
    for name, ok in checks:
        print(f"  [{'是' if ok else '否'}] {name}")
    allok = all(o for _, o in checks)
    print()
    if allok:
        print("  >>> 机制在独立的第二个队列上全部复现。")
        print("      而两个队列在受试者/扫描仪/站点/图谱/管线/数据格式上全都不同。")
    else:
        print("  >>> 机制未全部复现，需要讨论哪些环节依赖队列。")


if __name__ == "__main__":
    main()
