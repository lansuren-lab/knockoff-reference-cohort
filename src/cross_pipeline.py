"""
cross_pipeline.py -- 跨预处理管线对照（ABIDE I: CPAC vs NIAK）

问题：
    论文的核心警告是「参考队列 knockoffs 的行为很脆」。如果连同一队列、
    同一图谱、只换预处理管线都会让结论变化，那么跨队列复制必须极其谨慎。

    反过来，如果换管线后「机制」仍然复现（谱塌陷、收缩的作用方向一致），
    那么跨队列工作的可信度大增。

这一对照的变量是【单一】的：预处理管线。受试者完全相同（885 人）、
图谱完全相同（AAL116）、参考/主队列划分完全相同。

统计量用固定-lambda 套索（完全确定，无 CV 配置歧义），
以免把 §3.9 发现的「统计量配置依赖」混进管线效应。

Run:  python cross_pipeline.py
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

MAIN_SITES = ["NYU", "UM_1", "USM", "UCLA_1", "YALE", "PITT"]
MAX_ROI_MISS = 20
C_FIXED = 0.15
Q = 0.20


def corr_of(A):
    C = np.corrcoef(A, rowvar=False)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def pd_floor(C, eps=1e-6):
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    C = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def shrink_to(C, a):
    p = C.shape[0]
    return pd_floor((1 - a) * C + a * np.eye(p))


def ledoit_sigma(Xr):
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(Xr).covariance_
    d = np.sqrt(np.clip(np.diag(lw), 1e-300, None))
    return pd_floor(lw / np.outer(d, d))


def lcd_fixed(Xa, y, c=C_FIXED):
    Xa = Xa - Xa.mean(axis=0)
    y = y - y.mean()
    lam = c * np.abs(Xa.T @ y).max() / len(y)
    co = Lasso(alpha=lam, max_iter=100000, tol=1e-6).fit(Xa, y).coef_
    p = Xa.shape[1] // 2
    return np.abs(co[:p]) - np.abs(co[p:])


def load_raw(d):
    """读原始特征矩阵（不清洗），供交集对齐用。"""
    f = os.path.join(d, "X_meanconn.csv")
    X = np.loadtxt(f, delimiter=",", skiprows=1)
    names = open(f).readline().strip().split(",")
    man = list(csv.DictReader(open(os.path.join(d, "subject_manifest.csv"))))
    return X, names, man


def run(X, man, seed=0):
    ism = np.array([r["site"] in MAIN_SITES for r in man])
    y = np.array([int(r["y"]) for r in man], float)
    Xm, ym, Xr = X[ism], y[ism], X[~ism]
    p = X.shape[1]
    out = {}
    for tag, S in (("empirical", pd_floor(corr_of(Xr))),
                   ("LW", ledoit_sigma(Xr))):
        mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
        sd[sd == 0] = 1.0
        Xs = (Xm - mu) / sd
        s = solve_s_diag(S, method="equicorrelated")
        Xt = sample_knockoffs(Xs, S, s, rng=np.random.default_rng(seed))
        W = lcd_fixed(np.hstack([Xs, Xt]), ym)
        tau = knockoff_threshold(W, Q, plus_one=True)
        sel = np.where(W >= tau)[0] if np.isfinite(tau) else np.array([], int)
        if np.isfinite(tau) and len(sel):
            R = int((W >= tau).sum()); F = int((W <= -tau).sum())
            ratio = (1 + F) / R
        else:
            ratio = np.inf
        out[tag] = dict(lam=np.linalg.eigvalsh(S).min(), s=np.median(s),
                        ratio=ratio, sel=set(sel.tolist()), W=W)
    return out, ism, y


def main():
    print("=" * 80)
    print("跨预处理管线对照：ABIDE I 的 CPAC vs NIAK（同队列、同图谱、同划分）")
    print("=" * 80)

    Xc0, nc0, mc0 = load_raw("features")            # CPAC
    Xn0, nn0, mn0 = load_raw("features_niak")       # NIAK
    print(f"\n  原始：CPAC {Xc0.shape}（缺失 {int(np.isnan(Xc0).sum())} 格）   "
          f"NIAK {Xn0.shape}（缺失 {int(np.isnan(Xn0).sum())} 格）")

    # ---- 在两个管线的【交集】上对齐：共同 ROI + 共同受试者 ----
    rois = [r for r in nc0 if r in set(nn0)]
    ic = [nc0.index(r) for r in rois]
    inn = [nn0.index(r) for r in rois]
    fc = {r["file_id"]: i for i, r in enumerate(mc0)}
    fn = {r["file_id"]: i for i, r in enumerate(mn0)}
    subs = [f for f in fc if f in fn]
    subs = [r["file_id"] for r in mc0 if r["file_id"] in fn]   # 保持 CPAC 顺序

    Xc = Xc0[np.ix_([fc[s] for s in subs], ic)]
    Xn = Xn0[np.ix_([fn[s] for s in subs], inn)]
    mc = [mc0[fc[s]] for s in subs]
    mn = [mn0[fn[s]] for s in subs]

    # 用统一规则清洗：删掉任一管线缺失过多的 ROI，再删仍不完整的受试者
    bad_roi = ((np.isnan(Xc).sum(0) > MAX_ROI_MISS) |
               (np.isnan(Xn).sum(0) > MAX_ROI_MISS))
    keep_roi = ~bad_roi
    keep_sub = ((~np.isnan(Xc[:, keep_roi])).all(1) &
                (~np.isnan(Xn[:, keep_roi])).all(1))
    Xc = Xc[np.ix_(keep_sub, keep_roi)]
    Xn = Xn[np.ix_(keep_sub, keep_roi)]
    nc = [rois[j] for j in range(len(rois)) if keep_roi[j]]
    mc = [mc[i] for i in range(len(mc)) if keep_sub[i]]
    mn = [mn[i] for i in range(len(mn)) if keep_sub[i]]

    print(f"  对齐后（共同 ROI ∩ 共同受试者）：CPAC {Xc.shape}  NIAK {Xn.shape}")
    print(f"    共同 ROI {len(rois)} 个（保留 {len(nc)}），共同受试者 {len(subs)} 人"
          f"（保留 {len(mc)}）")
    print(f"    NIAK 侧被删的受试者 {len(subs)-len(mc)} 人（其为对齐 CPAC 的缺失）")

    # ---- 特征层一致性 ----
    rc = [np.corrcoef(Xc[:, j], Xn[:, j])[0, 1] for j in range(Xc.shape[1])]
    rc = np.array(rc)
    r_flat = np.corrcoef(Xc.ravel(), Xn.ravel())[0, 1]
    print(f"\n  特征层一致性（同一受试者、同一 ROI 在两个管线间的相关）：")
    print(f"    全体展平相关      {r_flat:.3f}")
    print(f"    逐 ROI 相关 中位  {np.median(rc):.3f}   最小 {rc.min():.3f}   最大 {rc.max():.3f}")
    print(f"    相关 < 0.5 的 ROI 数  {int((rc < 0.5).sum())} / {len(rc)}")

    # ---- knockoff 对照 ----
    oc, _, _ = run(Xc, mc)
    on, _, _ = run(Xn, mn)

    print(f"\n  {'管线':<8s}{'Sigma':<11s}{'lam_min':>9s}{'s中位':>8s}"
          f"{'阈值比':>9s}{'|S|':>6s}")
    for tag, o in (("CPAC", oc), ("NIAK", on)):
        for sig in ("empirical", "LW"):
            v = o[sig]
            print(f"  {tag:<8s}{sig:<11s}{v['lam']:9.5f}{v['s']:8.4f}"
                  f"{v['ratio']:9.3f}{len(v['sel']):6d}")

    # ---- 发现集比较 ----
    print()
    for sig in ("empirical", "LW"):
        sc, sn = oc[sig]["sel"], on[sig]["sel"]
        u = sc | sn
        jac = len(sc & sn) / len(u) if u else 1.0
        print(f"  {sig:<11s} CPAC |S|={len(sc):3d}   NIAK |S|={len(sn):3d}   "
              f"交集={len(sc & sn):3d}   Jaccard={jac:.3f}")
        if len(sc) and len(sn):
            print(f"      共同选中: {', '.join(sorted(nc[j] for j in (sc & sn))[:8])}")

    # ---- W 排序一致性 ----
    print()
    for sig in ("empirical", "LW"):
        wc, wn = oc[sig]["W"], on[sig]["W"]
        print(f"  {sig:<11s} W 向量的跨管线相关 = {np.corrcoef(wc, wn)[0,1]:.3f}")

    # ---- 结论 ----
    print()
    print("=" * 80)
    lam_ok = all(o["empirical"]["lam"] < 0.5 * o["LW"]["lam"] for o in (oc, on))
    dir_ok = all(o["LW"]["s"] > o["empirical"]["s"] for o in (oc, on))
    print("判读")
    print(f"  * 两管线下 λ_min(经验) 是否都明显小于 λ_min(LW)？  {'是' if lam_ok else '否'}")
    print(f"  * 两管线下收缩是否都抬高 s？                      {'是' if dir_ok else '否'}")
    print("  -> 机制层面若一致，说明跨管线复制了『谱塌陷』这一核心现象；")
    print("     发现集层面若不一致，则再次印证论文主题：输出不稳而机制稳。")


if __name__ == "__main__":
    main()
