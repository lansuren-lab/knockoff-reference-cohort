"""
evalue_derandomization.py -- 判决性实验：e-value 去随机化 vs 中位数聚合

背景：
    上一轮发现【中位数聚合是无效的】：在全局零（y 与 X 独立）下，
    M=10 的聚合凭空造出约 20 个发现，FDP = 0.98。
    我的"边际对称性"论证只证了中位数的边际分布对称，
    不足以支撑 knockoff+ 所需的联合符号对称性。

    Ren & Barber (2024, JRSS-B 86(1):122-154) 的 e-value 去随机化【有证明】。
    本脚本做三件事：
      1. 验证单次 e-value + e-BH 确实复现 knockoffs+（对拍）
      2. 在全局零下测 e-value 去随机化 —— FDP 应受控
      3. 与中位数聚合直接对比

e-value 构造（Ren & Barber Thm 1）：
    e_j = p * 1{W_j >= T} / (1 + #{k : W_k <= -T})
    其中 T 是该次运行的 knockoff 阈值。
    直觉：|S| = R 个变量共享同一个 e 值 p/(1+F)，而阈值保证 R >= (1+F)/q，
    故 e-BH 的 k* = R，与 knockoffs+ 的选择集完全相同。

Run:  python evalue_derandomization.py
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np

sys.path.insert(0, "knockoff_adni")
sys.path.insert(0, ".")
from knockoffs import solve_s_diag, sample_knockoffs, knockoff_threshold  # noqa: E402
from sklearn.linear_model import Lasso  # noqa: E402

C_FIXED = 0.15
Q = 0.20
MAIN_SITES = ["NYU", "UM_1", "USM", "UCLA_1", "YALE", "PITT"]
MAX_ROI_MISS = 20


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


def knockoff_evalues(W, q_kn):
    """Ren & Barber (2024) 的 knockoff e-value。"""
    p = len(W)
    T = knockoff_threshold(W, q_kn, plus_one=True)
    if not np.isfinite(T):
        return np.zeros(p)
    F = int((W <= -T).sum())
    e = np.zeros(p)
    e[W >= T] = p / (1.0 + F)
    return e


def eBH(e, q):
    """e-BH 程序。"""
    p = len(e)
    if p == 0:
        return np.array([], int)
    order = np.argsort(-e)
    es = e[order]
    k = np.arange(1, p + 1)
    ok = es >= p / (q * k)
    if not ok.any():
        return np.array([], int)
    kstar = int(np.max(np.where(ok)[0])) + 1
    return np.sort(order[:kstar])


def main():
    t0 = time.time()
    X = np.loadtxt("features_clean/X_meanconn.csv", delimiter=",", skiprows=1)
    man = list(csv.DictReader(open("features_clean/subject_manifest.csv")))
    ism = np.array([r["site"] in MAIN_SITES for r in man])
    Xm, Xr = X[ism], X[~ism]
    p = X.shape[1]
    mu, sd = Xm.mean(0), Xm.std(0, ddof=1); sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    S = ledoit_sigma(Xr)
    s = solve_s_diag(S, method="equicorrelated")

    print("=" * 78)
    print("e-value 去随机化 vs 中位数聚合")
    print("=" * 78)
    print(f"  真实协变量 {Xs.shape}  q={Q}  固定-lambda c={C_FIXED}")
    print()

    # ---------- 步骤 1：对拍，验证单次 e-BH == knockoffs+ ----------
    print("=" * 78)
    print("[1] 对拍：单次 e-value + e-BH 是否复现 knockoffs+ ？")
    print("=" * 78)
    agree = 0
    R = 25
    for r in range(R):
        y = np.random.default_rng(r).standard_normal(len(Xs))
        Xt = sample_knockoffs(Xs, S, s, rng=np.random.default_rng(500 + r))
        W = lcd_fixed(np.hstack([Xs, Xt]), y)
        T = knockoff_threshold(W, Q, plus_one=True)
        kn = set(np.where(W >= T)[0].tolist()) if np.isfinite(T) else set()
        eb = set(eBH(knockoff_evalues(W, Q), Q).tolist())
        agree += (kn == eb)
    print(f"  25 次随机试验中，两者给出相同选择集的次数：{agree}/25"
          f"  -> {'一致（公式正确）' if agree == 25 else '不一致，需排查'}")
    print()

    # ---------- 步骤 2：全局零 ----------
    print("=" * 78)
    print("[2] 全局零（y 与 X 完全独立）—— 正确行为是 |S| ≈ 0")
    print("=" * 78)
    print(f"  {'M':>4s}{'中位数聚合':>16s}{'e-value 去随机化':>20s}")
    print(f"  {'':>4s}{'|S|均值':>9s}{'FDP':>7s}{'|S|均值':>11s}{'FDP':>9s}")
    R2 = 60
    for M in (1, 5, 10, 20, 40):
        ns_med, ns_ev = [], []
        for r in range(R2):
            ynull = np.random.default_rng(9000 + r).standard_normal(len(Xs))
            Ws = []
            for m in range(M):
                Xt = sample_knockoffs(Xs, S, s,
                                      rng=np.random.default_rng(20_000 + r * 100 + m))
                Ws.append(lcd_fixed(np.hstack([Xs, Xt]), ynull))
            Wmat = np.vstack(Ws)
            # 中位数聚合
            Wa = Wmat[0] if M == 1 else np.median(Wmat, axis=0)
            ta = knockoff_threshold(Wa, Q, plus_one=True)
            ns_med.append(len(np.where(Wa >= ta)[0]) if np.isfinite(ta) else 0)
            # e-value 去随机化
            ebar = np.mean([knockoff_evalues(W, Q) for W in Wmat], axis=0)
            ns_ev.append(len(eBH(ebar, Q)))
        # 全局零下所有发现都是假的 -> FDP = 1 if |S|>0 else 0
        fdp_med = np.mean([1.0 if n > 0 else 0.0 for n in ns_med])
        fdp_ev = np.mean([1.0 if n > 0 else 0.0 for n in ns_ev])
        print(f"  {M:>4d}{np.mean(ns_med):>9.2f}{fdp_med:>7.2f}"
              f"{np.mean(ns_ev):>11.2f}{fdp_ev:>9.2f}", flush=True)

    print()
    print("=" * 78)
    print(f"总耗时 {time.time()-t0:.0f}s")
    print("读法：")
    print("  * 全局零下正确行为：|S| ≈ 0，FDP ≈ 0")
    print("  * 若 e-value 列正常而中位数列失控，则结论是"
          "『聚合必须用 e-value，中位数变体无效』")


if __name__ == "__main__":
    main()
