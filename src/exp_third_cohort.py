"""
exp_third_cohort.py -- 第三队列（ADHD-200）：机制是否可移植

前两队列都是 autism（ABIDE I/II）。ADHD-200 是不同疾病、不同站点、
不同采集协议，用来检验手稿里那条"可移植的是诊断量、不是常数"的论断。

产出 results_v3/third_cohort.csv：
  对每个队列、每个 n_ref，用 5 次随机切分报告 meanconn 的有效秩。
  n_ref 必须匹配 —— 样本相关矩阵在 n_ref/p 接近 1 时近乎必然近奇异，
  那是构造性的，不是数据的性质，不匹配就会得出错误结论。
"""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (corr_of, effective_rank, pd_floor, prepare,
                           provenance, split_xy, write_csv)

OUT = "results_v3"
os.makedirs(OUT, exist_ok=True)
COHORTS = (("ABIDE1-AAL116", "feat2_abide1_aal"),
           ("ABIDE2-Power2011", "feat2_abide2"),
           ("ADHD200-AAL116", "feat2_adhd200"))
NREFS = (80, 121, 200, 300)
S = 5

rows = []
for lbl, d in COHORTS:
    X, names, man, ism, ms = prepare(d, fname="X_meanconn.csv")
    Xm, ym, Xr = split_xy(X, man, ism)
    p = Xr.shape[1]
    print(f"\n{lbl}: n_ref={Xr.shape[0]}  p={p}  n_ref/p={Xr.shape[0]/p:.2f}")
    for n in NREFS:
        if Xr.shape[0] < n:
            continue
        ers = []
        for seed in range(S):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(Xr.shape[0])[:n]
            ers.append(effective_rank(pd_floor(corr_of(Xr[idx]))))
        rows.append({"cohort": lbl, "n_ref": n, "p": p, "draws": S,
                     "effrank_mean": float(np.mean(ers)),
                     "effrank_sd": float(np.std(ers)),
                     "n_ref_over_p": round(n / p, 3)})
        print(f"   n_ref={n:4d}  effrank={np.mean(ers):6.2f}±{np.std(ers):.2f}")

path = write_csv(os.path.join(OUT, "third_cohort.csv"), rows)
provenance(OUT, "exp_third_cohort", None)
print(f"\n-> {path}")


# --------------------------------------------------------------------------- #
def cohort_mismatch_with_noise_floor(n_null=12):
    """主/参考队列的不匹配度，以及【纯抽样噪声底】。

    不匹配度必须对照噪声底才有意义：把同一队列随机二分（两组同分布），
    测到的就是抽样噪声。观察值若只比噪声底高一点，那"不匹配"其实主要是
    估计噪声，不是真正的分布差异 —— 而两者的机制【不同】：
    分布差异让 Sigma_hat 有偏（反保守），估计噪声让它更噪（保守）。
    """
    out = []
    for lbl, d in COHORTS:
        X, names, man, ism, ms = prepare(d, fname="X_meanconn.csv")
        Xm, ym, Xr = split_xy(X, man, ism)
        nm, nr = Xm.shape[0], Xr.shape[0]
        p = X.shape[1]
        off = ~np.eye(p, dtype=bool)
        obs = float(np.sqrt(np.mean(
            (pd_floor(corr_of(Xm)) - pd_floor(corr_of(Xr)))[off] ** 2)))
        Xall = np.vstack([Xm, Xr])
        N = Xall.shape[0]
        null = []
        for seed in range(n_null):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(N)
            a = pd_floor(corr_of(Xall[idx[:nm]]))
            b = pd_floor(corr_of(Xall[idx[nm:nm + nr]]))
            null.append(float(np.sqrt(np.mean((a - b)[off] ** 2))))
        nb = float(np.mean(null))
        out.append({"cohort": lbl, "n_main": nm, "n_ref": nr, "p": p,
                    "mismatch": obs, "noise_floor": nb, "draws": n_null,
                    "excess": obs - nb, "ratio": obs / nb if nb else np.nan})
        print(f"  {lbl:20s} 观察 {obs:.4f}  噪声底 {nb:.4f}  "
              f"纯超出 {obs-nb:.4f}  ({obs/nb:.1f}x)")
    return write_csv(os.path.join(OUT, "cohort_mismatch.csv"), out)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "mismatch":
    print("=== 主/参考队列不匹配度与噪声底 ===")
    cohort_mismatch_with_noise_floor()
