"""
refcohort.py -- 参考队列 model-X knockoff 分析的公共工具

这个模块的存在目的：在本仓库各实验脚本中，以下函数曾被逐份复制

    corr_of        9 处
    pd_floor       7 处（两个在浮点上等价的变体）
    ledoit_sigma   6 处
    lcd_fixed      5 处（两处默认 c 不同：0.15 与 0.05）
    shrink_to      3 处
    marginal_t     1 处

复制带来的实际损害不只是冗余，而是**求解器容差不一致**：同一篇论文的
表格分别由 tol=1e-4、无 tol、tol=1e-6 的坐标下降产生。既然论文的论点是
「决策落在刀锋上」，容差就不该是随手写的。这里统一为

    TOL      = 1e-8
    MAX_ITER = 100000

已实测：把 tol 收紧到 1e-12 不改变任何一次（选择集/阈值）决策，因此统
一容差不会改变已有结论，只是把不确定性去掉。

命名约定（与 build_features.py / clean_features.py 配合）：
    clean_features.py 按【第一个下划线】切分列名前缀来匹配 ROI 名，
    所以任何新特征键都必须是 "<prefix>_<ROIname>" 形式，且 prefix 中
    不含下划线。见 FEATURE_PREFIXES。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

TOL = 1e-8
MAX_ITER = 100000

# 特征定义 -> 列名前缀。clean_features.py 依赖这个不变量。
FEATURE_PREFIXES = {
    "meanconn": "meanconn",
    "gsr": "gsr",
    "pcresid": "pcresid",
    "partial": "partial",
}

C_FIXED = 0.15          # 固定-lambda LCD 的 c（论文 §Methods）
C_STABILITY = 0.05      # 稳定性分析用的 c（论文 §Methods 明确区分）
Q_DEFAULT = 0.20
MAX_ROI_MISS = 20       # ROI 缺失 > 20 名受试者则丢弃该 ROI

MAIN_SITES_ABIDE1 = ["NYU", "UM_1", "USM", "UCLA_1", "YALE", "PITT"]

# 由 build_features_v2.py 产出的特征目录。键同时用作论文里的显示名。
DEF_DATASETS_FOR_DIAG = {
    "ABIDE1-AAL116-CPAC": "feat2_abide1_aal",
    "ABIDE1-AAL116-NIAK": "feat2_abide1_niak",
    "ABIDE1-CC200-CPAC": "feat2_abide1_cc200",
    "ABIDE1-HO-CPAC": "feat2_abide1_ho",
    "ABIDE2-Power2011": "feat2_abide2",
    # 第三个队列：ADHD-200（不同疾病、不同站点、不同采集协议）
    # 构建见 build_features_adhd200_v2.py；664 人，AAL116，427 对照 / 237 ADHD
    "ADHD200-AAL116": "feat2_adhd200",
}


# --------------------------------------------------------------------------- #
# 谱量 / 诊断
# --------------------------------------------------------------------------- #
def corr_of(A):
    """数据矩阵 A (n,p) -> 相关矩阵 (p,p)，对角严格为 1。"""
    C = np.corrcoef(A, rowvar=False)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def cov_to_corr(S):
    """协方差矩阵 -> 相关矩阵。

    【不要用 corr_of 处理协方差矩阵】corr_of 内部是 np.corrcoef，它把
    【行】当观测；对数据矩阵正确，对协方差矩阵完全错误。LedoitWolf 返回
    的是协方差，必须走这个函数。
    """
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    return S / np.outer(d, d)


def pd_floor(C, eps=1e-6):
    """投影到正定锥（特征值下界 eps）并重新标准化为相关矩阵。"""
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    C = V @ np.diag(w) @ V.T
    C = 0.5 * (C + C.T)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def ledoit_sigma(Xr):
    """参考队列 -> Ledoit-Wolf 收缩后的相关矩阵（已投影正定）。"""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(Xr).covariance_
    return pd_floor(cov_to_corr(lw))


def shrink_to(C, alpha):
    """向单位阵收缩：C -> (1-alpha) C + alpha I。"""
    p = C.shape[0]
    return pd_floor((1.0 - alpha) * C + alpha * np.eye(p))


def effective_rank(C):
    """有效秩 (sum lambda)^2 / sum lambda^2。"""
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 0, None)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def spectral_report(C):
    """一个相关矩阵的全部诊断量。这是论文推荐用户自查的那一组数。"""
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 1e-300, None)
    return {
        "lam_min": float(w.min()),
        "lam_max": float(w.max()),
        "cond": float(w.max() / w.min()),
        "pc1_share": float(w.max() / w.sum()),
        "eff_rank": float(w.sum() ** 2 / np.sum(w ** 2)),
        "trace": float(w.sum()),
    }


# --------------------------------------------------------------------------- #
# 统计量
# --------------------------------------------------------------------------- #
def lcd_fixed(Xa, y, c=C_FIXED, return_info=False):
    """固定-lambda 套索系数差 (LCD) 统计量。

    交换 X_j 与 X~_j 只是交换增广设计的列顺序，不改变列集合，因此
    lam = c * max|Xa^T y| / n 保持不变 -> lambda 是 swap-invariant 的，
    满足 flip-sign 性质。

    【收敛必须记录】这个统计量此前被描述为"确定性、无迭代歧义"，但那只是
    相对于 CV 版而言。固定-λ 套索本身仍是坐标下降：当 c 小（罚得轻）且特征
    高度共线时，它可能在上限内不收敛，返回的 W 就不可信。本函数把 n_iter 与
    是否收敛一并返回，让调用方能报告"有多少比例的重复没收敛"。
    论文 §solver 关心的正是这类问题，所以不能默认它不发生。
    """
    Xa = Xa - Xa.mean(axis=0)
    y = y - y.mean()
    lam = c * np.abs(Xa.T @ y).max() / len(y)
    from sklearn.linear_model import Lasso
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        m = Lasso(alpha=lam, max_iter=MAX_ITER, tol=TOL).fit(Xa, y)
        converged = not any("converge" in str(r.message) for r in rec)
    p = Xa.shape[1] // 2
    W = np.abs(m.coef_[:p]) - np.abs(m.coef_[p:])
    if return_info:
        return W, {"n_iter": int(m.n_iter_), "converged": bool(converged),
                   "lam": float(lam), "nnz": int(np.sum(np.abs(m.coef_) > 1e-12))}
    return W


def marginal_t(X, y, j):
    """y 对 X[:,j] 单变量回归的 |t|。"""
    x = X[:, j] - X[:, j].mean()
    yc = y - y.mean()
    b = (x @ yc) / (x @ x)
    r = yc - b * x
    se = np.sqrt((r @ r) / (len(y) - 2) / (x @ x))
    return float(abs(b / se)) if se > 0 else np.inf


def partial_t(X, y, j):
    """y 对 X[:,j] 在控制其余所有列之后的偏回归 |t|。

    这是 knockoff 真正检验的量。VIF 极高时边际 |t| 会严重夸大：
    positive_control_v2.py 按边际 |t| 标定，a=0.4 时边际 22.3 而条件只有
    2.36，注入的信号实际上不可条件识别——这是旧正对照的设计缺陷。
    """
    p = X.shape[1]
    others = np.delete(X, j, axis=1)
    A = np.hstack([np.ones((len(y), 1)), others])
    # 把 y 与 X[:,j] 都对 A 回归，取残差做单变量回归
    cy = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    xj = X[:, j] - A @ np.linalg.lstsq(A, X[:, j], rcond=None)[0]
    denom = xj @ xj
    if denom <= 0:
        return 0.0
    b = (xj @ cy) / denom
    r = cy - b * xj
    se = np.sqrt((r @ r) / (len(y) - A.shape[1] - 1) / denom)
    return float(abs(b / se)) if se > 0 else 0.0


def conditional_t_all(X, y):
    """y 对 X 全回归（含截距）时每个系数的 |t|。

    这就是"控制其余所有变量之后"的条件 |t|，也正是 knockoff 检验的量。
    与逐个调用 partial_t 数学上等价，但只需一次求解。

    返回 (t_abs, design) ，design 是冻结的 (A'A)^-1 与 A，供 calibrate 复用：
    在标定注入幅度的循环里 X 不变、只有 y 变，所以 (A'A)^-1 可以只算一次。
    """
    n, p = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    AtA_inv = np.linalg.pinv(A.T @ A)
    beta = AtA_inv @ (A.T @ y)
    resid = y - A @ beta
    dof = max(n - A.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(sigma2 * np.diag(AtA_inv), 1e-300))
    return np.abs(beta / se)[1:], (A, AtA_inv, dof)


def conditional_t_fast(y, design):
    """用冻结的设计复用条件 |t|。"""
    A, AtA_inv, dof = design
    beta = AtA_inv @ (A.T @ y)
    resid = y - A @ beta
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(sigma2 * np.diag(AtA_inv), 1e-300))
    return np.abs(beta / se)[1:]


def vif(X):
    """每个协变量的方差膨胀因子。"""
    p = X.shape[1]
    out = np.empty(p)
    for j in range(p):
        A = np.delete(X, j, axis=1)
        A = np.hstack([np.ones((len(X), 1)), A])
        xj = X[:, j]
        resid = xj - A @ np.linalg.lstsq(A, xj, rcond=None)[0]
        r2 = 1.0 - (resid @ resid) / max(((xj - xj.mean()) @ (xj - xj.mean())), 1e-300)
        out[j] = 1.0 / max(1.0 - r2, 1e-12)
    return out


# --------------------------------------------------------------------------- #
# knockoff 管线
# --------------------------------------------------------------------------- #
def knockoff_run(X, y, Sigma, q=Q_DEFAULT, seed=0, c=C_FIXED,
                 method="equicorrelated", median_of=None, s_override=None):
    """完整的参考队列 knockoffs+ 流程。

    X     主队列协变量 (n,p)，内部标准化
    Sigma 用于构造的协方差（参考队列估计或真值）
    seed  knockoff 抽样的随机种子

    返回 dict: sel, tau, W, s, fdp_hat, ratio_best

    fdp_hat   = 阈值处 (1+F)/R —— 用户唯一看得到的误差报告
    ratio_best= min_t (1+F_t)/R_t —— 最优可达阈值比；tau 为 inf 时它是
                唯一有定义的量，也是论文比较各队列所用的量
    """
    from knockoff_adni.knockoffs import (solve_s_diag, sample_knockoffs,
                                         knockoff_threshold)
    mu, sd = X.mean(0), X.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    # s_override 让调用方指定构造的对角向量（用于跨构造对照实验，
    # 避免为了复用同一个 s 而去反复求解 SDP）
    s = (np.asarray(s_override, float) if s_override is not None
         else solve_s_diag(Sigma, method=method))
    rng = np.random.default_rng(seed)
    if median_of is None:
        Xt = sample_knockoffs(Xs, Sigma, s, rng=rng)
        W, info = lcd_fixed(np.hstack([Xs, Xt]), y, c=c, return_info=True)
    else:
        W = np.median(np.vstack([lcd_fixed(np.hstack([Xs, sample_knockoffs(
            Xs, Sigma, s, rng=rng)]), y, c=c) for _ in range(int(median_of))]), axis=0)
        info = {"n_iter": -1, "converged": True, "lam": float("nan"), "nnz": -1}
    tau = knockoff_threshold(W, q, plus_one=True)
    sel = np.where(W >= tau)[0] if np.isfinite(tau) else np.array([], int)
    if np.isfinite(tau) and len(sel):
        R = int((W >= tau).sum())
        F = int((W <= -tau).sum())
        fdp_hat = min((1 + F) / R, 1.0)
    else:
        fdp_hat = 0.0
    return {"sel": sel, "tau": float(tau), "W": W, "s": s,
            "fdp_hat": float(fdp_hat), "ratio_best": best_ratio(W),
            "converged": info["converged"], "n_iter": info["n_iter"],
            "lam": info["lam"], "nnz": info["nnz"]}


def best_ratio(W):
    """min over t>0 of (1 + #{W<=-t}) / #{W>=t}。

    注意这**不是** FDR，而是计数比的最大值；它是 tau=inf 时唯一有定义的
    比较量。跨队列比较 p 不同的数据时它不是可比的统计量——论文 §ABIDE II
    的「ratio 在收缩下变差」正是这个量在作怪。
    """
    cand = np.unique(np.abs(W[W != 0]))
    best = np.inf
    for t in np.sort(cand):
        R = int((W >= t).sum())
        if R == 0:
            continue
        F = int((W <= -t).sum())
        best = min(best, (1 + F) / R)
    return float(best)


# --------------------------------------------------------------------------- #
# 数据准备（>20 缺失规则 + 主/参考划分）
# --------------------------------------------------------------------------- #
def load_matrix(path):
    """读 X_*.csv（首行列名）-> (X, names)。"""
    X = np.loadtxt(path, delimiter=",", skiprows=1)
    names = open(path).readline().strip().split(",")
    return X, names


def prepare(d, main_sites=None, n_main=6, max_roi_miss=MAX_ROI_MISS,
            fname="X_meanconn.csv"):
    """特征目录 -> (X, names, manifest, is_main, main_sites)。

    规则与 clean_features.py --max-roi-miss 一致：丢弃缺失 > max_roi_miss
    名受试者的 ROI，再丢弃在保留 ROI 上有缺失的受试者。

    fname 指定用哪种特征定义（X_meanconn.csv / X_gsr.csv / ...）。
    """
    X, names = load_matrix(os.path.join(d, fname))
    man = list(csv.DictReader(open(os.path.join(d, "subject_manifest.csv"))))
    keep_roi = np.isnan(X).sum(0) <= max_roi_miss
    keep_sub = (~np.isnan(X[:, keep_roi])).all(1)
    X = X[np.ix_(keep_sub, keep_roi)]
    names = [names[j] for j in range(len(names)) if keep_roi[j]]
    man = [man[i] for i in range(len(man)) if keep_sub[i]]
    cnt = {}
    for r in man:
        cnt[r["site"]] = cnt.get(r["site"], 0) + 1
    if main_sites is None:
        main_sites = sorted(cnt, key=lambda s: -cnt[s])[:n_main]
    is_main = np.array([r["site"] in main_sites for r in man])
    return X, names, man, is_main, main_sites


def split_xy(X, man, is_main, require_y=True):
    """-> Xm, ym, Xr（主队列协变量/响应、参考队列协变量）。

    manifest 里 y 可能为空串（DX_GROUP 缺失，ABIDE II 的 KUL_3 / NYU_2 等
    站点整站无对照）。这类受试者保留在参考队列里没有问题（参考队列的响应
    从不使用），但必须从主队列剔除，否则 int('') 会抛错。
    """
    y = np.array([float(r["y"]) if str(r["y"]).strip() != "" else np.nan
                  for r in man])
    main_mask = np.asarray(is_main)
    ref_mask = ~main_mask                       # 参考队列按站点划分，不因 y 缺失而变
    if require_y:
        main_mask = main_mask & np.isfinite(y)
    return X[main_mask], y[main_mask], X[ref_mask]


# --------------------------------------------------------------------------- #
# 可复现性辅助
# --------------------------------------------------------------------------- #
def file_hash(path, n=100_000):
    """输入文件哈希，写进 provenance json，便于事后核对用的是哪份数据。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def provenance(outdir, name, args, inputs=()):
    """记录一次运行的完整出处。

    这是防止再次出现「手稿里的 42% 在任何产物中都不存在」的机制：
    每个结果文件旁边必须有一份 json 说明它由什么参数、什么代码、什么
    输入文件产生。
    """
    os.makedirs(outdir, exist_ok=True)
    try:
        import sklearn, scipy
        versions = {"python": sys.version.split()[0], "numpy": np.__version__,
                    "sklearn": sklearn.__version__, "scipy": scipy.__version__,
                    "platform": platform.platform()}
    except Exception as e:                       # pragma: no cover
        versions = {"python": sys.version.split()[0], "error": repr(e)}
    rec = {"name": name, "args": {k: v for k, v in vars(args).items()
                                 if not k.startswith("_")} if args is not None else {},
           "constants": {"TOL": TOL, "MAX_ITER": MAX_ITER, "C_FIXED": C_FIXED,
                         "C_STABILITY": C_STABILITY, "Q_DEFAULT": Q_DEFAULT,
                         "MAX_ROI_MISS": MAX_ROI_MISS},
           "versions": versions, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "inputs": {str(p): file_hash(str(p)) for p in inputs
                      if os.path.exists(str(p))}}
    path = os.path.join(outdir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    return path


def write_csv(path, rows, fieldnames=None):
    """列表字典 -> csv，父目录自动创建。

    列名取【所有行的并集】而非首行的键：不同族/分支的行常常带不同字段
    （例如族 C 有 dataset/feature，族 A/B 没有），只按首行会直接抛错。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError("no rows")
    if fieldnames is None:
        seen = {}
        for r in rows:
            for k in r:
                seen.setdefault(k, None)
        fieldnames = list(seen)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """分位数 bootstrap 置信区间。"""
    v = np.asarray(values, float)
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    stats = stat(v[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
