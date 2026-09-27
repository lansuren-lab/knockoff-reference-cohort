"""
features.py -- 四种 ROI 特征定义，用于「诊断 + 修复」主线

论文旧版只用一种定义（meanconn），并把它造成的退化当成数据集的固有性质。
实测（150 被试、同 44 ROI）：

    定义        effective rank   PC1 占比   lambda_min
    meanconn         2.26         66.0%      0.0251
    gsr              7.75         28.9%      0.0317
    pcresid         16.54         15.3%      0.0702
    partial         24.88          7.7%      0.0764

秩差 11 倍 —— 退化是【特征定义的算术后果】，不是功能连接的固有性质。

原因：meanconn 是对相关矩阵取行均值，而行均值必然被第一主成分主导
（实测每个特征在 PC1 上的中位 R^2 = 0.69）。ABIDE 走的是
cpac/filt_noglobal 分支，**没有做全脑信号回归**，而全脑信号正是第一主成分
（实测其 PC1 与全脑均值信号相关 -0.83）。

【坏通道约定】与 build_features.py 一致：常量通道产生 NaN 相关，
绝不能当 0（那等于伪造"该脑区和谁都不连接"），必须整行整列标 NaN，
让 clean_features.py 的 ">20 名受试者缺失" 规则能生效。
"""
from __future__ import annotations

import numpy as np

N_PC_REMOVE = 5         # pcresid 去掉的主成分个数


def _corr(ts):
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(ts, rowvar=False)
    C = 0.5 * (C + C.T)
    return C


def _rowmean_z(C, n_valid):
    """相关矩阵 -> Fisher-z 行均值（对角置 0，除以 n_valid-1）。

    【这里修正了 build_features.py 的一处内部不一致】旧代码是

        "meanconn": np.nanmean(Z, axis=1) if len(bad) else Z.sum(axis=1)/(R-1)

    两个分支的除数不同：
        * 无坏通道时 -> sum/(R-1)                = sum/115   （正确）
        * 有坏通道时 -> nanmean = sum/(非NaN个数)
                        而对好的 ROI 该行有 R-n_bad 个非 NaN，
                        其中包含被置 0 的对角元 -> sum/(R-n_bad)（错误）

    分母里把"自己"算进去，却在分子里把自相关置 0 排除掉，是不自洽的：
    该特征的定义是"与其余脑区的平均 Fisher-z 连接"，除数应当是其余脑区的
    个数 n_valid-1。论文 Methods 写的也是 1/(p-1)。

    影响：仅对**无坏通道**的受试者（885 中的 817 个）产生 116/115 ≈ 0.87%
    的逐受试者缩放；由于缩放因子随受试者变化，它确实会改变相关系数，因此
    论文中所有基于 meanconn 的数字都会随之小幅变动（已在 build_features_v2
    的对照中量化）。我们重建全部结果，所以这里采用自洽的约定。
    """
    C = np.clip(C, -0.999999, 0.999999)
    Z = np.arctanh(C)
    np.fill_diagonal(Z, 0.0)
    return Z.sum(axis=1) / max(n_valid - 1, 1)


def _gsr(ts):
    """全脑信号回归：从每个 ROI 时间序列中回归掉全脑均值信号。

    这是影像里的标准步骤，ABIDE 的 filt_noglobal 分支恰恰跳过了它。
    """
    g = ts.mean(axis=1)
    if not np.all(np.isfinite(g)) or g.std() < 1e-12:
        return ts
    beta = np.linalg.lstsq(g[:, None], ts, rcond=None)[0]
    return ts - g[:, None] @ beta


def _pc_residual(ts, k=N_PC_REMOVE):
    """去掉前 k 个主成分后的残差时间序列。"""
    k = min(k, min(ts.shape) - 1)
    if k <= 0:
        return ts
    U, S, Vt = np.linalg.svd(ts - ts.mean(0), full_matrices=False)
    recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
    return ts - recon


def _partial_rowmean(ts, n_valid, regularize=True):
    """偏相关（精度矩阵）行均值。

    偏相关 k 与 j 衡量的是【条件】关联——去掉了其余所有 ROI 的影响。
    这与 knockoff 检验的量在概念上一致，而 meanconn 是边际量。

    【为什么必须正则化】偏相关需要相关矩阵的逆。当时间点数 T <= R 时相关
    矩阵秩亏（rank <= T-1 < R），逆不可估。这在本数据上是真实存在的：
    NIAK 管线有 145/885 名受试者 T < 116（CPAC 只有 23 名），最短 T=41。
    对秩亏矩阵用 pinv 得到的"偏相关"是数值噪声，实测会把 effective rank
    压到 2.98 —— 比它想修复的 meanconn（3.23）还差。

    因此默认走 Ledoit-Wolf 收缩估计精度矩阵，对任意 T 都有定义。这是
    标准的正则化精度估计，代价是偏相关向 0 收缩（这会抬高 effective rank，
    属于保守方向）。
    """
    C = _corr(ts)
    if regularize:
        try:
            from sklearn.covariance import LedoitWolf
            S = LedoitWolf().fit(ts).covariance_
            P = np.linalg.pinv(S, rcond=1e-12)
        except Exception:
            P = np.linalg.pinv(C, rcond=1e-10)
    else:
        try:
            P = np.linalg.pinv(C, rcond=1e-10)
        except np.linalg.LinAlgError:
            return np.full(ts.shape[1], np.nan)
    d = np.sqrt(np.clip(np.abs(np.diag(P)), 1e-300, None))
    PC = -P / np.outer(d, d)
    np.fill_diagonal(PC, 0.0)
    return _rowmean_z(PC, n_valid)


def all_features(ts, definitions=("meanconn", "gsr", "pcresid", "partial")):
    """ts: T x R 原始 ROI 时间序列。

    返回 (feats, info)：
        feats  dict  定义名 -> R 维向量（坏通道处为 NaN）
        info   dict  n_tr / n_bad_channel / bad_channels / n_valid_roi

    R 维输出（而非有效通道数）是刻意的：列顺序必须与 AAL116 名称表一一
    对应，否则 clean_features.py 按名字匹配列时会静默错配。
    """
    T, R = ts.shape
    # 坏通道 = 非常量或非有限。ABIDE II 的 DCAN/CIFTI 输出里确实存在
    # 非有限的 parcel 值，build_features_abide2.py:60 已显式处理这一点
    # （`~np.isfinite(sd) | (sd < 1e-8)`）；这里必须同样处理，否则 NaN 会
    # 一路传到 GSR 的 lstsq 并抛出 "SVD did not converge"。
    sd = ts.std(axis=0, ddof=1)
    bad = np.where(~np.isfinite(sd) | (sd < 1e-8)
                   | ~np.isfinite(ts).all(axis=0))[0]
    valid = np.setdiff1d(np.arange(R), bad)

    info = {"n_tr": int(T), "n_roi": int(R), "n_bad_channel": int(len(bad)),
            "bad_channels": bad.tolist(), "n_valid_roi": int(len(valid)),
            # T <= R 时无正则化的偏相关不可估；记录下来供论文报告
            "tr_le_p": bool(T <= len(valid))}
    feats = {}
    if len(valid) < 3:
        for name in definitions:
            feats[name] = np.full(R, np.nan)
        return feats, info

    Tv = ts[:, valid]
    nv = len(valid)

    def scatter(vec_v):
        out = np.full(R, np.nan)
        out[valid] = vec_v
        return out

    if "meanconn" in definitions:
        feats["meanconn"] = scatter(_rowmean_z(_corr(Tv), nv))
    if "gsr" in definitions:
        feats["gsr"] = scatter(_rowmean_z(_corr(_gsr(Tv)), nv))
    if "pcresid" in definitions:
        feats["pcresid"] = scatter(_rowmean_z(_corr(_pc_residual(Tv)), nv))
    if "partial" in definitions:
        feats["partial"] = scatter(_partial_rowmean(Tv, nv))
    return feats, info
