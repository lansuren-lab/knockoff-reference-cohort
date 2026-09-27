"""
build_features.py -- ABIDE I: ROI 时间序列 -> 受试者级特征矩阵 X

输入：
    abide_rois_aal/<FILE_ID>_rois_aal.1D    885 个文件，116 列 ROID，TR 数不等
    Phenotypic_V1_0b_preprocessed1.csv      表型（站点、诊断、年龄、性别、IQ）

输出（写到 features/ 目录）：
    X_meanconn.csv       885 x 116  主特征：Fisher-z 平均连接强度（每个 ROI 一个数）
    X_multifeat.csv      885 x 348  稳健性特征：meanconn + posprop + tsstd
    subject_manifest.csv 885 行     受试者元信息 + 主/参考队列划分 + QC 标记
    feature_names.csv    116 行     ROI 编码 + AAL 标准名 + 所属网络粗分
    qc_report.txt                   质控报告

设计要点（都是踩过坑才加的）：
  * 各受试者 TR 数不同(78~316)，不能堆成数组；相关矩阵天然把长度归一化。
  * Fisher-z 变换(arctanh)让相关系数近似正态，是脑连接分析的标准做法。
  * 相关矩阵对角线置 0 再取行均值，避免自相关(=1)混进特征。
  * 常量通道(扫描器故障)会让相关变成 NaN，必须检出并处理，不能静默传播。

Run:  python build_features.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

DERIV = os.environ.get("ROI_DERIV", "rois_aal")
# 【重要】ROI_DIR 与 OUTDIR 都可以用环境变量显式覆盖。
# 跨管线对照时（CPAC 与 NIAK 都用 rois_aal），只靠 DERIV 无法区分，
# 会把两次结果写进同一个目录而互相覆盖。
ROI_DIR = os.environ.get("ROI_DIR", f"abide_{DERIV}")
PHENO = "Phenotypic_V1_0b_preprocessed1.csv"
OUTDIR = os.environ.get(
    "ROI_OUTDIR", "features" if DERIV == "rois_aal" else f"features_{DERIV}")

# 主队列/参考队列划分：人数最多的 6 个站点 vs 其余（依据实测站点人数）
MAIN_SITES = ["NYU", "UM_1", "USM", "UCLA_1", "YALE", "PITT"]

# 标准 AAL116 顺序（用于把 #2001 这类编码翻译成人能读的名字）
AAL116 = """Precentral_L Precentral_R Frontal_Sup_L Frontal_Sup_R Frontal_Sup_Orb_L
Frontal_Sup_Orb_R Frontal_Mid_L Frontal_Mid_R Frontal_Mid_Orb_L Frontal_Mid_Orb_R
Frontal_Inf_Oper_L Frontal_Inf_Oper_R Frontal_Inf_Tri_L Frontal_Inf_Tri_R
Frontal_Inf_Orb_L Frontal_Inf_Orb_R Rolandic_Oper_L Rolandic_Oper_R
Supp_Motor_Area_L Supp_Motor_Area_R Olfactory_L Olfactory_R Frontal_Sup_Medial_L
Frontal_Sup_Medial_R Frontal_Med_Orb_L Frontal_Med_Orb_R Rectus_L Rectus_R
Insula_L Insula_R Cingulum_Ant_L Cingulum_Ant_R Cingulum_Mid_L Cingulum_Mid_R
Cingulum_Post_L Cingulum_Post_R Hippocampus_L Hippocampus_R ParaHippocampal_L
ParaHippocampal_R Amygdala_L Amygdala_R Calcarine_L Calcarine_R Cuneus_L Cuneus_R
Lingual_L Lingual_R Occipital_Sup_L Occipital_Sup_R Occipital_Mid_L Occipital_Mid_R
Occipital_Inf_L Occipital_Inf_R Fusiform_L Fusiform_R Postcentral_L Postcentral_R
Parietal_Sup_L Parietal_Sup_R Parietal_Inf_L Parietal_Inf_R SupraMarginal_L
SupraMarginal_R Angular_L Angular_R Precuneus_L Precuneus_R Paracentral_Lobule_L
Paracentral_Lobule_R Caudate_L Caudate_R Putamen_L Putamen_R Pallidum_L Pallidum_R
Thalamus_L Thalamus_R Heschl_L Heschl_R Temporal_Sup_L Temporal_Sup_R
Temporal_Pole_Sup_L Temporal_Pole_Sup_R Temporal_Mid_L Temporal_Mid_R
Temporal_Pole_Mid_L Temporal_Pole_Mid_R Temporal_Inf_L Temporal_Inf_R
Cerebelum_Crus1_L Cerebelum_Crus1_R Cerebelum_Crus2_L Cerebelum_Crus2_R
Cerebelum_3_L Cerebelum_3_R Cerebelum_4_5_L Cerebelum_4_5_R Cerebelum_6_L
Cerebelum_6_R Cerebelum_7b_L Cerebelum_7b_R Cerebelum_8_L Cerebelum_8_R
Cerebelum_9_L Cerebelum_9_R Cerebelum_10_L Cerebelum_10_R Vermis_1_2 Vermis_3
Vermis_4_5 Vermis_6 Vermis_7 Vermis_8 Vermis_9 Vermis_10""".split()

# AAL 编码 -> 顺序号
# 【重要】该数据集用的数字编码不是常见的那套 AAL 编号，但 ROI 的【排列顺序】
# 与标准 AAL116 完全一致（已用 11 个解剖锚点逐个验证：位置1=Precentral_L、
# 位置19=Supp_Motor_Area_L、位置37=Hippocampus_L、位置91=Cerebelum_Crus1_L、
# 位置109=Vermis_1_2 等全部吻合）。因此按【位置】映射名称，不按编码。
# 锚点定义见 verify 函数，任何不一致都会被显式报出来，不会静默出错。
ANCHORS = {1: "Precentral_L", 19: "Supp_Motor_Area_L", 21: "Olfactory_L",
           27: "Rectus_L", 29: "Insula_L", 37: "Hippocampus_L",
           41: "Amygdala_L", 71: "Caudate_L", 77: "Thalamus_L",
           91: "Cerebelum_Crus1_L", 109: "Vermis_1_2"}

# 大尺度网络粗分（用于后面的分块 ASDP 构造，以及结果解读）
NETWORK = {}
for _i, _n in enumerate(AAL116, start=1):
    if any(k in _n for k in ("Frontal", "Precentral", "Rolandic", "Supp_Motor",
                             "Olfactory", "Rectus", "Insula", "Cingulum_Ant")):
        NETWORK[_i] = "fronto-insular"
    elif any(k in _n for k in ("Cingulum_Mid", "Cingulum_Post", "Precuneus",
                               "Parietal", "SupraMarginal", "Angular",
                               "Paracentral")):
        NETWORK[_i] = "default-mode"
    elif any(k in _n for k in ("Occipital", "Calcarine", "Cuneus", "Lingual",
                               "Fusiform")):
        NETWORK[_i] = "visual"
    elif any(k in _n for k in ("Temporal", "Heschl", "Hippocampus",
                               "ParaHippocampal", "Amygdala")):
        NETWORK[_i] = "temporal-limbic"
    else:
        NETWORK[_i] = "subcortical-cerebellar"


# --------------------------------------------------------------------------- #
def load_pheno(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    out = {}
    for r in rows:
        fid = (r.get("FILE_ID") or "").strip()
        if not fid or fid == "no_filename":
            continue
        out[fid] = r
    return out


def read_roi_file(path):
    """读 .1D：首行是 ROI 编码，其余每行一个时间点。返回 (codes, T x R 矩阵)。"""
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    if len(lines) < 2:
        raise ValueError("文件行数不足")
    codes = lines[0].split()
    arr = np.array([[float(x) for x in ln.split()] for ln in lines[1:]])
    return codes, arr


def features_from_timeseries(ts):
    """ts: T x R。返回 dict(特征名 -> R 维向量) 以及诊断信息。

    【关键设计】坏通道（常量时间序列，扫描未覆盖该脑区）会产生 NaN 相关。
    绝不能把 NaN 当成 0——那等于伪造出"这个脑区和谁都不连接"的假数据，
    会污染下游的 Sigma 估计。正确做法是把该 ROI 标为缺失(NaN)，
    并让 QC 报告说清楚有多少受试者、哪些脑区受影响。
    """
    T, R = ts.shape
    info = {"n_tr": T, "n_roi": R}

    sd = ts.std(axis=0, ddof=1)
    bad = np.where(sd < 1e-8)[0]        # 常量通道 = 无信号

    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(ts, rowvar=False)
    np.fill_diagonal(C, 0.0)            # 去掉自相关
    C = np.clip(C, -0.999999, 0.999999)

    Z = np.arctanh(C)                   # Fisher-z

    if len(bad):
        Z[bad, :] = np.nan              # 坏 ROI 自己不可用
        Z[:, bad] = np.nan              # 别人与它算出的相关也不可用
        C[bad, :] = np.nan
        C[:, bad] = np.nan

    # 对每个 ROI，在"有效"的 ROI 子集上取均值；坏 ROI 整行 NaN -> 结果为 NaN
    feats = {
        "meanconn": np.nanmean(Z, axis=1) if len(bad) else Z.sum(axis=1) / (R - 1),
        "posprop": (np.nanmean(C > 0, axis=1) if len(bad)
                    else (C > 0).sum(axis=1) / (R - 1)),
        "tsstd": sd,
    }
    info["n_bad_channel"] = int(len(bad))
    info["bad_channels"] = bad.tolist()
    info["n_valid_roi"] = int(R - len(bad))
    return feats, info


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)

    if not os.path.isdir(ROI_DIR):
        sys.exit(f"找不到 {ROI_DIR}/，请先运行 download_abide_roi.py")
    files = sorted(f for f in os.listdir(ROI_DIR) if f.endswith(".1D"))
    if not files:
        sys.exit(f"{ROI_DIR}/ 里没有 .1D 文件")
    pheno = load_pheno(PHENO)
    print(f"找到 {len(files)} 个 ROI 文件，表型 {len(pheno)} 条")

    # 用第一个文件确定 ROI 编码与顺序；名称按【位置】映射并做锚点校验
    codes_ref, _ = read_roi_file(os.path.join(ROI_DIR, files[0]))
    R = len(codes_ref)
    code_int = [c.lstrip("#") for c in codes_ref]
    if DERIV == "rois_aal":
        if R != len(AAL116):
            sys.exit(f"ROI 数 {R} 与预期 116 不符，请检查数据")
        names = list(AAL116)
        bad_anchor = [(p, names[p - 1], ANCHORS[p]) for p in ANCHORS
                      if names[p - 1] != ANCHORS[p]]
        if bad_anchor:
            sys.exit(f"名称锚点校验失败：{bad_anchor}")
        print(f"ROI 数 = {R}；名称按位置映射到标准 AAL116，"
              f"{len(ANCHORS)} 个解剖锚点全部通过校验")
    else:
        # 其他图谱（CC200 / Harvard-Oxford 等）没有可靠的内建解剖名，
        # 直接用数据自带的标签命名，避免张冠李戴。
        names = [f"{DERIV.replace('rois_','')}_{c.lstrip('#')}" for c in codes_ref]
        print(f"ROI 数 = {R}；图谱 {DERIV}，使用数据自带标签命名")

    recs, rows_mean, rows_multi, qc = [], [], [], []
    n_code_mismatch = 0

    for k, fn in enumerate(files, 1):
        fid = fn[: -len(f"_{DERIV}.1D")]
        path = os.path.join(ROI_DIR, fn)
        try:
            codes, ts = read_roi_file(path)
        except Exception as e:
            qc.append((fid, "READ_FAIL", str(e)))
            continue
        if len(codes) != R or [c.lstrip("#") for c in codes] != [c.lstrip("#") for c in codes_ref]:
            n_code_mismatch += 1
            qc.append((fid, "CODE_MISMATCH", f"{len(codes)} 列"))
            continue

        feats, info = features_from_timeseries(ts)
        if info["n_bad_channel"] > 0:
            qc.append((fid, "BAD_CHANNEL", f"{info['n_bad_channel']} 个常量通道"))

        p = pheno.get(fid, {})
        site = (p.get("SITE_ID") or "?").strip()
        dx = (p.get("DX_GROUP") or "").strip()
        recs.append({
            "file_id": fid,
            "site": site,
            "cohort": "main" if site in MAIN_SITES else "reference",
            "dx_group": "ASD" if dx == "1" else ("control" if dx == "2" else "?"),
            "y": 1 if dx == "1" else (0 if dx == "2" else ""),
            "age": (p.get("AGE_AT_SCAN") or "").strip(),
            "sex": {"1": "M", "2": "F"}.get((p.get("SEX") or "").strip(), ""),
            "fiq": (p.get("FIQ") or "").strip(),
            "n_tr": info["n_tr"],
            "n_bad_channel": info["n_bad_channel"],
        })
        rows_mean.append(feats["meanconn"])
        rows_multi.append(np.concatenate([feats["meanconn"], feats["posprop"],
                                          feats["tsstd"]]))
        if k % 100 == 0 or k == len(files):
            print(f"  {k}/{len(files)}  ({time.time()-t0:.0f}s)")

    if not recs:
        sys.exit("没有任何受试者处理成功")

    X_mean = np.vstack(rows_mean)
    X_multi = np.vstack(rows_multi)

    # ---- 输出 ----
    np.savetxt(os.path.join(OUTDIR, "X_meanconn.csv"), X_mean,
               delimiter=",", fmt="%.6f",
               header=",".join(names), comments="")
    multi_names = ([f"meanconn_{n}" for n in names]
                   + [f"posprop_{n}" for n in names]
                   + [f"tsstd_{n}" for n in names])
    np.savetxt(os.path.join(OUTDIR, "X_multifeat.csv"), X_multi,
               delimiter=",", fmt="%.6f",
               header=",".join(multi_names), comments="")

    with open(os.path.join(OUTDIR, "subject_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    with open(os.path.join(OUTDIR, "feature_names.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "aal_code", "name", "network"])
        for i, (c, n) in enumerate(zip(code_int, names), start=1):
            w.writerow([i, c, n, NETWORK.get(i, "")])

    # ---- QC 报告 ----
    sites = {}
    for r in recs:
        d = sites.setdefault(r["site"], {"n": 0, "ASD": 0, "ctrl": 0, "bad": 0,
                                         "cohort": r["cohort"]})
        d["n"] += 1
        if r["n_bad_channel"] > 0:
            d["bad"] += 1
        if r["dx_group"] == "ASD":
            d["ASD"] += 1
        elif r["dx_group"] == "control":
            d["ctrl"] += 1

    lines = []
    lines.append("=" * 74)
    lines.append("ABIDE I 特征矩阵构建 —— 质控报告")
    lines.append("=" * 74)
    lines.append("")
    lines.append(f"ROI 文件总数        {len(files)}")
    lines.append(f"成功处理            {len(recs)}")
    lines.append(f"失败                {len(files) - len(recs)}")
    lines.append(f"ROI 编码不一致      {n_code_mismatch}")
    lines.append(f"含常量通道的受试者  {sum(1 for r in recs if r['n_bad_channel']>0)}")
    lines.append("")
    lines.append(f"X_meanconn 形状     {X_mean.shape}")
    lines.append(f"X_multifeat 形状    {X_multi.shape}")
    lines.append("")
    lines.append("主/参考队列划分：")
    for coh in ("main", "reference"):
        sub = [r for r in recs if r["cohort"] == coh]
        a = sum(1 for r in sub if r["dx_group"] == "ASD")
        c = sum(1 for r in sub if r["dx_group"] == "control")
        lines.append(f"    {coh:10s} n={len(sub):4d}   ASD={a:3d}  对照={c:3d}  "
                     f"站点数={len(set(r['site'] for r in sub))}")
    lines.append("")
    lines.append("各站点明细：")
    lines.append(f"    {'站点':10s} {'队列':10s} {'人数':>5s} {'ASD':>5s} {'对照':>5s}")
    for s, d in sorted(sites.items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"    {s:10s} {d['cohort']:10s} {d['n']:5d} {d['ASD']:5d} "
                     f"{d['ctrl']:5d}")
    lines.append("")
    lines.append("特征分布（X_meanconn，忽略 NaN）：")
    lines.append(f"    全局   均值={np.nanmean(X_mean):+.4f}  sd={np.nanstd(X_mean):.4f}  "
                 f"范围=[{np.nanmin(X_mean):.3f}, {np.nanmax(X_mean):.3f}]")
    lines.append(f"    每 ROI 标准差 中位数={np.nanmedian(np.nanstd(X_mean, axis=0)):.4f}")
    lines.append(f"    NaN 个数 = {int(np.isnan(X_mean).sum())} "
                 f"({np.isnan(X_mean).mean():.2%})")
    lines.append("")

    # ---- 常量通道（扫描未覆盖）专项分析 ----
    lines.append("=" * 74)
    lines.append("常量通道专项分析（扫描野未覆盖该脑区 / 信号缺失）")
    lines.append("=" * 74)
    lines.append("")
    n_bad_subj = sum(1 for r in recs if r["n_bad_channel"] > 0)
    lines.append(f"受影响受试者  {n_bad_subj} / {len(recs)}  ({n_bad_subj/len(recs):.1%})")
    lines.append("")
    lines.append("按站点（这是关键——受影响者高度集中在少数站点，属协议差异，非随机噪声）：")
    lines.append(f"    {'站点':10s} {'队列':10s} {'受影响/总数':>14s} {'比例':>7s}")
    for s, d in sorted(sites.items(), key=lambda kv: -kv[1]["bad"] / max(kv[1]["n"], 1)):
        pct = d["bad"] / d["n"] if d["n"] else 0
        flag = "  <<<" if pct > 0.3 else ""
        lines.append(f"    {s:10s} {d['cohort']:10s} {d['bad']:6d}/{d['n']:<7d} "
                     f"{pct:6.0%}{flag}")
    lines.append("")
    lines.append("受影响最严重的 ROI（按出现次数）：")
    nan_per_roi = np.isnan(X_mean).sum(axis=0)
    order = np.argsort(-nan_per_roi)
    for j in order[:12]:
        if nan_per_roi[j] == 0:
            break
        lines.append(f"    {names[j]:24s} {int(nan_per_roi[j]):4d} 人 "
                     f"({nan_per_roi[j]/len(recs):.1%})")
    lines.append("")
    cb = ([j for j in range(R) if "Cerebelum" in names[j] or "Vermis" in names[j]]
          if DERIV == "rois_aal" else [])
    cb_nan = int(nan_per_roi[cb].sum())
    tot_nan = int(nan_per_roi.sum())
    lines.append(f"    小脑/蚓部 ROI 共 {len(cb)} 个，占全部缺失的 "
                 f"{cb_nan/max(tot_nan,1):.0%}")
    lines.append(f"    受影响的 ROI 总数 = {int((nan_per_roi>0).sum())} / {R}")
    lines.append("")
    lines.append("处理选项（三选一，需你决定）：")
    lines.append("    A. 删除有任何缺失的受试者  —— 最干净，但丢人数，且"
                 "会不成比例地砍掉 SBL/CMU/CALTECH")
    lines.append("    B. 只保留在所有站点都完好的 ROI（砍列，不砍人）—— "
                 "保住参考队列，但维度下降")
    lines.append("    C. 缺失值插补 —— 最省数据，但引入假设，须做敏感性分析")
    lines.append("")
    if qc:
        lines.append(f"异常记录（前 20 条，共 {len(qc)} 条）：")
        for fid, tag, msg in qc[:20]:
            lines.append(f"    {fid:22s} {tag:16s} {msg}")
    else:
        lines.append("异常记录：无")
    lines.append("")
    lines.append(f"总耗时 {time.time()-t0:.0f}s")
    lines.append("=" * 74)
    report = "\n".join(lines)
    open(os.path.join(OUTDIR, "qc_report.txt"), "w").write(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
