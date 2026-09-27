"""
build_features_v2.py -- 一次读取、同时产出四种特征定义

与 build_features.py 的关系：QC 规则、坏通道约定、主/参考队列划分、
列序约定全部沿用；唯一新增的是同时计算 gsr / pcresid / partial 三种
替代定义（见 src/features.py）。

为什么要一次算完：读取 .1D / CIFTI 并做 arctanh 是全部成本所在，
四种定义共享同一份相关矩阵与同一批坏通道判定，分四次跑等于把 I/O
开销乘以四。

列名约定：每种定义单独写一个 X_<定义>.csv，列名是**裸 ROI 名**
（与现有 X_meanconn.csv 一致）。定义名放在文件名里而不是列名前缀里 ——
这样彻底绕开 clean_features.py 按第一个下划线切前缀的陷阱。

Run:
    python build_features_v2.py --dataset abide1_aal
    python build_features_v2.py --dataset abide1_cc200
    python build_features_v2.py --dataset abide1_ho
    python build_features_v2.py --dataset abide1_niak
    python build_features_v2.py --dataset abide2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import all_features                      # noqa: E402
from src.refcohort import MAIN_SITES_ABIDE1, file_hash      # noqa: E402

# --------------------------------------------------------------------------- #
# 数据集配置
# --------------------------------------------------------------------------- #
DATASETS = {
    "abide1_aal": dict(
        roi_dir="abide_rois_aal", outdir="feat2_abide1_aal", kind="1D",
        pheno="Phenotypic_V1_0b_preprocessed1.csv", pit="ablations",
        atlas="AAL116", main_sites=MAIN_SITES_ABIDE1),
    "abide1_niak": dict(
        roi_dir="abide_niak_rois_aal", outdir="feat2_abide1_niak", kind="1D",
        pheno="Phenotypic_V1_0b_preprocessed1.csv", atlas="AAL116",
        main_sites=MAIN_SITES_ABIDE1),
    "abide1_cc200": dict(
        roi_dir="abide_rois_cc200", outdir="feat2_abide1_cc200", kind="1D",
        pheno="Phenotypic_V1_0b_preprocessed1.csv", atlas="CC200",
        main_sites=MAIN_SITES_ABIDE1),
    "abide1_ho": dict(
        roi_dir="abide_rois_ho", outdir="feat2_abide1_ho", kind="1D",
        pheno="Phenotypic_V1_0b_preprocessed1.csv", atlas="HarvardOxford",
        main_sites=MAIN_SITES_ABIDE1),
    "abide2": dict(
        roi_dir="abide2_power2011", outdir="feat2_abide2", kind="cifti",
        master="abide2_master.csv", atlas="Power2011", main_sites=None),
}

DEFINITIONS = ("meanconn", "gsr", "pcresid", "partial")

# AAL116 顺序表与锚点（与 build_features.py 完全一致，用于"按位置而非按编码"映射）
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

ANCHORS = {1: "Precentral_L", 19: "Supp_Motor_Area_L", 21: "Olfactory_L",
           27: "Rectus_L", 29: "Insula_L", 37: "Hippocampus_L",
           41: "Amygdala_L", 71: "Caudate_L", 77: "Thalamus_L",
           91: "Cerebelum_Crus1_L", 109: "Vermis_1_2"}


# --------------------------------------------------------------------------- #
def read_1d(path):
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    codes = lines[0].split()
    ts = np.array([[float(x) for x in ln.split()] for ln in lines[1:]])
    return codes, ts


def read_cifti(path):
    import nibabel as nib
    return None, np.asarray(nib.load(path).dataobj, dtype=float)


def resolve_names(cfg, codes, R, src_dir):
    """ROI 名。AAL116 按【位置】映射（该数据的数字编码不是常见 AAL 编号），
    并用 11 个解剖锚点逐点校验；其它图谱用标签本身。"""
    if cfg["atlas"] == "AAL116":
        if R != 116:
            sys.exit(f"AAL116 期望 116 列，实际 {R}")
        names = list(AAL116)
        for pos, expect in ANCHORS.items():
            if names[pos - 1] != expect:
                sys.exit(f"锚点校验失败：位置 {pos} 期望 {expect}")
        if codes is not None and len(codes) != R:
            sys.exit(f"编码数 {len(codes)} != 列数 {R}")
        return names
    if cfg["kind"] == "cifti":
        import nibabel as nib
        fs = sorted(f for f in os.listdir(src_dir) if f.endswith(".nii"))
        ax = nib.load(os.path.join(src_dir, fs[0])).header.get_axis(1)
        raw = [(str(x).strip() or "UNLABELLED") for x in ax.name]
        if len(raw) != R:
            sys.exit(f"标签数 {len(raw)} 与脑区数 {R} 不符")
        return [f"{cfg['atlas']}_{k:03d}_{s}" for k, s in enumerate(raw)]
    return [f"{cfg['atlas']}_{c}" for c in (codes or range(R))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--max-files", type=int, default=0, help="调试用，0=全部")
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    t0 = time.time()

    src = cfg["roi_dir"]
    out = cfg["outdir"]
    os.makedirs(out, exist_ok=True)
    if not os.path.isdir(src):
        sys.exit(f"找不到 {src}/")

    ext = ".nii" if cfg["kind"] == "cifti" else ".1D"
    files = sorted(f for f in os.listdir(src) if f.endswith(ext))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        sys.exit(f"{src}/ 里没有 {ext}")

    # 表型 / 主参考划分
    if cfg["kind"] == "cifti":
        master = {r["sub_id"]: r for r in csv.DictReader(open(cfg["master"]))}
    else:
        pheno = {r["FILE_ID"]: r for r in csv.DictReader(
            open(cfg["pheno"], encoding="utf-8-sig"))}
        main_sites = set(cfg["main_sites"])

    # 用第一个文件定列数与列名
    fc, ts0 = (read_cifti(os.path.join(src, files[0])) if cfg["kind"] == "cifti"
               else read_1d(os.path.join(src, files[0])))
    R = ts0.shape[1]
    names = resolve_names(cfg, fc, R, src)
    print(f"[{args.dataset}]  {cfg['atlas']}  p={R}  文件 {len(files)}  "
          f"定义 {DEFINITIONS}")

    recs, bags, qc = [], {d: [] for d in DEFINITIONS}, []
    for k, fn in enumerate(files, 1):
        try:
            codes, ts = (read_cifti(os.path.join(src, fn)) if cfg["kind"] == "cifti"
                         else read_1d(os.path.join(src, fn)))
        except Exception as e:
            qc.append({"file": fn, "tag": "READ_FAIL", "detail": str(e)[:60]})
            continue
        if ts.ndim != 2 or ts.shape[1] != R:
            qc.append({"file": fn, "tag": "SHAPE", "detail": str(ts.shape)})
            continue

        feats, info = all_features(ts, DEFINITIONS)
        for d in DEFINITIONS:
            bags[d].append(feats[d])

        if cfg["kind"] == "cifti":
            sub = fn.split("_")[0].replace("sub-", "")
            ses = fn.split("_")[1] if "_" in fn else ""
            m = master.get(sub, {})
            dx = m.get("dx", "")
            rec = {"file_id": fn[: -len(".ptseries.nii")], "sub_id": sub,
                   "session": ses, "site": m.get("site", "?"),
                   "dx_group": "ASD" if dx == "1" else ("control" if dx == "2" else "?"),
                   "y": 1 if dx == "1" else (0 if dx == "2" else ""),
                   "n_tr": info["n_tr"], "n_bad_channel": info["n_bad_channel"]}
        else:
            # 文件名形如 CMU_a_0050649_rois_cc200.1D —— FILE_ID 是 "_rois_" 之前
            # 的部分。早先按 "_rois_aal.1D" 字面量裁剪，导致 CC200/HO 的
            # file_id 带着后缀，表型查不到，site/y 全空。
            fid = fn.split("_rois_")[0]
            p = pheno.get(fid, {})
            site = (p.get("SITE_ID") or "?").strip()
            dg = (p.get("DX_GROUP") or "").strip()
            rec = {"file_id": fid, "site": site,
                   "cohort": "main" if site in main_sites else "reference",
                   "dx_group": "ASD" if dg == "1" else ("control" if dg == "2" else "?"),
                   "y": 1 if dg == "1" else (0 if dg == "2" else ""),
                   "n_tr": info["n_tr"], "n_bad_channel": info["n_bad_channel"]}
        recs.append(rec)
        if info["n_bad_channel"]:
            qc.append({"file": fn, "tag": "BAD_CHANNEL",
                       "detail": str(info["n_bad_channel"])})
        if k % 150 == 0 or k == len(files):
            print(f"  {k}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)

    if not recs:
        sys.exit("没有任何文件处理成功")

    for d in DEFINITIONS:
        X = np.vstack(bags[d])
        np.savetxt(os.path.join(out, f"X_{d}.csv"), X, delimiter=",",
                   fmt="%.6f", header=",".join(names), comments="")
    with open(os.path.join(out, "subject_manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    with open(os.path.join(out, "feature_names.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "name"])
        for i, n in enumerate(names, 1):
            w.writerow([i, n])
    with open(os.path.join(out, "qc_log.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "n_files": len(files),
                   "n_ok": len(recs), "issues": qc}, f, indent=2)

    summary = {"dataset": args.dataset, "atlas": cfg["atlas"], "p": R,
               "n_subjects": len(recs),
               "seconds": round(time.time() - t0, 1),
               "source_hash_input": file_hash(os.path.join(src, files[0]))}
    with open(os.path.join(out, "build.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
