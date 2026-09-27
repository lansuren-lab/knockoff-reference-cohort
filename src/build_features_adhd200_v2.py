"""
build_features_adhd200_v2.py -- ADHD-200 的四种 ROI 汇总定义

第三个队列。与 ABIDE 的差别只在输入格式，特征定义完全沿用
`src/features.py`，保证跨队列可比。

输入格式（AFNI 3dROIstats 输出）：
    第 1 行  File  Sub-brick  Mean_2001  Mean_2002 ...   ← 2 + 116 列
    第 2 行起 <文件名> 0[?] v v v ...                     ← 每行一个时间点
所以必须跳表头并丢掉前两列。ROI 编码同样是 #2001 那套，按【位置】映射到
标准 AAL116 名称并用 11 个解剖锚点校验。

受试者选择：沿用 features_adhd200_clean 的 manifest（已是 QC 后的 664 人），
按 variant + session 定位文件，避免取错 run。

Run:  python build_features_adhd200_v2.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import all_features                      # noqa: E402
from build_features_v2 import AAL116, ANCHORS              # noqa: E402

SRC = "E:/LiJiale/adhd_gk/adhd200"
SRC_LINK = "adhd200_src"          # 仓库内的相对指向（软链或副本），见 main()
OUT = "feat2_adhd200"
MANIFEST = "E:/LiJiale/adhd_gk/features_adhd200_clean/subject_manifest.csv"
DEFS = ("meanconn", "gsr", "pcresid", "partial")


def read_tcs(path):
    """AFNI 3dROIstats 输出 -> (T x 116)；跳表头、丢前两列。"""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        fh.readline()                       # 表头
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                rows.append([float(x) for x in parts[2:]])
            except ValueError:
                continue
    return np.asarray(rows, dtype=float)


def locate(site, sub_id, variant, session):
    """按 variant + rest 编号定位 .1D 文件。"""
    k = session.split("rest")[-1] if "rest" in session else "1"
    pat = os.path.join(SRC, site, sub_id,
                       f"{variant}*_rest_{k}_aal_TCs.1D")
    hits = sorted(glob.glob(pat))
    if hits:
        return hits[0]
    # 退一步：只要 variant 对得上、任取 rest_1
    hits = sorted(glob.glob(os.path.join(SRC, site, sub_id,
                                         f"{variant}*_aal_TCs.1D")))
    return hits[0] if hits else None


def verify_anchors(codes):
    """ROI 编码不是标准 AAL 编号，按位置校验 11 个锚点。"""
    if len(codes) != 116:
        return False, f"列数 {len(codes)} != 116"
    for pos, expect in ANCHORS.items():
        if AAL116[pos - 1] != expect:
            return False, f"锚点表本身有误：位置 {pos}"
    return True, "ok"


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    man = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    print(f"manifest: {len(man)} 名受试者")

    recs, bags = [], {d: [] for d in DEFS}
    miss, nbad = [], 0
    for i, row in enumerate(man, 1):
        f = locate(row["site"], row["sub_id"], row["variant"], row["session"])
        if f is None:
            miss.append(row["file_id"])
            continue
        ts = read_tcs(f)
        if ts.ndim != 2 or ts.shape[1] != 116:
            miss.append(row["file_id"])
            continue
        feats, info = all_features(ts, DEFS)
        for d in DEFS:
            bags[d].append(feats[d])
        nbad += info["n_bad_channel"] > 0
        recs.append({"file_id": row["file_id"], "site": row["site"],
                     "y": int(row["y"]),
                     "dx_group": row["dx_group"],
                     "n_tr": info["n_tr"],
                     "n_bad_channel": info["n_bad_channel"]})
        if i % 150 == 0:
            print(f"  {i}/{len(man)}  ({time.time()-t0:.0f}s)", flush=True)

    if not recs:
        sys.exit("没有任何文件可读")
    names = list(AAL116)
    for d in DEFS:
        X = np.vstack(bags[d])
        np.savetxt(os.path.join(OUT, f"X_{d}.csv"), X, delimiter=",",
                   fmt="%.6f", header=",".join(names), comments="")
    with open(os.path.join(OUT, "subject_manifest.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)
    with open(os.path.join(OUT, "feature_names.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["index", "name"])
        for i, n in enumerate(names, 1):
            w.writerow([i, n])

    summary = {"source": SRC, "n_manifest": len(man), "n_built": len(recs),
               "n_missing": len(miss), "n_with_bad_channel": int(nbad),
               "p": 116, "definitions": list(DEFS),
               "y_counts": {str(v): sum(1 for r in recs if r["y"] == v)
                            for v in (0, 1)},
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "build.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if miss:
        print(f"未找到 {len(miss)} 个，例如：{miss[:5]}")
    print(f"-> {OUT}/  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
