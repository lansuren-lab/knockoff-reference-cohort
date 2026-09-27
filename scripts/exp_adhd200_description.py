"""
exp_adhd200_description.py -- ADHD-200 队列描述，落进存档供核对

手稿 §sec:data 的 ADHD-200 段引用了一批数字（768→664、93/11 剔除、
427/237 分组、9 站点、543/121 主/参照划分、年龄与头动范围）。
这些量由外部清洗步骤产生，原始 manifest 在仓库之外，因此把【结果】
写成 results_v3/adhd200_description.csv 存进存档，使手稿里的每个数字
都能追到一个仓库内的单元格——这正是 Code availability 里那条承诺。

数据来源（外部，公开可获取）：
  E:/LiJiale/adhd_gk/features_adhd200_clean/subject_manifest.csv
  （Neuro Bureau ADHD-200 Preprocessed 仓库，QC 后 664 人）
  其 qc_report.txt 记录了 768 -> 664 的剔除明细。

Run:  python exp_adhd200_description.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import write_csv                                  # noqa: E402

MANIFEST = "E:/LiJiale/adhd_gk/features_adhd200_clean/subject_manifest.csv"
OUT = "results_v3"
N_BEFORE_QC = 768
N_DROP_QC0 = 93
N_DROP_UNEVALUATED = 11
N_MAIN_SITES = 6          # 与 src/refcohort.prepare 的 n_main 默认一致


def main():
    if not os.path.exists(MANIFEST):
        raise SystemExit(
            f"manifest not found: {MANIFEST}\n"
            "This file is distributed with the Neuro Bureau ADHD-200 "
            "Preprocessed repository; see Data availability."
        )
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    n = len(rows)
    site = [r["site"] for r in rows]
    cnt = Counter(site)
    top = [s for s, _ in cnt.most_common(N_MAIN_SITES)]
    n_main = sum(cnt[s] for s in top)
    n_ref = n - n_main
    y = np.array([float(r["y"]) for r in rows])
    age = np.array([float(r["age"]) for r in rows if r["age"] not in ("", "nan")])
    qc = Counter(r["qc"] for r in rows)
    mot = np.array([float(r["max_motion_mm"]) for r in rows
                    if r["max_motion_mm"] not in ("", "nan")])
    n_bad_ch = sum(1 for r in rows if int(float(r["n_bad_channel"] or 0)) > 0)
    ntr = np.array([float(r["n_tr"]) for r in rows
                    if r["n_tr"] not in ("", "nan")])

    out = [
        {"quantity": "n_before_qc", "value": N_BEFORE_QC,
         "source": "qc_report.txt (external cleaning step)"},
        {"quantity": "n_dropped_qc_failed", "value": N_DROP_QC0,
         "source": "qc_report.txt"},
        {"quantity": "n_dropped_unevaluated", "value": N_DROP_UNEVALUATED,
         "source": "qc_report.txt (all three Peking sites)"},
        {"quantity": "n_subjects", "value": n, "source": "manifest"},
        # manifest 的编码：y=0 是对照（dx_group=0），y=1 是 ADHD
        # （dx_group 1/2/3 三种亚型，合计 132+8+97=237）
        {"quantity": "n_controls", "value": int((y == 0).sum()),
         "source": "manifest (y=0, dx_group=0)"},
        {"quantity": "n_adhd", "value": int((y == 1).sum()),
         "source": "manifest (y=1, dx_group 1/2/3)"},
        {"quantity": "p_roi", "value": 116, "source": "AAL116"},
        {"quantity": "n_sites", "value": len(cnt), "source": "manifest"},
        {"quantity": "n_main", "value": n_main,
         "source": f"manifest, {N_MAIN_SITES} largest sites"},
        {"quantity": "n_ref", "value": n_ref, "source": "manifest"},
        {"quantity": "n_ref_over_p", "value": round(n_ref / 116, 2),
         "source": "derived"},
        {"quantity": "age_min", "value": float(age.min()), "source": "manifest"},
        {"quantity": "age_max", "value": float(age.max()), "source": "manifest"},
        {"quantity": "n_tr_min", "value": float(ntr.min()), "source": "manifest"},
        {"quantity": "n_tr_max", "value": float(ntr.max()), "source": "manifest"},
        {"quantity": "n_bad_channel_subjects", "value": n_bad_ch,
         "source": "manifest"},
        {"quantity": "n_with_motion_record", "value": int(len(mot)),
         "source": "manifest (KKI only)"},
        {"quantity": "motion_min_mm", "value": float(mot.min()) if len(mot) else "",
         "source": "manifest"},
        {"quantity": "motion_median_mm",
         "value": float(np.median(mot)) if len(mot) else "", "source": "manifest"},
        {"quantity": "motion_max_mm", "value": float(mot.max()) if len(mot) else "",
         "source": "manifest"},
    ]
    for r in out:
        print(f"  {r['quantity']:26s} {r['value']}")
    print(f"\nQC 标志分布: {dict(qc)}   主队列站点: {sorted(top)}")
    path = write_csv(os.path.join(OUT, "adhd200_description.csv"), out)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
