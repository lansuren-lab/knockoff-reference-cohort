"""
exp_cohort_description.py -- 把手稿里"数据描述"那一段的数字也落盘

v3 的反向覆盖检查暴露出 56 个尚未有产物的数值，其中大部分是数据描述类
（队列人数、站点数、时间点分布、坏通道比例），它们此前散落在各次交互式
会话的输出里，没有统一的产物。手稿里每一个数字都必须可追，所以这里补齐。

产出 results_v3/cohort_description.csv：一行一个量。
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import prepare, provenance, write_csv          # noqa: E402

OUT = "results_v3"
os.makedirs(OUT, exist_ok=True)


def main():
    rows = []

    def add(name, value, source):
        rows.append({"quantity": name, "value": value, "source": source})
        print(f"  {name:44s} {value}")

    # ---- ABIDE I 队列构成 ----
    # 【必须读原始 manifest】prepare() 返回的是【已过滤】的 852 人；
    # 而手稿描述的"1035 -> 885，其中 68 人含坏通道（7.7%）"说的是
    # 过滤【之前】的 885 人。用 prepare() 的输出会把这两个数算错。
    RAW = "feat2_abide1_aal/subject_manifest.csv"
    man_raw = list(csv.DictReader(open(RAW)))
    X_raw = prepare("feat2_abide1_aal")[0]
    add("abide1_n_raw_after_motion_qc", len(man_raw), RAW)
    add("abide1_n_final", X_raw.shape[0], RAW)
    add("abide1_p_retained", X_raw.shape[1], RAW)

    Xa, na, mana, isma, msa = prepare("feat2_abide1_aal")
    ya = np.array([float(r["y"]) if str(r["y"]).strip() != "" else np.nan
                   for r in mana])
    add("abide1_main_n", int(isma.sum()), RAW)
    add("abide1_reference_n", int((~isma).sum()), RAW)
    add("abide1_main_asd", int((ya[isma] == 1).sum()), RAW)
    add("abide1_main_control", int((ya[isma] == 0).sum()), RAW)
    add("abide1_reference_asd", int((ya[~isma] == 1).sum()), RAW)
    add("abide1_reference_control", int((ya[~isma] == 0).sum()), RAW)

    # ---- 站点数与时间点分布（用过滤前的 885 人）----
    add("abide1_n_sites", len(set(r["site"] for r in man_raw)), RAW)
    Ts = np.array([int(r["n_tr"]) for r in man_raw])
    add("abide1_T_min", int(Ts.min()), RAW)
    add("abide1_T_max", int(Ts.max()), RAW)
    add("abide1_T_median", float(np.median(Ts)), RAW)
    bad = np.array([int(r["n_bad_channel"]) > 0 for r in man_raw])
    add("abide1_n_subjects_with_bad_channel", int(bad.sum()), RAW)
    add("abide1_pct_subjects_with_bad_channel",
        round(100.0 * bad.mean(), 1), RAW)
    agg = {}
    for r in man_raw:
        d = agg.setdefault(r["site"], [0, 0])
        d[0] += 1
        d[1] += int(r["n_bad_channel"]) > 0
    for s_name, (nn, bb) in sorted(
            agg.items(), key=lambda kv: -kv[1][1] / max(kv[1][0], 1))[:4]:
        add(f"abide1_bad_channel_pct_{s_name}", round(100.0 * bb / nn), RAW)

    # ---- ABIDE II ----
    X2, n2, man2, ism2, ms2 = prepare("feat2_abide2")
    add("abide2_n", len(man2), "feat2_abide2/manifest")
    add("abide2_main_n", int(ism2.sum()), "feat2_abide2/manifest")
    add("abide2_reference_n", int((~ism2).sum()), "feat2_abide2/manifest")
    add("abide2_p", X2.shape[1], "feat2_abide2/manifest")

    path = write_csv(os.path.join(OUT, "cohort_description.csv"), rows)
    provenance(OUT, "exp_cohort_description", None)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
