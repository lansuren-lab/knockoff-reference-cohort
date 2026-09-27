"""
make_figures_v3.py -- v3 手稿的十张图，全部从 results*/ 的数值重建。

设计契约（每张图先有一句话核心结论，再映射证据链；一个英雄面板 + 从属面板）：
  fig_flow         诊断量在任何 knockoff 抽取前算出；它预测功效，不预测有效性
  fig_geometry     同数据集内换 summary 让有效秩变 x44；固定 summary 换队列最多 x7.2
  fig_mechanism    meanconn 让每个特征成为 PC1 代理（中位 R2=0.677），偏相关不会（0.018）
  fig_limits       41 种协方差配置上功效从未低于 0.770；崩掉功效的是信号个数，不是条件数
  fig_machinery    没有替代方案能把"有功效"和"有有效性"分开
  fig_matrixpower  自报误差单调改善而真实误差单调恶化
  fig_shift        队列越漂移功效越升，FDP 越坏——失败是沉默的
  fig_penalty      罚常数须同时满足找回与收敛，只有 c=0.05 满足，而软件两个都不报
  fig_draws        真实响应上的输出是分布：多数配置中位抽签返回 0 个区域，尾部可达 28
  fig_third        换一种疾病后排序复现、量级不复现

硬性约束：
  * 灰度安全：每条序列除颜色外还必须靠 marker 形状与线型区分。
  * 绝不用双 y 轴；量纲不同的量一律分面板。
  * 按印张宽度设计（\\textwidth ~ 16.2 cm = 6.38 in），图内字号 1:1 复现。
  * 每张图同时导出 PNG(300 dpi) 与 PDF(矢量，fonttype=42 保证文字可编辑)，
    并把实际画出的数值写到 paper/figs/source_data/<stem>.csv。
  * 配色两套语义、同图内不混用：summary 族与角色族（见下）。

Run:  cd E:\\LiJiale\\sjyx  &&  python make_figures_v3.py [--only stem,...] [--list]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
# 运行位置守卫：OUT 是相对路径，跑错目录会把图写到别处
# --------------------------------------------------------------------------- #
if not (os.path.isdir("paper") and os.path.isdir("results_c0.05")):
    raise SystemExit(
        "run from the repo root (E:\\LiJiale\\sjyx): "
        "`paper/` and `results_c0.05/` not found in cwd = " + os.getcwd()
    )

OUT = "paper/figs"
SRC = os.path.join(OUT, "source_data")
os.makedirs(SRC, exist_ok=True)

# --------------------------------------------------------------------------- #
# 配色：两套语义，同一张图内不得混用
# --------------------------------------------------------------------------- #
# summary 族：凡"ROI summary 定义"作为序列时使用（沿用 v3 既有映射，勿改）
C_MEAN, C_GSR, C_PCR, C_PART = "#c0392b", "#1f6fb4", "#27ae60", "#8e44ad"
SUMMARY_COLOR = {"meanconn": C_MEAN, "gsr": C_GSR, "pcresid": C_PCR,
                 "partial": C_PART}
SUMMARY_LABEL = {"meanconn": "meanconn (default)", "gsr": "+GSR",
                 "pcresid": "pcresid", "partial": "partial correlation"}

# 角色族：实验类图使用
C_VALID, C_VIOL, C_ALT = "#1f6fb4", "#c0392b", "#8e44ad"
C_NEUTRAL, C_NEUTRAL_L = "#8a8a8a", "#c9c9c9"

MARKERS = ("o", "s", "^", "D", "v", "P")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

plt.rcParams.update({
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    # 不用 bbox="tight"：那会让每张 PDF 的实际宽度由内容决定（实测 5.66-6.78 in），
    # 被 \textwidth 缩放 0.94-1.13 倍，同名字号在不同图里差 20%，最小的掉到 4.7 pt。
    # 固定画布宽度 = 印张宽度，缩放因子恒为 1.000，图内字号才真的 1:1。
    "pdf.fonttype": 42,   # 默认 3(Type3) 在 Illustrator 里不可编辑
    "ps.fonttype": 42,
})

# \textwidth = a4paper 21 cm - 2 x 2.4 cm 边距 = 16.2 cm = 6.3779 in
FIGW = 6.378          # 单栏印张宽度，in
REG = {}              # 每张图登记实际画出的数值 -> source_data


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def g(rows, key):
    return np.array([float(r[key]) for r in rows])


def rows_of(rows, **eq):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in eq.items())]


def one(rows, **eq):
    m = rows_of(rows, **eq)
    if len(m) != 1:
        raise SystemExit(f"expected exactly 1 row for {eq}, got {len(m)}")
    return m[0]


def save(fig, stem):
    """同时写 300 dpi PNG 与矢量 PDF；漏写其一是结构上不可能的。"""
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def dump_source(stem, rows):
    """把该图实际画出的数值写出来，满足"源数据可追溯属于图的一部分"。
    允许多个面板用不同的行结构：取键的并集，缺的填空。"""
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(SRC, f"{stem}.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def panel(ax, letter):
    """面板字母，置于坐标区外。

    若该面板已有标题，就把字母并入标题并左对齐——否则居中的长标题会与
    左上角的字母相撞（fig_mechanism、fig_limits、fig_matrixpower 都撞过）。
    没有标题时直接写在坐标区左上外侧。
    """
    t = ax.get_title()
    if t:
        # 必须指定 loc="center"：matplotlib 对每个 loc 分开存标题，
        # 用 loc="left" 会新增一条而留下原来的居中标题，渲染成双标题。
        ax.set_title(f"({letter})  {t}", loc="center")
    else:
        ax.text(0.0, 1.015, f"({letter})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")


def short_ds(d):
    return (d.replace("ABIDE1-", "ABIDE I ").replace("ABIDE2-", "ABIDE II ")
             .replace("-CPAC", " (CPAC)").replace("-NIAK", " (NIAK)"))


def compact_ds(d):
    """轴标签形式：ABIDE1-AAL116-CPAC -> ABIDE I AAL116 CPAC。

    必须保留 pipeline 后缀：fig_mechanism 与 fig_draws 的结论正是
    "同一队列、同一 atlas，只有 summary/pipeline 不同"，若把 CPAC/NIAK
    抹掉会出现两行同名标签，读者无法把行映射回配置。
    """
    s = (d.replace("ABIDE1-", "ABIDE I ").replace("ABIDE2-", "ABIDE II ")
          .replace("ADHD200-", "ADHD-200 ")
          .replace("-CPAC", " CPAC").replace("-NIAK", " NIAK")
          .replace("-Power2011", " Power2011"))
    return " ".join(s.split())


def q_line(ax, q=0.20, xpos=0.97, ha="right"):
    ax.axhline(q, color="k", ls=":", lw=1.2, zorder=1)
    ax.text(xpos, q + 0.006, f"$q={q:.2f}$", transform=ax.get_yaxis_transform(),
            fontsize=7.5, ha=ha, va="bottom")


# --------------------------------------------------------------------------- #
# 1. 设计示意：诊断量预测功效，不预测有效性
# --------------------------------------------------------------------------- #
def fig_flow():
    diag = load("results/diagnostics.csv")
    ds = "ABIDE1-AAL116-CPAC"
    order = ("meanconn", "gsr", "pcresid", "partial")
    vals = [float(one(diag, dataset=ds, feature=f)["eff_rank"]) for f in order]

    fig = plt.figure(figsize=(FIGW, 5.0))
    ax = fig.add_axes([0, 0.28, 1, 0.72])
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 57)

    def box(x, y, w, h, text, fc="#f2f5f9", ec="#4a4a4a", fs=7.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.0",
                                    linewidth=0.8, edgecolor=ec,
                                    facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, ls="-", color="#4a4a4a", lw=0.9, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=8, linewidth=lw,
                                     linestyle=ls, color=color, zorder=1,
                                     shrinkA=0, shrinkB=0))

    # 左列：参考队列 -> Sigma 估计 -> 谱诊断
    box(1, 41, 26, 9, "reference cohort\n(covariates only, no response)")
    arrow(14, 41, 14, 36)
    box(1, 26, 26, 9, "$\\hat\\Sigma$ = Ledoit-Wolf\nshrinkage estimate")
    arrow(14, 26, 14, 21)
    box(1, 11, 26, 9, "spectral diagnostic\n$\\lambda_{\\min}$, cond., PC1 share,\n"
                     "effective rank", fc="#e8f0f8", bold=True)

    # 中列：主队列 -> knockoff -> W -> 阈值
    box(37, 41, 24, 9, "main cohort\n$X$, $y$")
    arrow(49, 41, 49, 36)
    box(37, 26, 24, 9, "knockoff draw\n$\\tilde X$")
    arrow(49, 26, 49, 21)
    box(37, 11, 24, 9, "$W$ with penalty $c$\n$\\to$ knockoff$+$ threshold\n"
                      "$\\to\\hat S$")

    # 诊断量 -> 程序：预测功效
    arrow(27, 15.5, 37, 15.5, ls="-", color=C_VALID, lw=1.1)
    ax.text(32, 17.4, "predicts\npower", fontsize=6.8, color=C_VALID,
            ha="center", va="bottom", linespacing=1.15)

    # 右列：两个必须分别测量的条件
    box(72, 32, 27, 13, "condition 1\ncohort mismatch $\\leq\\sim 0.12$\n"
                        "after subtracting\nthe sampling-noise floor",
        fc="#fdf2f2", ec=C_VIOL, fs=6.8)
    box(72, 12, 27, 13, "condition 2\npenalty $c$ satisfies\n"
                        "recovery $\\wedge$ convergence\nat once (both unreported)",
        fc="#fdf2f2", ec=C_VIOL, fs=6.8)

    # 诊断量 -> 条件：不预测有效性（向下绕行，避开中列）
    for k, seg in enumerate((((14, 11), (14, 4.5)), ((14, 4.5), (85.5, 4.5)),
                             ((85.5, 4.5), (85.5, 12)))):
        arrow(seg[0][0], seg[0][1], seg[1][0], seg[1][1], ls="--",
              color=C_VIOL, lw=1.0,
              style="-|>" if k == 2 else "-")   # 只在最后一段加箭头
    ax.text(50, 2.4, "does not predict validity", fontsize=6.8, color=C_VIOL,
            ha="center", va="top")

    ax.text(50, 55.5, "the diagnostic is computable before any knockoff is drawn",
            fontsize=7.4, ha="center", va="top", style="italic")

    # 从属面板：数值锚点
    ax2 = fig.add_axes([0.30, 0.04, 0.68, 0.22])
    y = np.arange(len(order))[::-1]
    for i, (f, v) in enumerate(zip(order, vals)):
        ax2.barh(y[i], v, height=0.55, color=SUMMARY_COLOR[f],
                 edgecolor="white", linewidth=0.7)
        ax2.text(v + 2, y[i], f"{v:.1f}", va="center", fontsize=7.2)
    ax2.set_yticks(y)
    ax2.set_yticklabels([SUMMARY_LABEL[f] for f in order], fontsize=7.2)
    ax2.set_xlim(0, 88)
    ax2.set_xlabel("effective rank of $\\hat\\Sigma$   "
                   "(ABIDE I, AAL116, CPAC; $p=113$)", fontsize=7.5)
    ax2.grid(True, axis="x")
    ax2.set_title("same subjects, same atlas: the summary is the only change",
                  fontsize=7.8, pad=3)

    save(fig, "fig_flow")
    dump_source("fig_flow",
                [{"dataset": ds, "feature": f, "eff_rank": v}
                 for f, v in zip(order, vals)])


# --------------------------------------------------------------------------- #
# 2. 有效秩：换 summary vs 换数据集
# --------------------------------------------------------------------------- #
def fig_geometry():
    rows = load("results/diagnostics.csv")
    exclude = {"ADHD200-AAL116"}          # 它的伏笔属于 sec:third
    datasets = []
    for r in rows:
        if r["dataset"] not in datasets and r["dataset"] not in exclude:
            datasets.append(r["dataset"])
    order = ("meanconn", "gsr", "pcresid", "partial")

    fig, ax = plt.subplots(figsize=(FIGW, 3.5), layout="constrained")
    x = np.arange(len(datasets))
    yy = {}
    for k, f in enumerate(order):
        v = np.array([float(one(rows, dataset=d, feature=f)["eff_rank"])
                      for d in datasets])
        yy[f] = v
        ax.plot(x, v, color=SUMMARY_COLOR[f], ls=LINESTYLES[k],
                marker=MARKERS[k], ms=4.5, lw=1.6, label=SUMMARY_LABEL[f],
                clip_on=False)
    # 右端直接标注：按末值排序后错开，避免四条线汇聚时文字互相压住
    ends = sorted(((float(yy[f][-1]), f) for f in order), reverse=True)
    for rank, (_, f) in enumerate(ends):
        dy = {0: 7, 1: 3, 2: -4, 3: -10}[rank]
        ax.annotate(SUMMARY_LABEL[f], (x[-1], float(yy[f][-1])),
                    textcoords="offset points", xytext=(6, dy), fontsize=7.0,
                    color=SUMMARY_COLOR[f], va="center", annotation_clip=False)

    ax.set_yscale("log")
    ax.set_ylim(1.7, 145)
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(x)
    ax.set_xticklabels([short_ds(d).replace(" (CPAC)", "\n(CPAC)")
                        .replace(" (NIAK)", "\n(NIAK)") for d in datasets],
                       fontsize=7.2)
    ax.set_xlim(-0.35, len(datasets) - 0.35)
    ax.set_ylabel("effective rank of the reference covariance")
    ax.grid(True, axis="y", which="both")

    # 标注 1：单个数据集内换 summary 的跨度（ABIDE I CC200 CPAC, x44）
    i_cc = datasets.index("ABIDE1-CC200-CPAC")
    lo, hi = float(yy["meanconn"][i_cc]), float(yy["partial"][i_cc])
    ax.annotate("", xy=(i_cc, hi), xytext=(i_cc, lo),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color="#333333",
                                shrinkA=0, shrinkB=0))
    ax.text(i_cc + 0.10, lo * 2.4, f"$\\times${hi / lo:.0f}\nwithin one dataset",
            fontsize=7.2, color="#333333", va="center")

    # 标注 2：固定 summary 换数据集的最大跨度（meanconn, x7.2）
    # 跨度线画在 meanconn 自身的纵向区间上，箭头指向该线中点，
    # 避免被误读成 pcresid/partial 的跨度
    mc = yy["meanconn"]
    i_lo, i_hi = int(np.argmin(mc)), int(np.argmax(mc))
    xm = (i_lo + i_hi) / 2
    y_lo, y_hi = float(mc[i_lo]), float(mc[i_hi])
    ax.plot([xm, xm], [y_lo, y_hi], color="#777777", ls=":", lw=1.1, zorder=1)
    ax.plot([xm - 0.10, xm + 0.10], [y_lo, y_lo], color="#777777", lw=1.1)
    ax.plot([xm - 0.10, xm + 0.10], [y_hi, y_hi], color="#777777", lw=1.1)
    ax.annotate(f"$\\times${mc.max() / mc.min():.1f} across\ncohorts, "
                "same summary",
                xy=(xm - 0.05, (y_lo * y_hi) ** 0.5), xytext=(0.02, 6.2),
                fontsize=6.8, color="#333333", va="center", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=1.2),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#777777"))

    save(fig, "fig_geometry")
    dump_source("fig_geometry",
                [{"dataset": d, "feature": f, "eff_rank": float(yy[f][i])}
                 for i, d in enumerate(datasets) for f in order])


# --------------------------------------------------------------------------- #
# 3. 机制：PC1 主导
# --------------------------------------------------------------------------- #
def fig_mechanism():
    per = load("results_c0.05/pc1_r2_per_roi.csv")
    dom = load("results_c0.05/pc1_dominance.csv")

    # 与 fig_geometry 一致：ADHD-200 的展开留给 sec:third，此处不提前剧透
    exclude = {"ADHD200-AAL116"}
    datasets = []
    for r in dom:
        if r["dataset"] not in datasets and r["dataset"] not in exclude:
            datasets.append(r["dataset"])
    order = ("meanconn", "gsr", "pcresid", "partial")
    combos = [(d, f) for d in datasets for f in order]

    fig, ax = plt.subplots(1, 2, figsize=(FIGW, 3.6), sharey=True,
                           gridspec_kw={"width_ratios": [2.5, 1.0]}, layout="constrained")

    # 灰度安全：四个 summary 除颜色外再各配一种 marker 形状
    msum = {"meanconn": "o", "gsr": "s", "pcresid": "^", "partial": "D"}

    rng = np.random.default_rng(0)
    for i, (d, f) in enumerate(combos):
        yv = i + 1
        v = np.array([float(r["r2_on_pc1"]) for r in per
                      if r["dataset"] == d and r["feature"] == f])
        vv = np.clip(v, 1e-4, None)
        ax[0].scatter(vv, yv + rng.uniform(-0.22, 0.22, len(vv)), s=3.4,
                      marker=msum[f], color=SUMMARY_COLOR[f], alpha=0.32,
                      linewidths=0)
        ax[0].plot([np.median(vv), np.median(vv)], [yv - 0.32, yv + 0.32],
                   color=SUMMARY_COLOR[f], lw=1.8, solid_capstyle="butt")
    ax[0].set_xscale("log")
    ax[0].set_xlim(5e-5, 1.5)
    ax[0].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(np.log10(v))}}}$"))
    ax[0].set_xlabel("$R^2$ of each feature on its own first principal component")
    ax[0].set_yticks(range(1, len(combos) + 1))
    ax[0].set_yticklabels([f"{compact_ds(d)} {f}" for d, f in combos], fontsize=5.8)
    ax[0].set_ylim(0.4, len(combos) + 0.6)
    ax[0].invert_yaxis()
    ax[0].grid(True, axis="x")

    i0 = combos.index(("ABIDE1-AAL116-CPAC", "meanconn"))
    i3 = combos.index(("ABIDE1-AAL116-CPAC", "partial"))
    ax[0].annotate("same subjects, same atlas:\nthe median $R^2$ moves "
                   "$0.018\\to0.677$",
                   xy=(0.66, i0 + 1), xytext=(1.6e-4, i0 + 4.6),
                   fontsize=7.6, color="#333333",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.88,
                             pad=1.6),
                   arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))
    ax[0].set_title("meanconn makes every feature a PC1 proxy",
                    fontsize=8.4, pad=4)
    panel(ax[0], "a")

    for i, (d, f) in enumerate(combos):
        yv = i + 1
        s = float(one(dom, dataset=d, feature=f)["pc1_share"])
        ax[1].barh(yv, s, height=0.55, color=SUMMARY_COLOR[f],
                   edgecolor="white", linewidth=0.6)
        ax[1].text(s + 0.012, yv, f"{s:.2f}", va="center", fontsize=6.2)
    ax[1].set_xlim(0, 0.78)
    ax[1].set_xlabel("PC1 share")
    ax[1].grid(True, axis="x")
    ax[1].set_title("variance in PC1", fontsize=8.2, pad=4)
    panel(ax[1], "b")

    save(fig, "fig_mechanism")
    # 面板 (a) 画的是逐 ROI 的散点，聚合的中位数不足以复现它，
    # 因此把每个 ROI 的值也写进本图的源数据
    dump_source("fig_mechanism",
                [{"panel": "a_summary", "dataset": d, "feature": f,
                  "roi": "", "r2_on_pc1": "",
                  "r2_on_pc1_median": float(one(dom, dataset=d, feature=f)["r2_on_pc1_median"]),
                  "pc1_share": float(one(dom, dataset=d, feature=f)["pc1_share"]),
                  "n_roi": len([r for r in per
                                if r["dataset"] == d and r["feature"] == f])}
                 for d, f in combos]
                + [{"panel": "a_roi", "dataset": r["dataset"],
                    "feature": r["feature"], "roi": r["roi"],
                    "r2_on_pc1": float(r["r2_on_pc1"]),
                    "r2_on_pc1_median": "", "pc1_share": "", "n_roi": ""}
                   for r in per
                   if r["dataset"] not in exclude])


# --------------------------------------------------------------------------- #
# 4. 什么限制功效：条件化 vs 信号个数
# --------------------------------------------------------------------------- #
def fig_limits():
    bs = load("results_c0.05/boundary_sweep.csv")
    pc = load("results_c0.05/positive_control_v3.csv")

    fam_col = {"A_blend_to_I": C_VALID, "B_AR1": C_ALT, "C_real_features": C_NEUTRAL}
    fam_mk = {"A_blend_to_I": "o", "B_AR1": "s", "C_real_features": "^"}
    fam_lab = {"A_blend_to_I": "blend to identity", "B_AR1": "AR(1)",
               "C_real_features": "real feature constructions"}

    fig, ax = plt.subplots(1, 3, figsize=(FIGW, 3.1),
                           gridspec_kw={"width_ratios": [1.30, 0.96, 1.14]},
                           layout="constrained")

    def plab(ax, s):
        panel(ax, s)

    # (a) 英雄面板：功效对有效秩——条件化基本无关
    a = ax[0]
    for fam in fam_col:
        sub = [r for r in bs if r["family"] == fam]
        a.scatter(g(sub, "eff_rank"), g(sub, "power"), s=15,
                  color=fam_col[fam], marker=fam_mk[fam], alpha=0.85,
                  linewidths=0.4, edgecolors="white", label=fam_lab[fam], zorder=3)
    pmin = min(float(r["power"]) for r in bs)
    a.axhline(pmin, color="#333333", ls="--", lw=1.0)
    a.text(0.97, pmin - 0.004, f"minimum ${pmin:.3f}$", color="#333333",
           transform=a.get_yaxis_transform(), fontsize=7.0, ha="right",
           va="top")
    r_col = one(bs, family="A_blend_to_I", param="0.0")
    a.annotate("collapsed covariance:\n14 points, not everything",
               xy=(float(r_col["eff_rank"]) * 1.15, float(r_col["power"])),
               xytext=(4.0, 0.815), fontsize=6.5, color="#333333",
               arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))
    a.set_xscale("log")
    a.set_xlim(1.8, 240)
    a.set_ylim(0.72, 1.16)
    a.set_xlabel("effective rank of $\\Sigma$", fontsize=7.5)
    a.set_ylabel("per-signal power", fontsize=7.5)
    a.grid(True)
    a.legend(loc="upper left", fontsize=6.0, borderpad=0.2)
    a.set_title("conditioning does not limit power", fontsize=7.8, pad=5)
    plab(a, "a")

    # (b) 同一批配置上的实现 FDP
    b = ax[1]
    for fam in fam_col:
        sub = [r for r in bs if r["family"] == fam]
        b.scatter(g(sub, "eff_rank"), g(sub, "fdp"), s=15,
                  color=fam_col[fam], marker=fam_mk[fam], alpha=0.85,
                  linewidths=0.4, edgecolors="white", zorder=3)
    q_line(b, 0.20, xpos=0.98)
    b.set_xscale("log")
    b.set_xlim(1.8, 240)
    b.set_ylim(0.11, 0.235)
    b.set_xlabel("effective rank of $\\Sigma$", fontsize=7.5)
    b.set_ylabel("realised FDP", fontsize=7.5)
    b.grid(True)
    n_over = sum(1 for r in bs if float(r["fdp"]) > 0.20)
    b.set_title(f"validity holds in {len(bs) - n_over} of {len(bs)}",
                fontsize=7.8, pad=5)
    plab(b, "b")

    # (c) 起作用的因素：注入信号个数
    c = ax[2]
    sup_style = {"random12": (C_VALID, "o", "-", "12 signals"),
                 "random3": (C_VIOL, "s", "--", "3 signals"),
                 "fixed_highVIF": (C_ALT, "^", "-.", "3 near-collinear")}
    for sup, (col, mk, ls, lab) in sup_style.items():
        sub = sorted([r for r in pc if r["support"] == sup],
                     key=lambda r: float(r["target_cond_t"]))
        if not sub:
            continue
        xs = g(sub, "target_cond_t")
        ys = g(sub, "pow_main_corr")
        c.plot(xs, ys, color=col, ls=ls, marker=mk, ms=3.6, lw=1.5, label=lab)
        c.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                   xytext=(3, -1), fontsize=6.6, color=col, va="center")
    c.set_xlabel("target conditional $|t|$", fontsize=7.5)
    c.set_ylabel("per-signal power", fontsize=7.5)
    c.set_ylim(-0.04, 1.14)
    c.set_xlim(0.6, 7.4)
    c.grid(True)
    c.legend(loc="lower right", fontsize=6.0, borderpad=0.2)
    c.set_title("what limits power is the\nnumber of true signals",
                fontsize=7.8, pad=5)
    plab(c, "c")

    save(fig, "fig_limits")
    dump_source("fig_limits",
                [{"panel": "a,b", "family": r["family"], "param": r["param"],
                  "eff_rank": float(r["eff_rank"]), "power": float(r["power"]),
                  "fdp": float(r["fdp"])} for r in bs]
                + [{"panel": "c", "family": r["support"],
                    "param": r["target_cond_t"],
                    "eff_rank": "", "power": float(r["pow_main_corr"]),
                    "fdp": ""} for r in pc])


# --------------------------------------------------------------------------- #
# 5. 穷尽替代方案：功效-FDP 平面
# --------------------------------------------------------------------------- #
def fig_machinery():
    cc = load("results_construct/construction_control.csv")
    gm = {r["kind"]: r for r in load("results_gm/gm_vs_knockoff.csv")}
    mech = load("results_v3/gm_mechanism.csv")

    fig, ax = plt.subplots(1, 2, figsize=(FIGW, 3.1),
                           gridspec_kw={"width_ratios": [1.45, 1.0]}, layout="constrained")

    # (a) 英雄面板：功效-FDP 平面
    a = ax[0]
    a.add_patch(plt.Rectangle((0.90, 0.0), 0.13, 0.20, facecolor=C_VALID,
                              alpha=0.07, zorder=0))
    a.text(0.965, 0.205, "power $>0.90$ and FDP $<q$", fontsize=6.2,
           color=C_VALID, ha="center", va="bottom")
    a.scatter(g(cc, "power"), g(cc, "fdp"), s=22, color=C_VALID, marker="o",
              alpha=0.9, linewidths=0.4, edgecolors="white", zorder=3)
    a.annotate("six knockoff constructions\n(indistinguishable here)",
               xy=(float(np.mean(g(cc, "power"))), float(np.mean(g(cc, "fdp")))),
               xytext=(0.30, 0.155), fontsize=6.8, color="#333333",
               arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))
    a.scatter([float(gm["ctrl"]["gm_power"])], [float(gm["ctrl"]["gm_fdp"])],
              s=26, color=C_ALT, marker="D", zorder=3,
              linewidths=0.4, edgecolors="white")
    a.annotate("Gaussian mirrors\nvalid, but $43\\%$ of the power",
               xy=(float(gm["ctrl"]["gm_power"]), float(gm["ctrl"]["gm_fdp"])),
               xytext=(0.42, 0.075), fontsize=6.8, color=C_ALT,
               arrowprops=dict(arrowstyle="->", lw=0.7, color=C_ALT))
    a.scatter([float(gm["ctrl"]["ko_power"])], [float(gm["ctrl"]["ko_fdp"])],
              s=26, color=C_NEUTRAL, marker="s", zorder=3,
              linewidths=0.4, edgecolors="white")
    q_line(a, 0.20, xpos=0.03, ha="left")
    a.set_xlim(0.30, 1.06)
    a.set_ylim(-0.012, 0.235)
    a.set_xlabel("per-signal power (known injected signal, $k=12$)")
    a.set_ylabel("realised FDP")
    a.grid(True)
    a.set_title("on this control the constructions hold;\n"
                "leaving the box costs power", fontsize=8.0)
    panel(a, "a")

    # (b) 从属：mirror 方法绕开 Sigma^-1，但每个变量仍被其余变量解释 90%
    b = ax[1]
    v = g(mech, "r2_on_others")
    b.hist(v, bins=18, color=C_ALT, alpha=0.75, edgecolor="white", linewidth=0.5)
    md = float(np.median(v))
    b.axvline(md, color="#333333", ls="--", lw=1.1)
    b.text(md - 0.012, b.get_ylim()[1] * 0.94, f"median $={md:.3f}$",
           fontsize=6.8, ha="right", va="top")
    b.set_xlabel("$R^2$ on the remaining $112$", fontsize=7.5)
    b.set_ylabel(f"ROIs ($n={len(v)}$)", fontsize=7.5)
    b.grid(True, axis="y")
    b.set_title("the mirror method still\nfaces collinearity", fontsize=8.4)
    panel(b, "b")

    save(fig, "fig_machinery")
    dump_source("fig_machinery",
                [{"panel": "a", "arm": r["method"], "power": float(r["power"]),
                  "fdp": float(r["fdp"])} for r in cc]
                + [{"panel": "a", "arm": "gaussian_mirrors",
                    "power": float(gm["ctrl"]["gm_power"]),
                    "fdp": float(gm["ctrl"]["gm_fdp"])},
                   {"panel": "a", "arm": "knockoff_equicorrelated",
                    "power": float(gm["ctrl"]["ko_power"]),
                    "fdp": float(gm["ctrl"]["ko_fdp"])}]
                + [{"panel": "b", "arm": "gm_mechanism",
                    "power": float(x), "fdp": ""} for x in v])


# --------------------------------------------------------------------------- #
# 6. 矩阵幂族：自报误差 vs 真实误差
# --------------------------------------------------------------------------- #
def fig_matrixpower():
    ctrl = load("results_matrixpow_step02/matrixpow_control.csv")
    real = load("results_matrixpow_step02/matrixpow_real.csv")
    null = load("results_matrixpow_step02/matrixpow_globalnull.csv")
    t = g(ctrl, "t")

    fig, ax = plt.subplots(1, 3, figsize=(FIGW, 2.9),
                           gridspec_kw={"width_ratios": [1.45, 1.0, 0.72]}, layout="constrained")

    def plab2(ax, s):
        panel(ax, s)

    a = ax[0]
    a.plot(t, g(ctrl, "fdp"), marker=MARKERS[0], ls=LINESTYLES[0], color=C_VIOL,
           lw=1.7, ms=4, label="true FDP (known truth)")
    a.plot(t, g(real, "ratio_median"), marker=MARKERS[1], ls=LINESTYLES[1],
           color=C_VALID, lw=1.7, ms=4, label="procedure's own report")
    q_line(a, 0.20, xpos=0.03, ha="left")
    # 标注两个极值（与图注的 0.826 / 0.091 对齐），而不是端点
    fdp = g(ctrl, "fdp")
    rat = g(real, "ratio_median")
    i_pk, i_vl = int(np.argmax(fdp)), int(np.argmin(rat))
    a.annotate(f"max {fdp[i_pk]:.2f}", (t[i_pk], fdp[i_pk]),
               textcoords="offset points", xytext=(0, 8), fontsize=6.8,
               color=C_VIOL, ha="center")
    a.annotate(f"min {rat[i_vl]:.2f}", (t[i_vl], rat[i_vl]),
               textcoords="offset points", xytext=(0, -12), fontsize=6.8,
               color=C_VALID, ha="center")
    a.set_xlabel("matrix-power order $t$", fontsize=7.5)
    a.set_ylabel("proportion", fontsize=7.5)
    a.set_ylim(0, 1.06)
    a.invert_xaxis()
    a.set_xticks(np.arange(-1.0, 1.01, 0.5))
    a.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.1f}".replace("-0.0", "0.0")))
    a.grid(True)
    # 图例放右下：左上留给 max 标注，右上留给曲线
    a.legend(loc="upper left", fontsize=6.0, borderpad=0.2)
    a.set_title("reported and true error\nmove apart", fontsize=7.8, pad=5)
    plab2(a, "a")

    b = ax[1]
    b.plot(t, g(null, "p_nonempty"), marker=MARKERS[2], ls=LINESTYLES[0],
           color=C_ALT, lw=1.7, ms=4)
    q_line(b, 0.20, xpos=0.05, ha="left")
    b.fill_between([-0.4, -0.2], 0, 0.36, color=C_ALT, alpha=0.10)
    b.text(-0.30, 0.345, "violation\nregion", fontsize=6.4, color=C_ALT,
           ha="center", va="top")
    b.set_xlabel("matrix-power order $t$", fontsize=7.5)
    b.set_ylabel("$P(\\hat S\\!\\neq\\!\\emptyset)=\\mathbb{E}[$FDP$]$",
                 fontsize=7.5)
    b.set_ylim(0, 0.40)
    b.invert_xaxis()
    b.set_xticks(np.arange(-1.0, 1.01, 0.5))
    b.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.1f}".replace("-0.0", "0.0")))
    b.grid(True)
    b.set_title("global null: the\nguarantee breaks", fontsize=7.8, pad=5)
    plab2(b, "b")

    c = ax[2]
    c.plot(t, g(ctrl, "eff_rank"), marker=MARKERS[3], ls=LINESTYLES[0],
           color=C_NEUTRAL, lw=1.7, ms=4)
    c.set_xlabel("matrix-power order $t$", fontsize=7.5)
    c.set_ylabel("effective rank", fontsize=7.5)
    c.invert_xaxis()
    # 窄面板只留三个刻度，否则 -1.0/-0.5/0.0 会互相压住
    c.set_xticks([-1.0, 0.0, 1.0])
    c.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.1f}".replace("-0.0", "0.0")))
    c.grid(True)
    c.set_title("what $t$ does to\nthe geometry", fontsize=7.8, pad=5)
    plab2(c, "c")

    save(fig, "fig_matrixpower")
    dump_source("fig_matrixpower",
                [{"t": float(t[i]), "true_fdp": float(g(ctrl, "fdp")[i]),
                  "reported_ratio": float(g(real, "ratio_median")[i]),
                  "global_null_p_nonempty": float(g(null, "p_nonempty")[i]),
                  "eff_rank": float(g(ctrl, "eff_rank")[i])} for i in range(len(t))])


# --------------------------------------------------------------------------- #
# 7. 队列漂移：功效上升而 FDP 恶化
# --------------------------------------------------------------------------- #
def fig_shift():
    rows = sorted(load("results_c0.05/cohort_shift.csv"),
                  key=lambda r: float(r["delta"]))
    d = g(rows, "delta")
    fdp, pow_, mism = g(rows, "fdp"), g(rows, "power"), g(rows, "mismatch")
    real_mis = float(rows[0]["real_mismatch_reference"])

    fig, ax = plt.subplots(2, 1, figsize=(FIGW, 3.6), sharex=True,
                           layout="constrained")

    a = ax[0]
    a.plot(d, fdp, color=C_VIOL, marker="o", ms=4, lw=1.7)
    q_line(a, 0.20, xpos=0.97)
    i_last_ok = max(i for i in range(len(d)) if fdp[i] <= 0.20)
    i_first_bad = min(i for i in range(len(d)) if fdp[i] > 0.20)
    a.axvspan(d[i_last_ok], d[i_first_bad], color=C_VIOL, alpha=0.10, lw=0)
    a.annotate("control lost here", xy=((d[i_last_ok] + d[i_first_bad]) / 2, 0.30),
               xytext=((d[i_last_ok] + d[i_first_bad]) / 2 - 0.02, 0.335),
               fontsize=6.8, color=C_VIOL, ha="center",
               arrowprops=dict(arrowstyle="->", lw=0.7, color=C_VIOL))
    a.axvline(0.10, color="#555555", ls=":", lw=1.0)
    a.annotate("ABIDE I mismatch", xy=(0.10, 1.0),
               xycoords=("data", "axes fraction"), xytext=(3, -3),
               textcoords="offset points", fontsize=6.4, color="#555555",
               ha="left", va="top")
    a.set_ylabel("realised FDP")
    a.set_ylim(0.12, 0.46)
    a.grid(True)
    a.set_title("cohort shift breaks validity while raising power", fontsize=8.6)
    panel(a, "a")

    b = ax[1]
    b.plot(d, pow_, color=C_VALID, marker="s", ms=4, lw=1.7, ls="--")
    b.axvline(0.10, color="#555555", ls=":", lw=1.0)
    b.annotate(f"${pow_[0]:.3f}\\to{pow_[-1]:.3f}$", xy=(0.42, 0.905),
               fontsize=7.0, color=C_VALID)
    b.text(0.40, 0.845, "the quantity a practitioner monitors",
           fontsize=6.6, color=C_VALID, style="italic")
    b.set_ylabel("per-signal power")
    b.set_xlabel("injected cohort shift $\\delta$")
    b.set_ylim(0.83, 1.04)
    b.grid(True)
    panel(b, "b")

    save(fig, "fig_shift")
    dump_source("fig_shift",
                [{"delta": float(d[i]), "mismatch": float(mism[i]),
                  "fdp": float(fdp[i]), "power": float(pow_[i]),
                  "real_mismatch_reference": real_mis} for i in range(len(d))])


# --------------------------------------------------------------------------- #
# 8. 罚常数的双向约束
# --------------------------------------------------------------------------- #
def fig_penalty():
    sw = sorted(load("results_c0.05/penalty_sweep.csv"), key=lambda r: float(r["c"]))
    cv = load("results_c0.05/penalty_convergence_real.csv")
    c_sw = g(sw, "c")
    rec = g(sw, "frac_recovered_12of12")
    sigmas = []
    for r in cv:
        if r["Sigma"] not in sigmas:
            sigmas.append(r["Sigma"])
    sig_col = {"empirical": C_MEAN, "LW": C_GSR, "in_sample": C_NEUTRAL}
    conv_min = {x: max(float(r["frac_nonconverged"]) for r in cv
                       if abs(float(r["c"]) - x) < 1e-12) for x in c_sw}

    fig, ax = plt.subplots(3, 1, figsize=(FIGW, 4.6),
                           gridspec_kw={"height_ratios": [1, 1, 0.30]},
                           layout="constrained")

    xt = [0.02, 0.03, 0.05, 0.1, 0.15, 0.3]
    xt_lab = ["0.02", "0.03", "0.05", "0.10", "0.15", "0.30"]

    def logx(a):
        a.set_xscale("log")
        a.set_xlim(0.017, 0.36)
        a.set_xticks(xt)
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ""))
        a.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        a.tick_params(axis="x", which="both", length=0)
        a.grid(True)

    a = ax[0]
    mean_frac = g(sw, "mean_recovered") / 12.0        # 平均恢复的信号占比
    a.plot(c_sw, mean_frac, color=C_VALID, marker="o", ms=4, lw=1.7,
           label="mean fraction of the $12$ recovered")
    a.plot(c_sw, rec, color=C_VALID, marker="s", ms=3.6, lw=1.4, ls="--",
           label="runs recovering all $12$")
    for x, y in zip(c_sw, mean_frac):
        a.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                   xytext=(0, 6), fontsize=6.6, ha="center", color=C_VALID)
    a.set_ylim(-0.06, 1.34)
    a.set_ylabel("fraction recovered", fontsize=7.5)
    logx(a)
    a.legend(loc="lower left", fontsize=6.0, borderpad=0.2)
    a.set_title("condition 1: the penalty must not destroy the signal",
                fontsize=8.2, pad=5)

    b = ax[1]
    for s in sigmas:
        sub = sorted([r for r in cv if r["Sigma"] == s], key=lambda r: float(r["c"]))
        b.plot(g(sub, "c"), g(sub, "frac_nonconverged"), color=sig_col[s],
               marker=MARKERS[sigmas.index(s)], ms=4, lw=1.6,
               ls=LINESTYLES[sigmas.index(s)], label=s)
    b.set_ylim(-0.05, 1.16)
    b.set_ylabel("fraction of fits where\nthe solver did not converge", fontsize=7.5)
    logx(b)
    b.legend(loc="upper right", fontsize=6.4, ncol=3, columnspacing=0.8)
    b.annotate("at $c=0.02$ almost every\nfit fails to converge", xy=(0.0203, 0.99),
               xytext=(0.033, 0.60), fontsize=6.8, color="#333333",
               arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))
    b.set_title("condition 2: and it must let the solver converge",
                fontsize=8.2, pad=5)

    # (c) 判定带：只有 c=0.05 两格全过。用 marker 网格，坐标系统比矩形更稳
    c = ax[2]
    for sp in c.spines.values():
        sp.set_visible(False)
    c.tick_params(axis="y", length=0, labelleft=False)
    c.set_xlim(-0.62, len(c_sw) - 0.38)
    c.set_ylim(-0.75, 2.75)   # 留出上方空间，斜体结论句不再横穿 Y/N 方块
    for i, x in enumerate(c_sw):
        for yv, ok in ((1, mean_frac[i] >= 0.90), (0, conv_min[x] == 0.0)):
            c.scatter([i], [yv], marker="s", s=95,
                      color=C_VALID if ok else C_NEUTRAL_L,
                      alpha=0.95 if ok else 0.5, linewidths=0)
            c.text(i, yv, "Y" if ok else "N", ha="center", va="center",
                   fontsize=6.4, color="white" if ok else "#4a4a4a",
                   fontweight="bold")
    c.text(-0.72, 1, "recovery", ha="right", va="center", fontsize=7.0)
    c.text(-0.72, 0, "convergence", ha="right", va="center", fontsize=7.0)
    c.text(-0.72, 1.78, "condition", ha="right", va="center", fontsize=6.4,
           color="#777777")
    c.set_xticks(range(len(c_sw)))
    c.set_xticklabels(xt_lab, fontsize=7.0)
    c.tick_params(axis="x", length=0, pad=1)
    c.set_xlabel("penalty constant $c$", fontsize=7.0, labelpad=1)
    c.text(-0.72, 2.55, "only $c=0.05$ satisfies both conditions; "
           "the software reports neither", ha="left", va="top",
           fontsize=7.2, style="italic")
    panel(a, "a")
    panel(b, "b")
    panel(c, "c")

    save(fig, "fig_penalty")
    # 面板 (b) 画的是三条 Sigma 曲线，只 dump 每档的最大值无法复现它
    dump_source("fig_penalty",
                [{"panel": "a,c", "c": float(x), "sigma": "",
                  "mean_frac_recovered": float(mean_frac[i]),
                  "frac_recovered_12of12": float(rec[i]),
                  "frac_nonconverged": "", "max_frac_nonconverged": float(conv_min[x])}
                 for i, x in enumerate(c_sw)]
                + [{"panel": "b", "c": float(r["c"]), "sigma": r["Sigma"],
                    "mean_frac_recovered": "", "frac_recovered_12of12": "",
                    "frac_nonconverged": float(r["frac_nonconverged"]),
                    "max_frac_nonconverged": ""} for r in cv])


# --------------------------------------------------------------------------- #
# 9. 真实数据上的输出是分布（重做：消灭 45 度标签）
# --------------------------------------------------------------------------- #
def fig_draws():
    rows = sorted(load("results_c0.05/draw_distribution.csv"),
                  key=lambda r: -float(r["p_nonempty"]))
    lab = [f"{compact_ds(r['dataset'])} {r['feature']}" for r in rows]
    med = g(rows, "nsel_median")
    p95 = g(rows, "nsel_p95")
    mx = g(rows, "nsel_max")
    pne = g(rows, "p_nonempty")
    conf = g(rows, "frac_converged")
    n = len(rows)

    fig, ax = plt.subplots(1, 2, figsize=(FIGW, 4.8), sharey=True,
                           gridspec_kw={"width_ratios": [1.75, 1.0]}, layout="constrained")
    xmax = float(mx.max()) + 3

    # 灰度安全：四个 summary 除颜色外再各配一种 marker 形状
    # （meanconn 与 gsr 的 Rec.601 亮度只差 1，仅靠颜色在灰度下不可分）
    msum = {"meanconn": "o", "gsr": "s", "pcresid": "^", "partial": "D"}

    a = ax[0]
    for i in range(n):
        feat = rows[i]["feature"]
        col = SUMMARY_COLOR[feat]
        mk = msum[feat]
        cc = C_NEUTRAL if conf[i] < 0.90 else col
        a.plot([0, max(p95[i], med[i])], [i, i], color=cc, lw=0.9,
               alpha=0.55, zorder=2)
        a.plot([med[i]], [i], marker=mk, ms=4.0, color=cc, zorder=3)
        a.plot([mx[i]], [i], marker=mk, ms=4.0, mfc="white", mec=cc,
               mew=1.0, zorder=3)
    a.set_yticks(np.arange(n))
    a.set_yticklabels(lab, fontsize=6.3)
    a.set_ylim(-2.1, n - 0.4)
    a.invert_yaxis()
    a.set_xlim(-1.4, xmax)
    a.set_xticks([0, 10, 20, 30, 40])
    a.set_xlabel("regions selected over $200$ independent draws",
                 fontsize=7.5, labelpad=2)
    a.grid(True, axis="x")
    a.text(0.985, 0.015, "line  median$\\to$95th pct     "
           "filled  median     open  maximum",
           transform=a.transAxes, ha="right", va="bottom", fontsize=6.2,
           color="#555555")
    panel(a, "a")

    i_bad = int(np.argmax(conf < 0.90))
    conv_txt = f"{np.floor(conf[i_bad] * 100 + 0.5) / 100:.2f}"   # 与正文同为 0.37
    a.annotate(f"converged in only {conv_txt} of draws --- excluded",
               xy=(float(p95[i_bad]) + 0.5, i_bad), xytext=(6.0, i_bad + 3.1),
               fontsize=6.4, color=C_NEUTRAL,
               arrowprops=dict(arrowstyle="->", lw=0.7, color=C_NEUTRAL))
    a.annotate(f"{pne[0]:.3f} of draws return selections,\n"
               f"median {med[0]:.0f}, up to {mx[0]:.0f} regions",
               xy=(float(mx[0]) - 0.8, 0.35), xytext=(15.0, 12.5),
               fontsize=6.4, color="#333333",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                         pad=1.4),
               arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))

    b = ax[1]
    for i in range(n):
        feat = rows[i]["feature"]
        cc = C_NEUTRAL if conf[i] < 0.90 else SUMMARY_COLOR[feat]
        b.plot([0, pne[i]], [i, i], color=cc, lw=1.1, alpha=0.6, zorder=2)
        b.plot([pne[i]], [i], marker=msum[feat], ms=4.0, color=cc, zorder=3)
    b.set_xlim(-0.03, 1.02)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_xlabel("fraction of draws\nreturning a nonempty selection",
                 fontsize=7.5, labelpad=2)
    b.grid(True, axis="x")
    panel(b, "b")
    i_last = n - 1
    b.annotate("nothing in any\nof the $200$ draws",
               xy=(0.005, i_last), xytext=(0.17, i_last - 2.6),
               fontsize=6.4, color="#333333",
               arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))

    save(fig, "fig_draws")
    dump_source("fig_draws",
                [{"dataset": r["dataset"], "feature": r["feature"],
                  "draws": int(float(r["draws"])),
                  "p_nonempty": float(pne[i]),
                  "nsel_median": float(med[i]), "nsel_p95": float(p95[i]),
                  "nsel_max": float(mx[i]),
                  "frac_converged": float(conf[i])}
                 for i, r in enumerate(rows)])


# --------------------------------------------------------------------------- #
# 10. 第三队列
# --------------------------------------------------------------------------- #
def fig_third():
    tc = load("results_v3/third_cohort.csv")
    mm = load("results_v3/cohort_mismatch.csv")
    cohorts = []
    for r in tc:
        if r["cohort"] not in cohorts:
            cohorts.append(r["cohort"])
    ccol = {"ABIDE1-AAL116": C_VALID, "ABIDE2-Power2011": C_VIOL,
            "ADHD200-AAL116": C_ALT}
    cmk = {"ABIDE1-AAL116": "o", "ABIDE2-Power2011": "s", "ADHD200-AAL116": "^"}

    fig, ax = plt.subplots(1, 2, figsize=(FIGW, 3.1),
                           gridspec_kw={}, layout="constrained")

    a = ax[0]
    for i, c in enumerate(cohorts):
        sub = sorted([r for r in tc if r["cohort"] == c],
                     key=lambda r: float(r["n_ref"]))
        a.errorbar(g(sub, "n_ref"), g(sub, "effrank_mean"),
                   yerr=g(sub, "effrank_sd"), color=ccol[c],
                   marker=cmk[c], ms=4, lw=1.6, ls=LINESTYLES[i],
                   capsize=2, elinewidth=0.8, label=c.split("-")[0])
        a.annotate(f"{g(sub, 'effrank_mean')[-1]:.1f}", (g(sub, "n_ref")[-1],
                   g(sub, "effrank_mean")[-1]), textcoords="offset points",
                   xytext=(4, 0), fontsize=6.6, color=ccol[c], va="center")
    a.axvline(121, color="#555555", ls=":", lw=1.0)
    a.text(124, 1.4, "matched $n_{\\mathrm{ref}}$", fontsize=7.4, color="#555555")
    a.set_xlabel("reference-cohort size $n_{\\mathrm{ref}}$")
    a.set_ylabel("effective rank of the\nconventional summary")
    a.set_ylim(0, 19)
    a.grid(True)
    a.legend(loc="upper left", fontsize=6.4)
    a.set_title("the ordering replicates,\nthe severity does not", fontsize=8.4)
    panel(a, "a")

    b = ax[1]
    for i, r in enumerate(mm):
        c = r["cohort"]
        nf, ob = float(r["noise_floor"]), float(r["mismatch"])
        b.plot([nf, ob], [i, i], color="#bbbbbb", lw=1.4, zorder=1)
        b.plot([nf], [i], marker="o", ms=5, mfc="white", mec=C_NEUTRAL,
               mew=1.1, zorder=3)
        b.plot([ob], [i], marker="o", ms=5.4, color=ccol[c], zorder=3)
        b.text(ob + 0.008, i, f"$\\times${float(r['ratio']):.1f}", fontsize=6.8,
               va="center", color=ccol[c])
    b.axvspan(0.12, 0.19, color=C_VIOL, alpha=0.08, lw=0)
    b.text(0.155, -0.62, "simulated violation region", fontsize=6.4,
           color=C_VIOL, ha="center")
    b.set_yticks(range(len(mm)))
    b.set_yticklabels(["ABIDE I", "ABIDE II", "ADHD-200"], fontsize=7.2)
    b.set_ylim(-0.85, len(mm) - 0.4)
    b.invert_yaxis()
    b.set_xlim(0, 0.37)
    b.set_xlabel("main/reference mismatch", fontsize=7.5)
    b.text(0.02, 0.03, "open: sampling-noise floor\nfilled: observed",
           transform=b.transAxes, fontsize=6.2, color="#555555",
           va="bottom", ha="left")
    b.grid(True, axis="x")
    b.set_title("the shift boundary holds\nout of sample", fontsize=8.4)
    panel(b, "b")

    save(fig, "fig_third")
    dump_source("fig_third",
                [{"panel": "a", "cohort": r["cohort"],
                  "n_ref": float(r["n_ref"]), "effrank_mean": float(r["effrank_mean"]),
                  "effrank_sd": float(r["effrank_sd"])} for r in tc]
                + [{"panel": "b", "cohort": r["cohort"],
                    "mismatch": float(r["mismatch"]),
                    "noise_floor": float(r["noise_floor"]),
                    "ratio": float(r["ratio"])} for r in mm])


# --------------------------------------------------------------------------- #
FIGURES = [
    ("fig_flow", fig_flow),
    ("fig_geometry", fig_geometry),
    ("fig_mechanism", fig_mechanism),
    ("fig_limits", fig_limits),
    ("fig_machinery", fig_machinery),
    ("fig_matrixpower", fig_matrixpower),
    ("fig_shift", fig_shift),
    ("fig_penalty", fig_penalty),
    ("fig_draws", fig_draws),
    ("fig_third", fig_third),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated stems")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, _ in FIGURES:
            print(name)
        return

    want = [s.strip() for s in args.only.split(",") if s.strip()]
    todo = [(n, f) for n, f in FIGURES if not want or n in want]
    unknown = set(want) - {n for n, _ in FIGURES}
    if unknown:
        raise SystemExit(f"unknown figure(s): {sorted(unknown)}")

    print(f"repo root : {os.path.abspath('.')}")
    print(f"output    : {os.path.abspath(OUT)}")
    print(f"source    : {os.path.abspath(SRC)}")
    print(f"figures   : {len(todo)} of {len(FIGURES)}\n")
    for name, fn in todo:
        print(f"[{name}]")
        fn()
    print(f"\ndone: {len(todo)} figure(s)")


if __name__ == "__main__":
    main()
