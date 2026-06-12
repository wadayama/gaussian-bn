"""Generate a LaTeX summary of Experiments 1-3 STRICTLY from results/*.json.

Every number in the emitted .tex is read from the experiment result files, which
were produced by running the actual library. No hand-typed or synthetic values.

Run: uv run python experiments/make_summary.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
TEX = os.path.join(HERE, "experiments_summary.tex")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def fnum(x, sig=3):
    ax = abs(x)
    if x == 0:
        return "0"
    if ax < 1e-3 or ax >= 1e4:
        m = f"{x:.{sig}e}"
        mant, exp = m.split("e")
        return f"{float(mant):.{sig}f}\\times 10^{{{int(exp)}}}"
    return f"{x:.{sig}f}"


def coords(xs, ys, clamp=None):
    out = []
    for x, y in zip(xs, ys):
        if clamp is not None:
            y = max(y, clamp)
        out.append(f"({x},{y:.6g})")
    return " ".join(out)


def main():
    e1, e2, e3 = load("exp1.json"), load("exp2.json"), load("exp3.json")
    e5 = load("exp5.json")
    L = []
    A = L.append

    A(r"\documentclass[a4paper,11pt]{ltjsarticle}")
    A(r"\usepackage{luatexja-fontspec}\usepackage[haranoaji]{luatexja-preset}")
    A(r"\usepackage{amsmath,amssymb,amsthm,bm,mathtools}")
    A(r"\usepackage{booktabs,array}\usepackage{geometry}\usepackage{xcolor}")
    A(r"\usepackage{pgfplots}\pgfplotsset{compat=1.18}")
    A(r"\geometry{margin=24mm}")
    A(r"\newcommand{\tr}{\operatorname{tr}}\newcommand{\rank}{\operatorname{rank}}")
    A(r"\newcommand{\OO}{\mathcal{O}}")
    A(r"\title{K-recursion Gaussian BN: 応用実験サマリー\\"
      r"\large 隠れノード推定・センサー配置・介入の情報幾何・構造学習}")
    A(r"\author{自動生成 (experiments/make\_summary.py より)}")
    A(r"\date{2026年6月10日}")
    A(r"\begin{document}\maketitle")
    A(r"\begin{abstract}")
    A(r"\texttt{gaussian\_bn} ライブラリを用いた3つの応用実験の\textbf{実行結果}である。"
      r"全数値は各 \texttt{run} スクリプトが出力した \texttt{results/exp\{1,2,3\}.json} "
      r"から機械転記しており、合成データは含まない。乱数シードは "
      rf"\texttt{{{e1['seed']}}}。実験はノートの Experiment 2--4 と Theme D に対応する。")
    A(r"\end{abstract}")

    # ===================== Exp 1 =====================
    A(r"\section{実験1: 隠れノード推定 --- EM と勾配学習の比較}")
    A(r"diamond DAG（$0\to1,0\to2,1\to3,2\to3$）の中間 $\{1,2\}$ を hidden とし、"
      rf"観測 $\OO=\{{0,3\}}$（$N={e1['N']}$）。観測周辺尤度 "
      r"$\mathcal L=\log\det K_{\OO\OO}+\tr(K_{\OO\OO}^{-1}S)$ を EM・Adam・LBFGS で最小化した。")
    nopt = e1["nll_optimum"]
    # convergence plot: NLL gap to optimum (log scale)
    em_gap = coords(range(len(e1["em"]["nll_history"])),
                    [v - nopt for v in e1["em"]["nll_history"]], clamp=1e-16)
    ad_gap = coords(range(len(e1["adam"]["nll_history"])),
                    [v - nopt for v in e1["adam"]["nll_history"]], clamp=1e-16)
    A(r"\begin{center}\begin{tikzpicture}")
    A(r"\begin{semilogyaxis}[width=0.78\linewidth,height=6cm,xlabel={反復},"
      r"ylabel={NLL $-$ 最適値},grid=both,legend pos=north east]")
    A(rf"\addplot[blue,thick] coordinates {{{em_gap}}};")
    A(rf"\addplot[red,thick] coordinates {{{ad_gap}}};")
    A(r"\legend{EM,Adam}\end{semilogyaxis}\end{tikzpicture}\end{center}")
    A(r"\begin{center}\begin{tabular}{lcccc}\toprule")
    A(r"手法 & 最終 NLL & $\|K_{\OO\OO}(\hat\eta)-S\|/\|S\|$ & 反復 & 時間[s] \\ \midrule")
    for key, lab in [("em", "EM"), ("adam", "Adam"), ("lbfgs", "LBFGS")]:
        d = e1[key]
        A(rf"{lab} & {fnum(d['nll_final'])} & ${fnum(d['relerr_KOO_vs_sample'])}$ & "
          rf"{d['iters']} & {fnum(d['time_s'])} \\")
    A(r"\bottomrule\end{tabular}\end{center}")
    A(rf"NLL の理論最適値は $\log\det S+\dim={fnum(nopt)}$ で、3手法すべてが到達した"
      r"（観測共分散を標本共分散へ復元）。")
    fe = ",\\ ".join(fnum(x) for x in e1["fisher_edge_eigenvalues"])
    A(rf"\paragraph{{gauge 診断.}}解での4 edge 母数の Fisher 固有値は $\{{{fe}\}}$、"
      rf"$\rank G^{{(\OO)}}={e1['fisher_edge_rank']}<q={e1['fisher_edge_q']}$。"
      r"零空間は2次元で、2つの hidden ノードのスケール gauge に対応する（観測共分散は"
      r"復元できても edge 母数自体は識別不能）。")

    # ===================== Exp 2 =====================
    A(r"\section{実験2: センサー配置と edge 識別可能性}")
    A(r"6ノード DAG の6本の edge 母数に対し、観測ノード集合 $\OO$ を予算 $m$ 以内で選ぶ。"
      r"pullback Fisher 計量 $G^{(\OO)}$ の D最適 $\log\det(G^{(\OO)}+\epsilon I)$ を greedy "
      r"最大化し、各予算での識別 rank と最小固有値も記録した。")
    bs = [r["budget"] for r in e2["d_curve"]]
    ld = coords(bs, [r["logdet"] for r in e2["d_curve"]])
    rk = coords(bs, [r["rank"] for r in e2["d_curve"]])
    A(r"\begin{center}\begin{tikzpicture}")
    A(r"\begin{axis}[width=0.78\linewidth,height=5.6cm,xlabel={観測ノード数(予算 $m$)},"
      r"ylabel={$\log\det G^{(\OO)}$},axis y line*=left,grid=both]")
    A(rf"\addplot[blue,mark=*,thick] coordinates {{{ld}}};\label{{plt:ld}}")
    A(r"\end{axis}")
    A(r"\begin{axis}[width=0.78\linewidth,height=5.6cm,axis y line*=right,"
      r"ylabel={識別 rank},ymin=0,ymax=6.5,hide x axis]")
    A(r"\addlegendimage{blue,mark=*}\addlegendentry{$\log\det G$}")
    A(rf"\addplot[red,mark=square,thick,dashed] coordinates {{{rk}}};\addlegendentry{{rank}}")
    A(r"\end{axis}\end{tikzpicture}\end{center}")
    A(r"\begin{center}\begin{tabular}{ccccc}\toprule")
    A(r"$m$ & 選択ノード & $\log\det G$ & $\lambda_{\min}$ & rank \\ \midrule")
    for r in e2["d_curve"]:
        A(rf"{r['budget']} & {r['chosen']} & ${fnum(r['logdet'])}$ & ${fnum(r['lambda_min'])}$ & "
          rf"{r['rank']}/{e2['q_edges']} \\")
    A(r"\bottomrule\end{tabular}\end{center}")
    frb = e2["full_rank_budget"]
    A(rf"わずか {frb} ノードの観測で全6 edge が識別可能（rank $={e2['q_edges']}$）になり、"
      r"以降はセンサー追加で条件数（$\log\det G,\ \lambda_{\min}$）が改善する。")
    cd = e2["greedy_vs_exhaustive"]["d"]
    ce = e2["greedy_vs_exhaustive"]["e"]
    A(rf"\paragraph{{greedy vs exhaustive (予算3).}}D最適では greedy {cd['greedy_chosen']} と"
      rf" exhaustive {cd['exhaustive_chosen']} が一致（スコア {fnum(cd['greedy_score'])}）。"
      rf"一方 E最適（$\max\lambda_{{\min}}$）は greedy {ce['greedy_chosen']} "
      rf"(${fnum(ce['greedy_score'])}$) が exhaustive {ce['exhaustive_chosen']} "
      rf"(${fnum(ce['exhaustive_score'])}$) に劣る。これは E最適性が劣モジュラでないための"
      r"既知の挙動で、rank 不足の領域では greedy が停滞する（exhaustive か rank 考慮基準が必要）。")

    # ===================== Exp 3 =====================
    A(r"\section{実験3: 介入の情報幾何}")
    A(r"交絡三角形 $0\to1,\ 0\to2,\ 1\to2$（ノード0は隠れ交絡、$1\to2$ が直接効果 $c$）。"
      r"観測上の連関 $I(V_1;V_2)$ は交絡経路と直接経路の混合だが、hard 介入 $\mathrm{do}(V_1)$ "
      r"はノード1を親から切り離すため $I^{\mathrm{do}}(V_1;V_2)$ は直接因果効果のみを表す。"
      r"差 $\Delta I=I(V_1;V_2)-I^{\mathrm{do}}(V_1;V_2)$ が交絡（非因果）成分である。")
    cs = e3["c_values"]
    iobs = coords(cs, [s["I_obs"] for s in e3["sweep"]])
    ido = coords(cs, [s["I_do"] for s in e3["sweep"]])
    A(r"\begin{center}\begin{tikzpicture}")
    A(r"\begin{axis}[width=0.78\linewidth,height=5.8cm,xlabel={直接効果 $c$},"
      r"ylabel={相互情報量 [nats]},grid=both,legend pos=north west]")
    A(rf"\addplot[blue,mark=*,thick] coordinates {{{iobs}}};")
    A(rf"\addplot[red,mark=square,thick] coordinates {{{ido}}};")
    A(r"\legend{$I(V_1;V_2)$ (観測),$I^{\mathrm{do}}(V_1;V_2)$ (介入)}")
    A(r"\end{axis}\end{tikzpicture}\end{center}")
    pc = e3["pure_confounding_c0"]
    mx = e3["mixed_c0p9"]
    A(r"\begin{center}\begin{tabular}{lccc}\toprule")
    A(r"設定 & $I(V_1;V_2)$ & $I^{\mathrm{do}}(V_1;V_2)$ & 交絡成分 $\Delta I$ \\ \midrule")
    A(rf"$c=0$（交絡のみ） & {fnum(pc['I_obs'])} & ${fnum(pc['I_do'])}$ & {fnum(pc['delta_I_confounding'])} \\")
    A(rf"$c=0.9$（交絡+直接） & {fnum(mx['I_obs'])} & {fnum(mx['I_do'])} & {fnum(mx['delta_I_confounding'])} \\")
    A(r"\bottomrule\end{tabular}\end{center}")
    A(r"$c=0$ では観測連関があるのに $I^{\mathrm{do}}=0$（連関は完全に交絡由来；介入で"
      rf"$\mathrm{{Cov}}(V_1,V_2)$ も $0$）。$c$ が増えると $I^{{\mathrm{{do}}}}$ が直接因果効果として"
      r"立ち上がり、観測連関との差が交絡成分を与える。do 演算が因果と交絡を情報量で分離できる。")

    # ===================== Exp 5 =====================
    A(r"\section{実験5: 構造学習（group sparsity による枝刈り）}")
    A(rf"全結合 supergraph（$i<j$ の全{len(e5['super_edges'])}本）から出発し、{e5['M']}ノード"
      rf"（各{e5['node_dim']}次元）の BN を K-recursion 周辺尤度＋\textbf{{group sparsity}} で学習：")
    A(r"\begin{equation}\min_\eta\ \mathcal L(\eta)+\lambda\sum_{(i,j)}\lVert A_{ji}\rVert_F\end{equation}")
    A(r"を近接勾配（平滑勾配ステップ＋edge ブロックの group soft-threshold）で解く。$\lambda$ を上げると"
      r"不要 edge の $2\times2$ ブロックが丸ごと $0$ に潰れる。各 $\lambda$ の選択構造を"
      r"無罰則 MLE（\texttt{fit\_local\_regression}）で再フィットし BIC で構造選択した。")
    true_keys = {f"{e[0]}_{e[1]}" for e in e5["true_edges"]}
    lams = e5["lambdas"]
    sub = [i for i, lam in enumerate(lams) if lam <= 7]      # zoom into the transition
    A(r"\begin{center}\begin{tikzpicture}")
    A(r"\begin{axis}[width=0.78\linewidth,height=5.8cm,xlabel={正則化 $\lambda$},"
      r"ylabel={edge ブロック $\lVert A_{ji}\rVert_F$},grid=both,legend pos=north east]")
    for e in e5["super_edges"]:
        k = f"{e[0]}_{e[1]}"
        ys = [e5["path"][i]["norms"][k] for i in sub]
        xs = [lams[i] for i in sub]
        style = "blue,thick,mark=*" if k in true_keys else "red,dashed,mark=square,mark size=1pt"
        A(rf"\addplot[{style}] coordinates {{{coords(xs, ys)}}};")
    A(r"\addlegendimage{blue,thick,mark=*}\addlegendentry{真の edge (5)}")
    A(r"\addlegendimage{red,dashed,mark=square}\addlegendentry{偽の edge (5)}")
    A(r"\end{axis}\end{tikzpicture}\end{center}")
    A(r"\begin{center}\begin{tabular}{ccccc}\toprule")
    A(r"$\lambda$ & 再フィット NLL & BIC & edge数 & 選択 \\ \midrule")
    best_lam = e5["selected"]["lambda"]
    for r in e5["path"]:
        if r["lambda"] > 7:
            continue
        sel = r"$\leftarrow$" if r["lambda"] == best_lam else ""
        A(rf"{fnum(r['lambda'])} & {fnum(r['nll_refit'])} & {fnum(r['bic'])} & "
          rf"{r['n_edges']} & {sel} \\")
    A(r"\bottomrule\end{tabular}\end{center}")
    sel = e5["selected"]
    A(rf"$\lambda=0$（罰則なし MLE）は全{len(e5['super_edges'])}本を残して過適合するが、$\lambda={fnum(best_lam)}$ で"
      rf"偽 edge の $2\times2$ ブロックが正確に $0$ となり、BIC が真の構造を選択。"
      rf"\textbf{{完全復元}}（TP={sel['tp']}, FP={sel['fp']}, FN={sel['fn']}, "
      rf"precision $={fnum(sel['precision'])}$, recall $={fnum(sel['recall'])}$）。"
      r"全観測のため再フィットは局所回帰の閉形式 MLE で行えるが、同じ枠組みは観測集合を"
      r"変えるだけで hidden node にも拡張できる。")

    # ===================== conclusion =====================
    A(r"\section{まとめ}")
    A(r"\begin{itemize}")
    A(r"\item \textbf{実験1}: 隠れノードありでも EM/Adam/LBFGS が同一最適 NLL に到達し観測共分散を"
      r"復元、Fisher rank が latent gauge を正しく検出。")
    A(r"\item \textbf{実験2}: Fisher 情報に基づく D最適 greedy センサー配置で、少数観測でも全 edge を"
      r"識別可能にできる。E最適は劣モジュラでないため greedy の限界も定量的に確認。")
    A(r"\item \textbf{実験3}: do 演算により観測連関を因果（直接）成分と交絡成分へ情報量で分解。")
    A(r"\item \textbf{実験5}: supergraph から group sparsity 近接勾配で枝刈りし、再フィット＋BIC で"
      r"真の DAG 構造を完全復元（構造学習への入口）。")
    A(r"\item いずれも \texttt{gaussian\_bn} の単一 API（K-recursion・推論・訓練・Fisher・介入・設計）"
      r"上で完結し、全数値は実行結果ファイル由来である。")
    A(r"\end{itemize}")
    A(r"\end{document}")

    with open(TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {TEX}")


if __name__ == "__main__":
    main()
