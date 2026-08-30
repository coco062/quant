"""
组合分析：美股 + BTC 混合持仓的绩效、风险、相关性、再平衡偏离。

用法:
    python portfolio.py                                  # 用下面 HOLDINGS 的配置
    python portfolio.py --holdings "SPY:0.6,BTC-USD:0.4"
    python portfolio.py --start 2023-01-01 --html        # 额外生成 quantstats HTML 报告

注意: 美股一周 5 个交易日, BTC 7 天都有价。本脚本以【股票交易日】为准对齐,
      BTC 的周末行情被折叠进下一个交易日。这样组合收益率才不会被虚增。
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------- 配置
# 改这里：你的目标权重（会自动归一化，不用凑到 1.0）
HOLDINGS = {
    "SPY": 0.40,
    "QQQ": 0.20,
    "AAPL": 0.15,
    "BTC-USD": 0.25,
}

BENCHMARK = "SPY"          # 对标基准
START = "2023-01-01"       # 回看起点
TRADING_DAYS = 252         # 年化用交易日数
RF = 0.0                   # 无风险利率（年化，0.04 = 4%）
OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------- 数据
def fetch_prices(tickers, start):
    """拉收盘价（后复权），并对齐到股票交易日历。"""
    tickers = list(dict.fromkeys(tickers))  # 去重保序
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.reindex(columns=tickers)

    missing = [t for t in tickers if close[t].notna().sum() == 0]
    if missing:
        raise SystemExit(f"拿不到数据，检查代码是否写错: {missing}")

    # 用非加密标的的交易日做基准日历；全是加密则退回全日历
    equity_cols = [t for t in tickers if not t.endswith("-USD")]
    if equity_cols:
        calendar = close[equity_cols].dropna(how="all").index
        close = close.reindex(calendar)

    close = close.ffill().dropna(how="any")
    if len(close) < 30:
        raise SystemExit(f"重叠的有效交易日只有 {len(close)} 天，样本太短。把 --start 往前调。")
    return close


# ---------------------------------------------------------------- 指标
def metrics(returns, rf=RF):
    """单条收益率序列的核心风险收益指标。"""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}

    cum = (1 + r).prod()
    years = n / TRADING_DAYS
    cagr = cum ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)

    excess = r - rf / TRADING_DAYS
    sharpe = excess.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan

    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = excess.mean() * TRADING_DAYS / downside if downside > 0 else np.nan

    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    max_dd = dd.min()

    return {
        "累计收益": cum - 1,
        "年化收益": cagr,
        "年化波动": vol,
        "夏普": sharpe,
        "索提诺": sortino,
        "最大回撤": max_dd,
        "卡玛": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "日胜率": (r > 0).mean(),
        "最差单日": r.min(),
        "样本天数": n,
    }


def pct(x, digits=2):
    return "—" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def num(x, digits=2):
    return "—" if pd.isna(x) else f"{x:.{digits}f}"


def show_table(title, rows, headers):
    width = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    print(f"\n{title}")
    print("  " + "  ".join(str(h).ljust(width[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("-" * width[i] for i in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(str(c).ljust(width[i]) for i, c in enumerate(row)))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", help='形如 "SPY:0.6,BTC-USD:0.4"')
    ap.add_argument("--start", default=START)
    ap.add_argument("--benchmark", default=BENCHMARK)
    ap.add_argument("--rf", type=float, default=RF, help="无风险年化利率，如 0.04")
    ap.add_argument("--html", action="store_true", help="额外生成 quantstats HTML 报告")
    ap.add_argument("--interactive", action="store_true", help="启动时提示输入持仓（给 .bat 双击用）")
    args = ap.parse_args()

    if args.interactive:
        print("=" * 44)
        print("  组合分析 — 美股 + BTC")
        print("=" * 44)
        print("\n直接回车 = 用脚本内置的默认权重")
        print("或输入持仓，格式:  SPY:0.6,BTC-USD:0.4\n")
        try:
            typed = input("持仓: ").strip()
            if typed:
                args.holdings = typed
            typed_start = input(f"起始日期 (回车默认 {args.start}): ").strip()
            if typed_start:
                args.start = typed_start
        except EOFError:
            pass
        print()

    weights = {}
    if args.holdings and args.holdings.strip():
        for part in args.holdings.split(","):
            if not part.strip():
                continue
            t, sep, w = part.partition(":")
            # 去掉 BOM / 零宽字符 / 全角空格，粘贴进来的文本常带这些
            t = t.strip().strip("﻿​‎‏　").strip().upper()
            if not t:
                continue
            if not sep or not w.strip():
                raise SystemExit(f'持仓格式错误: "{part.strip()}"，应为 代码:权重，例如 SPY:0.6')
            try:
                weights[t] = float(w)
            except ValueError:
                raise SystemExit(f'权重不是数字: "{part.strip()}"，例如 SPY:0.6,BTC-USD:0.4')
    if not weights:
        weights = dict(HOLDINGS)
        print("（未指定持仓，使用脚本内置的默认权重）")

    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("权重之和必须为正数。")
    weights = {k: v / total for k, v in weights.items()}

    tickers = list(weights) + ([args.benchmark] if args.benchmark not in weights else [])
    print(f"拉取 {len(tickers)} 个标的，起点 {args.start} ...")
    prices = fetch_prices(tickers, args.start)

    rets = prices.pct_change(fill_method=None).dropna()
    span = f"{rets.index[0]:%Y-%m-%d} ~ {rets.index[-1]:%Y-%m-%d}"
    print(f"有效区间 {span}，共 {len(rets)} 个交易日")

    w = pd.Series(weights)
    # 买入并持有：权重随行情漂移，这才是不再平衡的真实结果
    growth = (1 + rets[list(weights)]).cumprod()
    port_value = (growth * w).sum(axis=1)
    port_ret = port_value.pct_change(fill_method=None)
    port_ret.iloc[0] = port_value.iloc[0] - 1   # 首日相对期初本金 1.0，不能 dropna 后再补

    # --- 1. 组合 vs 基准 vs 各成分
    rows = []
    combo = {"【组合】": port_ret}
    for t in tickers:
        combo[t] = rets[t]
    for name, series in combo.items():
        m = metrics(series, args.rf)
        rows.append([
            name, pct(m["累计收益"]), pct(m["年化收益"]), pct(m["年化波动"]),
            num(m["夏普"]), num(m["索提诺"]), pct(m["最大回撤"]),
            num(m["卡玛"]), pct(m["日胜率"], 1),
        ])
    show_table(
        f"【绩效对比】{span}   无风险利率 {args.rf * 100:.1f}%",
        rows,
        ["标的", "累计收益", "年化收益", "年化波动", "夏普", "索提诺", "最大回撤", "卡玛", "日胜率"],
    )

    # --- 2. 相关性（分散化到底有没有用）
    corr = rets[tickers].corr()
    show_table(
        "【日收益相关性】越接近 1 越同涨同跌，分散化越无效",
        [[t] + [num(corr.loc[t, c]) for c in tickers] for t in tickers],
        ["  "] + tickers,
    )

    # --- 3. 权重漂移与再平衡
    end_w = (growth.iloc[-1] * w)
    end_w = end_w / end_w.sum()
    rows = []
    for t in weights:
        drift = end_w[t] - w[t]
        flag = "⚠ 超配" if drift > 0.05 else ("⚠ 低配" if drift < -0.05 else "")
        rows.append([t, pct(w[t], 1), pct(end_w[t], 1), f"{drift * 100:+.1f}pp", flag])
    show_table(
        "【权重漂移】期初目标 → 期末实际（未再平衡）",
        rows,
        ["标的", "目标权重", "当前权重", "偏离", "提示"],
    )

    # --- 4. 收益贡献拆解
    contrib = (growth.iloc[-1] - 1) * w
    rows = sorted(
        ([t, pct(w[t], 1), pct(growth.iloc[-1][t] - 1), pct(contrib[t])] for t in weights),
        key=lambda r: float(r[3].rstrip("%")), reverse=True,
    )
    rows.append(["合计", "100.0%", "—", pct(contrib.sum())])
    show_table(
        "【收益贡献】各标的对组合总收益的绝对贡献",
        rows,
        ["标的", "权重", "标的自身收益", "对组合贡献"],
    )

    # --- 5. 最差回撤区间
    curve = (1 + port_ret).cumprod()
    dd = curve / curve.cummax() - 1
    trough = dd.idxmin()
    peak = curve.loc[:trough].idxmax()
    rec = curve.loc[trough:][curve.loc[trough:] >= curve.loc[peak]]
    recovered = f"{rec.index[0]:%Y-%m-%d}（{(rec.index[0] - trough).days} 天）" if len(rec) else "尚未回本"
    print(f"\n【最大回撤区间】{peak:%Y-%m-%d} → {trough:%Y-%m-%d}  {pct(dd.min())}   回本: {recovered}")

    if args.html:
        OUT_DIR.mkdir(exist_ok=True)
        path = OUT_DIR / "report.html"
        try:
            import quantstats as qs
            qs.reports.html(port_ret, benchmark=rets[args.benchmark],
                            output=str(path), title="Portfolio vs " + args.benchmark)
            print(f"\nHTML 报告: {path}")
        except Exception as e:
            print(f"\nHTML 报告生成失败（不影响上面的结果）: {type(e).__name__}: {e}")

    print("\n提示: 以上为历史回看，不含手续费、滑点与税。历史表现不预示未来。")


if __name__ == "__main__":
    main()
