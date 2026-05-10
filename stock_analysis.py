"""
STOCK DATA ANALYSIS — 10 Years
Run this file directly: python stock_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import itertools

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\user\Desktop\DA\processed_stock_data.csv",
                 parse_dates=["date"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
tickers = df["ticker"].unique()
COLORS = ["#00d4ff","#ff6b6b","#ffd93d","#6bcb77","#c77dff",
          "#ff9f43","#48dbfb","#ff6b9d","#a29bfe","#55efc4"]

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor":   "#3a3d4d", "text.color":     "#c8ccd8",
    "xtick.color":      "#8a8d9a", "ytick.color":    "#8a8d9a",
    "axes.labelcolor":  "#c8ccd8", "grid.color":     "#2a2d3a",
    "grid.linestyle":   "--",      "grid.alpha":     0.5,
    "figure.dpi":       110,
})

def show(name):
    plt.savefig(f"{name}.png", bbox_inches="tight")
    plt.show(); plt.close()

print(f"✓ Loaded {len(df):,} rows | {len(tickers)} tickers | "
      f"{df['date'].min().date()} → {df['date'].max().date()}")


# =============================================================================
# SECTION 1 — PERFORMANCE & RETURNS
# =============================================================================
print("\n── SECTION 1: PERFORMANCE ──")

# 1A — Cumulative return curves
fig, ax = plt.subplots(figsize=(14, 5))
for i, tk in enumerate(tickers):
    sub = df[df["ticker"] == tk]
    ax.plot(sub["date"], sub["cum_return"] * 100,
            label=tk, color=COLORS[i % len(COLORS)], lw=1.8)
ax.set_title("Cumulative Return (%) — All Tickers", fontsize=13, color="white")
ax.set_ylabel("Cumulative Return (%)"); ax.legend(ncol=5, fontsize=8); ax.grid(True)
show("1A_cumulative_returns")

# 1B — Performance summary table + bar charts
perf = []
for tk in tickers:
    sub  = df[df["ticker"] == tk]
    yrs  = (sub["date"].max() - sub["date"].min()).days / 365.25
    cum  = sub["cum_return"].iloc[-1]
    ann  = (1 + cum) ** (1 / yrs) - 1
    vol  = sub["daily_return"].std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    perf.append({"Ticker":     tk,
                 "Cum Ret%":   round(cum * 100, 1),
                 "Ann Ret%":   round(ann * 100, 1),
                 "Ann Vol%":   round(vol * 100, 1),
                 "Sharpe":     round(sharpe, 2),
                 "Max DD%":    round(sub["drawdown"].min() * 100, 1)})

perf_df = pd.DataFrame(perf).sort_values("Cum Ret%", ascending=False)
print(perf_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col, title in zip(axes,
    ["Cum Ret%", "Sharpe", "Max DD%"],
    ["Cumulative Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]):
    srt = perf_df.sort_values(col, ascending=(col == "Max DD%"))
    ax.barh(srt["Ticker"], srt[col],
            color=[COLORS[i % len(COLORS)] for i in range(len(srt))])
    ax.set_title(title, color="white"); ax.axvline(0, color="white", lw=0.8)
plt.suptitle("Performance Summary", fontsize=13, color="white"); plt.tight_layout()
show("1B_performance_summary")

# 1C — Annual returns heatmap
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

annual = (df.groupby(["ticker", "year"])["daily_return"]
            .apply(lambda r: (1 + r).prod() - 1)
            .unstack(level="year") * 100)

fig, ax = plt.subplots(figsize=(max(12, len(annual.columns) * 1.1),
                                max(4, len(tickers) * 0.7)))
sns.heatmap(annual, annot=True, fmt=".0f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Annual Return (%) — Ticker × Year", color="white")
show("1C_annual_heatmap")

# 1D — Monthly seasonality heatmap
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
season = (df.groupby(["ticker", "month"])["monthly_return"]
            .mean().unstack(level="month") * 100)
season.columns = MONTH_LABELS

fig, ax = plt.subplots(figsize=(16, max(4, len(tickers) * 0.7)))
sns.heatmap(season, annot=True, fmt=".1f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Avg Monthly Return (%) — Seasonality", color="white")
show("1D_monthly_seasonality")

# 1E — Return KDE + stats
print("\nReturn Distribution Stats:")
for tk in tickers:
    r   = df[df["ticker"] == tk]["daily_return"].dropna()
    jbp = jarque_bera(r).pvalue
    print(f"  {tk}: skew={skew(r):.2f}  kurt={kurtosis(r):.2f}  "
          f"JB_p={jbp:.2e}  {'NOT Normal' if jbp < 0.05 else 'Normal'}")

fig, ax = plt.subplots(figsize=(12, 4))
for i, tk in enumerate(tickers):
    df[df["ticker"] == tk]["daily_return"].plot.kde(
        ax=ax, label=tk, color=COLORS[i % len(COLORS)], lw=1.8)
ax.set_xlim(-0.12, 0.12)
ax.set_title("Daily Return KDE — All Tickers", color="white")
ax.legend(ncol=5, fontsize=8)
show("1E_return_kde")


# =============================================================================
# SECTION 2 — CORRELATIONS
# =============================================================================
print("\n── SECTION 2: CORRELATIONS ──")

ret_wide = df.pivot_table(index="date", columns="ticker", values="log_return")

# 2A — Correlation heatmap
corr = ret_wide.corr()
fig, ax = plt.subplots(figsize=(max(7, len(tickers)), max(6, len(tickers))))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Log-Return Correlation Matrix", color="white")
show("2A_corr_matrix")

# 2B — Rolling 60-day correlation for top-5 pairs
pairs_sorted = sorted(itertools.combinations(tickers, 2),
                      key=lambda p: abs(corr.loc[p[0], p[1]]), reverse=True)

fig, ax = plt.subplots(figsize=(14, 5))
for i, (t1, t2) in enumerate(pairs_sorted[:5]):
    roll = ret_wide[t1].rolling(60).corr(ret_wide[t2])
    ax.plot(ret_wide.index, roll, label=f"{t1}/{t2}",
            color=COLORS[i % len(COLORS)], lw=1.6)
ax.axhline(0, color="white", lw=0.8, ls="--")
ax.set_title("Rolling 60-Day Correlation — Top 5 Pairs", color="white")
ax.set_ylabel("Pearson r"); ax.legend(ncol=3, fontsize=9)
show("2B_rolling_corr")

# 2C — Lead-lag table (lags 1-5d)
print("\nLead-Lag Cross-Correlation (lags 1–5 days):")
lag_rows = []
for t1, t2 in list(itertools.combinations(tickers, 2))[:10]:
    row = {"Pair": f"{t1}→{t2}"}
    for lag in range(1, 6):
        row[f"lag{lag}d"] = round(ret_wide[t1].corr(ret_wide[t2].shift(lag)), 3)
    lag_rows.append(row)
lag_df = pd.DataFrame(lag_rows).set_index("Pair")
print(lag_df.to_string())

fig, ax = plt.subplots(figsize=(10, max(4, len(lag_df) * 0.5)))
sns.heatmap(lag_df, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Lead-Lag Cross-Correlation (lags 1–5d)", color="white")
show("2C_lead_lag")


# =============================================================================
# SECTION 3 — TECHNICAL INDICATORS
# =============================================================================
print("\n── SECTION 3: TECHNICAL INDICATORS ──")

# RSI helper
def compute_rsi(series, period=14):
    d    = series.diff()
    gain = d.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-d.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)

df["rsi"]         = df.groupby("ticker")["close_price"].transform(compute_rsi)
df["touch_lower"] = df["close_price"] <= df["bollinger_lower"]
df["touch_upper"] = df["close_price"] >= df["bollinger_upper"]
df["bb_pct"]      = ((df["close_price"] - df["bollinger_lower"]) /
                     (df["bollinger_upper"] - df["bollinger_lower"] + 1e-9))
df["golden_50_200"] = (df["sma_50"]  > df["sma_200"]).astype(int)
df["golden_20_50"]  = (df["ema_20"]  > df["sma_50"]).astype(int)

# 3A — Bollinger band touch → win rate
print("\nBollinger Band Touch → Win Rate:")
bb_data = {}
for label, col in [("Touch Lower", "touch_lower"), ("Touch Upper", "touch_upper")]:
    wr = df[df[col]].groupby("ticker")["target"].mean().round(3)
    bb_data[label] = wr
    print(f"  {label}: {wr.to_dict()}")

bb_df = pd.DataFrame(bb_data)
fig, ax = plt.subplots(figsize=(10, 4))
bb_df.plot(kind="bar", ax=ax, color=COLORS[:2], width=0.6)
ax.axhline(0.5, color="white", ls="--", lw=1, label="50% baseline")
ax.set_title("Bollinger Band Touch → Next-Day Win Rate", color="white")
ax.set_xticklabels(bb_df.index, rotation=20); ax.legend(); plt.tight_layout()
show("3A_bollinger_winrate")

# 3B — MA crossover win rate + backtest
print("\nMA Crossover Win Rates:")
for name, col in [("SMA50>SMA200", "golden_50_200"), ("EMA20>SMA50", "golden_20_50")]:
    bull = df[df[col] == 1]["target"].mean()
    bear = df[df[col] == 0]["target"].mean()
    print(f"  {name} — Bull: {bull:.3f}  Bear: {bear:.3f}")

print("\nSMA50/200 Crossover Backtest:")
bt_rows = []
for tk in tickers:
    sub = df[df["ticker"] == tk].dropna(
        subset=["sma_50", "sma_200", "daily_return"]).copy()
    sub["pos"]   = sub["golden_50_200"].shift(1).fillna(0)
    sub["s_ret"] = sub["pos"] * sub["daily_return"]
    cum_s  = (1 + sub["s_ret"]).cumprod() - 1
    cum_bh = (1 + sub["daily_return"]).cumprod() - 1
    peak   = (1 + sub["s_ret"]).cumprod().cummax()
    max_dd = ((1 + sub["s_ret"]).cumprod() / peak - 1).min()
    wins   = (sub["s_ret"] > 0).sum()
    total  = (sub["s_ret"] != 0).sum()
    bt_rows.append({"Ticker": tk,
                    "Strategy%": round(cum_s.iloc[-1] * 100, 1),
                    "B&H%":      round(cum_bh.iloc[-1] * 100, 1),
                    "Win Rate":  round(wins / total if total > 0 else 0, 3),
                    "Max DD%":   round(max_dd * 100, 1)})
print(pd.DataFrame(bt_rows).to_string(index=False))

# 3C — RSI zone → win rate
df["rsi_zone"] = pd.cut(df["rsi"], bins=[0, 30, 70, 100],
                        labels=["Oversold <30", "Neutral 30-70", "Overbought >70"])
rsi_wr = df.groupby(["ticker", "rsi_zone"], observed=True)["target"].mean().unstack()
print("\nRSI Zone → Win Rate:")
print(rsi_wr.round(3))

fig, ax = plt.subplots(figsize=(10, 4))
rsi_wr.plot(kind="bar", ax=ax, color=COLORS[:3], width=0.6)
ax.axhline(0.5, color="white", ls="--", lw=1)
ax.set_title("RSI Zone → Next-Day Win Rate", color="white")
ax.set_xticklabels(rsi_wr.index, rotation=20); ax.legend(fontsize=8); plt.tight_layout()
show("3C_rsi_winrate")

# 3D — Volatility → next-day move magnitude
df["next_abs_ret"] = df.groupby("ticker")["daily_return"].shift(-1).abs()
vol_corr = df.groupby("ticker").apply(
    lambda x: x[["rolling_vol_20d", "std_dev_20", "next_abs_ret"]]
              .corr()["next_abs_ret"][["rolling_vol_20d", "std_dev_20"]]
)
print("\nVolatility → |Next-Day Return| Correlation:")
print(vol_corr.round(4))


# =============================================================================
# SECTION 4 — MARKET REGIMES
# =============================================================================
print("\n── SECTION 4: MARKET REGIMES ──")

REGIMES = {
    "Pre-COVID (2015–19)": ("2015-01-01", "2020-01-31"),
    "COVID Crash":         ("2020-02-01", "2020-04-30"),
    "Recovery":            ("2020-05-01", "2021-12-31"),
    "Bear 2022":           ("2022-01-01", "2022-12-31"),
    "AI Boom (2023+)":     ("2023-01-01", None),
}

reg_rows = []
for regime, (s, e) in REGIMES.items():
    s_dt = pd.Timestamp(s)
    e_dt = pd.Timestamp(e) if e else df["date"].max()
    sub  = df[(df["date"] >= s_dt) & (df["date"] <= e_dt)]
    for tk in tickers:
        t = sub[sub["ticker"] == tk]
        if len(t) < 5: continue
        ret = (1 + t["daily_return"]).prod() - 1
        vol = t["daily_return"].std() * np.sqrt(252)
        reg_rows.append({"Regime":   regime, "Ticker": tk,
                         "Return%":  round(ret * 100, 1),
                         "Vol%":     round(vol * 100, 1),
                         "Max DD%":  round(t["drawdown"].min() * 100, 1)})

reg_df = pd.DataFrame(reg_rows)

# Return heatmap by regime
ret_piv = reg_df.pivot_table(index="Ticker", columns="Regime", values="Return%")
ret_piv = ret_piv[[c for c in REGIMES if c in ret_piv.columns]]

fig, ax = plt.subplots(figsize=(14, max(4, len(tickers) * 0.7)))
sns.heatmap(ret_piv, annot=True, fmt=".0f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Return (%) by Regime × Ticker", color="white")
show("4A_regime_heatmap")

# COVID drawdown
fig, ax = plt.subplots(figsize=(14, 4))
for i, tk in enumerate(tickers):
    sub = df[(df["ticker"] == tk) &
             (df["date"] >= "2020-01-01") &
             (df["date"] <= "2021-06-30")]
    ax.plot(sub["date"], sub["drawdown"] * 100,
            label=tk, color=COLORS[i % len(COLORS)], lw=1.8)
ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-23"),
           alpha=0.15, color="red", label="Crash window")
ax.set_title("Drawdown (%) — COVID Period", color="white")
ax.set_ylabel("Drawdown (%)"); ax.legend(ncol=5, fontsize=8)
show("4B_covid_drawdown")

# Rolling 20d vol with regime shading
fig, axes = plt.subplots(len(tickers), 1,
                         figsize=(14, 2.8 * len(tickers)), sharex=True)
if len(tickers) == 1: axes = [axes]
for i, tk in enumerate(tickers):
    sub = df[df["ticker"] == tk]
    axes[i].plot(sub["date"], sub["rolling_vol_20d"] * 100,
                 color=COLORS[i % len(COLORS)], lw=1.3)
    axes[i].set_ylabel(f"{tk}\n20d Vol%", fontsize=8)
    shade = {"COVID Crash": "red", "Bear 2022": "orange", "AI Boom (2023+)": "green"}
    for regime, (s, e) in REGIMES.items():
        c = shade.get(regime)
        if c:
            axes[i].axvspan(pd.Timestamp(s),
                            pd.Timestamp(e) if e else df["date"].max(),
                            alpha=0.08, color=c)
axes[0].set_title("Rolling 20-Day Volatility — All Tickers", color="white")
plt.tight_layout()
show("4C_rolling_vol")


# =============================================================================
# SECTION 5 — PREDICTIVE MODELING
# =============================================================================
print("\n── SECTION 5: PREDICTIVE MODELING ──")

# Feature engineering
df["lag1"]       = df.groupby("ticker")["daily_return"].shift(1)
df["lag2"]       = df.groupby("ticker")["daily_return"].shift(2)
df["lag5"]       = df.groupby("ticker")["daily_return"].shift(5)
df["dist_sma50"] = (df["close_price"] - df["sma_50"])  / df["sma_50"]
df["dist_sma200"]= (df["close_price"] - df["sma_200"]) / df["sma_200"]
df["vol_ratio"]  = df["rolling_vol_20d"] / (df["rolling_vol_60d"] + 1e-9)
df["vol_chg"]    = df.groupby("ticker")["trading_volume"].pct_change()

FEATURES = ["daily_return", "log_return", "lag1", "lag2", "lag5",
            "rsi", "bb_pct", "dist_sma50", "dist_sma200",
            "rolling_vol_20d", "std_dev_20", "vol_ratio",
            "drawdown", "monthly_return", "vol_chg"]
FEATURES = [f for f in FEATURES if f in df.columns]

mdf = df.dropna(subset=FEATURES + ["target"])
X, y = mdf[FEATURES].values, mdf["target"].values

# 5A — Class balance
bal = pd.Series(y).value_counts(normalize=True)
print(f"\nClass Balance — Up(1): {bal.get(1, 0):.1%}  |  Down(0): {bal.get(0, 0):.1%}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.pie(bal.values, labels=["Down(0)", "Up(1)"],
       colors=COLORS[:2], autopct="%1.1f%%", startangle=90)
ax.set_title("Target Class Balance", color="white")
show("5A_class_balance")

# 5B — Mutual Information + Random Forest importance
mi    = mutual_info_classif(X, y, random_state=42)
mi_df = (pd.DataFrame({"Feature": FEATURES, "MI Score": mi})
           .sort_values("MI Score", ascending=False))

rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                             random_state=42, n_jobs=-1)
rf.fit(X, y)
imp_df = (pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_})
            .sort_values("Importance", ascending=False))
print("\nTop Features (Random Forest):")
print(imp_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(FEATURES) * 0.4)))
for ax, data, col, title, c in [
    (axes[0], mi_df,  "MI Score",   "Mutual Information",      COLORS[0]),
    (axes[1], imp_df, "Importance", "Random Forest Importance", COLORS[3])]:
    srt = data.sort_values(col)
    ax.barh(srt["Feature"], srt[col], color=c)
    ax.set_title(title, color="white")
plt.suptitle("Feature Importance → Target", fontsize=13, color="white")
plt.tight_layout()
show("5B_feature_importance")

# 5C — Performance by market regime
print("\nModel Performance by Regime:")
cv_rows = []
for regime, (s, e) in REGIMES.items():
    s_dt = pd.Timestamp(s)
    e_dt = pd.Timestamp(e) if e else mdf["date"].max()
    mask = (mdf["date"] >= s_dt) & (mdf["date"] <= e_dt)
    Xt, yt = mdf.loc[mask, FEATURES].values, mdf.loc[mask, "target"].values
    if len(yt) < 30 or len(set(yt)) < 2:
        continue
    pred  = rf.predict(Xt)
    proba = rf.predict_proba(Xt)[:, 1]
    cv_rows.append({"Regime":   regime, "N": len(yt),
                    "Accuracy": round(accuracy_score(yt, pred), 3),
                    "F1":       round(f1_score(yt, pred, zero_division=0), 3),
                    "AUC":      round(roc_auc_score(yt, proba), 3)})

cv_df = pd.DataFrame(cv_rows)
print(cv_df.to_string(index=False))

if len(cv_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(cv_df))
    for j, (metric, c) in enumerate(zip(["Accuracy", "F1", "AUC"], COLORS[:3])):
        ax.bar(x + j * 0.25, cv_df[metric], width=0.25, label=metric, color=c)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(cv_df["Regime"], rotation=15, ha="right")
    ax.axhline(0.5, color="white", lw=0.8, ls="--")
    ax.set_title("Model Performance by Market Regime", color="white")
    ax.legend(fontsize=9); plt.tight_layout()
    show("5C_regime_performance")

print("\n✓ Done — all PNG charts saved in the current working directory.")
