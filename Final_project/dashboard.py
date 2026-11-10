"""
Renewable Energy Dashboard  (All India + top states)
----------------------------------------------------
Run it with:   streamlit run app.py

Put these files in the SAME folder as this app.py:
  1. monthly_renewable_energy2.csv   (your original historical data)
  2. forecast_2030.csv               (made by your forecast cell: state_name, date, forecast_renewable_energy)
  3. state_r2_scores.csv             (optional: made by results_df1.to_csv(...), used for the Accuracy tab)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# SETTINGS (you can change these)
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent
HISTORY_FILE = HERE / "monthly_renewable_energy.csv"
FORECAST_FILE = HERE / "forecast_2030.csv"
EVAL_FILE = HERE / "state_r2_scores.csv"
UNIT = "MU"                      # unit of your energy numbers
DEFAULT_STATE = "All India"      # shown first when the dashboard opens
TOTAL_NAME = "All India"         # name of the India-total row in your data

# colours (light theme)
C_HIST = "#1D4ED8"   # blue   = actual
C_FC = "#EA580C"     # orange = forecast
C_WIND = "#3B82F6"
C_SOLAR = "#F59E0B"
C_OTHER = "#10B981"
C_TEXT = "#0F172A"
C_MUTED = "#64748B"
C_GRID = "#E2E8F0"
REL_COLORS = {"GOOD": "#16A34A", "MODERATE": "#F59E0B", "LOW": "#DC2626"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------------------
# Force a LIGHT theme (creates .streamlit/config.toml once, if missing)
# ----------------------------------------------------------------------------
def ensure_light_theme():
    try:
        cfg = Path.cwd() / ".streamlit" / "config.toml"
        if not cfg.exists():
            cfg.parent.mkdir(exist_ok=True)
            cfg.write_text(
                "[theme]\n"
                'base = "light"\n'
                'primaryColor = "#1D4ED8"\n'
                'backgroundColor = "#F8FAFC"\n'
                'secondaryBackgroundColor = "#FFFFFF"\n'
                'textColor = "#0F172A"\n'
            )
    except Exception:
        pass


ensure_light_theme()

st.set_page_config(page_title="Renewable Energy Dashboard", page_icon="⚡", layout="wide")

st.markdown(
    """
<style>
.stApp { background: #F8FAFC; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1300px; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E2E8F0; }
div[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(15,23,42,.06);
}
div[data-testid="stMetric"] label { color: #64748B !important; }
h1, h2, h3, h4 { color: #0F172A; }
.badge { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.82rem;
         font-weight: 600; color: white; vertical-align: middle; margin-left: 8px; }
.subtle { color:#64748B; font-size:0.95rem; margin-top:-6px; }
button[data-baseweb="tab"] { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": C_GRID, "axes.labelcolor": C_MUTED,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlecolor": C_TEXT, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": C_GRID,
    "grid.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False,
})


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def thousands(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def compact(v):
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.0f}"


def rating_from_r2(r2):
    if r2 >= 0.9:
        return "GOOD"
    if r2 >= 0.7:
        return "MODERATE"
    return "LOW"


def badge(label):
    color = REL_COLORS.get(label, "#64748B")
    return f'<span class="badge" style="background:{color}">{label.title()}</span>'


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_history(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    parts = ["wind_energy", "solar_energy", "other_renewable_energy"]
    for c in parts:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df else 0.0
    df["total_renewable_energy"] = pd.to_numeric(df["total_renewable_energy"], errors="coerce")
    df = df.drop_duplicates(["state_name", "date"]).dropna(subset=["state_name", "date", "total_renewable_energy"])
    df = df[df["total_renewable_energy"] >= 0]
    cols = parts + ["total_renewable_energy"]
    return df[["state_name", "date"] + cols].sort_values(["state_name", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_forecast(path):
    f = pd.read_csv(path)
    f["date"] = pd.to_datetime(f["date"])
    f = f.rename(columns={"forecast_renewable_energy": "forecast"})
    return f[["state_name", "date", "forecast"]].sort_values(["state_name", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_eval(path):
    e = pd.read_csv(path)
    e = e.rename(columns={"state_name": "state", "r2_score": "R2"}).set_index("state")
    e["rating"] = e["R2"].apply(rating_from_r2)
    return e


missing = [p.name for p in (HISTORY_FILE, FORECAST_FILE) if not p.exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Put them in the same folder as app.py.")
    st.stop()

hist = load_history(HISTORY_FILE)
fc_all = load_forecast(FORECAST_FILE)
eval_df = load_eval(EVAL_FILE) if EVAL_FILE.exists() else None

total_hist = hist[hist["state_name"] == TOTAL_NAME]
if total_hist.empty:
    total_hist = hist
last_hist_date = total_hist["date"].max()
first_hist_date = total_hist["date"].min()


def total_series(state):
    s = hist[hist["state_name"] == state].set_index("date")["total_renewable_energy"]
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(idx).interpolate()


# state list: All India first, then the other states, biggest first
modelled = [s for s in fc_all["state_name"].unique() if s != TOTAL_NAME]
last12 = {s: total_series(s).iloc[-12:].sum() for s in modelled}
state_options = ([TOTAL_NAME] if TOTAL_NAME in fc_all["state_name"].unique() else []) + \
                sorted(modelled, key=lambda s: -last12[s])

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("## ⚡ Controls")
default_idx = state_options.index(DEFAULT_STATE) if DEFAULT_STATE in state_options else 0
state = st.sidebar.selectbox("Select state / region", state_options, index=default_idx)

fc_years = sorted(fc_all["date"].dt.year.unique())
horizon_options = [y for y in fc_years if y >= last_hist_date.year] or fc_years
end_year = st.sidebar.select_slider("Forecast until year", options=horizon_options, value=horizon_options[-1])

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data: {first_hist_date.strftime('%b %Y')} to {last_hist_date.strftime('%b %Y')}\n\n"
    f"Unit: {UNIT}\n\n"
    "Forecasts are scenarios, not certainties."
)

# ----------------------------------------------------------------------------
# Data for the selected state
# ----------------------------------------------------------------------------
hs = total_series(state)
fc = fc_all[(fc_all["state_name"] == state) & (fc_all["date"].dt.year <= end_year)].copy()

rel = None
if eval_df is not None and state in eval_df.index:
    rel = eval_df.loc[state, "rating"]

last12_total = hs.iloc[-12:].sum()
next12_total = fc["forecast"].iloc[:12].sum()
end_total = fc.loc[fc["date"].dt.year == end_year, "forecast"].sum()
chg = (end_total / last12_total - 1) * 100 if last12_total > 0 else np.nan

# ----------------------------------------------------------------------------
# Header + KPI cards
# ----------------------------------------------------------------------------
badge_html = badge(rel) if rel else ""
st.markdown(f"# {state} {badge_html}", unsafe_allow_html=True)
st.markdown('<p class="subtle">Renewable energy generation: history, trends, comparison and forecast</p>',
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Last 12 months (actual)", f"{last12_total:,.0f} {UNIT}")
k2.metric("Next 12 months (forecast)", f"{next12_total:,.0f} {UNIT}",
          delta=f"{(next12_total / last12_total - 1) * 100:+.1f}% vs last 12 mo" if last12_total > 0 else None)
k3.metric(f"Year {end_year} (forecast)", f"{end_total:,.0f} {UNIT}",
          delta=f"{chg:+.1f}% vs last 12 mo" if pd.notna(chg) else None)
k4.metric("Forecast model", "Trend + Random Forest")

if rel == "LOW":
    st.warning("The model fits this state's history **poorly**. Use the forecast only as a rough guide.")
elif rel == "MODERATE":
    st.info("The model fits this state's history **moderately**. Use the forecast with care.")

st.write("")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_trend():
    comb = pd.concat([hs, fc.set_index("date")["forecast"]])
    roll = comb.rolling(12, min_periods=12).mean()
    cut = hs.index[-1]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(hs.index, hs.values, color=C_HIST, lw=2, label="Monthly actual")
    if len(fc):
        ax.plot(fc["date"], fc["forecast"], color=C_FC, lw=2, ls="--", label="Monthly forecast")
    ax.plot(roll[roll.index <= cut].index, roll[roll.index <= cut].values, color=C_HIST, lw=1.5,alpha=0.9,
            label="12-month average (actual)")
    ax.plot(roll[roll.index >= cut].index, roll[roll.index >= cut].values, color=C_FC, lw=1.5,alpha=0.7,
            label="12-month average (forecast)")
    ax.axvline(cut, color="#94A3B8", ls=":", lw=1.3)
    ax.text(cut, ax.get_ylim()[1], "  forecast →", color=C_MUTED, va="top", fontsize=9)
    ax.set_ylabel(f"Energy ({UNIT} per month)")
    ax.set_title(f"{state}: monthly generation, trend and forecast")
    ax.legend(ncol=2, loc="upper left", fontsize=8.5)
    thousands(ax)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def fig_yearly():
    a = hs.groupby(hs.index.year).sum()
    f = fc.groupby(fc["date"].dt.year)["forecast"].sum() if len(fc) else pd.Series(dtype=float)
    years = sorted(set(a.index) | set(f.index))
    a = a.reindex(years).fillna(0)
    f = f.reindex(years).fillna(0)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(years, a.values, color=C_HIST, width=0.7, label="Actual")
    ax.bar(years, f.values, bottom=a.values, color=C_FC, width=0.7, label="Forecast")
    for y, tot in zip(years, (a + f).values):
        ax.text(y, tot, compact(tot), ha="center", va="bottom", fontsize=7.5, color=C_TEXT)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=8, rotation=45)
    ax.set_ylabel(f"Energy ({UNIT} per year)")
    ax.set_title("Yearly total")
    ax.legend(loc="upper left", fontsize=8.5)
    thousands(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_mix():
    d = hist[hist["state_name"] == state].set_index("date")
    d = d[["wind_energy", "solar_energy", "other_renewable_energy"]]
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="MS")).fillna(0)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    series, labels, cols = [d["wind_energy"], d["solar_energy"]], ["Wind", "Solar"], [C_WIND, C_SOLAR]
    if d["other_renewable_energy"].sum() > 0:
        series.append(d["other_renewable_energy"]); labels.append("Hydro & other"); cols.append(C_OTHER)
    ax.stackplot(d.index, *[s.values for s in series], labels=labels, colors=cols, alpha=0.9)
    ax.set_ylabel(f"Energy ({UNIT} per month)")
    ax.set_title("Energy mix (actual)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    thousands(ax)
    fig.tight_layout()
    return fig


def fig_heatmap():
    df = hs.to_frame("v")
    piv = df.pivot_table(index=df.index.year, columns=df.index.month, values="v").reindex(columns=range(1, 13))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#F1F5F9")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(np.ma.masked_invalid(piv.values), aspect="auto", cmap=cmap)
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS, fontsize=8.5)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index.astype(str), fontsize=9)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    ax.set_title("Heat map: which months are strongest each year")
    fig.tight_layout()
    return fig


def fig_month_avg():
    df = hs.to_frame("v")
    avg = df.groupby(df.index.month)["v"].mean().reindex(range(1, 13))
    cols = [C_FC if m == avg.idxmax() else C_HIST for m in avg.index]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.bar(MONTHS, avg.values, color=cols, width=0.7)
    ax.set_ylabel(f"Average ({UNIT})")
    ax.set_title("Average by month")
    thousands(ax)
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    fig.tight_layout()
    return fig, MONTHS[int(avg.idxmax()) - 1], MONTHS[int(avg.idxmin()) - 1]


def comparison_table():
    rows = []
    for s in modelled:
        f = fc_all[(fc_all["state_name"] == s) & (fc_all["date"].dt.year == end_year)]["forecast"].sum()
        l12 = last12[s]
        rows.append({"State": s, "Last 12 months": l12, f"{end_year} forecast": f,
                     "Change %": (f / l12 - 1) * 100 if l12 > 0 else np.nan})
    return pd.DataFrame(rows).sort_values(f"{end_year} forecast", ascending=False).reset_index(drop=True)


def fig_compare(comp):
    c = comp.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(c)); h = 0.38
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(c) + 1.6))
    ax.barh(y + h / 2, c["Last 12 months"], h, color=C_HIST, label="Last 12 months (actual)")
    ax.barh(y - h / 2, c[f"{end_year} forecast"], h, color=C_FC, label=f"{end_year} (forecast)")
    ax.set_yticks(y)
    ax.set_yticklabels(c["State"], fontsize=9.5)
    for lab, name in zip(ax.get_yticklabels(), c["State"]):
        if name == state:
            lab.set_fontweight("bold"); lab.set_color(C_FC)
    ax.set_xlabel(f"Energy ({UNIT} per year)")
    ax.set_title(f"Comparison of states: now vs forecast ({TOTAL_NAME} excluded)")
    ax.legend(loc="lower right", fontsize=9)
    thousands(ax, "x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_eval():
    e = eval_df.sort_values("R2")
    y = np.arange(len(e))
    fig, ax = plt.subplots(figsize=(10, 0.46 * len(e) + 1.8))
    ax.barh(y, e["R2"], color=[REL_COLORS.get(r, "#64748B") for r in e["rating"]], height=0.62)
    ax.axvline(0.9, color="#94A3B8", ls=":", lw=1)
    ax.axvline(0.7, color="#94A3B8", ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(e.index, fontsize=9.5)
    for lab, name in zip(ax.get_yticklabels(), e.index):
        if name == state:
            lab.set_fontweight("bold"); lab.set_color(C_FC)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("R² score (closer to 1 is better)")
    ax.set_title("How well the model fits each state")
    ax.legend(handles=[Patch(color=REL_COLORS[k], label=k.title()) for k in REL_COLORS],
              loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends & Forecast", "🗓️ Seasonality", "⚖️ Compare states", "🎯 Model accuracy"])

with tab1:
    show(fig_trend())
    c1, c2 = st.columns(2)
    with c1:
        show(fig_yearly())
        st.caption(f"{last_hist_date.year} = actual up to {last_hist_date.strftime('%b')} + forecast for the remaining months.")
    with c2:
        show(fig_mix())
        st.caption("Wind, solar and other renewable generation from the original data.")
    with st.expander("See the forecast table and download it"):
        tbl = fc[["date", "forecast"]].copy()
        tbl["date"] = tbl["date"].dt.strftime("%b %Y")
        tbl.columns = ["Month", f"Forecast ({UNIT})"]
        st.dataframe(tbl.round(1), hide_index=True)
        st.download_button("Download forecast as CSV", data=tbl.to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_{state.replace(' ', '_')}.csv", mime="text/csv")

with tab2:
    c1, c2 = st.columns([3, 2])
    with c1:
        show(fig_heatmap())
    with c2:
        fig, best, worst = fig_month_avg()
        show(fig)
    st.markdown(f"**{state}:** strongest month on average is **{best}**, weakest is **{worst}**.")

with tab3:
    comp = comparison_table()
    show(fig_compare(comp))
    st.caption(f"{TOTAL_NAME} is left out of this chart because it is the total of all states and would dwarf the rest. "
               "The selected state is highlighted in orange.")
    shown = comp.copy()
    shown["Last 12 months"] = shown["Last 12 months"].round(0)
    shown[f"{end_year} forecast"] = shown[f"{end_year} forecast"].round(0)
    shown["Change %"] = shown["Change %"].round(1)
    st.dataframe(shown, hide_index=True)

with tab4:
    if eval_df is None:
        st.info("Accuracy file not found. Save `state_r2_scores.csv` from the notebook next to app.py to see this tab.")
    else:
        st.markdown("**What is shown?** The R² score tells how closely the model follows the real historical data "
                    "(1 = perfect). It is measured on the same history the model learned from, so it shows "
                    "how well the model fits, not a guarantee for the future.")
        if state in eval_df.index:
            r = eval_df.loc[state]
            st.metric("R² score", f"{r['R2']:.2f}", help="1 = perfect fit. Below 0.7 = weak fit.")
            msg = {"GOOD": "The model follows this state's history closely.",
                   "MODERATE": "The model follows the history moderately. Use with care.",
                   "LOW": "The model fits poorly. Treat the forecast as a rough guide only."}[r["rating"]]
            st.markdown(f"**Verdict for {state}:** {badge(r['rating'])} &nbsp; {msg}", unsafe_allow_html=True)
        show(fig_eval())

st.markdown("---")
st.caption("Forecasts up to 2030 are scenarios based on 5 years of history. They are not guaranteed values.")
