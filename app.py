import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy_financial as npf
import requests

st.set_page_config(page_title="Portfolio Analyzer", layout="wide", page_icon="📊")

st.title("📊 Portfolio Analyzer — EUR")
st.caption("Análisis de carteras con conversión de divisa, frontera eficiente y proyección de riqueza")

# ============================================================
# SEARCH
# ============================================================
def yahoo_search(query: str, max_results: int = 25):
    query = (query or "").strip()
    if not query:
        return []
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": max_results, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    type_order = {"ETF": 0, "MUTUALFUND": 1, "EQUITY": 2, "CRYPTOCURRENCY": 3, "INDEX": 4, "FUTURE": 5}
    out = []
    seen = set()
    for it in data.get("quotes", []):
        tkr = it.get("symbol")
        if not tkr or tkr in seen:
            continue
        seen.add(tkr)
        longn  = it.get("longname")  or it.get("longName")  or ""
        shortn = it.get("shortname") or it.get("shortName") or ""
        name   = (longn or shortn or "").strip()
        out.append({
            "ticker":   tkr,
            "name":     name,
            "type":     it.get("quoteType") or "",
            "exchange": it.get("exchDisp")  or it.get("exchange") or "",
            "_rank":    type_order.get(it.get("quoteType", ""), 99),
            "_has":     1 if name and name != tkr else 0,
        })
    out.sort(key=lambda x: (-x["_has"], x["_rank"]))
    for r in out:
        r.pop("_rank"); r.pop("_has")
    return out


# ============================================================
# DOWNLOADS (cached)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def download_close(ticker: str, start: str) -> pd.Series:
    # Intentar con auto_adjust y sin él como fallback
    df = None
    for adj in [True, False]:
        try:
            df = yf.download(ticker, start=start, auto_adjust=adj, progress=False)
            if df is not None and not df.empty:
                break
        except Exception:
            continue

    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}. Comprueba que el ticker es correcto.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col_name in ["Adj Close", "Close"] + list(df.columns):
        if col_name in df.columns:
            s = df[col_name].astype(float).dropna()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            if len(s) > 0:
                s.name = ticker
                return s

    raise ValueError(f"Sin datos para {ticker}.")

@st.cache_data(ttl=3600, show_spinner=False)
def get_eurusd(start: str) -> pd.Series:
    s = download_close("EURUSD=X", start=start)
    s.name = "EURUSD"
    return s

def download_prices(tickers, start):
    series = []
    for t in tickers:
        series.append(download_close(t, start))
    return pd.concat(series, axis=1).sort_index().ffill().dropna()


# ============================================================
# FX + RETURNS
# ============================================================
def convert_to_eur(prices_df: pd.DataFrame, start: str, ccy_map: dict) -> pd.DataFrame:
    has_usd = any(v == "USD" for v in ccy_map.values())
    if not has_usd:
        return prices_df
    eurusd = get_eurusd(start).reindex(prices_df.index).ffill().bfill()
    out = prices_df.copy()
    for col in prices_df.columns:
        if (ccy_map.get(col) or "EUR").upper() == "USD":
            out[col] = prices_df[col] / eurusd
    return out

def clean_returns(ret_df: pd.DataFrame) -> pd.DataFrame:
    return ret_df.replace([np.inf, -np.inf], np.nan).clip(lower=-0.99).dropna()


# ============================================================
# METRICS
# ============================================================
def metrics_from_returns(ret: pd.Series, rf=0.02):
    ret = ret.dropna()
    if len(ret) < 10:
        return {"CAGR": np.nan, "Volatilidad": np.nan, "Sharpe": np.nan,
                "Max Drawdown": np.nan, "Total Return": np.nan}
    eq    = (1 + ret).cumprod()
    dd    = eq / eq.cummax() - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol   = float(ret.std() * np.sqrt(252))
    rf_d  = (1 + rf) ** (1 / 252) - 1
    sharpe = float((ret.mean() - rf_d) / (ret.std() + 1e-12) * np.sqrt(252))
    return {"CAGR": cagr, "Volatilidad": vol, "Sharpe": sharpe,
            "Max Drawdown": float(dd.min()), "Total Return": float(eq.iloc[-1] - 1)}

def portfolio_returns(returns_df, weights):
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return pd.Series(returns_df.values @ w, index=returns_df.index), w


# ============================================================
# EFFICIENT FRONTIER
# ============================================================
def eff_frontier(returns_df, rf=0.02, n=60000, gold_ticker=None, gold_max=None):
    assets = returns_df.columns.tolist()
    mu  = (returns_df.mean() * 252).values
    cov = (returns_df.cov()  * 252).values
    n_a = len(assets)
    rng = np.random.default_rng(42)
    W   = rng.random((n, n_a))
    W   = W / W.sum(axis=1, keepdims=True)

    if gold_ticker and gold_max and gold_max < 1.0 and gold_ticker in assets:
        gi   = assets.index(gold_ticker)
        over = W[:, gi] > gold_max
        if np.any(over):
            W[over, gi] = gold_max
            rest = W[over].copy(); rest[:, gi] = 0.0
            rs   = rest.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            W[over] = rest / rs * (1 - gold_max)
            W[over, gi] = gold_max

    rets    = W @ mu
    vols    = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    sharpes = (rets - rf) / (vols + 1e-12)

    cloud   = pd.DataFrame({"Return": rets, "Volatility": vols, "Sharpe": sharpes})
    cloud["row_id"] = np.arange(len(cloud))
    cloud["bin"]    = pd.qcut(cloud["Volatility"], q=200, duplicates="drop")
    frontier = cloud.loc[cloud.groupby("bin")["Return"].idxmax()].sort_values("Volatility").reset_index(drop=True)

    picks = np.linspace(0, len(frontier) - 1, min(6, len(frontier))).round().astype(int)
    f6    = frontier.iloc[picks].copy().reset_index(drop=True)
    f6.insert(0, "Portfolio", [f"{i}/6" for i in range(1, len(f6) + 1)])
    W6    = W[f6["row_id"].values]
    for j, a in enumerate(assets):
        f6[a] = W6[:, j]

    return cloud, frontier, f6.drop(columns=["row_id", "bin"], errors="ignore")


# ============================================================
# WEALTH PROJECTION + IRR
# ============================================================
def project_wealth(cagr, pv, pmt_monthly, years_list):
    if cagr <= -0.999:
        return {y: np.nan for y in years_list}
    r_m = (1 + cagr) ** (1 / 12) - 1
    out = {}
    for y in years_list:
        n      = 12 * y
        fv_pv  = pv * (1 + cagr) ** y
        fv_pmt = pmt_monthly * n if abs(r_m) < 1e-12 else pmt_monthly * (((1 + r_m) ** n - 1) / r_m)
        out[y] = fv_pv + fv_pmt
    return out

def compute_irr(port_ret, pv, pmt):
    monthly = port_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna()
    if len(monthly) < 2:
        return np.nan
    value = float(pv)
    cf    = [-value]
    for r in monthly:
        value = value * (1 + float(r)) + float(pmt)
        cf.append(-float(pmt))
    cf[-1] += value
    irr_m = npf.irr(cf)
    return float((1 + irr_m) ** 12 - 1) if irr_m is not None and not np.isnan(irr_m) else np.nan


# ============================================================
# SESSION STATE HELPERS
# ============================================================
def get_rows(key):
    return st.session_state[f"{key}_rows"]

def add_row(key):
    st.session_state[f"{key}_rows"].append(("", 0.0, "EUR"))

def remove_row(key, idx):
    rows = st.session_state[f"{key}_rows"]
    if 0 <= idx < len(rows):
        rows.pop(idx)

def add_from_search(key, tickers):
    existing = {r[0] for r in st.session_state[f"{key}_rows"]}
    for tkr in tickers:
        if tkr not in existing:
            default_ccy = "EUR" if any(tkr.endswith(s) for s in [".DE", ".AS", ".PA", ".MI", ".MC", ".LS", ".L"]) else "USD"
            st.session_state[f"{key}_rows"].append((tkr, 0.0, default_ccy))


# ============================================================
# SCENARIO BLOCK
# ============================================================
def render_scenario(key, title, start_default, bench_default,
                    rf_default=0.02, pv_default=10000.0, pmt_default=1000.0):

    st.subheader(title)

    # Controls
    c1, c2, c3, c4 = st.columns([2, 1.2, 1.5, 1.2])
    start     = c1.text_input("Fecha inicio",  value=start_default,  key=f"{key}_start")
    rf        = c2.number_input("RF anual",     value=rf_default,     step=0.005, format="%.3f", key=f"{key}_rf")
    bench     = c3.text_input("Benchmark",      value=bench_default,  key=f"{key}_bench")
    bench_ccy = c4.selectbox("Bench CCY",       ["USD", "EUR"],       key=f"{key}_bench_ccy")

    # Search
    with st.expander("🔍 Buscar activos", expanded=False):
        query = st.text_input("Buscar ticker o nombre", key=f"{key}_query",
                              placeholder="Ej: azvalor, iShares MSCI, BTC...")
        if query and len(query) >= 2:
            with st.spinner("Buscando..."):
                results = yahoo_search(query)
            if results:
                options = {}
                for r in results:
                    label = f"{r['ticker']} — {r['name']}" if r['name'] else r['ticker']
                    if r['exchange']:
                        label += f" [{r['exchange']}]"
                    options[label] = r["ticker"]
                selected_labels = st.multiselect("Selecciona uno o varios:", list(options.keys()),
                                                  key=f"{key}_search_sel")
                if st.button("➕ Añadir seleccionados", key=f"{key}_add_sel"):
                    add_from_search(key, [options[l] for l in selected_labels])
                    st.rerun()
            else:
                st.caption("Sin resultados.")

    # Ticker rows
    st.markdown("**Ticker &nbsp;&nbsp;&nbsp; Peso &nbsp;&nbsp;&nbsp; CCY**")
    rows = get_rows(key)
    updated_rows = []
    for i, (t, w, c) in enumerate(rows):
        col1, col2, col3, col4 = st.columns([3, 2, 1.5, 0.8])
        t_new = col1.text_input("t", value=t,     key=f"{key}_t_{i}", label_visibility="collapsed")
        w_new = col2.number_input("w", value=float(w), step=0.05, min_value=0.0,
                                   key=f"{key}_w_{i}", label_visibility="collapsed")
        c_new = col3.selectbox("c", ["EUR", "USD"], index=0 if c == "EUR" else 1,
                                key=f"{key}_c_{i}", label_visibility="collapsed")
        if col4.button("❌", key=f"{key}_rm_{i}"):
            remove_row(key, i)
            st.rerun()
        updated_rows.append((t_new, w_new, c_new))
    st.session_state[f"{key}_rows"] = updated_rows

    if st.button("➕ Añadir fila", key=f"{key}_addrow"):
        add_row(key)
        st.rerun()

    # Gold cap + investment params
    g1, g2, g3, g4 = st.columns(4)
    gold_ticker = g1.text_input("Gold ticker", value="GLD",  key=f"{key}_gold")
    gold_cap    = g2.number_input("Gold máx",  value=1.0, min_value=0.0, max_value=1.0,
                                   step=0.05, key=f"{key}_goldcap",
                                   help="1.0 = sin límite")
    pv          = g3.number_input("Capital inicial €", value=pv_default,  step=1000.0, key=f"{key}_pv")
    pmt         = g4.number_input("Aportación mensual €", value=pmt_default, step=100.0, key=f"{key}_pmt")

    if st.button("⚡ Calcular", type="primary", key=f"{key}_calc"):
        _run_analysis(key, start, rf, bench, bench_ccy, gold_ticker, gold_cap, pv, pmt)

    if f"{key}_result" in st.session_state:
        _display_results(st.session_state[f"{key}_result"])


# ============================================================
# ANALYSIS ENGINE
# ============================================================
def _run_analysis(key, start, rf, bench, bench_ccy, gold_ticker, gold_cap, pv, pmt):
    rows    = get_rows(key)
    tickers = [r[0].strip() for r in rows if r[0].strip()]
    weights = [float(r[1]) for r in rows if r[0].strip()]
    ccy_map = {r[0].strip(): r[2] for r in rows if r[0].strip()}

    if not tickers:
        st.error("No hay tickers definidos."); return
    if sum(weights) == 0:
        st.error("La suma de pesos es 0."); return

    with st.spinner("Descargando y calculando..."):
        try:
            prices_raw = download_prices(tickers, start)
            prices_eur = convert_to_eur(prices_raw, start, ccy_map)
            ret_assets = clean_returns(prices_eur.pct_change().dropna())

            bench_px_raw = download_close(bench, start).reindex(prices_eur.index).ffill().dropna()
            b_ccy = ccy_map.get(bench, bench_ccy)
            if b_ccy == "USD":
                fx = get_eurusd(start).reindex(bench_px_raw.index).ffill().bfill()
                bench_px_eur = (bench_px_raw / fx).dropna()
            else:
                bench_px_eur = bench_px_raw.copy()

            ret_bench = clean_returns(bench_px_eur.pct_change().dropna().to_frame("bench")).iloc[:, 0]
            common    = ret_assets.index.intersection(ret_bench.index)

            if len(common) < 50:
                st.error("Muy pocos días solapados. Revisa Start/tickers."); return

            ret_assets = ret_assets.loc[common]
            ret_bench  = ret_bench.loc[common]

            port_ret, w_norm = portfolio_returns(ret_assets, weights)
            m_user  = metrics_from_returns(port_ret,  rf)
            m_bench = metrics_from_returns(ret_bench, rf)

            eq_user  = (1 + port_ret).cumprod()
            eq_bench = (1 + ret_bench).cumprod()
            dd_user  = eq_user  / eq_user.cummax()  - 1
            dd_bench = eq_bench / eq_bench.cummax() - 1

            gt = gold_ticker if gold_cap < 1.0 else None
            gc = float(gold_cap) if gold_cap < 1.0 else None
            cloud, frontier, f6 = eff_frontier(ret_assets, rf=rf, gold_ticker=gt, gold_max=gc)

            irr  = compute_irr(port_ret, pv, pmt)
            proj = project_wealth(float(m_user["CAGR"]), pv, pmt, list(range(1, 11)) + [15, 20, 30])

            asset_rows = []
            for t in tickers:
                r = ret_assets[t].dropna() if t in ret_assets.columns else pd.Series(dtype=float)
                if len(r) < 10:
                    asset_rows.append({"Activo": t, "CCY": ccy_map.get(t, "EUR"),
                                       "CAGR": np.nan, "Vol": np.nan, "Total Return": np.nan})
                else:
                    eq  = (1 + r).cumprod()
                    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
                    asset_rows.append({"Activo": t, "CCY": ccy_map.get(t, "EUR"),
                                       "CAGR":         float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan,
                                       "Vol":          float(r.std() * np.sqrt(252)),
                                       "Total Return": float(eq.iloc[-1] - 1)})

            st.session_state[f"{key}_result"] = dict(
                tickers=tickers, weights=w_norm, ccy_map=ccy_map,
                bench=bench, common=common,
                port_ret=port_ret, ret_bench=ret_bench,
                m_user=m_user, m_bench=m_bench,
                eq_user=eq_user, eq_bench=eq_bench,
                dd_user=dd_user, dd_bench=dd_bench,
                cloud=cloud, frontier=frontier, f6=f6,
                user_ret_ann=float(port_ret.mean() * 252),
                user_vol_ann=float(port_ret.std()  * np.sqrt(252)),
                irr=irr, proj=proj, asset_rows=asset_rows, rf=rf, pv=pv, pmt=pmt,
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())


# ============================================================
# DISPLAY RESULTS
# ============================================================
def _display_results(res):
    tickers  = res["tickers"]
    weights  = res["weights"]
    bench    = res["bench"]
    m_user   = res["m_user"]
    m_bench  = res["m_bench"]
    eq_user  = res["eq_user"]
    eq_bench = res["eq_bench"]
    dd_user  = res["dd_user"]
    dd_bench = res["dd_bench"]
    cloud    = res["cloud"]
    frontier = res["frontier"]
    f6       = res["f6"]
    user_ret = res["user_ret_ann"]
    user_vol = res["user_vol_ann"]
    irr      = res["irr"]
    proj     = res["proj"]
    common   = res["common"]

    st.markdown("---")
    st.caption(f"📅 Rango: **{common.min().date()}** → **{common.max().date()}** ({len(common)} días)")

    # Weights
    st.markdown("**Pesos normalizados:**")
    st.write({t: round(float(w), 3) for t, w in zip(tickers, weights)})

    # Per-asset table
    st.markdown("**Rendimientos por activo (EUR):**")
    df_assets = pd.DataFrame(res["asset_rows"]).set_index("Activo")
    st.dataframe(df_assets.style.format({"CAGR": "{:.2%}", "Vol": "{:.2%}", "Total Return": "{:.2%}"}),
                 use_container_width=True)

    # Metrics
    st.markdown("**Métricas User vs Benchmark:**")
    m_table = pd.DataFrame({"Tu cartera": m_user, f"Benchmark ({bench})": m_bench}).T
    st.dataframe(m_table.style.format({
        "CAGR": "{:.2%}", "Volatilidad": "{:.2%}", "Sharpe": "{:.2f}",
        "Max Drawdown": "{:.2%}", "Total Return": "{:.2%}"
    }), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("CAGR (time-weighted)", f"{m_user['CAGR']:.2%}")
    if not np.isnan(irr):
        col2.metric("IRR (money-weighted)", f"{irr:.2%}")

    # Equity
    fig, ax = plt.subplots(figsize=(8, 3.2))
    pd.DataFrame({"Tu cartera": eq_user, f"Benchmark ({bench})": eq_bench}).plot(ax=ax)
    ax.set_title("Equity curve (EUR)"); ax.set_ylabel("Valor base 1"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    # Drawdown
    fig2, ax2 = plt.subplots(figsize=(8, 2.5))
    pd.DataFrame({"Tu cartera": dd_user, f"Benchmark ({bench})": dd_bench}).plot(ax=ax2)
    ax2.set_title("Drawdown (EUR)"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    # Frontier table
    st.markdown("**Frontera Eficiente — 6 carteras:**")
    asset_cols  = [c for c in f6.columns if c not in ["Portfolio", "Return", "Volatility", "Sharpe"]]
    display_f6  = f6[["Portfolio", "Return", "Volatility", "Sharpe"] + asset_cols]
    st.dataframe(display_f6.style.format({
        "Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}",
        **{a: "{:.1%}" for a in asset_cols}
    }), use_container_width=True)

    # Frontier chart
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.scatter(cloud["Volatility"], cloud["Return"], s=3, alpha=0.08, color="steelblue")
    ax3.plot(frontier["Volatility"], frontier["Return"], lw=2, color="navy", label="Frontera")
    ax3.scatter(f6["Volatility"], f6["Return"], s=60, zorder=5, color="orange")
    for _, row in f6.iterrows():
        ax3.annotate(row["Portfolio"], (row["Volatility"], row["Return"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax3.scatter([user_vol], [user_ret], s=150, marker="X", color="red", zorder=6, label="Tu cartera")
    ax3.annotate("Tu cartera", (user_vol, user_ret), xytext=(6, 4),
                 textcoords="offset points", fontsize=9, color="red")
    ax3.set_xlabel("Volatilidad (anual)"); ax3.set_ylabel("Retorno (anual)")
    ax3.set_title("Frontera Eficiente"); ax3.legend(); ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True); plt.close(fig3)

    # Wealth projection
    st.markdown("**Proyección de riqueza (CAGR constante):**")
    years_list = list(range(1, 11)) + [15, 20, 30]
    proj_df = pd.DataFrame({
        "Horizonte (años)": years_list,
        "Valor esperado (€)": [f"{proj[y]:,.0f} €" for y in years_list]
    })
    st.dataframe(proj_df, use_container_width=True, hide_index=True)


# ============================================================
# INIT SESSION STATE
# ============================================================
if "scenA_rows" not in st.session_state:
    st.session_state["scenA_rows"] = [("SPY", 0.8, "USD"), ("GLD", 0.2, "USD")]
if "scenB_rows" not in st.session_state:
    st.session_state["scenB_rows"] = [("IBCF.DE", 0.4, "EUR"), ("GLD", 0.6, "USD")]

# ============================================================
# LAYOUT — 2 columnas
# ============================================================
colA, colB = st.columns(2, gap="large")
with colA:
    render_scenario("scenA", "Escenario A", start_default="2015-01-01", bench_default="SPY")
with colB:
    render_scenario("scenB", "Escenario B", start_default="2010-01-01", bench_default="SPY")
