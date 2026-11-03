from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# ── PALETTE (true black + cyan) ────────────────────────────────────────────
BG    = "#080808"
CARD  = "#111111"
CHART = "#0C0C0C"
CYAN  = "#00E5E5"
TEAL  = "#00A8B5"
BRDR  = "#1E1E1E"
TEXT  = "#F2F2F2"
DIM   = "#9A9A9A"
RED   = "#FB7185"
GOLD  = "#F0A020"

P6 = [CYAN, TEAL, RED, GOLD, "#9B7AE0", "#48C78E"]

S_CYAN = [[0,"#050E0E"],[.35,"#006060"],[.65,"#009898"],[1,"#00E5E5"]]
S_TEAL = [[0,"#040C10"],[.35,"#004858"],[.65,"#007888"],[1,"#00A8B5"]]
S_RED  = [[0,"#150308"],[.35,"#7A0020"],[.65,"#C03050"],[1,"#FB7185"]]
S_WARM = [[0,"#0F0A00"],[.35,"#7A4800"],[.65,"#C07800"],[1,"#F0A020"]]
D_CR   = [[0,"#FB7185"],[.5,"#1A1A1A"],[1,"#00E5E5"]]

# Minimum rows needed before attempting a chart
MIN_ROWS = 3

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*,html,body,[class*="css"]{
  font-family:'Inter',sans-serif !important;
  box-sizing:border-box;
}
.stApp{background:#080808 !important;}
.block-container{padding-top:.6rem !important;max-width:1580px;}

::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:#080808;}
::-webkit-scrollbar-thumb{background:#00E5E5;border-radius:2px;}

h1,h2,h3{color:#F2F2F2 !important;}

/* ── custom nav bar ─────────────────────── */
.nav-bar{
  display:flex;gap:6px;padding:6px;
  background:#111111;border:1px solid #1E1E1E;
  border-radius:10px;margin-bottom:18px;
}
.nav-btn{
  flex:1;padding:8px 0;border:none;border-radius:7px;
  font-family:'Inter',sans-serif;font-size:13px;font-weight:500;
  cursor:pointer;transition:background .15s,color .15s;
  background:transparent;color:#9A9A9A;
}
.nav-btn.active{
  background:#00E5E5;color:#080808;font-weight:600;
}
.nav-btn:hover:not(.active){background:#1E1E1E;color:#F2F2F2;}

/* ── KPI card ───────────────────────────── */
.kpi{
  background:#111111;border:1px solid #1E1E1E;
  border-radius:10px;padding:16px 18px;
  border-top:2px solid #00E5E5;
}
.kpi-label{font-size:11px;font-weight:500;color:#9A9A9A;letter-spacing:.06em;text-transform:uppercase;}
.kpi-value{font-size:28px;font-weight:700;color:#00E5E5;margin:6px 0 2px;line-height:1.1;}
.kpi-note{font-size:12px;color:#9A9A9A;}

/* ── section label ──────────────────────── */
.sec{
  font-size:11px;font-weight:600;letter-spacing:.14em;
  text-transform:uppercase;color:#00E5E5;margin:16px 0 8px;
}

/* ── chart wrapper ──────────────────────── */
div[data-testid="stPlotlyChart"]{
  background:#0C0C0C !important;
  border:1px solid #1E1E1E !important;
  border-radius:10px !important;
  padding:4px !important;
}

/* ── sidebar ─────────────────────────────── */
div[data-testid="stSidebar"]{
  background:#1C1C1C !important;
  border-right:1px solid #2A2A2A !important;
}
div[data-testid="stSidebar"] *{color:#F2F2F2 !important;}
div[data-testid="stSidebar"] .stSlider>div>div>div{background:#00E5E5 !important;}

/* ── tabs ───────────────────────────────── */
div[data-baseweb="tab-list"]{background:#111111 !important;border-radius:8px !important;border:1px solid #1E1E1E !important;}
div[data-baseweb="tab"]{color:#9A9A9A !important;font-size:13px !important;}
div[data-baseweb="tab"][aria-selected="true"]{color:#00E5E5 !important;font-weight:600 !important;}
div[data-baseweb="tab-highlight"]{background:#00E5E5 !important;}

/* ── buttons ─────────────────────────────── */
div[data-testid="stDownloadButton"]>button{
  background:#00E5E5 !important;color:#080808 !important;
  border:none !important;font-weight:600 !important;
  border-radius:8px !important;font-size:13px !important;
}
/* nav buttons (st.button used for page nav) */
div[data-testid="stButton"]>button{
  border-radius:7px !important;
  font-size:13px !important;font-weight:500 !important;
  border:1px solid #1E1E1E !important;
  background:#111111 !important;color:#9A9A9A !important;
  width:100% !important;padding:8px !important;
  transition:all .15s !important;
}
div[data-testid="stButton"]>button:hover{
  background:#1E1E1E !important;color:#F2F2F2 !important;
  border-color:#333 !important;
}

/* inputs */
.stTextInput input{
  background:#111111 !important;color:#F2F2F2 !important;
  border:1px solid #1E1E1E !important;border-radius:8px !important;font-size:13px !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"]{
  background:#111111 !important;border-color:#1E1E1E !important;
}

/* dataframe */
div[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;}

/* alerts */
div[data-testid="stAlert"]{
  background:#111111 !important;border:1px solid #1E1E1E !important;
  color:#F2F2F2 !important;border-radius:10px !important;
}
hr{border-color:#1E1E1E !important;}
</style>
""", unsafe_allow_html=True)


# ── DATA ───────────────────────────────────────────────────────────────────
def read(name):
    p = DATA / name
    return pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()

raw   = read("movie_metadata(3).csv")
clean = read("movie_dataset_cleaned.csv")
eng   = read("feature_engineered_dataset.csv")
enc   = read("movie_dataset_encoded.csv")
sel   = read("movie_dataset_selected_features.csv")
pca_df = read("movie_dataset_pca.csv")
scaled = {n: read(f) for n, f in [
    ("Min-max",      "minmax_scaled_dataset.csv"),
    ("Standardized", "standardized_dataset.csv"),
    ("Normalized",   "normalized_dataset.csv"),
]}
if raw.empty:
    st.error("Raw dataset not found in data/."); st.stop()

for c in ["gross","budget","duration","title_year","imdb_score","num_voted_users",
          "cast_total_facebook_likes","director_facebook_likes","actor_1_facebook_likes",
          "num_critic_for_reviews","num_user_for_reviews","movie_facebook_likes"]:
    if c in raw:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")


# ── HELPERS ────────────────────────────────────────────────────────────────
def cash(x):
    if pd.isna(x): return "—"
    return f"${x/1e9:.2f}B" if abs(x)>=1e9 else (f"${x/1e6:.1f}M" if abs(x)>=1e6 else f"${x:,.0f}")

def kpi(label, value, note=""):
    st.markdown(
        f'<div class="kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div>'
        f'</div>', unsafe_allow_html=True)

def sec(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)

def _style(fig, h):
    """Apply black+cyan theme styling to a Plotly figure."""
    fig.update_layout(
        height=h,
        margin=dict(l=14, r=14, t=42, b=14),
        paper_bgcolor=CHART,
        plot_bgcolor=CHART,
        font=dict(family="Inter", color=TEXT, size=12),
        title_font=dict(family="Inter", size=14, color=TEXT, weight=600),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=DIM, size=11)),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#1A1A1A", font=dict(color=TEXT, size=12), bordercolor=BRDR),
        coloraxis_colorbar=dict(
            tickfont=dict(color=TEXT, size=11),
            title_font=dict(color=TEXT, size=12),
            outlinecolor=BRDR,
        ),
    )
    fig.update_xaxes(
        showgrid=False, linecolor=BRDR,
        tickfont=dict(color=TEXT, size=11),
        title_font=dict(color=TEXT, size=12),
    )
    fig.update_yaxes(
        gridcolor=BRDR, zeroline=False,
        tickfont=dict(color=TEXT, size=11),
        title_font=dict(color=TEXT, size=12),
    )


def plot(fig, h=370):
    """Style and render a Plotly figure; silently skip if it errors."""
    try:
        _style(fig, h)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    except Exception:
        st.caption("_Not enough data to render this chart for the current selection._")


def no_data(msg="Not enough data for the current filter selection."):
    st.markdown(
        f'<div style="background:#111111;border:1px solid #1E1E1E;border-radius:10px;'
        f'padding:20px 18px;color:#9A9A9A;font-size:13px;">{msg}</div>',
        unsafe_allow_html=True)


def enough(data, cols=None, n=MIN_ROWS):
    """Return True only when data has at least n non-null rows for given cols."""
    if isinstance(data, pd.DataFrame):
        if data.empty or len(data) < n:
            return False
        if cols:
            sub = data.dropna(subset=[c for c in cols if c in data])
            return len(sub) >= n
        return True
    if isinstance(data, pd.Series):
        return data.dropna().shape[0] >= n
    return False


# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span style="font-size:16px;font-weight:700;color:#00E5E5">🎬 Movie Success Predictor</span>',
                unsafe_allow_html=True)
    st.markdown('<span style="font-size:10px;color:#9A9A9A;letter-spacing:.1em">BOX OFFICE ANALYTICS</span>',
                unsafe_allow_html=True)
    st.divider()
    st.markdown('<span style="font-size:12px;font-weight:600;color:#F2F2F2">Filters</span>',
                unsafe_allow_html=True)

    df = raw.copy()
    if "title_year" in df and df.title_year.notna().any():
        lo, hi = int(df.title_year.min()), int(df.title_year.max())
        years = st.slider("Year range", lo, hi, (lo, hi))
        df = df[df.title_year.between(*years) | df.title_year.isna()]

    for col, label in [("country","Country"),("content_rating","Rating"),("language","Language")]:
        if col in df:
            opts = sorted(df[col].dropna().astype(str).unique())
            sel_opts = st.multiselect(label, opts)
            if sel_opts:
                df = df[df[col].astype(str).isin(sel_opts)]

    if "genres" in df:
        g_opts = sorted({g.strip() for x in df.genres.dropna().astype(str)
                         for g in x.split("|") if g.strip()})
        g_sel = st.multiselect("Genre", g_opts)
        if g_sel:
            df = df[df.genres.fillna("").astype(str).apply(
                lambda x: any(g in x.split("|") for g in g_sel))]

    if "imdb_score" in df and df.imdb_score.notna().any():
        mn = float(df.imdb_score.min())
        mx = float(df.imdb_score.max())
        if mn < mx:
            sr = st.slider("IMDb score", mn, mx, (mn, mx), step=0.1)
            df = df[df.imdb_score.between(*sr) | df.imdb_score.isna()]
        else:
            st.caption(f"IMDb score: {mn:.1f} (single value)")

    st.divider()
    st.markdown(f'<span style="font-size:12px;color:#9A9A9A">{len(df):,} / {len(raw):,} films</span>',
                unsafe_allow_html=True)


# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="padding:4px 0 14px 0">'
    '<span style="font-size:12px;font-weight:500;color:#00E5E5;'
    'letter-spacing:.14em;text-transform:uppercase">Movie Success Predictor</span>'
    '<h1 style="font-size:28px;font-weight:700;color:#F2F2F2;margin:4px 0 2px">Box-office analytics</h1>'
    '<p style="font-size:13px;color:#9A9A9A;margin:0;line-height:1.5">'
    'Revenue patterns, genre dynamics, production economics and the full '
    'data-preparation trail — from raw metadata to model-ready features.</p>'
    '</div>',
    unsafe_allow_html=True)

# ── CUSTOM NAV (session-state buttons) ─────────────────────────────────────
PAGES = ["Overview", "Patterns", "Talent & Audience", "Data Pipeline", "Model Inputs"]
if "page" not in st.session_state:
    st.session_state.page = "Overview"

# Render nav strip inside a styled container
nav_cols = st.columns(len(PAGES))
for col, page in zip(nav_cols, PAGES):
    with col:
        active = st.session_state.page == page
        # Use st.button; override color via CSS for active state with a unique key
        label_html = f'<span style="color:#080808;font-weight:600">{page}</span>' if active else page
        if st.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page
            st.rerun()

# Inject active styling after buttons render
active_page = st.session_state.page
active_idx = PAGES.index(active_page)
st.markdown(f"""
<style>
/* highlight active nav button — target nth button in the nav row */
div[data-testid="stHorizontalBlock"] > div:nth-child({active_idx + 1})
  div[data-testid="stButton"] > button {{
  background:#00E5E5 !important;
  color:#080808 !important;
  border-color:#00E5E5 !important;
  font-weight:600 !important;
}}
</style>
""", unsafe_allow_html=True)

view = st.session_state.page
st.divider()


# ══════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if view == "Overview":
    gross  = df.gross.dropna()  if "gross"  in df else pd.Series(dtype=float)
    budget = df.budget.dropna() if "budget" in df else pd.Series(dtype=float)
    score  = df.imdb_score.dropna() if "imdb_score" in df else pd.Series(dtype=float)

    if len(df) == 0:
        no_data("No films match the current filter selection. Try broadening your filters.")
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: kpi("Films selected",  f"{len(df):,}",                f"of {len(raw):,}")
        with c2: kpi("Median gross",    cash(gross.median())  if len(gross)  else "—", "box office")
        with c3: kpi("Median budget",   cash(budget.median()) if len(budget) else "—", "production")
        with c4: kpi("Avg IMDb score",  f"{score.mean():.1f}" if len(score) else "—", "out of 10")
        with c5: kpi("Years on record", f"{int(df.title_year.nunique())}" if "title_year" in df else "—", "distinct")

        st.markdown("")
        a, b = st.columns([1.3, 0.9])
        with a:
            if "title_year" in df:
                yr = df.dropna(subset=["title_year"]).groupby("title_year").size().reset_index(name="Films")
                if enough(yr):
                    fig = px.area(yr, x="title_year", y="Films",
                                  title="Films produced per year",
                                  color_discrete_sequence=[CYAN])
                    fig.update_traces(fill="tozeroy", fillcolor="rgba(0,229,229,.08)",
                                      line_color=CYAN, line_width=1.8)
                    plot(fig)
                else:
                    no_data("Not enough data points to draw the timeline.")
        with b:
            if "genres" in df:
                g = (df.genres.dropna().astype(str).str.split("|").explode().str.strip()
                       .value_counts().head(12).reset_index())
                g.columns = ["Genre","Count"]
                if enough(g):
                    fig = px.bar(g.sort_values("Count"), x="Count", y="Genre", orientation="h",
                                 title="Most common genres",
                                 color="Count", color_continuous_scale=S_CYAN)
                    plot(fig)

        a, b = st.columns(2)
        with a:
            if enough(df, ["imdb_score"]):
                d_sc = df.dropna(subset=["imdb_score"])
                fig = px.histogram(d_sc, x="imdb_score",
                                   nbins=min(50, max(5, len(d_sc))),
                                   title="IMDb score distribution",
                                   color_discrete_sequence=[TEAL])
                fig.update_traces(marker_line_width=0, opacity=0.85)
                if len(score):
                    fig.add_vline(x=float(score.mean()), line_dash="dash", line_color=CYAN,
                                  annotation_text=f"avg {score.mean():.1f}",
                                  annotation_font_color=CYAN)
                plot(fig)
            else:
                no_data("Not enough IMDb score data.")
        with b:
            if {"title_year","gross","genres"}.issubset(df.columns) and enough(df, ["gross","genres","title_year"], n=5):
                try:
                    dg = (df.dropna(subset=["gross","genres","title_year"]).copy()
                            .assign(_d=lambda d:(d.title_year//10*10).astype("Int64").astype(str)+"s")
                            .assign(G=lambda d: d.genres.str.split("|"))
                            .explode("G").assign(G=lambda d: d.G.str.strip()))
                    tg = dg.G.value_counts().head(8).index.tolist()
                    dg = dg[dg.G.isin(tg)]
                    if len(dg) >= MIN_ROWS and dg["_d"].nunique() >= 2:
                        pv = (dg.groupby(["_d","G"]).gross.median()/1e6).reset_index()
                        pv.columns = ["Decade","Genre","Med $M"]
                        pv2 = pv.pivot(index="Genre",columns="Decade",values="Med $M").fillna(0)
                        fig = go.Figure(go.Heatmap(
                            z=pv2.values, x=pv2.columns.tolist(), y=pv2.index.tolist(),
                            colorscale=S_CYAN, colorbar=dict(title="$M"),
                            hovertemplate="Decade: %{x}<br>Genre: %{y}<br>$%{z:.1f}M<extra></extra>"))
                        fig.update_layout(title="Genre median gross by decade ($M)")
                        plot(fig)
                    else:
                        no_data("Need films from at least 2 decades for this chart.")
                except Exception:
                    no_data("Could not build the decade heatmap for this selection.")
            else:
                no_data("Not enough data for decade heatmap.")

        a, b = st.columns(2)
        with a:
            if "language" in df:
                lang = df.language.value_counts().head(8).reset_index()
                lang.columns = ["Language","Count"]
                if enough(lang):
                    fig = px.pie(lang, names="Language", values="Count",
                                 title="Language breakdown",
                                 color_discrete_sequence=P6, hole=0.45)
                    fig.update_traces(textfont=dict(color="#F2F2F2"),
                                      marker=dict(line=dict(color=BG, width=2)))
                    plot(fig)
        with b:
            if {"content_rating","gross"}.issubset(df.columns):
                cr = (df.dropna(subset=["content_rating","gross"])
                        .groupby("content_rating")
                        .agg(Films=("gross","size"), Med=("gross","median")).reset_index())
                if len(cr) >= 2:
                    fig = px.treemap(cr, path=["content_rating"], values="Films",
                                     color="Med", color_continuous_scale=S_CYAN,
                                     title="Content rating — size=count, colour=median gross")
                    fig.update_traces(textfont=dict(color="#F2F2F2"))
                    plot(fig)

        a, b = st.columns(2)
        with a:
            if enough(df, ["gross"]):
                fig = px.histogram(df.dropna(subset=["gross"]), x="gross", nbins=60,
                                   title="Box-office gross distribution",
                                   color_discrete_sequence=[RED])
                fig.update_traces(marker_line_width=0, opacity=0.8)
                plot(fig)
        with b:
            bdf = df.dropna(subset=["budget"]) if "budget" in df else pd.DataFrame()
            bdf = bdf[bdf.budget > 0] if not bdf.empty else bdf
            if enough(bdf):
                fig = px.histogram(bdf, x="budget", nbins=60,
                                   title="Production budget distribution",
                                   color_discrete_sequence=[GOLD])
                fig.update_traces(marker_line_width=0, opacity=0.8)
                plot(fig)

    sec("Search the catalogue")
    q = st.text_input("Search by title, director or cast",
                      placeholder="e.g. Nolan, Inception, Meryl Streep …")
    res = df.copy()
    if q:
        cols_ = [c for c in ["movie_title","director_name",
                              "actor_1_name","actor_2_name","actor_3_name"] if c in res]
        mask = pd.Series(False, index=res.index)
        for c in cols_:
            mask |= res[c].fillna("").astype(str).str.contains(q, case=False, regex=False)
        res = res[mask]
    show = [c for c in ["movie_title","title_year","genres","director_name","country",
                         "budget","gross","imdb_score","content_rating"] if c in res]
    if "gross" in res:
        res = res.sort_values("gross", ascending=False, na_position="last")
    st.dataframe(res[show], use_container_width=True, hide_index=True, height=320)
    if len(df) > 0:
        st.download_button("Download filtered records",
                           df.to_csv(index=False).encode(), "filtered_movies.csv","text/csv")


# ══════════════════════════════════════════════════════════════════════════
#  PATTERNS
# ══════════════════════════════════════════════════════════════════════════
elif view == "Patterns":
    if len(df) == 0:
        no_data("No films match the current filters. Try broadening your selection.")
    else:
        sec("Budget & revenue")
        a, b = st.columns(2)
        if {"budget","gross"}.issubset(df.columns):
            d = df.dropna(subset=["budget","gross"]).query("budget>0 and gross>0")
            with a:
                if enough(d):
                    fig = px.scatter(d, x="budget", y="gross",
                                     color="imdb_score" if "imdb_score" in d else None,
                                     size="num_voted_users" if "num_voted_users" in d else None,
                                     hover_name="movie_title",
                                     title="Budget vs gross",
                                     opacity=0.7, color_continuous_scale=S_CYAN)
                    plot(fig)
                else:
                    no_data("Not enough data for budget vs gross.")
            with b:
                if enough(d):
                    d2 = d.copy()
                    d2["ROI %"] = (d2.gross - d2.budget) / d2.budget * 100
                    d2 = d2[(d2["ROI %"] > -100) & (d2["ROI %"] < 1500)]
                    if enough(d2):
                        fig = px.scatter(d2, x="budget", y="ROI %", color="ROI %",
                                         hover_name="movie_title",
                                         title="Budget vs return on investment",
                                         opacity=0.7, color_continuous_scale=D_CR)
                        fig.add_hline(y=0, line_dash="dash", line_color=CYAN, line_width=1,
                                      annotation_text="Break-even", annotation_font_color=CYAN)
                        plot(fig)

        a, b = st.columns(2)
        if {"duration","gross"}.issubset(df.columns):
            dur_d = df.dropna(subset=["duration","gross"])
            with a:
                if enough(dur_d):
                    fig = px.scatter(dur_d, x="duration", y="gross",
                                     color="content_rating" if "content_rating" in df else None,
                                     hover_name="movie_title",
                                     title="Runtime vs gross",
                                     opacity=0.65, color_discrete_sequence=P6)
                    plot(fig)
                else:
                    no_data("Not enough data for runtime chart.")
        if {"title_year","gross"}.issubset(df.columns):
            with b:
                _tyr = df.dropna(subset=["title_year","gross"])
                if _tyr.title_year.nunique() >= 2:
                    try:
                        grp = _tyr.groupby("title_year").gross
                        yt  = grp.agg(Median="median", Mean="mean").reset_index()
                        p90 = grp.quantile(.90).rename("P90").reset_index()
                        p10 = grp.quantile(.10).rename("P10").reset_index()
                        yt  = yt.merge(p90, on="title_year").merge(p10, on="title_year")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=pd.concat([yt.title_year, yt.title_year[::-1]]),
                            y=pd.concat([yt.P90, yt.P10[::-1]]),
                            fill="toself", fillcolor="rgba(0,229,229,.07)",
                            line_color="rgba(0,0,0,0)", showlegend=False))
                        fig.add_trace(go.Scatter(x=yt.title_year, y=yt.Median, name="Median",
                                                 line=dict(color=CYAN, width=2)))
                        fig.add_trace(go.Scatter(x=yt.title_year, y=yt.Mean, name="Mean",
                                                 line=dict(color=RED, width=2, dash="dot")))
                        fig.update_layout(title="Median and mean gross over time")
                        plot(fig)
                    except Exception:
                        no_data("Not enough year data for trend chart.")
                else:
                    no_data("Need films from at least 2 years for the trend chart.")

        sec("Genre breakdown")
        if {"genres","gross"}.issubset(df.columns):
            ex = df.dropna(subset=["genres","gross"]).copy()
            if len(ex) >= MIN_ROWS:
                ex["Genre"] = ex.genres.astype(str).str.split("|")
                ex = ex.explode("Genre"); ex["Genre"] = ex.Genre.str.strip()
                gs = ex.groupby("Genre").gross.agg(
                    Films="size", Median="median", Mean="mean", Std="std").reset_index()
                # Dynamic threshold: at least 1 film per genre when data is small
                min_films = max(1, min(8, len(ex) // 10))
                gs = gs[gs.Films >= min_films]
                if enough(gs):
                    a, b, c = st.columns(3)
                    with a:
                        fig = px.bar(gs.nlargest(14,"Median").sort_values("Median"),
                                     x="Median", y="Genre", orientation="h",
                                     title="Genres by median gross",
                                     color="Median", color_continuous_scale=S_CYAN)
                        plot(fig)
                    with b:
                        if enough(gs, n=2):
                            fig = px.scatter(gs, x="Films", y="Median", size="Mean",
                                             title="Volume vs earning power",
                                             color="Median", color_continuous_scale=S_TEAL,
                                             hover_data={"Genre": True})
                            fig.update_traces(marker=dict(line=dict(color=BRDR, width=0.5)))
                            plot(fig)
                    with c:
                        if gs.Std.notna().any():
                            fig = px.bar(gs.nlargest(12,"Std").sort_values("Std"),
                                         x="Std", y="Genre", orientation="h",
                                         title="Earnings variability",
                                         color="Std", color_continuous_scale=S_RED)
                            plot(fig)
                else:
                    no_data("Not enough genre data for the current selection.")

        a, b = st.columns(2)
        if {"country","gross"}.issubset(df.columns):
            cs = (df.dropna(subset=["country","gross"]).groupby("country").gross
                    .agg(Films="size", Median="median").reset_index())
            # Show all countries that have data; no minimum film floor when filtered
            cs = cs.nlargest(14,"Median")
            with a:
                if enough(cs):
                    fig = px.bar(cs.sort_values("Median"), x="Median", y="country",
                                 orientation="h", title="Countries by median gross",
                                 color="Median", color_continuous_scale=S_TEAL)
                    plot(fig)
                else:
                    no_data("Not enough country data.")
        if {"content_rating","gross"}.issubset(df.columns):
            rs = (df.dropna(subset=["content_rating","gross"]).groupby("content_rating").gross
                    .agg(Median="median", Mean="mean").reset_index())
            with b:
                if enough(rs):
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Median", x=rs.content_rating, y=rs.Median,
                                         marker_color=CYAN, marker_line_color=BG))
                    fig.add_trace(go.Bar(name="Mean",   x=rs.content_rating, y=rs.Mean,
                                         marker_color=TEAL, marker_line_color=BG))
                    fig.update_layout(barmode="group", title="Revenue by content rating")
                    plot(fig)

        a, b = st.columns(2)
        if {"imdb_score","gross"}.issubset(df.columns):
            sc_d = df.dropna(subset=["imdb_score","gross"])
            with a:
                if enough(sc_d):
                    fig = px.scatter(sc_d, x="imdb_score", y="gross", color="imdb_score",
                                     hover_name="movie_title",
                                     title="IMDb score vs gross",
                                     opacity=0.65, color_continuous_scale=S_CYAN)
                    plot(fig)
                else:
                    no_data("Not enough score data.")
        num_cs = [c for c in ["gross","budget","imdb_score","duration","num_voted_users",
                               "cast_total_facebook_likes","num_critic_for_reviews",
                               "movie_facebook_likes"] if c in df]
        if len(num_cs) >= 4:
            with b:
                if len(df) >= 5:
                    try:
                        cm = df[num_cs].dropna().corr()
                        fig = go.Figure(go.Heatmap(
                            z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
                            colorscale=D_CR, zmin=-1, zmax=1,
                            text=np.round(cm.values, 2), texttemplate="%{text}",
                            textfont=dict(color=TEXT, size=10),
                            hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
                            colorbar=dict(title="r")))
                        fig.update_layout(title="Feature correlation matrix")
                        plot(fig, h=400)
                    except Exception:
                        no_data("Correlation matrix unavailable for this selection.")
                else:
                    no_data("Need at least 5 films for a correlation matrix.")

        a, b = st.columns(2)
        if {"budget","gross"}.issubset(df.columns):
            roi = df.dropna(subset=["budget","gross"]).query("budget>0").copy()
            roi["Multiple"] = roi.gross / roi.budget
            roi = roi.replace([np.inf,-np.inf], np.nan).dropna(subset=["Multiple"])
            with a:
                roi_u = roi[roi.Multiple < 10]
                if enough(roi_u):
                    fig = px.histogram(roi_u, x="Multiple", nbins=50,
                                       title="Gross-to-budget multiple (under 10×)",
                                       color_discrete_sequence=[TEAL])
                    fig.update_traces(marker_line_width=0, opacity=0.85)
                    fig.add_vline(x=1, line_dash="dash", line_color=CYAN,
                                  annotation_text="Break-even", annotation_font_color=CYAN)
                    plot(fig)
            with b:
                sec("Best-performing films by ROI")
                sc = [c for c in ["movie_title","title_year","genres",
                                   "budget","gross","Multiple","imdb_score"] if c in roi]
                if not roi.empty:
                    st.dataframe(roi.nlargest(15,"Multiple")[sc],
                                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
#  TALENT & AUDIENCE
# ══════════════════════════════════════════════════════════════════════════
elif view == "Talent & Audience":
    if len(df) == 0:
        no_data("No films match the current filters. Try broadening your selection.")
    else:
        sec("Directors")
        a, b = st.columns(2)
        if {"director_name","gross"}.issubset(df.columns):
            dg = (df.dropna(subset=["director_name","gross"])
                    .groupby("director_name")
                    .agg(Films=("gross","size"), Total=("gross","sum"),
                         Median=("gross","median"),
                         Score=("imdb_score","mean") if "imdb_score" in df else ("gross","mean"))
                    .reset_index())
            # Dynamic film threshold: lower to 1 when few directors
            dir_min = max(1, min(3, len(dg) // 3))
            dg_filtered = dg[dg.Films >= dir_min]
            with a:
                top = dg_filtered.nlargest(15,"Total")
                if enough(top):
                    fig = px.bar(top.sort_values("Total"), x="Total", y="director_name",
                                 orientation="h", title="Directors by total gross",
                                 color="Total", color_continuous_scale=S_CYAN)
                    plot(fig, h=420)
                else:
                    no_data("Not enough director data for this selection.")
            with b:
                if enough(dg_filtered, n=2):
                    fig = px.scatter(dg_filtered, x="Films", y="Median",
                                     size="Total", hover_name="director_name",
                                     title="Films made vs median gross",
                                     color="Score", color_continuous_scale=S_TEAL)
                    fig.update_traces(marker=dict(line=dict(color=BRDR, width=0.5)))
                    plot(fig, h=420)

        sec("Lead actors")
        a, b = st.columns(2)
        if {"actor_1_name","actor_1_facebook_likes","gross"}.issubset(df.columns):
            ad = (df.dropna(subset=["actor_1_name","actor_1_facebook_likes","gross"])
                    .groupby("actor_1_name")
                    .agg(Films=("gross","size"), Likes=("actor_1_facebook_likes","mean"),
                         Median=("gross","median")).reset_index())
            actor_min = max(1, min(3, len(ad) // 3))
            top_a = ad[ad.Films >= actor_min].nlargest(30,"Likes")
            with a:
                if enough(top_a, n=2):
                    fig = px.scatter(top_a, x="Likes", y="Median", size="Films",
                                     hover_name="actor_1_name", color="Median",
                                     title="Actor social reach vs median gross",
                                     color_continuous_scale=S_TEAL)
                    fig.update_traces(marker=dict(line=dict(color=BRDR, width=0.5)))
                    plot(fig, h=420)
                else:
                    no_data("Not enough actor data for this selection.")
        if {"actor_1_name","gross"}.issubset(df.columns):
            ag = (df.dropna(subset=["actor_1_name","gross"])
                    .groupby("actor_1_name")
                    .agg(Films=("gross","size"), Total=("gross","sum")).reset_index())
            actor_min2 = max(1, min(3, len(ag) // 3))
            with b:
                top_ag = ag[ag.Films >= actor_min2].nlargest(20,"Total")
                if enough(top_ag):
                    fig = px.bar(top_ag.sort_values("Total"),
                                 x="Total", y="actor_1_name", orientation="h",
                                 title="Lead actors by total gross",
                                 color="Total", color_continuous_scale=S_RED)
                    plot(fig, h=420)
                else:
                    no_data("Not enough actor data for this selection.")

    sec("Audience signals")
    a, b, c = st.columns(3)
    if {"movie_facebook_likes","gross"}.issubset(df.columns):
        with a:
            fig = px.scatter(df.dropna(subset=["movie_facebook_likes","gross"])
                               .query("movie_facebook_likes>0"),
                             x="movie_facebook_likes", y="gross",
                             color="imdb_score" if "imdb_score" in df else None,
                             hover_name="movie_title",
                             title="Movie FB likes vs gross",
                             opacity=0.65, color_continuous_scale=S_CYAN)
            plot(fig, h=360)
    if {"num_voted_users","imdb_score"}.issubset(df.columns):
        with b:
            fig = px.scatter(df.dropna(subset=["num_voted_users","imdb_score"]),
                             x="num_voted_users", y="imdb_score",
                             color="gross" if "gross" in df else None,
                             hover_name="movie_title",
                             title="Vote count vs IMDb score",
                             opacity=0.65, color_continuous_scale=S_TEAL)
            plot(fig, h=360)
    if {"num_critic_for_reviews","imdb_score"}.issubset(df.columns):
        with c:
            fig = px.scatter(df.dropna(subset=["num_critic_for_reviews","imdb_score"]),
                             x="num_critic_for_reviews", y="imdb_score",
                             color="gross" if "gross" in df else None,
                             hover_name="movie_title",
                             title="Critic reviews vs IMDb score",
                             opacity=0.65, color_continuous_scale=S_CYAN)
            plot(fig, h=360)

    a, b = st.columns(2)
    if {"cast_total_facebook_likes","gross"}.issubset(df.columns):
        with a:
            fig = px.scatter(df.dropna(subset=["cast_total_facebook_likes","gross"])
                               .query("cast_total_facebook_likes>0"),
                             x="cast_total_facebook_likes", y="gross",
                             hover_name="movie_title",
                             title="Cast social reach vs gross",
                             color="imdb_score" if "imdb_score" in df else None,
                             opacity=0.65, color_continuous_scale=S_TEAL)
            plot(fig, h=360)
    if {"content_rating","imdb_score"}.issubset(df.columns):
        with b:
            fig = px.box(df.dropna(subset=["content_rating","imdb_score"]),
                         x="content_rating", y="imdb_score",
                         title="IMDb score by content rating",
                         color="content_rating",
                         color_discrete_sequence=P6)
            fig.update_traces(marker_color=CYAN)
            plot(fig, h=360)

    sec("All-time highest grossing films")
    sc = [c for c in ["movie_title","title_year","director_name","actor_1_name",
                       "genres","budget","gross","imdb_score"] if c in df]
    st.dataframe(df.dropna(subset=["gross"]).nlargest(25,"gross")[sc],
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════
elif view == "Data Pipeline":
    sec("Preprocessing steps")
    t1,t2,t3,t4 = st.tabs(["Cleaning","Feature engineering","Encoding","Scaling"])

    with t1:
        if not clean.empty:
            before = (raw.isna().sum().nlargest(15)
                        .rename("Missing").reset_index().rename(columns={"index":"Column"}))
            after  = (clean.isna().sum().nlargest(15)
                        .rename("Missing").reset_index().rename(columns={"index":"Column"}))
            a, b = st.columns(2)
            with a:
                fig = px.bar(before.sort_values("Missing"), x="Missing", y="Column",
                             orientation="h", title="Missing values — raw",
                             color="Missing", color_continuous_scale=S_RED)
                plot(fig, h=400)
            with b:
                fig = px.bar(after.sort_values("Missing"), x="Missing", y="Column",
                             orientation="h", title="Missing values — after cleaning",
                             color="Missing", color_continuous_scale=S_CYAN)
                plot(fig, h=400)
            a, b = st.columns(2)
            with a:
                pct = (raw.isna().mean()*100).sort_values(ascending=False).head(15)
                fig = go.Figure(go.Bar(
                    x=pct.values, y=pct.index.tolist(),
                    orientation="h", marker_color=RED, marker_line_color=BG))
                fig.update_layout(title="Missing % per column (raw)")
                plot(fig, h=380)
            with b:
                dt = raw.dtypes.value_counts().reset_index()
                dt.columns = ["dtype","Count"]
                dt["dtype"] = dt["dtype"].astype(str)
                fig = px.pie(dt, names="dtype", values="Count",
                             title="Column type breakdown",
                             color_discrete_sequence=P6, hole=0.42)
                fig.update_traces(textfont=dict(color="#F2F2F2"),
                                  marker=dict(line=dict(color=BG, width=2)))
                plot(fig, h=380)
            st.dataframe(clean.head(30), use_container_width=True, hide_index=True)
        else:
            st.warning("Cleaned dataset not found.")

    with t2:
        if not eng.empty:
            new_cols = [c for c in eng.columns if c not in raw.columns]
            desc = {
                "release_decade":"Decade bucket from release year",
                "number_of_genres":"Genre tag count per film",
                "duration_category":"Short / medium / long bucket",
                "budget_category":"Budget tier: low / mid / high / blockbuster",
                "number_of_plot_keywords":"Count of plot keywords listed",
                "cast_popularity":"Sum of lead-cast Facebook likes",
                "movie_age":"Years since release",
                "production_country_count":"Number of production countries",
            }
            st.dataframe(pd.DataFrame(
                [{"Feature":c,"Description":desc.get(c,"Derived field")} for c in new_cols]),
                use_container_width=True, hide_index=True)
            a, b, c = st.columns(3)
            if "duration_category" in eng:
                with a:
                    dc = eng.duration_category.value_counts().reset_index()
                    dc.columns = ["Category","Count"]
                    fig = px.bar(dc, x="Category", y="Count",
                                 title="Runtime category split",
                                 color="Category", color_discrete_sequence=P6)
                    plot(fig)
            if "number_of_genres" in eng:
                with b:
                    fig = px.histogram(eng, x="number_of_genres", nbins=12,
                                       title="Genre count per film",
                                       color_discrete_sequence=[CYAN])
                    fig.update_traces(marker_line_width=0)
                    plot(fig)
            if "budget_category" in eng:
                with c:
                    bc = eng.budget_category.value_counts().reset_index()
                    bc.columns = ["Category","Count"]
                    fig = px.pie(bc, names="Category", values="Count",
                                 title="Budget tier split",
                                 color_discrete_sequence=P6, hole=0.42)
                    fig.update_traces(textfont=dict(color="#F2F2F2"),
                                      marker=dict(line=dict(color=BG, width=2)))
                    plot(fig)
            if "gross" in eng:
                a, b = st.columns(2)
                nf = [c for c in new_cols if c in eng and eng[c].dtype in [np.float64,np.int64]]
                if nf:
                    cr2 = eng[nf+["gross"]].corr(numeric_only=True)["gross"].drop("gross").sort_values()
                    with a:
                        fig = px.bar(cr2.reset_index(), x="gross", y="index",
                                     orientation="h",
                                     title="New features — correlation with gross",
                                     color="gross", color_continuous_scale=D_CR)
                        plot(fig)
                if "movie_age" in eng:
                    with b:
                        fig = px.scatter(eng.dropna(subset=["movie_age","gross"]),
                                         x="movie_age", y="gross",
                                         color="number_of_genres" if "number_of_genres" in eng else None,
                                         title="Film age vs gross",
                                         opacity=0.65, color_continuous_scale=S_CYAN)
                        plot(fig)
            st.dataframe(eng.head(20), use_container_width=True, hide_index=True)
        else:
            st.warning("Feature-engineered dataset not found.")

    with t3:
        if not enc.empty:
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Rows",    f"{enc.shape[0]:,}")
            with c2: st.metric("Columns", f"{enc.shape[1]:,}")
            indicators = [c for c in enc if any(c.startswith(p+"_")
                          for p in ["content_rating","language","country","color"])]
            with c3: st.metric("One-hot columns", len(indicators))
            a, b = st.columns(2)
            with a:
                pfc = {}
                for col in indicators:
                    p = col.split("_")[0]; pfc[p] = pfc.get(p,0)+1
                pf_df = pd.DataFrame(list(pfc.items()), columns=["Prefix","Columns"])
                fig = px.pie(pf_df, names="Prefix", values="Columns",
                             title="One-hot column groups",
                             color_discrete_sequence=P6, hole=0.42)
                fig.update_traces(textfont=dict(color="#F2F2F2"),
                                  marker=dict(line=dict(color=BG, width=2)))
                plot(fig)
            with b:
                ne = enc.select_dtypes(include=np.number).shape[1]
                ce = enc.shape[1] - ne
                fig = go.Figure(go.Bar(
                    x=["Numeric","Categorical / binary"], y=[ne, ce],
                    marker_color=[CYAN, TEAL], marker_line_color=BG))
                fig.update_layout(title="Column types after encoding")
                plot(fig)
            st.dataframe(enc.head(15), use_container_width=True, hide_index=True)
        else:
            st.warning("Encoded dataset not found.")

    with t4:
        sets = [(k,v) for k,v in scaled.items() if not v.empty]
        if sets:
            num_cols = [c for c in sets[0][1].select_dtypes(include=np.number) if c != "gross"]
            feature = st.selectbox("Compare feature across scaling methods", num_cols)
            colors  = [CYAN, RED, GOLD]
            fig = go.Figure()
            for (lbl, data), col in zip(sets, colors):
                if feature in data:
                    fig.add_trace(go.Violin(
                        y=pd.to_numeric(data[feature], errors="coerce").dropna(),
                        name=lbl, box_visible=True, meanline_visible=True,
                        fillcolor=col+"28", line_color=col))
            fig.update_layout(title=f"'{feature}' by scaling method")
            plot(fig, 400)
            fig2 = go.Figure()
            for (lbl, data), col in zip(sets, colors):
                if feature in data:
                    vals = pd.to_numeric(data[feature], errors="coerce").dropna()
                    fig2.add_trace(go.Histogram(x=vals, name=lbl,
                                                opacity=0.6, marker_color=col, nbinsx=50))
            fig2.update_layout(barmode="overlay", title=f"Overlaid distributions — '{feature}'")
            plot(fig2, 340)
            st.dataframe(pd.DataFrame(
                [{"Method":n,"Rows":len(d),"Columns":d.shape[1]} for n,d in sets]),
                use_container_width=True, hide_index=True)
        else:
            st.warning("Scaling output files not found.")


# ══════════════════════════════════════════════════════════════════════════
#  MODEL INPUTS
# ══════════════════════════════════════════════════════════════════════════
elif view == "Model Inputs":
    sec("Feature selection")
    t1,t2,t3 = st.tabs(["Features","PCA","Training setup"])

    with t1:
        if not sel.empty:
            feats = [c for c in sel if c != "gross"]
            c1,c2 = st.columns(2)
            with c1: st.metric("Selected predictors", len(feats))
            with c2: st.metric("Rows available",      f"{sel.shape[0]:,}")
            if "gross" in sel:
                corr = sel.corr(numeric_only=True).gross.drop("gross").sort_values().reset_index()
                corr.columns = ["Feature","r"]
                a, b = st.columns(2)
                with a:
                    fig = px.bar(corr, x="r", y="Feature", orientation="h",
                                 title="All features — correlation with gross",
                                 color="r", color_continuous_scale=D_CR)
                    plot(fig, h=450)
                with b:
                    top15 = corr.reindex(corr.r.abs().sort_values(ascending=False).index).head(15)
                    fig = px.bar(top15, x="r", y="Feature", orientation="h",
                                 title="Top 15 by absolute correlation",
                                 color="r", color_continuous_scale=S_CYAN)
                    plot(fig, h=450)
            st.dataframe(sel.head(25), use_container_width=True, hide_index=True)
        else:
            st.warning("Selected features file not found.")

    with t2:
        if not pca_df.empty:
            pcs = [c for c in pca_df if c.startswith("PC")]
            c1,c2 = st.columns(2)
            with c1: st.metric("Principal components", len(pcs))
            with c2: st.metric("Rows", f"{pca_df.shape[0]:,}")
            st.caption("15 components retain 95.57% of variance.")
            if len(pcs) >= 2:
                a, b = st.columns(2)
                with a:
                    fig = px.scatter(pca_df, x=pcs[0], y=pcs[1],
                                     color="gross" if "gross" in pca_df else None,
                                     title=f"{pcs[0]} vs {pcs[1]}",
                                     opacity=0.55, color_continuous_scale=S_CYAN)
                    plot(fig, h=380)
                with b:
                    fig = px.box(pca_df[pcs[:min(10,len(pcs))]],
                                 title="Score spread across components",
                                 color_discrete_sequence=P6)
                    fig.update_traces(marker_color=CYAN)
                    plot(fig, h=380)
            if len(pcs) >= 3:
                a, b = st.columns(2)
                with a:
                    sample = pca_df.sample(min(1000,len(pca_df)), random_state=42)
                    fig = px.scatter_3d(sample, x=pcs[0], y=pcs[1], z=pcs[2],
                                        color="gross" if "gross" in pca_df else None,
                                        title="3-D component view",
                                        opacity=0.55, color_continuous_scale=S_TEAL)
                    fig.update_layout(
                        height=420, paper_bgcolor=CHART,
                        font=dict(color=TEXT, family="Inter", size=11),
                        title_font=dict(color=TEXT, size=14, family="Inter"))
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
                with b:
                    stds = pca_df[pcs].std().values
                    vr = (stds**2)/(stds**2).sum()
                    cv = np.cumsum(vr)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=pcs, y=vr*100, name="Per component",
                                         marker_color=CYAN, marker_line_color=BG))
                    fig.add_trace(go.Scatter(x=pcs, y=cv*100, name="Cumulative",
                                             mode="lines+markers",
                                             line=dict(color=RED, width=2),
                                             marker=dict(size=6, color=GOLD)))
                    fig.update_layout(title="Variance explained", yaxis_title="Variance (%)")
                    plot(fig, h=420)
            if len(pcs) >= 4:
                a, b = st.columns(2)
                with a:
                    fig = px.scatter(pca_df, x=pcs[1], y=pcs[2],
                                     color="gross" if "gross" in pca_df else None,
                                     title=f"{pcs[1]} vs {pcs[2]}",
                                     opacity=0.55, color_continuous_scale=S_RED)
                    plot(fig, h=350)
                with b:
                    fig = px.scatter(pca_df, x=pcs[2], y=pcs[3],
                                     color="gross" if "gross" in pca_df else None,
                                     title=f"{pcs[2]} vs {pcs[3]}",
                                     opacity=0.55, color_continuous_scale=S_TEAL)
                    plot(fig, h=350)
            st.dataframe(pca_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.warning("PCA output not found.")

    with t3:
        st.info("Target: **gross** (box-office revenue). Model training is the next step.")
        c1,c2,c3 = st.columns(3)
        with c1: kpi("Target variable",    "gross",          "USD box-office revenue")
        with c2: kpi("Suggested model",    "Random Forest",  "Handles mixed types well")
        with c3: kpi("Train / test split", "80 / 20",        "Stratified by year")
        st.markdown("""
**Before training**

- Use the selected-feature dataset, not the full encoded matrix.
- Fit scalers on the training fold only — never on the full dataset.
- Log-transform `gross` and `budget` to reduce skew.
- Run 5-fold cross-validation for a reliable performance estimate.
- Avoid post-release features (vote counts, review counts) for pre-release predictions.
""")
        if not sel.empty:
            train_n = int(0.8*len(sel))
            test_n  = len(sel) - train_n
            a, b = st.columns(2)
            with a:
                fig = go.Figure(go.Pie(
                    labels=["Train (80%)","Test (20%)"],
                    values=[train_n, test_n], hole=0.5,
                    marker=dict(colors=[CYAN, TEAL],
                                line=dict(color=BG, width=2))))
                fig.update_traces(textfont=dict(color="#F2F2F2"))
                fig.update_layout(
                    title="Train / test split",
                    paper_bgcolor=CHART,
                    font=dict(color=TEXT, family="Inter"),
                    title_font=dict(color=TEXT, size=14, family="Inter"),
                    legend=dict(font=dict(color=DIM)))
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
            with b:
                if "gross" in sel:
                    fig = px.histogram(sel.dropna(subset=["gross"]), x="gross",
                                       nbins=50, title="Target variable (gross) distribution",
                                       color_discrete_sequence=[GOLD])
                    fig.update_traces(marker_line_width=0)
                    plot(fig, h=320)
            st.dataframe(sel.head(25), use_container_width=True, hide_index=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<span style="font-size:12px;color:#2E2E2E;">'
    'Movie Success Predictor · built on IMDb metadata and project preprocessing outputs'
    '</span>', unsafe_allow_html=True)
