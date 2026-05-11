"""
IBIS — Impulse Buying Intelligence System
Cyberpunk-clean dark theme. Top nav with proper headings.
Run with:  streamlit run ibis_app_clean.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import norm, binom, poisson

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IBIS — Impulse Buying Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main { background: #080C14 !important; }

[data-testid="stAppViewContainer"]::before { display:none !important; }
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

section.main > div.block-container {
    padding: 0 0 60px 0 !important;
    max-width: 100% !important;
}

[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed; inset: 0; pointer-events: none; z-index: 9999;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,200,0.012) 2px, rgba(0,255,200,0.012) 4px
    );
}

.ibis-navbar {
    background: rgba(8,12,20,0.97);
    border-bottom: 1px solid rgba(0,255,180,0.2);
    padding: 0 44px;
    height: 70px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky; top: 0; z-index: 999;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 40px rgba(0,255,180,0.07), 0 1px 0 rgba(0,255,180,0.15);
}
.ibis-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.15em;
    color: #00FFBC;
    text-shadow: 0 0 30px rgba(0,255,188,0.6), 0 0 60px rgba(0,255,188,0.2);
    line-height: 1;
}
.ibis-logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: rgba(0,255,188,0.4);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    display: block;
    margin-top: 2px;
}
.ibis-nav-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: rgba(255,255,255,0.22);
    letter-spacing: 0.1em;
}

.ibis-body {
    padding: 36px 44px 0 44px;
    max-width: 1340px;
    margin: 0 auto;
}

.ibis-page-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #00FFBC;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.7;
}
.ibis-page-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.4rem;
    letter-spacing: 0.06em;
    color: #EEEEF4;
    line-height: 0.95;
    text-shadow: 0 0 60px rgba(0,255,188,0.1);
    margin-bottom: 8px;
}
.ibis-page-desc {
    font-family: 'Syne', sans-serif;
    font-size: 0.86rem;
    color: rgba(255,255,255,0.35);
    letter-spacing: 0.02em;
    font-weight: 400;
    max-width: 600px;
    line-height: 1.65;
    margin-bottom: 28px;
}

[data-testid="stRadio"] > label { display: none !important; }
.stRadio > div {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap;
    gap: 6px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 0 28px 0 !important;
}
.stRadio div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center;
    cursor: pointer;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px;
    padding: 8px 18px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: rgba(255,255,255,0.4) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: all 0.2s ease;
}
.stRadio div[role="radiogroup"] > label:hover {
    border-color: rgba(0,255,188,0.35);
    color: rgba(0,255,188,0.8) !important;
    background: rgba(0,255,188,0.04);
}
.stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(0,255,188,0.1);
    border-color: #00FFBC;
    color: #00FFBC !important;
    box-shadow: 0 0 16px rgba(0,255,188,0.18);
}
.stRadio div[role="radiogroup"] > label input[type="radio"] { display: none !important; }

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: 2px solid #00FFBC !important;
    border-radius: 6px !important;
    padding: 20px 22px !important;
    box-shadow: none !important;
    transform: none !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #EEEEF4 !important;
    text-shadow: 0 0 18px rgba(0,255,188,0.2) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: rgba(255,255,255,0.32) !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #00FFBC !important;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    color: #EEEEF4 !important;
    letter-spacing: 0.08em !important;
    font-weight: 400 !important;
    text-shadow: none !important;
}
h2 { font-size: 1.7rem !important; margin-bottom: 14px !important; }
h3 { font-size: 1.35rem !important; margin-bottom: 10px !important; }

hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.07) !important; margin: 28px 0 !important; }

.insight-card {
    background: rgba(0,255,188,0.05);
    border: 1px solid rgba(0,255,188,0.18);
    border-left: 3px solid #00FFBC;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 18px 0;
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.65);
    line-height: 1.65;
}
.insight-card strong { color: #00FFBC; }

.info-banner {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    padding: 12px 18px;
    margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.03em;
}
.info-banner strong { color: #00FFBC; }

.dist-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    padding: 18px 20px;
    margin: 8px 0;
}
.dist-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 0.08em;
    color: #EEEEF4;
    margin-bottom: 6px;
}
.discrete-badge {
    display: inline-block;
    background: rgba(255,190,0,0.1);
    color: #FFBE00;
    border: 1px solid rgba(255,190,0,0.25);
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.continuous-badge {
    display: inline-block;
    background: rgba(0,180,255,0.1);
    color: #00B4FF;
    border: 1px solid rgba(0,180,255,0.25);
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.dist-desc {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.4);
    line-height: 1.6;
}

.badge-buy {
    background: rgba(0,255,188,0.1);
    color: #00FFBC;
    border: 1px solid rgba(0,255,188,0.35);
    padding: 10px 26px;
    border-radius: 4px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 0.15em;
    display: inline-block;
    margin: 10px 0;
    box-shadow: 0 0 22px rgba(0,255,188,0.15);
}
.badge-no {
    background: rgba(255,60,80,0.1);
    color: #FF3C50;
    border: 1px solid rgba(255,60,80,0.35);
    padding: 10px 26px;
    border-radius: 4px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 0.15em;
    display: inline-block;
    margin: 10px 0;
    box-shadow: 0 0 22px rgba(255,60,80,0.12);
}

.prob-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 3px; height: 10px;
    overflow: hidden; margin: 4px 0 12px 0;
}
.prob-fill {
    background: #00FFBC; height: 10px; border-radius: 3px;
    box-shadow: 0 0 10px rgba(0,255,188,0.4);
}
.prob-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.38);
    margin-bottom: 3px;
}

.reason-yes {
    background: rgba(0,255,188,0.05);
    border: 1px solid rgba(0,255,188,0.14);
    border-left: 2px solid #00FFBC;
    border-radius: 4px; padding: 9px 14px; margin: 4px 0;
    font-family: 'Syne', sans-serif; font-size: 0.85rem;
    color: rgba(255,255,255,0.62);
}
.reason-no {
    background: rgba(255,60,80,0.05);
    border: 1px solid rgba(255,60,80,0.14);
    border-left: 2px solid #FF3C50;
    border-radius: 4px; padding: 9px 14px; margin: 4px 0;
    font-family: 'Syne', sans-serif; font-size: 0.85rem;
    color: rgba(255,255,255,0.52);
}

.stButton button {
    background: transparent !important;
    color: #00FFBC !important;
    border: 1px solid rgba(0,255,188,0.45) !important;
    border-radius: 4px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    padding: 10px 28px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    transform: none !important;
    box-shadow: none !important;
}
.stButton button:hover {
    background: rgba(0,255,188,0.07) !important;
    border-color: #00FFBC !important;
    box-shadow: 0 0 18px rgba(0,255,188,0.18) !important;
    transform: none !important;
}

.stAlert {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
}

.stDataFrame { border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 6px !important; }
[data-testid="stDataFrame"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.79rem !important;
}

[data-testid="stSlider"] label,
.stSelectbox label, .stToggle label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    color: rgba(255,255,255,0.5) !important;
    font-weight: 600 !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ─── COLOURS ────────────────────────────────────────────────────────────────
CYAN  = "#00FFBC"
PINK  = "#FF3C50"
AMBER = "#FFBE00"
BLUE  = "#00B4FF"
BG    = "#080C14"
CARD  = "#0D1320"
TEXT  = "#EEEEF4"
MUTED = "#3A4055"

MONTH_ORDER = ["Feb","Mar","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─── MATPLOTLIB THEME ───────────────────────────────────────────────────────
def cp_fig(w=10, h=4, subplots=(1,1)):
    fig, axes = plt.subplots(*subplots, figsize=(w,h))
    fig.patch.set_facecolor(CARD)
    ax_list = axes.flatten() if hasattr(axes,"flatten") else [axes]
    for ax in ax_list:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color("#1A2035"); spine.set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        ax.xaxis.label.set_color(MUTED); ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_color(MUTED); ax.yaxis.label.set_fontsize(9)
        ax.title.set_color(TEXT); ax.title.set_fontsize(11); ax.title.set_fontweight("bold")
        ax.grid(True, color="#131A28", linestyle="-", linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, (axes if subplots!=(1,1) else ax_list[0])

# ─── DATA ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading shopper data…")
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "online_shoppers_intention.csv")
    if not os.path.exists(path):
        st.error(f"CSV not found at: {path}"); st.stop()
    df = pd.read_csv(path)
    df["Revenue"] = df["Revenue"].astype(bool)
    df["Weekend"] = df["Weekend"].astype(bool)
    return df

@st.cache_resource(show_spinner="Training regression model…")
def train_models(df_hash):
    df = load_data(); df2 = df.copy()
    le_m = LabelEncoder().fit(df2["Month"])
    le_v = LabelEncoder().fit(df2["VisitorType"])
    df2["Month_enc"]       = le_m.transform(df2["Month"])
    df2["VisitorType_enc"] = le_v.transform(df2["VisitorType"])
    df2["Weekend_int"]     = df2["Weekend"].astype(int)
    feats = ["ProductRelated","ProductRelated_Duration","BounceRates","ExitRates",
             "PageValues","SpecialDay","Month_enc","VisitorType_enc",
             "Weekend_int","Administrative","Informational"]
    X, y = df2[feats], df2["Revenue"].astype(int)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    pipe = Pipeline([("sc",StandardScaler()),
                     ("lr",LogisticRegression(max_iter=1000,C=1.0,random_state=42))])
    pipe.fit(Xtr,ytr)
    acc = accuracy_score(yte, pipe.predict(Xte))
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:,1])
    coefs = pipe.named_steps["lr"].coef_[0]
    return {"lr":pipe,"features":feats,"le_month":le_m,"le_visitor":le_v,
            "lr_acc":acc,"lr_auc":auc,"coefs":coefs,"odds_ratios":np.exp(coefs),
            "X_test":Xte,"y_test":yte}

# ─── TOP NAV ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ibis-navbar">
  <div>
    <div class="ibis-logo">IBIS
      <span class="ibis-logo-sub">Impulse Buying Intelligence System</span>
    </div>
  </div>
  <div class="ibis-nav-meta">12,330 sessions &nbsp;·&nbsp; UCI Dataset &nbsp;·&nbsp; Regression Model</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="ibis-body">', unsafe_allow_html=True)

# ─── NAVIGATION ─────────────────────────────────────────────────────────────
page = st.radio("nav",
    ["Overview","Monthly Trends","Visitor Profiles","Behaviour Analysis",
     "Traffic Sources","Temporal Patterns","Distributions","Regression Model","Purchase Predictor"],
    horizontal=True, label_visibility="collapsed")

df   = load_data()
mods = train_models(id(df))

PAGE_META = {
    "Overview":           ("01 — OVERVIEW",           "Command Centre",            "High-level snapshot of 12,330 online shopper sessions"),
    "Monthly Trends":     ("02 — MONTHLY TRENDS",      "Monthly Intelligence",      "Which months bring real buyers vs window-shoppers?"),
    "Visitor Profiles":   ("03 — VISITOR PROFILES",    "Visitor Profiles",          "New vs returning — and do sale days actually work?"),
    "Behaviour Analysis": ("04 — BEHAVIOUR ANALYSIS",  "Behaviour Analysis",        "What buyers actually do differently from non-buyers"),
    "Traffic Sources":    ("05 — TRAFFIC SOURCES",     "Traffic Sources",           "Which channels bring real buyers?"),
    "Temporal Patterns":  ("06 — TEMPORAL PATTERNS",   "Temporal Patterns",         "Weekend vs weekday — does timing change buying intent?"),
    "Distributions":      ("07 — DISTRIBUTIONS",       "Probability Distributions", "Discrete & continuous distributions applied to shopper data"),
    "Regression Model":   ("08 — REGRESSION MODEL",    "Regression Model",          "Linear regression y = a + bx, single & multiple"),
    "Purchase Predictor": ("09 — PREDICTOR",           "Purchase Predictor",        "Enter visitor signals — regression model predicts buy probability"),
}
eyebrow, title, desc = PAGE_META[page]
st.markdown(f"""
<div class="ibis-page-eyebrow">{eyebrow}</div>
<div class="ibis-page-title">{title}</div>
<div class="ibis-page-desc">{desc}</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Overview":
    total  = len(df); buyers = int(df["Revenue"].sum()); pct = buyers/total*100
    wc = df[df["Weekend"]]["Revenue"].mean()*100
    nc = df[df["VisitorType"]=="New_Visitor"]["Revenue"].mean()*100
    rc = df[df["VisitorType"]=="Returning_Visitor"]["Revenue"].mean()*100
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Sessions",f"{total:,}")
    c2.metric("Conversion Rate",f"{pct:.1f}%",f"{buyers:,} purchases")
    c3.metric("Returning CVR",f"{rc:.1f}%",f"+{rc-nc:.1f}% vs new")
    c4.metric("Weekend CVR",f"{wc:.1f}%")
    st.markdown("---")
    m1,m2 = st.columns(2)
    m1.metric("Regression Accuracy",f"{mods['lr_acc']*100:.1f}%")
    m2.metric("AUC-ROC Score",f"{mods['lr_auc']:.3f}")
    st.markdown("---")
    st.markdown("### Purchase Split & Session Volume")
    ca,cb = st.columns(2)
    with ca:
        fig,ax = cp_fig(5.5,4.5)
        ax.set_facecolor(CARD)
        _,_,auts = ax.pie([buyers,total-buyers],labels=["Purchased","No Purchase"],
            autopct="%1.1f%%",colors=[CYAN,MUTED],startangle=140,
            wedgeprops={"edgecolor":BG,"linewidth":2.5},
            textprops={"color":TEXT,"fontsize":10,"fontfamily":"Syne"})
        for a in auts: a.set_color(TEXT); a.set_fontsize(10); a.set_fontweight("bold")
        ax.set_title("Buyers vs Non-Buyers",color=TEXT,fontsize=11,fontweight="bold")
        st.pyplot(fig, use_container_width=True)
    with cb:
        fig2,ax2 = cp_fig(5.5,4.5)
        bars = ax2.bar(["Non-Buyers","Buyers"],[total-buyers,buyers],
                       color=[MUTED,CYAN],width=0.5,edgecolor=BG,linewidth=2)
        for bar,val in zip(bars,[total-buyers,buyers]):
            ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+60,
                     f"{val:,}",ha="center",color=TEXT,fontsize=11,fontweight="bold")
        ax2.set_title("Session Volume: Buyers vs Non-Buyers")
        ax2.set_ylabel("Sessions")
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True)
    st.markdown("""<div class="insight-card">⚡ Only <strong>1 in 6</strong> visitors makes a purchase.
Returning shoppers convert at nearly <strong>3× the rate</strong> of first-time visitors —
retention is the highest-leverage growth lever available.</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  MONTHLY TRENDS
# ════════════════════════════════════════════════════════════
elif page == "Monthly Trends":
    monthly = df.groupby("Month").agg(Visits=("Revenue","count"),Purchases=("Revenue","sum")).reset_index()
    monthly["Conversion %"] = monthly["Purchases"]/monthly["Visits"]*100
    monthly["Month_sort"] = monthly["Month"].map({m:i for i,m in enumerate(MONTH_ORDER)}).fillna(99)
    monthly = monthly.sort_values("Month_sort").reset_index(drop=True)
    fig,axes = cp_fig(13,5,(1,2)); ax1,ax2 = axes.flatten()
    x = np.arange(len(monthly))
    ax1.bar(x-0.2,monthly["Visits"],width=0.38,color=MUTED,label="Total Visits",edgecolor=BG,linewidth=1)
    ax1.bar(x+0.2,monthly["Purchases"],width=0.38,color=CYAN,label="Purchases",edgecolor=BG,linewidth=1)
    ax1.set_xticks(x); ax1.set_xticklabels(monthly["Month"],rotation=30)
    ax1.set_title("Monthly Traffic vs Purchases")
    ax1.legend(facecolor=CARD,labelcolor=TEXT,fontsize=9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    colours = [CYAN if v>15 else (AMBER if v>8 else MUTED) for v in monthly["Conversion %"]]
    ax2.bar(x,monthly["Conversion %"],color=colours,width=0.6,edgecolor=BG,linewidth=1)
    ax2.plot(x,monthly["Conversion %"],color=TEXT,linewidth=1.8,marker="o",markersize=4.5,zorder=5)
    avg_cvr = monthly["Conversion %"].mean()
    ax2.axhline(avg_cvr,color=PINK,linestyle="--",linewidth=1.5,label=f"Avg {avg_cvr:.1f}%")
    ax2.set_xticks(x); ax2.set_xticklabels(monthly["Month"],rotation=30)
    ax2.set_title("Monthly Conversion Rate (%)")
    ax2.legend(facecolor=CARD,labelcolor=TEXT,fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)
    st.markdown("---")
    disp = monthly[["Month","Visits","Purchases","Conversion %"]].copy()
    disp["Conversion %"] = disp["Conversion %"].round(1)
    st.dataframe(disp.rename(columns={"Conversion %":"Conversion Rate (%)"}),use_container_width=True,hide_index=True)
    best = monthly.loc[monthly["Conversion %"].idxmax()]
    st.markdown(f"""<div class="insight-card">⚡ <strong>{best['Month']}</strong> achieves the highest conversion at
<strong>{best['Conversion %']:.1f}%</strong> — likely driven by seasonal demand or holiday promotions.</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  VISITOR PROFILES
# ════════════════════════════════════════════════════════════
elif page == "Visitor Profiles":
    VLABELS = {"Returning_Visitor":"Returning Shoppers","New_Visitor":"First-Time Visitors","Other":"Other / Unknown"}
    vt = df.groupby("VisitorType")["Revenue"].agg(["count","sum","mean"]).reset_index()
    vt.columns = ["Type","Visits","Purchases","Rate"]; vt["Rate"]*=100
    vt["Label"] = vt["Type"].map(VLABELS)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Visitor Mix")
        fig,ax = cp_fig(5.5,4.5); ax.set_facecolor(CARD)
        ax.pie(vt["Visits"],labels=vt["Label"],autopct="%1.1f%%",colors=[CYAN,AMBER,BLUE],
               wedgeprops={"edgecolor":BG,"linewidth":2},textprops={"color":TEXT,"fontsize":9,"fontfamily":"Syne"})
        ax.set_title("Share of Visitor Types",color=TEXT); st.pyplot(fig,use_container_width=True)
    with c2:
        st.markdown("### Conversion Rate by Visitor Type")
        fig2,ax2 = cp_fig(5.5,4.5)
        cv = [CYAN if r>15 else (AMBER if r>8 else MUTED) for r in vt["Rate"]]
        bars = ax2.barh(vt["Label"],vt["Rate"],color=cv,height=0.45,edgecolor=BG,linewidth=1)
        for bar,val in zip(bars,vt["Rate"]):
            ax2.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                     f"{val:.1f}%",va="center",color=TEXT,fontsize=9,fontweight="bold")
        ax2.set_title("Buy Rate by Visitor Type")
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig2,use_container_width=True)
    st.markdown("---")
    st.markdown("### Sale Day Proximity Effect")
    sd = df.groupby("SpecialDay")["Revenue"].agg(["count","sum","mean"]).reset_index()
    sd["mean"]*=100
    sd_labels = {0:"Normal Day",0.2:"Near a Sale",0.4:"Mid Sale",0.6:"Close to Sale",0.8:"Almost Sale",1.0:"Peak Sale Day"}
    sd["Label"] = sd["SpecialDay"].map(sd_labels)
    fig3,ax3 = cp_fig(9,4.5)
    cs = [CYAN if v>15 else (AMBER if v>8 else MUTED) for v in sd["mean"]]
    bars3 = ax3.bar(sd["Label"],sd["mean"],color=cs,width=0.55,edgecolor=BG,linewidth=1)
    for bar,val in zip(bars3,sd["mean"]):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                 f"{val:.1f}%",ha="center",color=TEXT,fontsize=9,fontweight="bold")
    ax3.set_title("Purchase Rate (%) vs Sale Day Proximity"); ax3.set_ylabel("Conversion Rate (%)")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig3,use_container_width=True)
    nr = float(sd[sd["SpecialDay"]==0]["mean"].iloc[0]) if not sd[sd["SpecialDay"]==0].empty else 0.0
    st.markdown(f"""<div class="insight-card">⚡ On a normal day the purchase rate is <strong>{nr:.1f}%</strong>.
Near peak sale days, traffic surges but many are window-shopping — conversion can actually dip.</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  BEHAVIOUR ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "Behaviour Analysis":
    bdf = df[df["Revenue"]]; ndf = df[~df["Revenue"]]
    metrics = [("Product Pages Browsed","ProductRelated"),("Time on Product Pages (secs)","ProductRelated_Duration"),
               ("Page Value Score","PageValues"),("Bounce Rate","BounceRates"),
               ("Exit Rate","ExitRates"),("Account Pages Visited","Administrative")]
    rows = []
    for label,col in metrics:
        b,n = bdf[col].mean(), ndf[col].mean()
        rows.append({"Signal":label,"Buyers (avg)":round(b,2),"Non-Buyers (avg)":round(n,2),"Difference":round(b-n,2)})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("---"); st.markdown("### Buyer vs Non-Buyer Comparison")
    fig,axes = cp_fig(14,8,(2,3)); al = axes.flatten()
    for i,(label,col) in enumerate(metrics):
        ax = al[i]; ba,na = bdf[col].mean(),ndf[col].mean()
        bars = ax.bar([0,1],[ba,na],color=[CYAN,PINK],width=0.45,edgecolor=BG,linewidth=1)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Buyers","Non-Buyers"],fontsize=9)
        ax.set_title(label)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for bar,val in zip(bars,[ba,na]):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.03,
                    f"{val:.1f}",ha="center",color=TEXT,fontsize=8,fontweight="bold")
    plt.tight_layout(pad=2.5); st.pyplot(fig,use_container_width=True)
    st.markdown("""<div class="insight-card">⚡ <strong>Page Value Score</strong> is the clearest separator —
buyers land on high-value checkout-linked pages far more often than non-buyers.</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TRAFFIC SOURCES
# ════════════════════════════════════════════════════════════
elif page == "Traffic Sources":
    SOURCE = {1:"Direct Link",2:"Organic Search",3:"Social Media",4:"Email Campaign",
              5:"Display Ad",6:"Referral Site",7:"Affiliate",8:"Paid Search",
              9:"Video Ad",10:"Newsletter",11:"Other Ad",12:"Push Notification"}
    tr = df.groupby("TrafficType")["Revenue"].agg(["count","sum","mean"]).reset_index()
    tr.columns = ["Type","Visits","Purchases","Rate"]; tr["Rate"]*=100
    tr["Source"] = tr["Type"].map(SOURCE).fillna(tr["Type"].apply(lambda x:f"Source {int(x)}"))
    tr = tr.sort_values("Visits",ascending=False).head(12)
    fig,axes = cp_fig(13,5,(1,2)); ax1,ax2 = axes.flatten()
    ts = tr.sort_values("Visits")
    ax1.barh(ts["Source"],ts["Visits"],color=CYAN,height=0.6,edgecolor=BG,linewidth=0.5)
    ax1.set_title("Total Visits by Source")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    c2 = [CYAN if r>15 else (AMBER if r>8 else MUTED) for r in ts["Rate"]]
    ax2.barh(ts["Source"],ts["Rate"],color=c2,height=0.6,edgecolor=BG,linewidth=0.5)
    ax2.axvline(tr["Rate"].mean(),color=PINK,linestyle="--",linewidth=1.5,label=f"Avg {tr['Rate'].mean():.1f}%")
    ax2.set_title("Conversion Rate by Source (%)"); ax2.legend(facecolor=CARD,labelcolor=TEXT,fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True)
    bs = tr.loc[tr["Rate"].idxmax(),"Source"]; br = tr["Rate"].max()
    st.markdown(f"""<div class="insight-card">⚡ <strong>{bs}</strong> achieves the highest conversion at
<strong>{br:.1f}%</strong>. High-volume, low-converting channels need better landing pages.</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TEMPORAL PATTERNS
# ════════════════════════════════════════════════════════════
elif page == "Temporal Patterns":
    wk = df.groupby("Weekend")["Revenue"].agg(["count","sum","mean"]).reset_index()
    wk.columns = ["Weekend","Visits","Purchases","Rate"]; wk["Rate"]*=100
    wk["Label"] = wk["Weekend"].map({False:"Weekday",True:"Weekend"})
    c1,c2 = st.columns(2)
    for _,row in wk.iterrows():
        col = c1 if not row["Weekend"] else c2
        col.metric(f"{row['Label']} — Conversion",f"{row['Rate']:.1f}%")
        col.metric(f"{row['Label']} — Sessions",f"{row['Visits']:,}")
    st.markdown("---"); st.markdown("### Weekend vs Weekday Comparison")
    fig,axes = cp_fig(10,4.5,(1,2)); ax1,ax2 = axes.flatten()
    ax1.bar(wk["Label"],wk["Visits"],color=[MUTED,CYAN],width=0.45,edgecolor=BG,linewidth=1)
    ax1.set_title("Total Session Volume"); ax1.set_ylabel("Sessions")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    for bar,val in zip(ax1.patches,wk["Visits"]):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+50,
                 f"{val:,}",ha="center",color=TEXT,fontsize=11,fontweight="bold")
    ax2.bar(wk["Label"],wk["Rate"],color=[MUTED,CYAN],width=0.45,edgecolor=BG,linewidth=1)
    ax2.set_title("Purchase Rate (%)"); ax2.set_ylabel("Conversion Rate (%)")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    for bar,val in zip(ax2.patches,wk["Rate"]):
        ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                 f"{val:.1f}%",ha="center",color=TEXT,fontsize=11,fontweight="bold")
    plt.tight_layout(); st.pyplot(fig,use_container_width=True)
    wr = float(wk[wk["Weekend"]==True]["Rate"].iloc[0]); dr = float(wk[wk["Weekend"]==False]["Rate"].iloc[0])
    diff = wr-dr
    st.markdown(f"""<div class="insight-card">⚡ Weekend purchase rates are <strong>{abs(diff):.1f}%</strong>
{"higher" if diff>0 else "lower"} than weekdays.
{"Shoppers browse leisurely on weekends and buy more impulsively." if diff>0 else "Weekday visitors tend to be more purposeful."}</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  DISTRIBUTIONS
# ════════════════════════════════════════════════════════════
elif page == "Distributions":
    cd1,cd2 = st.columns(2)
    with cd1:
        st.markdown("""
<div class="dist-card"><div class="dist-name">Binomial Distribution</div><span class="discrete-badge">Discrete</span>
<div class="dist-desc">Number of successes in <em>n</em> independent trials each with probability <em>p</em>.
Applied: probability of exactly k buyers in 20 sessions.</div></div>
<div class="dist-card" style="margin-top:10px"><div class="dist-name">Poisson Distribution</div><span class="discrete-badge">Discrete</span>
<div class="dist-desc">Count of events in a fixed interval at average rate λ.
Applied: average purchases per day in November.</div></div>
<div class="dist-card" style="margin-top:10px"><div class="dist-name">Hypergeometric Distribution</div><span class="discrete-badge">Discrete</span>
<div class="dist-desc">Sampling without replacement from a finite population.
Applied: probability exactly k of 50 sessions are buyers from our 12,330 records.</div></div>
""", unsafe_allow_html=True)
    with cd2:
        st.markdown("""
<div class="dist-card"><div class="dist-name">Uniform Distribution</div><span class="continuous-badge">Continuous</span>
<div class="dist-desc">Equal probability across interval [a, b].
Applied: bounce rate data spreads relatively evenly between its min and max values.</div></div>
<div class="dist-card" style="margin-top:10px"><div class="dist-name">Normal Distribution</div><span class="continuous-badge">Continuous</span>
<div class="dist-desc">Bell-curve defined by mean μ and standard deviation σ.
Applied: product page session duration follows an approximately normal shape.</div></div>
<div class="dist-card" style="margin-top:10px"><div class="dist-name">Inverse Normal</div><span class="continuous-badge">Continuous</span>
<div class="dist-desc">Finds x such that P(X ≤ x) = p.
Applied: what session time puts a visitor in the top 10% of engagement?</div></div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Binomial Distribution — Buyers in 20 Sessions")
    p_buy = df["Revenue"].mean(); n_t = 20
    kv = np.arange(0,n_t+1); bp = binom.pmf(kv,n_t,p_buy)
    fig1,ax1 = cp_fig(10,4)
    ax1.bar(kv,bp,color=CYAN,width=0.7,edgecolor=BG,linewidth=1,alpha=0.9)
    ax1.axvline(n_t*p_buy,color=AMBER,linewidth=2,linestyle="--",label=f"Mean = {n_t*p_buy:.1f}")
    ax1.set_xlabel("Number of Buyers (k)"); ax1.set_ylabel("Probability P(X = k)")
    ax1.set_title(f"Binomial(n=20, p={p_buy:.3f}) — Buyers in 20 Random Sessions")
    ax1.legend(facecolor=CARD,labelcolor=TEXT,fontsize=9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig1,use_container_width=True)
    st.markdown(f"""<div class="insight-card">⚡ With p = {p_buy:.3f}, in 20 random sessions we expect about
<strong>{n_t*p_buy:.1f} buyers</strong>. The binomial PMF shows the probability of each exact count.</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Poisson Distribution — Daily Purchases in November")
    nov = df[df["Month"]=="Nov"]; lam = nov["Revenue"].sum()/30
    kp = np.arange(0,int(lam*3)+1); pp = poisson.pmf(kp,lam)
    fig2,ax2 = cp_fig(10,4)
    ax2.bar(kp,pp,color=BLUE,width=0.7,edgecolor=BG,linewidth=1,alpha=0.9)
    ax2.axvline(lam,color=PINK,linewidth=2,linestyle="--",label=f"λ = {lam:.1f} (mean)")
    ax2.set_xlabel("Purchases per Day (k)"); ax2.set_ylabel("Probability P(X = k)")
    ax2.set_title(f"Poisson(λ={lam:.1f}) — Daily Purchases in November")
    ax2.legend(facecolor=CARD,labelcolor=TEXT,fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig2,use_container_width=True)
    st.markdown("---")
    st.markdown("### Normal Distribution — Time on Product Pages")
    dur = df["ProductRelated_Duration"].dropna(); dur = dur[dur<dur.quantile(0.97)]
    mu,sg = dur.mean(),dur.std()
    xn = np.linspace(mu-4*sg,mu+4*sg,300); yn = norm.pdf(xn,mu,sg); x90 = norm.ppf(0.90,mu,sg)
    fig3,ax3 = cp_fig(10,4)
    ax3.hist(dur,bins=50,density=True,color=MUTED,edgecolor=BG,linewidth=0.5,label="Actual Data",alpha=0.55)
    ax3.plot(xn,yn,color=CYAN,linewidth=2.5,label=f"Normal  μ={mu:.0f}s  σ={sg:.0f}s")
    ax3.axvline(mu,color=AMBER,linestyle="--",linewidth=1.8,label=f"Mean {mu:.0f}s")
    ax3.axvline(x90,color=PINK,linestyle=":",linewidth=1.8,label=f"Inverse Normal P=0.90 → {x90:.0f}s")
    ax3.fill_between(xn[xn>=x90],yn[xn>=x90],alpha=0.18,color=PINK)
    ax3.set_xlabel("Time on Product Pages (secs)"); ax3.set_ylabel("Density")
    ax3.set_title("Normal Distribution — Product Page Session Duration")
    ax3.legend(facecolor=CARD,labelcolor=TEXT,fontsize=9)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig3,use_container_width=True)
    st.markdown(f"""<div class="insight-card">⚡ Session duration fits Normal with mean = <strong>{mu:.0f}s</strong>.
Inverse Normal: a visitor must spend at least <strong>{x90:.0f}s</strong> to be in the top 10% — a strong purchase-intent signal.</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  REGRESSION MODEL
# ════════════════════════════════════════════════════════════
elif page == "Regression Model":
    m1,m2 = st.columns(2)
    m1.metric("Regression Accuracy",f"{mods['lr_acc']*100:.1f}%")
    m2.metric("AUC-ROC",f"{mods['lr_auc']:.3f}")
    st.markdown("---")
    st.markdown("### Single Regression — y = a + bx")
    st.markdown("""<div class="info-banner">📐 Simple linear regression: <strong>y = a + bx</strong>
— b is the slope, a is the intercept. Regressing purchase probability on Page Value Score.</div>""", unsafe_allow_html=True)
    pv = df["PageValues"].values.reshape(-1,1); rv = df["Revenue"].astype(float).values
    lrs = LinearRegression().fit(pv,rv); a_s,b_s = lrs.intercept_,lrs.coef_[0]
    si = np.random.default_rng(42).choice(len(df),min(600,len(df)),replace=False)
    fig_s,ax_s = cp_fig(10,5)
    jit = np.random.default_rng(1).uniform(-0.03,0.03,len(si))
    ax_s.scatter(df["PageValues"].values[si],df["Revenue"].astype(float).values[si]+jit,
                 c=[CYAN if r==1 else PINK for r in df["Revenue"].astype(int).values[si]],
                 alpha=0.22,s=12,edgecolors="none")
    xl = np.linspace(0,df["PageValues"].max(),200)
    ax_s.plot(xl,a_s+b_s*xl,color=AMBER,linewidth=2.5,label=f"y = {a_s:.3f} + {b_s:.4f}x")
    ax_s.set_xlabel("Page Value Score (x)"); ax_s.set_ylabel("Purchase Probability (y)")
    ax_s.set_title("Simple Linear Regression: Purchase ~ Page Value Score")
    ax_s.legend(facecolor=CARD,labelcolor=TEXT,fontsize=10)
    ax_s.spines["top"].set_visible(False); ax_s.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig_s,use_container_width=True)
    st.markdown(f"""<div class="insight-card">📐 Equation: <strong>y = {a_s:.3f} + {b_s:.4f}x</strong><br>
· a (intercept) = {a_s:.3f} — baseline buy probability when Page Value = 0<br>
· b (slope) = {b_s:.4f} — each unit increase in Page Value raises buy probability by {b_s:.4f}</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Multiple Regression — Coefficients Table")
    st.markdown("""<div class="info-banner">📐 Multiple regression: <strong>y = a + b₁x₁ + b₂x₂ + … + b₁₁x₁₁</strong>
Each β shows the slope (effect) of that variable on purchase probability.</div>""", unsafe_allow_html=True)
    FL = {"ProductRelated":"Product Pages Visited","ProductRelated_Duration":"Time on Product Pages",
          "BounceRates":"Bounce Rate","ExitRates":"Exit Rate","PageValues":"Page Value Score",
          "SpecialDay":"Closeness to Sale Day","Month_enc":"Month of Visit",
          "VisitorType_enc":"New vs Returning Visitor","Weekend_int":"Weekend vs Weekday",
          "Administrative":"Account Pages Visited","Informational":"Info / Help Pages Visited"}
    fll = [FL.get(f,f) for f in mods["features"]]
    cdf = pd.DataFrame({"Feature":fll,"Coefficient β":mods["coefs"].round(4),
                        "Direction":["↑ Increases" if c>0 else "↓ Decreases" for c in mods["coefs"]]
                        }).sort_values("Coefficient β",ascending=False).reset_index(drop=True)
    st.dataframe(cdf,use_container_width=True,hide_index=True)
    cg1,cg2 = st.columns(2)
    with cg1:
        st.markdown("### Coefficient Chart")
        fig_c,ax_c = cp_fig(6,5)
        si2 = np.argsort(mods["coefs"]); sl = [fll[i] for i in si2]; sc = mods["coefs"][si2]
        ax_c.barh(sl,sc,color=[CYAN if c>0 else PINK for c in sc],edgecolor=BG,linewidth=0.5)
        ax_c.axvline(0,color=MUTED,linewidth=1.2)
        ax_c.set_xlabel("Coefficient β (slope)"); ax_c.set_title("Regression Coefficients")
        ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_c,use_container_width=True)
    with cg2:
        st.markdown("### Predicted Probability Distribution")
        probs = mods["lr"].predict_proba(mods["X_test"])[:,1]; yte = mods["y_test"]
        fig_p,ax_p = cp_fig(6,5); bins = np.linspace(0,1,35)
        ax_p.hist(probs[yte==0],bins=bins,color=PINK,alpha=0.65,label="Non-Buyers",density=True)
        ax_p.hist(probs[yte==1],bins=bins,color=CYAN,alpha=0.65,label="Buyers",density=True)
        ax_p.axvline(0.5,color=AMBER,linestyle="--",linewidth=1.8,label="Decision Line (0.5)")
        ax_p.set_xlabel("Predicted Purchase Probability"); ax_p.set_ylabel("Density")
        ax_p.set_title("Predicted Probabilities by Actual Outcome")
        ax_p.legend(facecolor=CARD,labelcolor=TEXT,fontsize=9)
        ax_p.spines["top"].set_visible(False); ax_p.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_p,use_container_width=True)
    st.markdown("""<div class="insight-card">⚡ <strong>Page Value Score</strong> carries the strongest positive β —
every unit increase meaningfully raises predicted buy probability.
<strong>Bounce Rate</strong> has the strongest negative β, sharply reducing it.</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  PURCHASE PREDICTOR
# ════════════════════════════════════════════════════════════
elif page == "Purchase Predictor":
    st.markdown("""<div class="info-banner">📐 Multiple regression: <strong>y = a + b₁x₁ + b₂x₂ + … + b₁₁x₁₁</strong></div>""", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Browsing Behaviour**")
        pp  = st.slider("Product Pages Browsed",0,100,5)
        pt  = st.slider("Time on Product Pages (secs)",0,5000,300,step=10)
        pv  = st.slider("Page Value Score (0–360)",0.0,360.0,10.0,step=1.0)
        sp  = st.slider("Sale Day Proximity (0–1)",0.0,1.0,0.0,step=0.2)
        adm = st.slider("Account Pages Visited",0,20,0)
        inf = st.slider("Info / Help Pages Visited",0,20,0)
    with c2:
        st.markdown("**Exit Signals & Context**")
        br  = st.slider("Bounce Rate (0–1)",0.0,1.0,0.02,step=0.01)
        er  = st.slider("Exit Rate (0–1)",0.0,1.0,0.03,step=0.01)
        mon = st.selectbox("Month",MONTH_ORDER,index=9)
        vtype = st.selectbox("Visitor Type",["Returning Shopper","Brand-New Visitor","Other / Unknown"])
        wknd = st.toggle("Weekend Visit?",value=False)
    vtm = {"Returning Shopper":"Returning_Visitor","Brand-New Visitor":"New_Visitor","Other / Unknown":"Other"}
    vt  = vtm[vtype]
    try:    me = mods["le_month"].transform([mon])[0]
    except: me = 0
    try:    ve = mods["le_visitor"].transform([vt])[0]
    except: ve = 0
    st.markdown("---")
    if st.button("Run Regression Prediction"):
        Xn = pd.DataFrame([[pp,pt,br,er,pv,sp,me,ve,int(wknd),adm,inf]],columns=mods["features"])
        lbuy = mods["lr"].predict_proba(Xn)[0][1]*100; final = 1 if lbuy>=50 else 0
        st.markdown("### Prediction Result")
        r1,r2 = st.columns(2)
        r1.metric("Buy Probability",f"{lbuy:.1f}%"); r2.metric("Model Accuracy",f"{mods['lr_acc']*100:.1f}%")
        if final==1:
            st.markdown('<span class="badge-buy">⚡ LIKELY TO PURCHASE</span>',unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-no">✗ UNLIKELY TO PURCHASE</span>',unsafe_allow_html=True)
        st.markdown(f"""<p class="prob-label">Buy probability — {lbuy:.1f}%</p>
<div class="prob-wrap"><div class="prob-fill" style="width:{int(lbuy)}%"></div></div>""",unsafe_allow_html=True)
        st.markdown("---"); st.markdown("### Where Does This Visitor Rank?")
        ap = mods["lr"].predict_proba(mods["X_test"])[:,1]*100
        fh,ah = cp_fig(10,4)
        ah.hist(ap,bins=40,color=MUTED,edgecolor=BG,linewidth=0.5,density=True,label="All Visitors",alpha=0.8)
        ah.axvline(lbuy,color=CYAN,linewidth=2.5,label=f"This Visitor ({lbuy:.1f}%)")
        ah.axvline(50,color=PINK,linewidth=1.5,linestyle="--",label="Decision Threshold (50%)")
        ah.set_xlabel("Predicted Buy Probability (%)"); ah.set_ylabel("Density")
        ah.set_title("This Visitor vs All Visitors — Predicted Probability Distribution")
        ah.legend(facecolor=CARD,labelcolor=TEXT,fontsize=9)
        ah.spines["top"].set_visible(False); ah.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fh,use_container_width=True)
        st.markdown("---"); st.markdown("### Why the Model Predicted This")
        reasons = []
        if pv>20: reasons.append(("yes",f"Page Value {pv:.0f} — visitor on checkout-linked pages."))
        else:     reasons.append(("no", f"Page Value {pv:.0f} — not yet near checkout."))
        if pp>10: reasons.append(("yes",f"{pp} product pages browsed — high shopping intent."))
        elif pp<3:reasons.append(("no", f"Only {pp} product pages — still exploring."))
        if br<0.05:reasons.append(("yes",f"Low bounce rate ({br:.2f}) — engaged visitor."))
        else:      reasons.append(("no", f"High bounce rate ({br:.2f}) — many like this leave."))
        if vt=="Returning_Visitor": reasons.append(("yes","Returning shopper — converts ~3× more than first-timers."))
        else:                       reasons.append(("no", "New visitor — still building trust with the store."))
        if sp>0.4: reasons.append(("yes",f"Sale day proximity {sp} — deal-hunting mode active."))
        if mon in ["Nov","Dec"]: reasons.append(("yes",f"{mon} — holiday season drives impulse purchases."))
        if pt>500: reasons.append(("yes",f"{pt:.0f}s on product pages — serious purchase intent."))
        for kind,reason in reasons:
            css = "reason-yes" if kind=="yes" else "reason-no"
            icon = "⚡" if kind=="yes" else "✗"
            st.markdown(f'<div class="{css}">{icon} {reason}</div>',unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
