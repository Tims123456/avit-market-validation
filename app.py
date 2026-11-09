"""
app.py — Streamlit dashboard
Indian AV/IT Infrastructure Manufacturing Brand: market-validation analytics
Run:  streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from analysis import (load, run_classification, run_regression, run_clustering, run_arm, pretty,
                      IMP_COLS, NUM_FEATS)

st.set_page_config(page_title="AV/IT Infra Brand — Market Validation", layout="wide")

# On a fresh deploy the data files don't exist yet: build them once from the reproducible scripts (seed 42).
import os, runpy
if not os.path.exists("data/cleaned_survey_data.csv"):
    with st.spinner("Generating synthetic survey data and cleaning it (first run only)…"):
        os.makedirs("data", exist_ok=True)
        runpy.run_path("generate_data.py", run_name="__main__")
        runpy.run_path("clean_data.py", run_name="__main__")

# ----------------------------------------------------------------- chart style (validated reference palette)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                           "#e87ba4", "#008300", "#6250d6", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = [[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#256abf"]]
pio.templates["avit"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#0b0b0b"),
    paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", colorway=CAT,
    xaxis=dict(gridcolor="#ebeae6", zerolinecolor="#d6d5cf", linecolor="#d6d5cf"),
    yaxis=dict(gridcolor="#ebeae6", zerolinecolor="#d6d5cf", linecolor="#d6d5cf"),
    margin=dict(l=10, r=10, t=50, b=10), bargap=0.25, hoverlabel=dict(bgcolor="white")))
pio.templates.default = "avit"
YESNO = {"Yes": BLUE, "No": ORANGE}


def bar(series, title, xlab="", ylab="Respondents", horizontal=True, pct=False, color=BLUE, height=380):
    s = series.copy()
    if horizontal:
        s = s.sort_values()
        fig = px.bar(x=s.values, y=s.index, orientation="h", labels={"x": ylab, "y": xlab}, title=title)
    else:
        fig = px.bar(x=s.index, y=s.values, labels={"x": xlab, "y": ylab}, title=title)
    fig.update_traces(marker_color=color, texttemplate="%{x:.0%}" if (pct and horizontal) else ("%{y:.0%}" if pct else None),
                      textposition="outside", cliponaxis=False)
    if pct:
        (fig.update_xaxes if horizontal else fig.update_yaxes)(tickformat=".0%")
    fig.update_layout(height=height)
    return fig


def insight(text):
    import re as _re
    text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = _re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    st.markdown(f"<div style='background:#f3f6fb;border-left:4px solid {BLUE};padding:10px 14px;border-radius:4px;"
                f"margin:-6px 0 22px 0;font-size:0.93rem'>💡 {text}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------- data & models (cached)
@st.cache_data
def get_data():
    raw = pd.read_csv("data/raw_survey_responses.csv", keep_default_na=False, na_values=[""])
    clean = load()
    logdf = pd.read_csv("data/cleaning_log.csv")
    return raw, clean, logdf


@st.cache_resource
def get_models(_df):
    return run_classification(_df), run_regression(_df), run_clustering(_df), run_arm(_df)


raw, df_all, cleaning_log = get_data()
with st.spinner("Training models (first load only)…"):
    CLF, REG, CLU, ARM = get_models(df_all)
df_all = df_all.assign(Cluster=CLU["labels"])
persona_map = dict(zip(CLU["profile"].index, CLU["profile"]["Persona"]))
df_all["Persona"] = df_all["Cluster"].map(persona_map)

# ----------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters (apply to EDA tabs)")
f_region = st.sidebar.multiselect("Region", sorted(df_all["Region"].unique()), default=sorted(df_all["Region"].unique()))
f_tier = st.sidebar.multiselect("City tier", sorted(df_all["City_Tier"].unique()), default=sorted(df_all["City_Tier"].unique()))
SIZES = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
f_size = st.sidebar.multiselect("Company size (employees)", SIZES, default=SIZES)
f_ind = st.sidebar.multiselect("Industry", sorted(df_all["Q2_Industry"].unique()), default=sorted(df_all["Q2_Industry"].unique()))
df = df_all[df_all["Region"].isin(f_region) & df_all["City_Tier"].isin(f_tier) & df_all["Q3_Company_Size"].isin(f_size)
            & df_all["Q2_Industry"].isin(f_ind)]
st.sidebar.caption(f"Showing **{len(df):,}** of {len(df_all):,} cleaned respondents.")
if len(df) < 30:
    st.warning("Too few respondents for the current filters — widen them.")
    st.stop()

# ----------------------------------------------------------------- header
st.title("Indian AV/IT Infrastructure Brand — Market Validation Dashboard")
st.caption("Racks · Smart PDUs · Rack cooling & monitoring · Custom AV/industrial enclosures · Meeting-room connectivity  |  "
           "Synthetic B2B buyer survey (n = 3,245 raw → 3,200 clean)")

tabs = st.tabs(["① Overview & verdict", "② Data quality", "③ Descriptive EDA", "④ Diagnostic & correlation",
                "⑤ Classification", "⑥ Regression", "⑦ Clustering", "⑧ Association rules", "⑨ Feasibility & market"])

# ===================================================================== ① OVERVIEW
with tabs[0]:
    adopt = df["Target_Likely_Buy"].mean()
    adopters = df[df["Target_Likely_Buy"] == 1]
    c = st.columns(5)
    c[0].metric("Respondents (filtered)", f"{len(df):,}")
    c[1].metric("Say 'likely to buy' in 12 months", f"{adopt:.0%}")
    c[2].metric("Avg 1st-year spend per likely buyer", f"₹{adopters['Q25_Expected_Spend_New_Brand_Lakh'].mean():.1f} L")
    c[3].metric("Median discount needed to switch", f"{df['Q18_Discount_Needed_To_Switch_Pct'].median():.0f}%")
    c[4].metric("Need Make-in-India Class-I", f"{df['MII_Required'].mean():.0%}")

    st.subheader("What the data says")
    best_clf = CLF["results"].iloc[0]
    prof = CLU["profile"].sort_values("Exp_Spend_New_Brand", ascending=False)
    top_rule = ARM["rules"].iloc[0]
    st.markdown(f"""
- **Demand exists but is conditional.** {adopt:.0%} of buyers say they are likely to buy from a new Indian brand — but stated intent
  always overstates real orders. Even at a conservative 30% conversion of stated intent, that is ~{adopt*0.3:.0%} of contacted accounts.
- **Openness and price are the switch.** The best classifier (*{best_clf['Model']}*, ROC-AUC {best_clf['ROC-AUC']:.2f}) ranks openness
  to an Indian brand, the discount needed to switch, lead time and current-supplier satisfaction as the strongest drivers.
- **Money is concentrated.** The **{prof.iloc[0]['Persona']}** cluster plans ₹{prof.iloc[0]['Exp_Spend_New_Brand']:.1f} L per account vs
  ₹{prof.iloc[-1]['Exp_Spend_New_Brand']:.1f} L for **{prof.iloc[-1]['Persona']}**. Volume and value sit in different segments.
- **Sell systems, not boxes.** Strongest product rule: *{top_rule['Antecedent']} → {top_rule['Consequent']}* (lift {top_rule['lift']:.1f}).
  Rack + smart PDU + cooling + sensors, and AV rack + table box + floor box, are natural bundles.
""")
    st.subheader("Verdict")
    st.info("**Sustainable — if you enter narrow, not broad.** The market is growing fast and buyers are open, but a new brand cannot beat "
            "Schneider, Vertiv and Rittal on enterprise data-centre trust in year 1. The data points to a beachhead in "
            "(1) Make-in-India government/PSU tenders and (2) customised AV and industrial enclosures, where lead time and "
            "customisation are the pain and local manufacturing is the advantage. Smart PDUs and monitoring should follow once "
            "the rack business funds certification and firmware. See tab ⑨ for the numbers and the break-even scenario.")
    st.caption("This dataset is synthetic. It was built to mirror plausible market behaviour for coursework; it is not evidence of real demand. "
               "Validate with 30–50 real buyer interviews before investing.")

# ===================================================================== ② DATA QUALITY
with tabs[1]:
    st.subheader("Raw vs cleaned data")
    c = st.columns(4)
    c[0].metric("Raw rows", f"{len(raw):,}")
    c[1].metric("Clean rows", f"{len(df_all):,}", delta=f"{len(df_all) - len(raw)}")
    c[2].metric("Missing cells (raw)", f"{int(raw.isna().sum().sum()):,}")
    c[3].metric("Missing cells (clean)", f"{int(df_all.drop(columns=['Cluster','Persona']).isna().sum().sum()):,}")
    miss = (raw.isna().mean().sort_values(ascending=False) * 100).head(12)
    fig = bar(miss / 100, "Share of missing answers by question (raw data)", ylab="Missing", pct=True, color=ORANGE)
    st.plotly_chart(fig, width='stretch')
    insight(f"Missingness is highest in sensitive or effort-heavy questions — smart-PDU willingness to pay "
            f"({raw['Q21_WTP_Smart_PDU_INR'].isna().mean():.0%}) and discount needed ({raw['Q18_Discount_Needed_To_Switch_Pct'].isna().mean():.0%}). "
            "That is typical of real B2B surveys: people skip price questions. Numeric gaps were filled with the median *within the same company-size band* "
            "so a 10-person integrator is not given an enterprise's budget.")
    st.subheader("Cleaning & transformation log")
    st.dataframe(cleaning_log, width='stretch', hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        rs = pd.to_numeric(raw["Q7_Annual_AVIT_Spend_Lakh"], errors="coerce").dropna()
        fig = px.box(y=rs, points="outliers", title="BEFORE: annual AV/IT spend (₹ lakh)", log_y=True)
        fig.update_traces(marker_color=ORANGE, line_color=ORANGE); st.plotly_chart(fig, width='stretch')
    with col2:
        fig = px.box(y=df_all["Q7_Annual_AVIT_Spend_Lakh"], points="outliers", title="AFTER: capped within size band", log_y=True)
        fig.update_traces(marker_color=BLUE, line_color=BLUE); st.plotly_chart(fig, width='stretch')
    insight(f"Raw spend reached ₹{rs.max():,.0f} L — a respondent typing rupees-thousands into a ₹-lakh box. "
            f"After capping at log-IQR×3 within each size band the maximum is ₹{df_all['Q7_Annual_AVIT_Spend_Lakh'].max():,.0f} L, "
            "which is a credible budget for a 1000+ employee enterprise. Genuine large buyers are kept; only impossible values are pulled in.")
    with st.expander("Preview cleaned data"):
        st.dataframe(df_all.head(50), width='stretch')

# ===================================================================== ③ DESCRIPTIVE
with tabs[2]:
    st.caption("Descriptive analytics: *who* answered and *what* they buy today. Filters in the sidebar apply.")
    c1, c2 = st.columns(2)
    with c1:
        s = df["Q1_Role"].value_counts()
        st.plotly_chart(bar(s, "D1. Respondent role"), width='stretch')
        insight(f"**{s.index[0]}** is the largest group ({s.iloc[0]/len(df):.0%}). Integrators specify and resell racks for many end-clients, "
                "so a few integrator relationships can unlock many projects — the channel matters as much as the end-user.")
    with c2:
        s = df["Q2_Industry"].value_counts()
        st.plotly_chart(bar(s, "D2. End-user industry"), width='stretch')
        insight(f"Demand is spread across {df['Q2_Industry'].nunique()} industries; the top three (**{', '.join(s.index[:3])}**) make up "
                f"{s.iloc[:3].sum()/len(df):.0%}. No single vertical dominates, so the product range needs to be configurable rather than vertical-specific.")
    c1, c2 = st.columns(2)
    with c1:
        s = df["Q3_Company_Size"].value_counts().reindex(SIZES).fillna(0)
        st.plotly_chart(bar(s, "D3. Company size (employees)", horizontal=False), width='stretch')
        small = df["Size_Ord"].le(2).mean()
        insight(f"{small:.0%} of respondents are firms with ≤50 employees. They are numerous but individually small buyers; "
                "the 201+ employee firms are fewer but carry most of the spend (see D5).")
    with c2:
        t = pd.crosstab(df["Region"], df["City_Tier"]).drop(columns="Unknown", errors="ignore").drop(index="Unknown", errors="ignore")
        fig = px.bar(t, barmode="group", title="D4. Respondents by region and city tier", labels={"value": "Respondents"},
                     color_discrete_sequence=[BLUE, ORANGE, AQUA])
        st.plotly_chart(fig, width='stretch')
        insight(f"Metros account for {df['City_Tier'].eq('Metro').mean():.0%} of respondents, led by the South and West — consistent with "
                "India's data-centre clusters in Mumbai, Chennai, Hyderabad and Bengaluru. A first plant near Pune/Mumbai or Chennai/Bengaluru "
                "keeps freight short to the biggest demand pools.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Q7_Annual_AVIT_Spend_Lakh", nbins=60, log_x=False, title="D5. Annual AV/IT infrastructure spend (₹ lakh)",
                           labels={"Q7_Annual_AVIT_Spend_Lakh": "₹ lakh / year"})
        fig.update_traces(marker_color=BLUE); fig.update_xaxes(type="log")
        st.plotly_chart(fig, width='stretch')
        sp = df["Q7_Annual_AVIT_Spend_Lakh"]
        top20 = sp.sort_values(ascending=False).head(int(len(sp) * 0.2)).sum() / sp.sum()
        insight(f"Spend is heavily right-skewed: median ₹{sp.median():.1f} L vs mean ₹{sp.mean():.1f} L. The top 20% of accounts hold "
                f"**{top20:.0%}** of all spend (a Pareto pattern), which is why spend is log-transformed for modelling and why key-account selling matters.")
    with c2:
        s = df["Q9_Current_Primary_Brand"].value_counts(normalize=True)
        st.plotly_chart(bar(s, "D6. Current primary brand", ylab="Share", pct=True), width='stretch')
        glob = df["Q9_Current_Primary_Brand"].isin(["Schneider/APC", "Vertiv", "Rittal", "Legrand", "Eaton"]).mean()
        loc = df["Q9_Current_Primary_Brand"].isin(["Local fabricator/unbranded", "Imported (China/Taiwan)"]).mean()
        insight(f"Global brands are primary supplier for {glob:.0%} of buyers; **{loc:.0%}** use unbranded local fabricators or imports. "
                "That second group is the easiest to win — they already buy on price and have no brand loyalty — while the global-brand base is the harder, higher-value prize.")
    c1, c2 = st.columns(2)
    with c1:
        s = df["Q10_Purchase_Channel"].value_counts(normalize=True)
        st.plotly_chart(bar(s, "D7. How they buy today", ylab="Share", pct=True), width='stretch')
        insight(f"Distributors/dealers handle {s.get('Distributor/Dealer', 0):.0%} of purchases and tenders/GeM {s.get('Tender / GeM portal', 0):.0%}. "
                "A direct-only model would miss most of the market: plan for a dealer network plus GeM registration from day one.")
    with c2:
        fig = px.histogram(df, x="Q11_Avg_Lead_Time_Weeks", nbins=30, title="D8. Lead time experienced today (weeks)")
        fig.update_traces(marker_color=BLUE); st.plotly_chart(fig, width='stretch')
        insight(f"Median lead time is **{df['Q11_Avg_Lead_Time_Weeks'].median():.1f} weeks**; {df['Q11_Avg_Lead_Time_Weeks'].gt(8).mean():.0%} wait more than 8 weeks. "
                "A local plant that ships standard racks in 2–3 weeks and custom builds in 4–5 would have a visible, sellable edge.")
    c1, c2 = st.columns(2)
    with c1:
        pain_cols = [c for c in df.columns if c.startswith("Pain_") and c != "Pain_Count"]
        s = df[pain_cols].mean(); s.index = [pretty(i).replace("Pain: ", "") for i in s.index]
        st.plotly_chart(bar(s, "D9. Pain points with current suppliers", ylab="Share of respondents", pct=True), width='stretch')
        insight(f"**{s.idxmax()}** ({s.max():.0%}) and **{s.drop(s.idxmax()).idxmax()}** ({s.drop(s.idxmax()).max():.0%}) are the top complaints. "
                "These map directly to the value proposition: price-competitive local manufacturing with short lead times.")
    with c2:
        s = df[IMP_COLS].mean(); s.index = [i.replace("Q16_Imp_", "").replace("_", " ") for i in s.index]
        fig = bar(s, "D10. What matters when choosing a supplier (mean 1–5)", ylab="Mean importance")
        fig.update_xaxes(range=[1, 5]); st.plotly_chart(fig, width='stretch')
        insight(f"On average **{s.idxmax()}** ({s.max():.2f}) and **{s.drop(s.idxmax()).idxmax()}** lead; **{s.idxmin()}** is lowest ({s.min():.2f}). "
                "Averages hide the segments, though — clustering (tab ⑦) shows each group weights these very differently.")
    c1, c2 = st.columns(2)
    with c1:
        s = ARM["item_support"] if len(df) == len(df_all) else df[[c for c in df.columns if c.startswith("Plan_")]].mean()
        s.index = [i.replace("Plan_", "").replace("_", " ") for i in s.index]
        st.plotly_chart(bar(s, "D11. Products planned in the next 12 months", ylab="Share planning to buy", pct=True), width='stretch')
        insight(f"**{s.idxmax()}** is the most widely planned item ({s.max():.0%}), followed by {', '.join(s.sort_values(ascending=False).index[1:4])}. "
                "High-frequency, low-complexity items (network racks, accessories, cable management) are the volume base; smart PDUs and cooling are lower-frequency but higher-value.")
    with c2:
        t = df["Q26_Likely_To_Buy_12M"].value_counts()
        fig = px.histogram(df, x="Q17_Openness_New_Indian_Brand", color="Q26_Likely_To_Buy_12M", barmode="group",
                           color_discrete_map=YESNO, title="D12. Openness to a new Indian brand (1–5) and purchase intent",
                           labels={"Q17_Openness_New_Indian_Brand": "Openness rating", "Q26_Likely_To_Buy_12M": "Likely to buy"})
        st.plotly_chart(fig, width='stretch')
        insight(f"{df['Q17_Openness_New_Indian_Brand'].ge(4).mean():.0%} rate their openness 4 or 5. Intent climbs steeply with openness: "
                f"{df.loc[df['Q17_Openness_New_Indian_Brand']>=4,'Target_Likely_Buy'].mean():.0%} likely-to-buy among 4–5 raters vs "
                f"{df.loc[df['Q17_Openness_New_Indian_Brand']<=2,'Target_Likely_Buy'].mean():.0%} among 1–2 raters.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Q18_Discount_Needed_To_Switch_Pct", nbins=30, title="D13. Discount needed vs current brand to switch (%)")
        fig.update_traces(marker_color=BLUE); st.plotly_chart(fig, width='stretch')
        insight(f"Median required discount is **{df['Q18_Discount_Needed_To_Switch_Pct'].median():.0f}%**; a quarter need ≥{df['Q18_Discount_Needed_To_Switch_Pct'].quantile(.75):.0f}%. "
                "Sheet-metal racks typically run thin margins, so competing purely on a 20%+ discount is not sustainable — it must be paired with lead-time and customisation advantages.")
    with c2:
        fig = px.box(df, x="Q3_Company_Size", y="Q21_WTP_Smart_PDU_INR", category_orders={"Q3_Company_Size": SIZES},
                     title="D14. Willingness to pay for a smart PDU (₹) by company size", labels={"Q21_WTP_Smart_PDU_INR": "₹", "Q3_Company_Size": "Employees"})
        fig.update_traces(marker_color=BLUE, line_color=BLUE); st.plotly_chart(fig, width='stretch')
        w = df.groupby("Q3_Company_Size")["Q21_WTP_Smart_PDU_INR"].median().reindex(SIZES)
        insight(f"Median WTP rises from ₹{w.iloc[0]:,.0f} (1–10 staff) to ₹{w.iloc[-1]:,.0f} (1000+). Smart PDUs are an enterprise and "
                "data-centre product; smaller buyers want a basic or metered unit at a lower price point.")

# ===================================================================== ④ DIAGNOSTIC
with tabs[3]:
    st.caption("Diagnostic analytics: *why* some buyers intend to switch and others do not.")
    num = ["Target_Likely_Buy", "Q25_Expected_Spend_New_Brand_Lakh", "Q17_Openness_New_Indian_Brand", "Q18_Discount_Needed_To_Switch_Pct",
           "Q12_Current_Supplier_Satisfaction", "Q11_Avg_Lead_Time_Weeks", "MII_Required", "Log_AVIT_Spend", "Size_Ord",
           "Q8_Racks_Purchased_12M", "Q20_Interest_Smart_Monitoring", "Pain_Count", "Planned_Basket_Size"] + IMP_COLS
    corr = df[num].corr()
    labels = [pretty(c).replace("Target Likely Buy", "LIKELY TO BUY").replace("Q25 Expected Spend New Brand Lakh", "Exp. spend new brand")
              .replace("Openness New Indian Brand", "Openness").replace("Discount Needed To Switch Pct", "Discount needed")
              .replace("Current Supplier Satisfaction", "Satisfaction").replace("Avg Lead Time Weeks", "Lead time")
              .replace("Racks Purchased 12M", "Racks bought").replace("Interest Smart Monitoring", "Smart interest") for c in num]
    fig = px.imshow(corr.values, x=labels, y=labels, color_continuous_scale=DIV, zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                    title="G1. Correlation matrix (Pearson)")
    fig.update_traces(textfont_size=9); fig.update_layout(height=720)
    st.plotly_chart(fig, width='stretch')
    tc = corr["Target_Likely_Buy"].drop("Target_Likely_Buy")
    insight(f"Strongest positive link with purchase intent: **{labels[num.index(tc.idxmax())]}** (r = {tc.max():.2f}); strongest negative: "
            f"**{labels[num.index(tc.idxmin())]}** (r = {tc.min():.2f}). Spend, size and racks bought are tightly inter-correlated (r ≈ "
            f"{corr.loc['Log_AVIT_Spend','Size_Ord']:.2f}), so regression needs regularisation or tree models to avoid double-counting.")

    c1, c2 = st.columns(2)
    with c1:
        s = tc.sort_values(); s.index = [labels[num.index(i)] for i in s.index]
        fig = px.bar(x=s.values, y=s.index, orientation="h", title="G2. Correlation of each variable with 'likely to buy'",
                     labels={"x": "r", "y": ""}, color=np.where(s.values > 0, "Positive", "Negative"),
                     color_discrete_map={"Positive": BLUE, "Negative": RED})
        fig.update_layout(height=600, showlegend=False); st.plotly_chart(fig, width='stretch')
        insight("Read this as the shortlist of levers. Buyers lean in when they are dissatisfied, wait long, and need Make-in-India content; "
                "they lean out when they need a big discount. Brand reputation and price importance are weak on their own.")
    with c2:
        s = df.groupby("Q9_Current_Primary_Brand")["Target_Likely_Buy"].mean().sort_values()
        st.plotly_chart(bar(s, "G3. Purchase intent by current primary brand", ylab="Likely to buy", pct=True, height=600), width='stretch')
        insight(f"Intent ranges only from {s.min():.0%} ({s.idxmin()}) to {s.max():.0%} ({s.idxmax()}). The incumbent brand on its own is a weak signal — "
                "what pushes buyers to switch is their *experience* with it (satisfaction, lead time; see G4–G5). Users of imported products, who face the longest waits, lean slightly more open.")
    c1, c2 = st.columns(2)
    with c1:
        b = pd.cut(df["Q11_Avg_Lead_Time_Weeks"], [0, 4, 6, 8, 10, 14, 40], labels=["≤4", "4–6", "6–8", "8–10", "10–14", "14+"])
        s = df.groupby(b, observed=True)["Target_Likely_Buy"].mean()
        fig = px.line(x=s.index.astype(str), y=s.values, markers=True, title="G4. Purchase intent vs current lead time",
                      labels={"x": "Lead time (weeks)", "y": "Likely to buy"})
        fig.update_traces(line_color=BLUE, line_width=2, marker_size=9); fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')
        insight(f"Intent rises from {s.iloc[0]:.0%} for buyers who wait ≤4 weeks to {s.iloc[-1]:.0%} for those waiting 14+. "
                "Lead time is a genuine, measurable pain — and the one a domestic factory is best placed to fix.")
    with c2:
        s = df.groupby("Q12_Current_Supplier_Satisfaction")["Target_Likely_Buy"].mean()
        fig = bar(s, "G5. Purchase intent by satisfaction with current supplier", xlab="Satisfaction (1–5)", ylab="Likely to buy",
                  horizontal=False, pct=True)
        st.plotly_chart(fig, width='stretch')
        insight(f"Dissatisfied buyers (rating 1) are {s.iloc[0]/max(s.iloc[-1],0.01):.1f}× as likely to switch as delighted ones (rating 5). "
                "Unhappy customers of incumbents are the warmest leads.")
    c1, c2 = st.columns(2)
    with c1:
        t = df.groupby(["Q10_Purchase_Channel", "Q19_Requires_MakeInIndia_ClassI"])["Target_Likely_Buy"].mean().reset_index()
        fig = px.bar(t, x="Q10_Purchase_Channel", y="Target_Likely_Buy", color="Q19_Requires_MakeInIndia_ClassI", barmode="group",
                     title="G6. Intent by channel and Make-in-India Class-I requirement",
                     labels={"Target_Likely_Buy": "Likely to buy", "Q10_Purchase_Channel": "", "Q19_Requires_MakeInIndia_ClassI": "Needs Class-I"},
                     color_discrete_map={"Yes": BLUE, "No": ORANGE, "Not sure": AQUA})
        fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig, width='stretch')
        m = df.groupby("MII_Required")["Target_Likely_Buy"].mean()
        insight(f"Buyers who must meet Class-I local content show {m.get(1,0):.0%} intent vs {m.get(0,0):.0%} for those who don't. "
                "Under the Public Procurement (Make in India) Order, Class-I suppliers (≥50% local content) get purchase preference — a structural advantage imports cannot match.")
    with c2:
        fig = px.scatter(df.sample(min(len(df), 1500), random_state=1), x="Q7_Annual_AVIT_Spend_Lakh", y="Q25_Expected_Spend_New_Brand_Lakh",
                         color="Q26_Likely_To_Buy_12M", log_x=True, log_y=True, opacity=0.6, color_discrete_map=YESNO,
                         title="G7. Current budget vs spend planned with the new brand (log–log)",
                         labels={"Q7_Annual_AVIT_Spend_Lakh": "Current annual AV/IT spend (₹ L)",
                                 "Q25_Expected_Spend_New_Brand_Lakh": "Planned 1st-year spend with new brand (₹ L)", "Q26_Likely_To_Buy_12M": "Likely to buy"})
        fig.update_traces(marker_size=8, marker_line_width=1, marker_line_color="#fcfcfb")
        st.plotly_chart(fig, width='stretch')
        a = df[df.Target_Likely_Buy == 1]
        insight(f"Two bands appear: likely buyers would move about {(a.Q25_Expected_Spend_New_Brand_Lakh/a.Q7_Annual_AVIT_Spend_Lakh).median():.0%} of their budget "
                "(a trial share, not a full switch), while others plan token amounts. Wallet share grows only after a successful first order.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="Q17_Openness_New_Indian_Brand", y="Q18_Discount_Needed_To_Switch_Pct", title="G8. Discount needed falls as openness rises",
                     labels={"Q17_Openness_New_Indian_Brand": "Openness (1–5)", "Q18_Discount_Needed_To_Switch_Pct": "Discount needed (%)"})
        fig.update_traces(marker_color=BLUE, line_color=BLUE); st.plotly_chart(fig, width='stretch')
        g = df.groupby("Q17_Openness_New_Indian_Brand")["Q18_Discount_Needed_To_Switch_Pct"].median()
        insight(f"Median discount needed drops from {g.iloc[0]:.0f}% (openness 1) to {g.iloc[-1]:.0f}% (openness 5). "
                "Targeting already-open buyers protects margin; converting sceptics is expensive.")
    with c2:
        t = df[df.Region != "Unknown"].pivot_table(index="Region", columns="City_Tier", values="Target_Likely_Buy", aggfunc="mean")
        t = t.drop(columns="Unknown", errors="ignore")
        fig = px.imshow(t, text_auto=".0%", color_continuous_scale=SEQ, title="G9. Purchase intent by region × city tier", aspect="auto")
        st.plotly_chart(fig, width='stretch')
        mt = t[[c for c in ["Metro", "Tier-2"] if c in t.columns]]
        insight(f"Metro and Tier-2 intent sits in a narrow band ({mt.min().min():.0%}–{mt.max().max():.0%}), so location should be chosen for logistics and demand volume. "
                "Tier-3 cells swing more widely but rest on small samples — treat them as noise until confirmed with real data.")
    s = df.groupby("Q1_Role")["Target_Likely_Buy"].agg(["mean", "count"]).sort_values("mean")
    fig = bar(s["mean"], "G10. Purchase intent by respondent role", ylab="Likely to buy", pct=True, height=460)
    st.plotly_chart(fig, width='stretch')
    insight(f"**{s.index[-1]}** ({s['mean'].iloc[-1]:.0%}) and **{s.index[-2]}** ({s['mean'].iloc[-2]:.0%}) are the most receptive roles; "
            f"**{s.index[0]}** is least ({s['mean'].iloc[0]:.0%}). Sales effort should be sequenced by role, not spread evenly.")

# ===================================================================== ⑤ CLASSIFICATION
with tabs[4]:
    st.subheader("Who will buy? — predicting Q26 'Likely to buy in 12 months' (Yes/No)")
    st.caption("Features: firmographics, spend, lead time, satisfaction, importance ratings, openness, discount needed, Make-in-India need, "
               "pain points, channel, current brand. Excluded as leakage: trial-order willingness and planned spend. 75/25 stratified split + 5-fold CV.")
    res = CLF["results"].copy()
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"}).highlight_max(subset=res.columns[1:], color="#cde2fb"),
                 width='stretch', hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for i, (n, (fpr, tpr)) in enumerate(CLF["rocs"].items()):
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{n} (AUC {res.set_index('Model').loc[n, 'ROC-AUC']:.2f})", line=dict(width=2, color=CAT[i])))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(dash="dash", color="#a3a29c", width=1)))
        fig.update_layout(title="C1. ROC curves", xaxis_title="False-positive rate", yaxis_title="True-positive rate", height=440)
        st.plotly_chart(fig, width='stretch')
    with c2:
        cm = CLF["cm"]
        fig = px.imshow(cm, text_auto=True, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"], color_continuous_scale=SEQ,
                        title=f"C2. Confusion matrix — {CLF['best']} (test set)")
        fig.update_layout(height=440); st.plotly_chart(fig, width='stretch')
    b = res.iloc[0]
    insight(f"**{b['Model']}** performs best (ROC-AUC {b['ROC-AUC']:.2f}, accuracy {b['Accuracy']:.0%}, precision {b['Precision']:.0%}, recall {b['Recall']:.0%}) "
            f"against a base rate of {CLF['base_rate']:.0%}. AUC ≈ 0.8 means the model ranks a random buyer above a random non-buyer about 8 times in 10 — "
            "good enough to score a lead list so sales calls go to the top-ranked accounts first. Cross-validated AUC is close to the test AUC, so the model is not overfitting.")
    c1, c2 = st.columns(2)
    with c1:
        s = CLF["rf_importance"].head(15).sort_values(); s.index = [pretty(i) for i in s.index]
        st.plotly_chart(bar(s, "C3. Random-forest feature importance (top 15)", ylab="Importance", height=520), width='stretch')
    with c2:
        co = CLF["lr_coefs"]; co = pd.concat([co.head(8), co.tail(8)]); co.index = [pretty(i) for i in co.index]
        fig = px.bar(x=co.values, y=co.index, orientation="h", title="C4. Logistic-regression coefficients (standardised)",
                     labels={"x": "Effect on log-odds of buying", "y": ""}, color=np.where(co.values > 0, "Raises intent", "Lowers intent"),
                     color_discrete_map={"Raises intent": BLUE, "Lowers intent": RED})
        fig.update_layout(height=520); st.plotly_chart(fig, width='stretch')
    insight("Both models agree: openness, the discount needed, lead time, satisfaction and the Make-in-India requirement drive intent. "
            "The logistic model gives direction (blue raises, red lowers) — e.g. each step up in required discount pushes intent down, while long lead times and Class-I requirements push it up.")

# ===================================================================== ⑥ REGRESSION
with tabs[5]:
    st.subheader("How much will they spend? — predicting Q25 planned 1st-year spend with the new brand (₹ lakh)")
    st.caption("Target is right-skewed (many near-zero, a few large enterprises), so models are trained on log(1+spend) and errors are converted back to ₹ lakh.")
    res = REG["results"].copy()
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"}), width='stretch', hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=REG["y_test_log"], y=REG["pred_best_log"], opacity=0.5, title=f"R1. Actual vs predicted — {REG['best']} (log scale)",
                         labels={"x": "Actual log(1+₹ L)", "y": "Predicted log(1+₹ L)"})
        mx = max(REG["y_test_log"].max(), REG["pred_best_log"].max())
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="#a3a29c"), name="Perfect fit"))
        fig.update_traces(marker=dict(color=BLUE, size=8), selector=dict(mode="markers")); fig.update_layout(height=460)
        st.plotly_chart(fig, width='stretch')
    with c2:
        s = REG["rf_importance"].head(12).sort_values(); s.index = [pretty(i) for i in s.index]
        st.plotly_chart(bar(s, "R2. What drives planned spend (random-forest importance)", ylab="Importance", height=460), width='stretch')
    b = res.iloc[0]
    insight(f"**{b['Model']}** explains {b['R² (log scale)']:.0%} of the variation in (log) planned spend, with a typical error of ₹{b['MAE (₹ lakh)']:.1f} L. "
            "Account size — racks bought and current budget — sets *how much* a buyer could spend; openness and the discount needed set *how much of it* comes to you. "
            "The remaining unexplained variance is the yes/no switch decision itself, which is why classification and regression are used together: "
            "score likelihood first, then estimate ticket size.")
    co = REG["lin_coefs"]; co = pd.concat([co.head(6), co.tail(8)]); co.index = [pretty(i) for i in co.index]
    fig = px.bar(x=co.values, y=co.index, orientation="h", title="R3. Linear-regression coefficients (standardised, log target)",
                 labels={"x": "Effect on log spend", "y": ""}, color=np.where(co.values > 0, "Raises spend", "Lowers spend"),
                 color_discrete_map={"Raises spend": BLUE, "Lowers spend": RED})
    st.plotly_chart(fig, width='stretch')
    insight("Positive coefficients for spend size, openness and Class-I requirement; negative for the discount needed. "
            "Several firmographic coefficients are small or unstable because size, revenue and spend are correlated (see G1) — the tree models handle that better.")

# ===================================================================== ⑦ CLUSTERING
with tabs[6]:
    st.subheader("Who are the buyer segments? — K-means on importance ratings + spend + size + openness")
    c1, c2 = st.columns(2)
    sc = CLU["scan"]
    with c1:
        fig = px.line(sc, x="k", y="Inertia (WCSS)", markers=True, title="K1. Elbow method")
        fig.update_traces(line_color=BLUE, marker_size=9); st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.line(sc, x="k", y="Silhouette", markers=True, title="K2. Silhouette score by k")
        fig.update_traces(line_color=ORANGE, marker_size=9); st.plotly_chart(fig, width='stretch')
    insight(f"The elbow flattens after k = 4, and k = 4 gives four distinct, actionable personas. Silhouette scores are modest "
            f"({sc.set_index('k').loc[4,'Silhouette']:.2f} at k=4) — normal for Likert survey data, where people sit on a continuum rather than in tight clumps. "
            "k = 2 scores higher mathematically but only splits big vs small buyers, which is not useful for targeting.")
    prof = CLU["profile"].copy()
    lab = df_all.assign(PC1=CLU["xy"][:, 0], PC2=CLU["xy"][:, 1])
    fig = px.scatter(lab.sample(1800, random_state=2), x="PC1", y="PC2", color="Persona", opacity=0.6,
                     title=f"K3. Clusters projected onto 2 principal components ({CLU['explained_var'].sum():.0%} of variance shown)",
                     color_discrete_sequence=CAT)
    fig.update_traces(marker_size=8, marker_line_width=1, marker_line_color="#fcfcfb"); fig.update_layout(height=520)
    st.plotly_chart(fig, width='stretch')
    show = prof.set_index("Persona")[["Respondents", "Median_Spend_Lakh", "Openness", "Likely_Buy_Rate", "Exp_Spend_New_Brand",
                                      "Discount_Needed", "MII_Required", "Smart_Interest"]]
    show.columns = ["Respondents", "Median spend ₹L", "Openness (1–5)", "Likely to buy", "Planned spend new brand ₹L", "Discount needed %",
                    "Needs Class-I", "Smart interest (1–5)"]
    st.dataframe(show.style.format({"Median spend ₹L": "{:.1f}", "Openness (1–5)": "{:.2f}", "Likely to buy": "{:.0%}",
                                    "Planned spend new brand ₹L": "{:.1f}", "Discount needed %": "{:.1f}", "Needs Class-I": "{:.0%}",
                                    "Smart interest (1–5)": "{:.2f}"}), width='stretch')
    imp = prof.set_index("Persona")[[c for c in prof.columns if c.startswith("Imp_")]]
    imp.columns = [c.replace("Imp_", "").replace("_", " ") for c in imp.columns]
    fig = px.imshow(imp, text_auto=".1f", color_continuous_scale=SEQ, zmin=1, zmax=5, aspect="auto", title="K4. What each segment values (mean importance 1–5)")
    fig.update_layout(height=380); st.plotly_chart(fig, width='stretch')
    p = prof.set_index("Persona")
    val = (p["Respondents"] * p["Likely_Buy_Rate"] * p["Exp_Spend_New_Brand"]).sort_values(ascending=False)
    insight(" ".join([f"**{n}** — {p.loc[n,'Respondents']:.0f} buyers, {p.loc[n,'Likely_Buy_Rate']:.0%} likely, ₹{p.loc[n,'Exp_Spend_New_Brand']:.1f} L planned each."
                      for n in val.index]) +
            f" Weighted by size × intent × ticket, **{val.index[0]}** is the most valuable segment to target first. "
            "Price-driven SMB integrators are many but want deep discounts and small tickets — serve them via dealers with a standard catalogue, not a direct sales team.")

# ===================================================================== ⑧ ASSOCIATION RULES
with tabs[7]:
    st.subheader("What do buyers purchase together? — Apriori on Q15 planned-product baskets")
    c = st.columns(3)
    smin = c[0].slider("Min support", 0.02, 0.20, 0.05, 0.01)
    cmin = c[1].slider("Min confidence", 0.2, 0.9, 0.45, 0.05)
    lmin = c[2].slider("Min lift", 1.0, 4.0, 1.1, 0.1)
    arm = run_arm(df_all, smin, cmin, lmin) if (smin, cmin, lmin) != (0.05, 0.45, 1.1) else ARM
    r = arm["rules"]
    st.caption(f"{arm['n_itemsets']} frequent itemsets → **{len(r)} rules** pass the thresholds.")
    st.dataframe(r.head(25).style.format({"support": "{:.3f}", "confidence": "{:.2f}", "lift": "{:.2f}"}), width='stretch', hide_index=True)
    if len(r):
        top = r.head(20)
        fig = px.scatter(top, x="support", y="confidence", size="lift", color="lift", hover_data=["Antecedent", "Consequent"],
                         color_continuous_scale=SEQ[2:], title="A1. Top rules — support vs confidence (bubble size = lift)")
        fig.update_layout(height=460); st.plotly_chart(fig, width='stretch')
    insight("Two clear bundles appear. **Data-centre bundle**: server rack → smart PDU → cooling unit → environmental sensors (lifts ≈3). "
            "**Meeting-room bundle**: AV rack → table connectivity box → floor box (lifts ≈3–4.5). Lift > 1 means the items are bought together more often than chance. "
            "Sell these as pre-configured kits with one price — it raises order value and makes the new brand a single-source supplier.")
    st.subheader("Diagnostic rules: which conditions lead to 'LIKELY TO BUY'?")
    br = arm["buy_rules"].head(12)
    st.dataframe(br.style.format({"support": "{:.3f}", "confidence": "{:.2f}", "lift": "{:.2f}"}), width='stretch', hide_index=True)
    if len(br):
        insight(f"Top condition: **{br.iloc[0]['Antecedent']}** → {br.iloc[0]['confidence']:.0%} likely to buy (lift {br.iloc[0]['lift']:.2f}, i.e. "
                f"{br.iloc[0]['lift']:.1f}× the average). These combinations are ready-made targeting rules for the sales team.")

# ===================================================================== ⑨ FEASIBILITY
with tabs[8]:
    st.subheader("Industry context (secondary research, 2025–26)")
    ctx = pd.DataFrame([
        ["India data-centre rack market", "USD 211 M (2025) → USD 746 M (2032)", "21.9% CAGR", "MarkNtel Advisors"],
        ["India data-centre PDU market", "USD 118.7 M (2026) → USD 518 M (2031)", "27.8% CAGR", "MarketsandMarkets"],
        ["India PDU market (all uses)", "USD 0.34 B (FY25) → USD 0.62 B (FY33)", "7.7% CAGR", "Markets & Data"],
        ["India operational DC capacity", "1,789 MW (H1 2026); 3,860 MW more planned by 2030", "—", "Cushman & Wakefield"],
        ["Top-5 rack vendors' share", "Schneider, Vertiv, Rittal, Eaton, nVent ≈ 38%", "Moderately fragmented", "MarkNtel Advisors"],
        ["Public procurement", "Class-I local supplier (≥50% local content) gets purchase preference", "—", "DPIIT PPP-MII Order 2017 (rev. 2024)"],
    ], columns=["Indicator", "Value", "Growth", "Source"])
    st.dataframe(ctx, width='stretch', hide_index=True)
    st.caption("Market-size estimates vary widely between research firms (some put India DC racks at only a few USD million) — treat them as direction, not precision.")

    st.subheader("Break-even scenario (edit the assumptions)")
    p = CLU["profile"].set_index("Persona")
    c = st.columns(4)
    reach = c[0].number_input("Accounts actively pitched per year", 100, 10000, 1200, 100)
    haircut = c[1].slider("Share of stated intent that converts", 0.1, 0.6, 0.30, 0.05)
    fixed = c[2].number_input("Annual fixed costs (₹ crore)", 1.0, 50.0, 6.0, 0.5)
    gm = c[3].slider("Gross margin", 0.15, 0.45, 0.28, 0.01)
    mix = p["Respondents"] / p["Respondents"].sum()
    rev_per_pitch = (mix * p["Likely_Buy_Rate"] * haircut * p["Exp_Spend_New_Brand"]).sum()  # ₹ lakh
    revenue_cr = reach * rev_per_pitch / 100
    be_cr = fixed / gm
    k = st.columns(3)
    k[0].metric("Projected revenue", f"₹{revenue_cr:.1f} Cr")
    k[1].metric("Break-even revenue", f"₹{be_cr:.1f} Cr")
    k[2].metric("Gap", f"₹{revenue_cr - be_cr:+.1f} Cr", delta="above break-even" if revenue_cr >= be_cr else "below break-even",
                delta_color="normal" if revenue_cr >= be_cr else "inverse")
    need = be_cr * 100 / max(rev_per_pitch, 1e-6)
    insight(f"With the survey's segment mix, each pitched account is worth about ₹{rev_per_pitch*100000:,.0f} of first-year revenue after the intent haircut. "
            f"To cover ₹{fixed:.1f} Cr fixed costs at {gm:.0%} gross margin you need ₹{be_cr:.1f} Cr revenue, i.e. roughly **{need:,.0f} accounts pitched per year**. "
            "Fixed costs, margin and conversion here are editable assumptions, not sourced figures — replace them with quotes for your plant, team and pricing.")
