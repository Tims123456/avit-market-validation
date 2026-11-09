"""
generate_data.py
Synthetic survey data for: an Indian AV/IT infrastructure manufacturing brand
(racks, smart PDUs, rack cooling & monitoring, custom enclosures, meeting-room connectivity).

Respondents = B2B buyers / decision influencers across India.
Each question is designed to feed one or more algorithms:
  REG  = Regression       (target: Q25_Expected_Spend_New_Brand_Lakh)
  CLF  = Classification   (target: Q26_Likely_To_Buy_12M)
  CLU  = Clustering       (Q17 importance ratings + spend/size behaviour)
  ARM  = Association rules (Q16 planned product basket, Q14 pain points)

Realism is driven by 4 HIDDEN buyer segments (the analyst does not see them;
clustering should rediscover something close). Noise, missing values,
outliers, inconsistent labels and duplicates are injected deliberately.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 3200          # unique respondents
N_DUP = 45        # duplicate submissions injected

SEGMENTS = {
    # name: weight
    "Cost-driven SMB integrators": 0.34,
    "Enterprise & data-centre buyers": 0.22,
    "Govt / PSU compliance buyers": 0.18,
    "AV & custom-build specialists": 0.26,
}
SEG_NAMES = list(SEGMENTS)
SEG_W = np.array(list(SEGMENTS.values()))

ROLES = {
    0: (["IT System Integrator", "Network/Cabling Contractor", "IT Reseller/Dealer", "SME IT Head"], [0.40, 0.25, 0.20, 0.15]),
    1: (["Enterprise IT/Infra Head", "Data-Centre Contractor", "Colocation/DC Operator", "Telecom/ISP Engineer"], [0.35, 0.25, 0.20, 0.20]),
    2: (["Govt/PSU Procurement Officer", "PSU IT Head", "Defence/Railways Contractor", "Govt Project System Integrator"], [0.35, 0.20, 0.15, 0.30]),
    3: (["AV System Integrator", "AV Consultant/Designer", "Facility/Admin Manager", "Industrial Panel Builder/OEM"], [0.45, 0.15, 0.20, 0.20]),
}
INDUSTRY = {
    0: (["IT/ITeS", "Manufacturing", "Retail", "Healthcare", "Education", "BFSI"], [0.30, 0.20, 0.15, 0.12, 0.13, 0.10]),
    1: (["IT/ITeS", "BFSI", "Data Centre/Colocation", "Telecom", "Healthcare", "Manufacturing"], [0.25, 0.22, 0.25, 0.15, 0.06, 0.07]),
    2: (["Government", "PSU/Utilities", "Defence", "Railways/Transport", "Education", "Healthcare"], [0.35, 0.25, 0.12, 0.13, 0.10, 0.05]),
    3: (["Corporate Offices", "Education", "Hospitality", "Media/Broadcast", "Manufacturing", "Healthcare"], [0.35, 0.20, 0.15, 0.10, 0.12, 0.08]),
}
SIZE_BANDS = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
SIZE_P = {
    0: [0.28, 0.42, 0.22, 0.06, 0.02],
    1: [0.01, 0.06, 0.20, 0.35, 0.38],
    2: [0.02, 0.10, 0.28, 0.30, 0.30],
    3: [0.15, 0.40, 0.30, 0.11, 0.04],
}
SIZE_SPEND_MEDIAN = [3, 8, 22, 70, 210]      # ₹ lakh / year on AV-IT infrastructure
REVENUE_BANDS = ["<1 Cr", "1-10 Cr", "10-100 Cr", "100-500 Cr", "500+ Cr"]

CITIES = {
    # city: (region, tier, list of messy spellings respondents type)
    "Mumbai": ("West", "Metro", ["Mumbai", "mumbai", "MUMBAI ", "Bombay", "Navi Mumbai"]),
    "Delhi NCR": ("North", "Metro", ["Delhi", "New Delhi", "delhi ncr", "Gurgaon", "Gurugram", "Noida"]),
    "Bengaluru": ("South", "Metro", ["Bengaluru", "Bangalore", "bangalore", "BLR"]),
    "Chennai": ("South", "Metro", ["Chennai", "chennai", "Madras"]),
    "Hyderabad": ("South", "Metro", ["Hyderabad", "hyderabad", "Hyd", "Secunderabad"]),
    "Pune": ("West", "Metro", ["Pune", "pune", "PUNE"]),
    "Kolkata": ("East", "Metro", ["Kolkata", "Calcutta", "kolkata"]),
    "Ahmedabad": ("West", "Tier-2", ["Ahmedabad", "ahmedabad", "Amdavad"]),
    "Jaipur": ("North", "Tier-2", ["Jaipur", "jaipur"]),
    "Lucknow": ("North", "Tier-2", ["Lucknow", "lucknow"]),
    "Kochi": ("South", "Tier-2", ["Kochi", "Cochin"]),
    "Coimbatore": ("South", "Tier-2", ["Coimbatore", "Kovai"]),
    "Indore": ("West", "Tier-2", ["Indore", "indore"]),
    "Chandigarh": ("North", "Tier-2", ["Chandigarh", "chandigarh", "Mohali"]),
    "Bhubaneswar": ("East", "Tier-2", ["Bhubaneswar", "BBSR"]),
    "Nagpur": ("West", "Tier-2", ["Nagpur"]),
    "Visakhapatnam": ("South", "Tier-2", ["Visakhapatnam", "Vizag"]),
    "Surat": ("West", "Tier-2", ["Surat"]),
    "Guwahati": ("East", "Tier-3", ["Guwahati"]),
    "Raipur": ("East", "Tier-3", ["Raipur"]),
    "Mysuru": ("South", "Tier-3", ["Mysuru", "Mysore"]),
    "Dehradun": ("North", "Tier-3", ["Dehradun"]),
    "Ranchi": ("East", "Tier-3", ["Ranchi"]),
    "Vadodara": ("West", "Tier-3", ["Vadodara", "Baroda"]),
    "Madurai": ("South", "Tier-3", ["Madurai"]),
    "Patna": ("East", "Tier-3", ["Patna"]),
}
CITY_LIST = list(CITIES)
TIER_W = {  # probability of a city by tier, per segment
    0: {"Metro": 0.45, "Tier-2": 0.38, "Tier-3": 0.17},
    1: {"Metro": 0.82, "Tier-2": 0.15, "Tier-3": 0.03},
    2: {"Metro": 0.50, "Tier-2": 0.32, "Tier-3": 0.18},
    3: {"Metro": 0.65, "Tier-2": 0.28, "Tier-3": 0.07},
}

BRANDS = ["Schneider/APC", "Vertiv", "Rittal", "Legrand", "Eaton", "Netrack", "Valrack",
          "Local fabricator/unbranded", "Imported (China/Taiwan)"]
BRAND_P = {
    0: [0.10, 0.03, 0.02, 0.04, 0.02, 0.17, 0.12, 0.33, 0.17],
    1: [0.30, 0.21, 0.19, 0.08, 0.08, 0.07, 0.03, 0.02, 0.02],
    2: [0.18, 0.10, 0.10, 0.06, 0.05, 0.17, 0.12, 0.17, 0.05],
    3: [0.08, 0.03, 0.05, 0.20, 0.03, 0.10, 0.06, 0.30, 0.15],
}
CHANNELS = ["Direct from OEM", "Distributor/Dealer", "Online B2B (IndiaMART etc.)", "Tender / GeM portal"]
CHANNEL_P = {0: [0.08, 0.55, 0.32, 0.05], 1: [0.55, 0.38, 0.04, 0.03], 2: [0.10, 0.20, 0.05, 0.65], 3: [0.18, 0.52, 0.25, 0.05]}

PAIN = ["Long lead times", "High price", "Poor customisation", "Weak after-sales service",
        "Import dependence / forex risk", "Inconsistent build quality", "No local engineering support",
        "No smart monitoring", "Compliance paperwork (local content)"]
PAIN_P = {
    0: [0.35, 0.70, 0.20, 0.40, 0.15, 0.45, 0.25, 0.10, 0.03],
    1: [0.55, 0.30, 0.25, 0.30, 0.40, 0.10, 0.20, 0.35, 0.05],
    2: [0.45, 0.35, 0.20, 0.35, 0.30, 0.25, 0.30, 0.15, 0.55],
    3: [0.40, 0.35, 0.65, 0.30, 0.25, 0.30, 0.35, 0.10, 0.04],
}

PRODUCTS = ["Server Rack", "Network Rack", "Wall-Mount Rack", "Rack Accessories", "Rack Cooling Unit",
            "Env. Monitoring Sensors", "Basic PDU", "Smart PDU", "Custom Industrial Enclosure",
            "AV Rack", "Table Connectivity Box", "Floor Box", "Cable Management"]
BASE_PROD_P = {
    0: [0.25, 0.60, 0.65, 0.55, 0.06, 0.06, 0.55, 0.08, 0.06, 0.06, 0.05, 0.03, 0.55],
    1: [0.80, 0.55, 0.15, 0.65, 0.45, 0.40, 0.25, 0.55, 0.05, 0.04, 0.06, 0.03, 0.55],
    2: [0.55, 0.50, 0.35, 0.45, 0.20, 0.20, 0.35, 0.25, 0.30, 0.10, 0.15, 0.08, 0.40],
    3: [0.08, 0.20, 0.25, 0.35, 0.05, 0.05, 0.25, 0.08, 0.35, 0.70, 0.55, 0.30, 0.60],
}
# conditional lifts: if A is in basket, P(B) is boosted -> the association rules we expect to find
LIFTS = [("Server Rack", "Smart PDU", 0.30), ("Server Rack", "Rack Accessories", 0.15),
         ("Smart PDU", "Env. Monitoring Sensors", 0.35), ("Smart PDU", "Rack Cooling Unit", 0.20),
         ("Rack Cooling Unit", "Env. Monitoring Sensors", 0.30),
         ("AV Rack", "Cable Management", 0.20), ("AV Rack", "Table Connectivity Box", 0.20),
         ("Table Connectivity Box", "Floor Box", 0.35), ("Wall-Mount Rack", "Basic PDU", 0.20),
         ("Network Rack", "Cable Management", 0.15)]

IMPORTANCE = ["Price", "Build_Quality", "Customisation", "Delivery_Speed", "After_Sales",
              "Smart_Features", "Brand_Reputation", "Make_In_India", "Energy_Efficiency", "Warranty"]
IMP_MEAN = {  # mean importance (1-5) per segment
    0: [4.6, 3.2, 2.6, 3.8, 3.4, 2.0, 2.5, 2.6, 2.4, 3.6],
    1: [3.0, 4.6, 3.2, 3.9, 4.4, 4.3, 4.4, 2.6, 4.2, 4.0],
    2: [3.6, 3.9, 2.9, 3.3, 3.8, 3.0, 3.0, 4.7, 3.2, 3.9],
    3: [3.4, 3.8, 4.6, 4.1, 3.6, 2.7, 3.1, 2.8, 2.7, 3.4],
}
PAY_TERMS = ["100% advance", "30 days credit", "60 days credit", "90 days credit", "As per tender (post-delivery)"]
PAY_P = {0: [0.05, 0.25, 0.40, 0.30, 0.00], 1: [0.03, 0.40, 0.42, 0.15, 0.00],
         2: [0.01, 0.06, 0.10, 0.08, 0.75], 3: [0.08, 0.45, 0.32, 0.15, 0.00]}
ENGAGE = ["Direct sales engineer", "Channel partner / distributor", "Online configurator + instant quote", "GeM / tender portal"]
ENG_P = {0: [0.12, 0.45, 0.40, 0.03], 1: [0.62, 0.25, 0.12, 0.01], 2: [0.15, 0.15, 0.05, 0.65], 3: [0.40, 0.30, 0.28, 0.02]}


def pick(options, p):
    return RNG.choice(options, p=np.array(p) / np.sum(p))


def clip_round(x, lo=1, hi=5):
    return int(np.clip(np.round(x), lo, hi))


rows = []
seg_ids = RNG.choice(4, size=N, p=SEG_W)
for i, s in enumerate(seg_ids):
    r = {"Respondent_ID": f"R{i + 1:05d}"}
    r["Q1_Role"] = pick(*ROLES[s])
    r["Q2_Industry"] = pick(*INDUSTRY[s])
    size_idx = RNG.choice(5, p=SIZE_P[s])
    r["Q3_Company_Size"] = SIZE_BANDS[size_idx]
    # city: choose tier then city
    tier = pick(list(TIER_W[s]), list(TIER_W[s].values()))
    c_in_tier = [c for c in CITY_LIST if CITIES[c][1] == tier]
    w = np.array([3.0 if c in ("Mumbai", "Delhi NCR", "Bengaluru") else 1.0 for c in c_in_tier])
    city = RNG.choice(c_in_tier, p=w / w.sum())
    r["_city_true"] = city
    r["Q4_City"] = RNG.choice(CITIES[city][2])
    r["Q5_Years_In_Business"] = int(np.clip(RNG.gamma(2.2 + size_idx * 0.8, 3.0), 1, 60))
    rev_idx = int(np.clip(size_idx + RNG.choice([-1, 0, 0, 0, 1]), 0, 4))
    r["Q6_Annual_Revenue_Band"] = REVENUE_BANDS[rev_idx]

    seg_mult = [0.75, 1.45, 1.10, 0.85][s]
    spend = RNG.lognormal(np.log(SIZE_SPEND_MEDIAN[size_idx] * seg_mult), 0.55)
    r["Q7_Annual_AVIT_Spend_Lakh"] = round(spend, 2)
    rack_share = [0.35, 0.22, 0.30, 0.20][s]
    r["Q8_Racks_Purchased_12M"] = int(RNG.poisson(max(spend * rack_share / 0.55, 0.3)))
    r["Q9_Current_Primary_Brand"] = pick(BRANDS, BRAND_P[s])
    r["Q10_Purchase_Channel"] = pick(CHANNELS, CHANNEL_P[s])
    is_import = r["Q9_Current_Primary_Brand"] in ("Imported (China/Taiwan)", "Vertiv", "Rittal", "Eaton")
    lead = RNG.gamma(4, 1.5) + (2.5 if is_import else 0) + (1.5 if s == 3 else 0)
    r["Q11_Avg_Lead_Time_Weeks"] = round(float(np.clip(lead, 1, 26)), 1)
    sat = 3.6 - 0.12 * (lead - 6) + RNG.normal(0, 0.8) + (0.4 if s == 1 else 0) \
        - (0.4 if r["Q9_Current_Primary_Brand"] == "Local fabricator/unbranded" else 0)
    r["Q12_Current_Supplier_Satisfaction"] = clip_round(sat)

    # Pain points (multi-select) — longer lead time raises "Long lead times"
    pp = np.array(PAIN_P[s], dtype=float)
    pp[0] = np.clip(pp[0] + 0.04 * (lead - 6), 0.05, 0.95)
    if r["Q12_Current_Supplier_Satisfaction"] <= 2:
        pp[3] = min(pp[3] + 0.25, 0.95)
    pains = [p for p, prob in zip(PAIN, pp) if RNG.random() < prob]
    r["Q13_Pain_Points"] = "; ".join(pains) if pains else "None"

    # Products bought last 12M (multi) & planned next 12M (multi, ARM basket)
    def basket(base, boost=0.0):
        pr = dict(zip(PRODUCTS, np.clip(np.array(base) + boost, 0, 0.95)))
        chosen = [p for p in PRODUCTS if RNG.random() < pr[p]]
        for a, b, lift in LIFTS:
            if a in chosen and b not in chosen and RNG.random() < lift:
                chosen.append(b)
        return [p for p in PRODUCTS if p in chosen]
    bought = basket(BASE_PROD_P[s])
    r["Q14_Products_Bought_12M"] = "; ".join(bought) if bought else "None"
    planned = basket(BASE_PROD_P[s], boost=-0.05)
    if not planned:
        planned = [RNG.choice(PRODUCTS, p=np.array(BASE_PROD_P[s]) / sum(BASE_PROD_P[s]))]
    r["Q15_Products_Planned_12M"] = "; ".join(planned)

    # Importance ratings (Likert 1-5)
    imp = {}
    for k, m in zip(IMPORTANCE, IMP_MEAN[s]):
        imp[k] = clip_round(RNG.normal(m, 0.75))
        r[f"Q16_Imp_{k}"] = imp[k]

    # Openness to a NEW Indian manufacturer brand
    open_lat = [3.2, 2.6, 3.9, 3.4][s] + 0.30 * (3 - r["Q12_Current_Supplier_Satisfaction"]) \
        + 0.06 * (lead - 6) + 0.25 * (imp["Make_In_India"] - 3) + RNG.normal(0, 0.7)
    r["Q17_Openness_New_Indian_Brand"] = clip_round(open_lat)
    disc = [20, 12, 8, 12][s] - 2.2 * (r["Q17_Openness_New_Indian_Brand"] - 3) + 1.5 * (imp["Price"] - 3) + RNG.normal(0, 4)
    r["Q18_Discount_Needed_To_Switch_Pct"] = round(float(np.clip(disc, 0, 45)), 1)
    mii_p = [0.08, 0.15, 0.85, 0.12][s]
    r["Q19_Requires_MakeInIndia_ClassI"] = pick(["Yes", "No", "Not sure"], [mii_p, (1 - mii_p) * 0.75, (1 - mii_p) * 0.25])
    r["Q20_Interest_Smart_Monitoring"] = clip_round(0.6 * imp["Smart_Features"] + 0.4 * [2.2, 4.3, 3.0, 2.5][s] + RNG.normal(0, 0.6))
    wtp = RNG.lognormal(np.log([14000, 42000, 28000, 20000][s]), 0.30) * (1 + 0.08 * (r["Q20_Interest_Smart_Monitoring"] - 3))
    r["Q21_WTP_Smart_PDU_INR"] = int(round(wtp, -2))
    r["Q22_Preferred_Payment_Terms"] = pick(PAY_TERMS, PAY_P[s])
    r["Q23_Preferred_Engagement"] = pick(ENGAGE, ENG_P[s])

    # ---- Classification target: likely to buy from the new brand in 12 months
    logit = (-1.10 + 0.95 * (r["Q17_Openness_New_Indian_Brand"] - 3)
             + 0.85 * (r["Q19_Requires_MakeInIndia_ClassI"] == "Yes")
             + 0.30 * (3 - r["Q12_Current_Supplier_Satisfaction"])
             + 0.05 * (lead - 6)
             + 0.30 * ("Poor customisation" in pains) + 0.25 * ("Long lead times" in pains)
             - 0.045 * (r["Q18_Discount_Needed_To_Switch_Pct"] - 15)
             + 0.25 * (r["Q16_Imp_Customisation"] - 3) * (s == 3)
             + 0.20 * (r["Q10_Purchase_Channel"] == "Online B2B (IndiaMART etc.)")
             + RNG.normal(0, 0.6))
    p_buy = 1 / (1 + np.exp(-logit))
    likely = RNG.random() < p_buy
    r["Q26_Likely_To_Buy_12M"] = "Yes" if likely else "No"
    if likely:
        r["Q24_Trial_Order_Willingness"] = pick(["Yes", "Maybe", "No"], [0.62, 0.33, 0.05])
    else:
        r["Q24_Trial_Order_Willingness"] = pick(["Yes", "Maybe", "No"], [0.10, 0.40, 0.50])

    # ---- Regression target: expected first-year spend with the new brand (₹ lakh)
    if likely:
        share = np.clip(0.10 + 0.045 * (r["Q17_Openness_New_Indian_Brand"] - 3)
                        + 0.07 * (r["Q19_Requires_MakeInIndia_ClassI"] == "Yes")
                        + 0.03 * (s == 3) + RNG.normal(0, 0.04), 0.01, 0.6)
    else:
        share = np.clip(RNG.normal(0.012, 0.012), 0, 0.08)
    r["Q25_Expected_Spend_New_Brand_Lakh"] = round(float(spend * share * RNG.lognormal(0, 0.15)), 2)
    r["_segment_true"] = SEG_NAMES[s]
    rows.append(r)

df = pd.DataFrame(rows)

# ------------------------------------------------------------------ NOISE INJECTION
n = len(df)
def idx(frac):
    return RNG.choice(n, size=int(frac * n), replace=False)

# 1) Missing values (item non-response)
for col, frac in [("Q5_Years_In_Business", 0.04), ("Q7_Annual_AVIT_Spend_Lakh", 0.035),
                  ("Q8_Racks_Purchased_12M", 0.03), ("Q11_Avg_Lead_Time_Weeks", 0.05),
                  ("Q12_Current_Supplier_Satisfaction", 0.025), ("Q18_Discount_Needed_To_Switch_Pct", 0.06),
                  ("Q21_WTP_Smart_PDU_INR", 0.07), ("Q6_Annual_Revenue_Band", 0.05),
                  ("Q16_Imp_Energy_Efficiency", 0.02), ("Q16_Imp_Warranty", 0.02),
                  ("Q20_Interest_Smart_Monitoring", 0.02), ("Q4_City", 0.015)]:
    df[col] = df[col].astype(object)
    df.loc[idx(frac), col] = np.nan

# 2) Outliers / data-entry errors
o = idx(0.008); df.loc[o, "Q7_Annual_AVIT_Spend_Lakh"] = pd.to_numeric(df.loc[o, "Q7_Annual_AVIT_Spend_Lakh"], errors="coerce") * 100   # typed rupees-thousands
o = idx(0.004); df.loc[o, "Q8_Racks_Purchased_12M"] = 999
o = idx(0.003); df.loc[o, "Q8_Racks_Purchased_12M"] = -1 * RNG.integers(1, 10, len(o))
o = idx(0.004); df.loc[o, "Q11_Avg_Lead_Time_Weeks"] = RNG.choice([0, 52, 80, 120], len(o))
o = idx(0.004); df.loc[o, "Q5_Years_In_Business"] = RNG.choice([-3, 0, 150, 210], len(o))
for c in ["Q16_Imp_Price", "Q16_Imp_Build_Quality", "Q12_Current_Supplier_Satisfaction"]:
    o = idx(0.003); df[c] = df[c].astype(object); df.loc[o, c] = RNG.choice([0, 6, 7, 9, 10], len(o))
o = idx(0.004); df.loc[o, "Q18_Discount_Needed_To_Switch_Pct"] = RNG.choice([95, 120, 150], len(o))
o = idx(0.005); df.loc[o, "Q21_WTP_Smart_PDU_INR"] = RNG.choice([1, 50, 2500000, 9999999], len(o))
o = idx(0.006); df.loc[o, "Q25_Expected_Spend_New_Brand_Lakh"] = pd.to_numeric(df.loc[o, "Q25_Expected_Spend_New_Brand_Lakh"]) * RNG.choice([20, 50], len(o))

# 3) Inconsistent categorical labels
o = idx(0.06); df.loc[o, "Q3_Company_Size"] = df.loc[o, "Q3_Company_Size"].replace({"1000+": "Above 1000", "1-10": "1 - 10", "11-50": "11 to 50", "51-200": "51-200 ", "201-1000": "201 - 1000"})
for c in ["Q26_Likely_To_Buy_12M", "Q19_Requires_MakeInIndia_ClassI", "Q24_Trial_Order_Willingness"]:
    o = idx(0.05)
    df.loc[o, c] = df.loc[o, c].map(lambda v: RNG.choice({"Yes": ["Y", "yes", "YES", " Yes"], "No": ["N", "no", "NO "],
                                                          "Maybe": ["maybe", "MAYBE"], "Not sure": ["not sure", "NS", "Don't know"]}[v]))
o = idx(0.04); df.loc[o, "Q9_Current_Primary_Brand"] = df.loc[o, "Q9_Current_Primary_Brand"].replace(
    {"Schneider/APC": "APC", "Netrack": "NetRack", "Valrack": "VALRACK", "Local fabricator/unbranded": "local"})
o = idx(0.03); df.loc[o, "Q2_Industry"] = df.loc[o, "Q2_Industry"].str.upper()

# 4) Numbers typed as text
o = idx(0.01); df["Q7_Annual_AVIT_Spend_Lakh"] = df["Q7_Annual_AVIT_Spend_Lakh"].astype(object)
df.loc[o, "Q7_Annual_AVIT_Spend_Lakh"] = df.loc[o, "Q7_Annual_AVIT_Spend_Lakh"].map(lambda v: f"Rs {v} L" if pd.notna(v) else v)

# 5) Duplicate submissions
dups = df.sample(N_DUP, random_state=7)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=11).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
truth = df[["Respondent_ID", "_segment_true", "_city_true"]].drop_duplicates("Respondent_ID")
truth.to_csv("data/hidden_truth_segments.csv", index=False)
df.drop(columns=["_segment_true", "_city_true"]).to_csv("data/raw_survey_responses.csv", index=False)
print("raw rows:", len(df), "cols:", df.shape[1] - 2)
print(df["Q26_Likely_To_Buy_12M"].value_counts().head())
