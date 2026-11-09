"""
clean_data.py  —  Data cleaning & transformation
Input : data/raw_survey_responses.csv
Output: data/cleaned_survey_data.csv, data/cleaning_log.csv
Every step is logged (rows/cells affected) so it can be quoted in the report.
"""
import re
import numpy as np
import pandas as pd

raw = pd.read_csv("data/raw_survey_responses.csv")
df = raw.copy()
log = []

def note(step, action, affected):
    log.append({"Step": step, "Action": action, "Records/cells affected": int(affected)})

# 1. Duplicates ---------------------------------------------------------------
before = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset="Respondent_ID", keep="first")
note("1. Duplicates", "Removed duplicate survey submissions (exact copies / repeated Respondent_ID)", before - len(df))

# 2. Whitespace & case --------------------------------------------------------
obj_cols = df.select_dtypes(include=["object", "string"]).columns
n_ws = 0
for c in obj_cols:
    s = df[c].astype("string")
    stripped = s.str.strip()
    n_ws += int((s != stripped).fillna(False).sum())
    df[c] = stripped.astype(object).where(stripped.notna(), np.nan)
note("2. Text hygiene", "Trimmed leading/trailing spaces in text answers", n_ws)

# 3. City -> canonical city, Region, Tier ----------------------------------------
CITY_MAP = {
    "mumbai": "Mumbai", "bombay": "Mumbai", "navi mumbai": "Mumbai",
    "delhi": "Delhi NCR", "new delhi": "Delhi NCR", "delhi ncr": "Delhi NCR", "gurgaon": "Delhi NCR", "gurugram": "Delhi NCR", "noida": "Delhi NCR",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "blr": "Bengaluru",
    "chennai": "Chennai", "madras": "Chennai", "hyderabad": "Hyderabad", "hyd": "Hyderabad", "secunderabad": "Hyderabad",
    "pune": "Pune", "kolkata": "Kolkata", "calcutta": "Kolkata", "ahmedabad": "Ahmedabad", "amdavad": "Ahmedabad",
    "jaipur": "Jaipur", "lucknow": "Lucknow", "kochi": "Kochi", "cochin": "Kochi", "coimbatore": "Coimbatore", "kovai": "Coimbatore",
    "indore": "Indore", "chandigarh": "Chandigarh", "mohali": "Chandigarh", "bhubaneswar": "Bhubaneswar", "bbsr": "Bhubaneswar",
    "nagpur": "Nagpur", "visakhapatnam": "Visakhapatnam", "vizag": "Visakhapatnam", "surat": "Surat", "guwahati": "Guwahati",
    "raipur": "Raipur", "mysuru": "Mysuru", "mysore": "Mysuru", "dehradun": "Dehradun", "ranchi": "Ranchi",
    "vadodara": "Vadodara", "baroda": "Vadodara", "madurai": "Madurai", "patna": "Patna",
}
CITY_INFO = {
    "Mumbai": ("West", "Metro"), "Delhi NCR": ("North", "Metro"), "Bengaluru": ("South", "Metro"), "Chennai": ("South", "Metro"),
    "Hyderabad": ("South", "Metro"), "Pune": ("West", "Metro"), "Kolkata": ("East", "Metro"), "Ahmedabad": ("West", "Tier-2"),
    "Jaipur": ("North", "Tier-2"), "Lucknow": ("North", "Tier-2"), "Kochi": ("South", "Tier-2"), "Coimbatore": ("South", "Tier-2"),
    "Indore": ("West", "Tier-2"), "Chandigarh": ("North", "Tier-2"), "Bhubaneswar": ("East", "Tier-2"), "Nagpur": ("West", "Tier-2"),
    "Visakhapatnam": ("South", "Tier-2"), "Surat": ("West", "Tier-2"), "Guwahati": ("East", "Tier-3"), "Raipur": ("East", "Tier-3"),
    "Mysuru": ("South", "Tier-3"), "Dehradun": ("North", "Tier-3"), "Ranchi": ("East", "Tier-3"), "Vadodara": ("West", "Tier-3"),
    "Madurai": ("South", "Tier-3"), "Patna": ("East", "Tier-3"),
}
orig = df["Q4_City"].copy()
df["City"] = df["Q4_City"].str.lower().map(CITY_MAP)
note("3. City standardisation", "City answers rewritten to one of 26 canonical names (Bombay, BLR, Gurgaon, lower-case…)", int((orig.notna() & (orig != df["City"])).sum()))
note("3. City standardisation", "City left blank -> 'Unknown'", int(df["City"].isna().sum()))
df["City"] = df["City"].fillna("Unknown")
df["Region"] = df["City"].map(lambda c: CITY_INFO.get(c, ("Unknown", "Unknown"))[0])
df["City_Tier"] = df["City"].map(lambda c: CITY_INFO.get(c, ("Unknown", "Unknown"))[1])

# 4. Categorical label harmonisation -----------------------------------------
size_map = {"1-10": "1-10", "1 - 10": "1-10", "11-50": "11-50", "11 to 50": "11-50", "51-200": "51-200",
            "201-1000": "201-1000", "201 - 1000": "201-1000", "1000+": "1000+", "Above 1000": "1000+"}
n = int((df["Q3_Company_Size"] != df["Q3_Company_Size"].map(size_map)).sum())
df["Q3_Company_Size"] = df["Q3_Company_Size"].map(size_map)
note("4. Label harmonisation", "Company-size labels unified (e.g. 'Above 1000' -> '1000+')", n)

def yn(v):
    if pd.isna(v): return np.nan
    v = str(v).strip().lower()
    return {"y": "Yes", "yes": "Yes", "n": "No", "no": "No", "maybe": "Maybe",
            "not sure": "Not sure", "ns": "Not sure", "don't know": "Not sure"}.get(v, v.title())
for c in ["Q26_Likely_To_Buy_12M", "Q19_Requires_MakeInIndia_ClassI", "Q24_Trial_Order_Willingness"]:
    new = df[c].map(yn); n = int((new != df[c]).sum()); df[c] = new
    note("4. Label harmonisation", f"{c}: Y/yes/YES/N… unified", n)

brand_map = {"APC": "Schneider/APC", "NetRack": "Netrack", "VALRACK": "Valrack", "local": "Local fabricator/unbranded"}
n = int(df["Q9_Current_Primary_Brand"].isin(brand_map).sum())
df["Q9_Current_Primary_Brand"] = df["Q9_Current_Primary_Brand"].replace(brand_map)
note("4. Label harmonisation", "Brand aliases unified (APC, NetRack, VALRACK, local)", n)

ind_canon = {s.upper(): s for s in df["Q2_Industry"].dropna().unique() if not s.isupper() or s in ("IT/ITES",)}
ind_canon.update({"IT/ITES": "IT/ITeS"})
n = int(df["Q2_Industry"].str.isupper().sum())
df["Q2_Industry"] = df["Q2_Industry"].map(lambda v: ind_canon.get(v, v) if isinstance(v, str) and v.isupper() else v)
note("4. Label harmonisation", "Industry names in ALL CAPS converted to standard case", n)

# 5. Numbers stored as text ----------------------------------------------------
def to_num(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float)): return float(v)
    m = re.search(r"[-+]?\d*\.?\d+", str(v))
    return float(m.group()) if m else np.nan
txt = df["Q7_Annual_AVIT_Spend_Lakh"].astype(str).str.contains("Rs", na=False).sum()
num_cols = ["Q5_Years_In_Business", "Q7_Annual_AVIT_Spend_Lakh", "Q8_Racks_Purchased_12M", "Q11_Avg_Lead_Time_Weeks",
            "Q12_Current_Supplier_Satisfaction", "Q18_Discount_Needed_To_Switch_Pct", "Q21_WTP_Smart_PDU_INR",
            "Q25_Expected_Spend_New_Brand_Lakh", "Q17_Openness_New_Indian_Brand", "Q20_Interest_Smart_Monitoring"] + \
           [c for c in df.columns if c.startswith("Q16_Imp_")]
for c in num_cols:
    df[c] = df[c].map(to_num)
note("5. Type conversion", "Spend typed as text ('Rs 12.4 L') parsed to numbers", txt)

# 6. Invalid / impossible values -> NaN -------------------------------------------
likert = ["Q12_Current_Supplier_Satisfaction", "Q17_Openness_New_Indian_Brand", "Q20_Interest_Smart_Monitoring"] + \
         [c for c in df.columns if c.startswith("Q16_Imp_")]
n = 0
for c in likert:
    bad = ~df[c].between(1, 5) & df[c].notna(); n += int(bad.sum()); df.loc[bad, c] = np.nan
note("6. Validity rules", "Likert ratings outside 1-5 (0, 6, 7, 9, 10) set to missing", n)
rules = [("Q8_Racks_Purchased_12M", lambda s: (s < 0) | (s >= 900), "Racks bought negative or 999 (placeholder) set to missing"),
         ("Q5_Years_In_Business", lambda s: (s < 1) | (s > 100), "Years in business <1 or >100 set to missing"),
         ("Q11_Avg_Lead_Time_Weeks", lambda s: (s <= 0) | (s > 40), "Lead time 0 or >40 weeks set to missing"),
         ("Q18_Discount_Needed_To_Switch_Pct", lambda s: s > 60, "Discount >60% (not a credible switching threshold) set to missing"),
         ("Q21_WTP_Smart_PDU_INR", lambda s: (s < 1000) | (s > 500000), "Smart-PDU WTP <₹1,000 or >₹5,00,000 set to missing")]
for c, f, msg in rules:
    bad = f(df[c]) & df[c].notna(); note("6. Validity rules", msg, bad.sum()); df.loc[bad, c] = np.nan

# 7. Outliers on money columns: IQR on log scale within company-size band, then cap ---------
def iqr_cap(col, group):
    capped = 0
    for g, idx in df.groupby(group).groups.items():
        s = np.log1p(df.loc[idx, col])
        q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
        hi = np.expm1(q3 + 3 * iqr); lo = np.expm1(max(q1 - 3 * iqr, 0))
        over = df.loc[idx, col] > hi; under = df.loc[idx, col] < lo
        capped += int(over.sum() + under.sum())
        df.loc[over[over].index, col] = hi; df.loc[under[under].index, col] = lo
    return capped
df["Q3_Company_Size"] = df["Q3_Company_Size"].fillna(df["Q3_Company_Size"].mode()[0])
note("7. Outliers", "Annual AV/IT spend capped (log-IQR ×3 within company-size band) — mainly ₹ vs ₹-lakh entry errors",
     iqr_cap("Q7_Annual_AVIT_Spend_Lakh", "Q3_Company_Size"))
note("7. Outliers", "Expected spend with new brand capped (log-IQR ×3 within size band)",
     iqr_cap("Q25_Expected_Spend_New_Brand_Lakh", "Q3_Company_Size"))
# new-brand spend cannot exceed total AV/IT budget
over = df["Q25_Expected_Spend_New_Brand_Lakh"] > df["Q7_Annual_AVIT_Spend_Lakh"]
note("7. Outliers", "Expected new-brand spend > total budget -> capped at 60% of budget", over.sum())
df.loc[over, "Q25_Expected_Spend_New_Brand_Lakh"] = (df.loc[over, "Q7_Annual_AVIT_Spend_Lakh"] * 0.6).round(2)

# 8. Imputation -------------------------------------------------------------------
num_impute = ["Q5_Years_In_Business", "Q7_Annual_AVIT_Spend_Lakh", "Q8_Racks_Purchased_12M", "Q11_Avg_Lead_Time_Weeks",
              "Q18_Discount_Needed_To_Switch_Pct", "Q21_WTP_Smart_PDU_INR"]
for c in num_impute:
    k = int(df[c].isna().sum())
    df[c] = df[c].fillna(df.groupby("Q3_Company_Size")[c].transform("median"))
    note("8. Imputation", f"{c}: median within company-size band", k)
for c in likert:
    k = int(df[c].isna().sum())
    if k:
        df[c] = df[c].fillna(df[c].median()); note("8. Imputation", f"{c}: overall median rating", k)
for c in ["Q6_Annual_Revenue_Band", "Q19_Requires_MakeInIndia_ClassI", "Q26_Likely_To_Buy_12M", "Q24_Trial_Order_Willingness"]:
    k = int(df[c].isna().sum())
    if k:
        if c == "Q6_Annual_Revenue_Band":
            df[c] = df[c].fillna(df.groupby("Q3_Company_Size")[c].transform(lambda s: s.mode()[0]))
        else:
            df[c] = df[c].fillna(df[c].mode()[0])
        note("8. Imputation", f"{c}: mode", k)
df["Q8_Racks_Purchased_12M"] = df["Q8_Racks_Purchased_12M"].round().astype(int)
df["Q5_Years_In_Business"] = df["Q5_Years_In_Business"].round().astype(int)

for c in ["Q13_Pain_Points", "Q14_Products_Bought_12M"]:
    df[c] = df[c].fillna("None")  # "None" was a valid answer that pandas reads as missing

# 9. Feature engineering / transformation ------------------------------------------------
SIZE_ORD = {"1-10": 1, "11-50": 2, "51-200": 3, "201-1000": 4, "1000+": 5}
REV_ORD = {"<1 Cr": 1, "1-10 Cr": 2, "10-100 Cr": 3, "100-500 Cr": 4, "500+ Cr": 5}
df["Size_Ord"] = df["Q3_Company_Size"].map(SIZE_ORD)
df["Revenue_Ord"] = df["Q6_Annual_Revenue_Band"].map(REV_ORD)
df["Log_AVIT_Spend"] = np.log1p(df["Q7_Annual_AVIT_Spend_Lakh"]).round(4)
df["MII_Required"] = (df["Q19_Requires_MakeInIndia_ClassI"] == "Yes").astype(int)
df["Target_Likely_Buy"] = (df["Q26_Likely_To_Buy_12M"] == "Yes").astype(int)
df["Trial_Score"] = df["Q24_Trial_Order_Willingness"].map({"No": 0, "Maybe": 1, "Yes": 2})

PAIN = ["Long lead times", "High price", "Poor customisation", "Weak after-sales service", "Import dependence / forex risk",
        "Inconsistent build quality", "No local engineering support", "No smart monitoring", "Compliance paperwork (local content)"]
for p in PAIN:
    df["Pain_" + re.sub(r"[^A-Za-z]+", "_", p).strip("_")] = df["Q13_Pain_Points"].fillna("").str.contains(re.escape(p)).astype(int)
df["Pain_Count"] = df[[c for c in df.columns if c.startswith("Pain_") and c != "Pain_Count"]].sum(axis=1)

PRODUCTS = ["Server Rack", "Network Rack", "Wall-Mount Rack", "Rack Accessories", "Rack Cooling Unit", "Env. Monitoring Sensors",
            "Basic PDU", "Smart PDU", "Custom Industrial Enclosure", "AV Rack", "Table Connectivity Box", "Floor Box", "Cable Management"]
for p in PRODUCTS:
    df["Plan_" + re.sub(r"[^A-Za-z]+", "_", p).strip("_")] = df["Q15_Products_Planned_12M"].fillna("").str.split("; ").map(lambda l: p in l).astype(int)
df["Planned_Basket_Size"] = df[[c for c in df.columns if c.startswith("Plan_")]].sum(axis=1)
df["Spend_per_Rack"] = (df["Q7_Annual_AVIT_Spend_Lakh"] / df["Q8_Racks_Purchased_12M"].clip(lower=1)).round(3)
note("9. Feature engineering", "Created ordinal, log, binary (one-hot) and count features: Size_Ord, Log_AVIT_Spend, 9 Pain_* flags, 13 Plan_* flags, Pain_Count, Planned_Basket_Size, Spend_per_Rack, Target_Likely_Buy", len(df))

df = df.drop(columns=["Q4_City"])
df.to_csv("data/cleaned_survey_data.csv", index=False)
pd.DataFrame(log).to_csv("data/cleaning_log.csv", index=False)
print(pd.DataFrame(log).to_string())
print("final shape", df.shape, "missing cells:", int(df.isna().sum().sum()))
