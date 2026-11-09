"""
analysis.py — the four algorithms, shared by the Streamlit dashboard (app.py)
and by export_results.py (numbers used in the report).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, silhouette_score,
                             adjusted_rand_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

IMP_COLS = ["Q16_Imp_Price", "Q16_Imp_Build_Quality", "Q16_Imp_Customisation", "Q16_Imp_Delivery_Speed", "Q16_Imp_After_Sales",
            "Q16_Imp_Smart_Features", "Q16_Imp_Brand_Reputation", "Q16_Imp_Make_In_India", "Q16_Imp_Energy_Efficiency", "Q16_Imp_Warranty"]
NUM_FEATS = ["Size_Ord", "Revenue_Ord", "Q5_Years_In_Business", "Log_AVIT_Spend", "Q8_Racks_Purchased_12M", "Q11_Avg_Lead_Time_Weeks",
             "Q12_Current_Supplier_Satisfaction", "Q17_Openness_New_Indian_Brand", "Q18_Discount_Needed_To_Switch_Pct",
             "MII_Required", "Q20_Interest_Smart_Monitoring", "Pain_Count", "Planned_Basket_Size"] + IMP_COLS
CAT_FEATS = ["City_Tier", "Q10_Purchase_Channel", "Q9_Current_Primary_Brand"]
LEAKY = ["Q24_Trial_Order_Willingness", "Q25_Expected_Spend_New_Brand_Lakh", "Q26_Likely_To_Buy_12M"]


def load(path="data/cleaned_survey_data.csv"):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def feature_matrix(df):
    pain = [c for c in df.columns if c.startswith("Pain_") and c != "Pain_Count"]
    X = pd.concat([df[NUM_FEATS + pain], pd.get_dummies(df[CAT_FEATS], drop_first=True, dtype=int)], axis=1)
    return X


def pretty(name):
    return (name.replace("Q16_Imp_", "Importance: ").replace("Q17_", "").replace("Q18_", "").replace("Q20_", "")
            .replace("Q12_", "").replace("Q11_", "").replace("Q8_", "").replace("Q5_", "").replace("Q10_Purchase_Channel_", "Channel: ")
            .replace("Q9_Current_Primary_Brand_", "Current brand: ").replace("City_Tier_", "City tier: ").replace("Pain_", "Pain: ")
            .replace("_", " "))


# ---------------------------------------------------------------- CLASSIFICATION
def run_classification(df, seed=42):
    X = feature_matrix(df); y = df["Target_Likely_Buy"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=seed),
    }
    rows, rocs, fitted = [], {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for name, m in models.items():
        m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; yhat = (p >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(yte, p); rocs[name] = (fpr, tpr)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, yhat), "Precision": precision_score(yte, yhat),
                     "Recall": recall_score(yte, yhat), "F1": f1_score(yte, yhat), "ROC-AUC": roc_auc_score(yte, p),
                     "CV ROC-AUC (5-fold)": cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()})
        fitted[name] = m
    res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best = res.iloc[0]["Model"]
    cm = confusion_matrix(yte, (fitted[best].predict_proba(Xte)[:, 1] >= 0.5).astype(int))
    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    lr = fitted["Logistic Regression"][-1]
    coefs = pd.Series(lr.coef_[0], index=X.columns).sort_values()
    return {"results": res, "rocs": rocs, "best": best, "cm": cm, "rf_importance": imp, "lr_coefs": coefs,
            "fitted": fitted, "columns": X.columns, "base_rate": y.mean()}


# ---------------------------------------------------------------- REGRESSION
def run_regression(df, seed=42):
    """Target is heavily right-skewed (many ~0 values, a few large enterprises), so models are fitted on
    log(1 + spend) and errors are reported back in ₹ lakh."""
    X = feature_matrix(df); y_raw = df["Q25_Expected_Spend_New_Brand_Lakh"]; y = np.log1p(y_raw)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=5)),
        "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=5, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=seed),
    }
    rows, preds, fitted = [], {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr); p = m.predict(Xte); preds[name] = p; fitted[name] = m
        rows.append({"Model": name, "R² (log scale)": r2_score(yte, p), "R² (₹ lakh)": r2_score(np.expm1(yte), np.expm1(p)),
                     "MAE (₹ lakh)": mean_absolute_error(np.expm1(yte), np.expm1(p)),
                     "RMSE (log)": mean_squared_error(yte, p) ** 0.5})
    res = pd.DataFrame(rows).sort_values("R² (log scale)", ascending=False).reset_index(drop=True)
    best = res.iloc[0]["Model"]
    imp = pd.Series(fitted["Random Forest"].feature_importances_, index=X.columns).sort_values(ascending=False)
    lin = fitted["Linear Regression"][-1]
    coefs = pd.Series(lin.coef_, index=X.columns).sort_values()
    return {"results": res, "best": best, "y_test": np.expm1(yte.values), "pred_best": np.expm1(preds[best]),
            "y_test_log": yte.values, "pred_best_log": preds[best], "rf_importance": imp, "lin_coefs": coefs}


# ---------------------------------------------------------------- CLUSTERING
CLU_FEATS = IMP_COLS + ["Log_AVIT_Spend", "Size_Ord", "Q17_Openness_New_Indian_Brand", "Q20_Interest_Smart_Monitoring"]


def run_clustering(df, k=4, seed=42):
    Z = StandardScaler().fit_transform(df[CLU_FEATS])
    scan = []
    for kk in range(2, 9):
        km = KMeans(kk, n_init=10, random_state=seed).fit(Z)
        scan.append({"k": kk, "Inertia (WCSS)": km.inertia_, "Silhouette": silhouette_score(Z, km.labels_, sample_size=2000, random_state=seed)})
    km = KMeans(k, n_init=20, random_state=seed).fit(Z)
    labels = km.labels_
    pcs = PCA(2, random_state=seed).fit(Z)
    xy = pcs.transform(Z)
    prof = df.assign(Cluster=labels).groupby("Cluster").agg(
        Respondents=("Respondent_ID", "count"),
        Avg_Spend_Lakh=("Q7_Annual_AVIT_Spend_Lakh", "mean"),
        Median_Spend_Lakh=("Q7_Annual_AVIT_Spend_Lakh", "median"),
        Size_Ord=("Size_Ord", "mean"),
        Openness=("Q17_Openness_New_Indian_Brand", "mean"),
        Likely_Buy_Rate=("Target_Likely_Buy", "mean"),
        Exp_Spend_New_Brand=("Q25_Expected_Spend_New_Brand_Lakh", "mean"),
        MII_Required=("MII_Required", "mean"),
        Discount_Needed=("Q18_Discount_Needed_To_Switch_Pct", "mean"),
        **{c.replace("Q16_Imp_", "Imp_"): (c, "mean") for c in IMP_COLS},
        Smart_Interest=("Q20_Interest_Smart_Monitoring", "mean"),
    )
    prof["Persona"] = name_clusters(prof)
    return {"scan": pd.DataFrame(scan), "labels": labels, "xy": xy, "profile": prof,
            "explained_var": pcs.explained_variance_ratio_}


def name_clusters(prof):
    names = {}
    remaining = set(prof.index)
    order = [("Imp_Make_In_India", "Govt / Make-in-India buyers"), ("Imp_Smart_Features", "Enterprise & data-centre buyers"),
             ("Imp_Customisation", "AV & custom-build specialists"), ("Imp_Price", "Price-driven SMB integrators")]
    for col, nm in order:
        c = prof.loc[list(remaining), col].idxmax(); names[c] = nm; remaining.discard(c)
    for c in remaining:
        names[c] = f"Cluster {c}"
    return [names[i] for i in prof.index]


# ---------------------------------------------------------------- ASSOCIATION RULES
def run_arm(df, min_support=0.05, min_conf=0.45, min_lift=1.1):
    plan_cols = [c for c in df.columns if c.startswith("Plan_")]
    basket = df[plan_cols].astype(bool)
    basket.columns = [c.replace("Plan_", "").replace("_", " ").replace("Env Monitoring", "Env. Monitoring") for c in plan_cols]
    fi = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(fi, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] >= min_lift].copy()
    rules["Antecedent"] = rules["antecedents"].map(lambda s: " + ".join(sorted(s)))
    rules["Consequent"] = rules["consequents"].map(lambda s: " + ".join(sorted(s)))
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    out = rules[["Antecedent", "Consequent", "support", "confidence", "lift"]].reset_index(drop=True)

    # diagnostic rules: which pain points / attributes go with a "Yes, likely to buy"
    pain_cols = [c for c in df.columns if c.startswith("Pain_") and c != "Pain_Count"]
    d = df[pain_cols].astype(bool).copy()
    d.columns = [c.replace("Pain_", "Pain: ").replace("_", " ") for c in pain_cols]
    d["MII Class-I required"] = df["MII_Required"].astype(bool)
    d["Low satisfaction (≤2)"] = df["Q12_Current_Supplier_Satisfaction"] <= 2
    d["Lead time > 8 wks"] = df["Q11_Avg_Lead_Time_Weeks"] > 8
    d["LIKELY TO BUY"] = df["Target_Likely_Buy"].astype(bool)
    fi2 = apriori(d, min_support=0.04, use_colnames=True, max_len=3)
    r2 = association_rules(fi2, metric="confidence", min_threshold=0.3)
    r2 = r2[r2["consequents"].map(lambda s: s == frozenset({"LIKELY TO BUY"}))].copy()
    r2["Antecedent"] = r2["antecedents"].map(lambda s: " + ".join(sorted(s)))
    r2 = r2.sort_values("lift", ascending=False)[["Antecedent", "support", "confidence", "lift"]].reset_index(drop=True)
    item_support = basket.mean().sort_values(ascending=False)
    return {"rules": out, "buy_rules": r2, "item_support": item_support, "n_itemsets": len(fi)}
