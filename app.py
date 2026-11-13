import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime, timedelta

st.set_page_config(page_title="FinGuard AI", page_icon="🛡️", layout="wide")

FEATURES = ["amount", "hour", "tx_count_1h", "avg_amount", "amount_ratio",
            "unique_receivers_1h", "international", "new_device"]

@st.cache_data
def make_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    normal_n = int(n * 0.94)
    fraud_n = n - normal_n

    def block(k, fraud=False):
        amount = rng.lognormal(mean=7.0 if not fraud else 10.1, sigma=0.75, size=k)
        hour = rng.integers(6, 23, size=k) if not fraud else rng.choice([0,1,2,3,4,23], size=k)
        freq = rng.poisson(2.0 if not fraud else 10.0, size=k) + 1
        avg = rng.lognormal(mean=6.9, sigma=0.65, size=k)
        receivers = rng.integers(1, 3, size=k) if not fraud else rng.integers(3, 9, size=k)
        international = rng.binomial(1, 0.015 if not fraud else 0.35, size=k)
        new_device = rng.binomial(1, 0.08 if not fraud else 0.65, size=k)
        ratio = amount / np.maximum(avg, 1)
        return pd.DataFrame({
            "amount": amount.round(2), "hour": hour, "tx_count_1h": freq,
            "avg_amount": avg.round(2), "amount_ratio": ratio.round(2),
            "unique_receivers_1h": receivers, "international": international,
            "new_device": new_device, "is_fraud": int(fraud)
        })

    df = pd.concat([block(normal_n), block(fraud_n, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Synthetic transaction identifiers and account network
    df.insert(0, "transaction_id", [f"TX{i:05d}" for i in range(1, len(df)+1)])
    senders = [f"ACC{rng.integers(100, 450):03d}" for _ in range(len(df))]
    receivers = [f"ACC{rng.integers(450, 750):03d}" for _ in range(len(df))]
    df.insert(1, "sender_id", senders)
    df.insert(2, "receiver_id", receivers)
    base = datetime.now().replace(second=0, microsecond=0)
    mins = rng.integers(0, 60*24*7, size=len(df))
    df.insert(3, "timestamp", [base - timedelta(minutes=int(x)) for x in mins])
    df["aml_flag"] = (
        ((df["tx_count_1h"] >= 8) & (df["amount"].between(15000, 50000))) |
        ((df["unique_receivers_1h"] >= 6) & (df["amount"] >= 20000)) |
        ((df["international"] == 1) & (df["new_device"] == 1))
    ).astype(int)
    return df

@st.cache_resource
def train_models():
    df = make_data()
    X = df[FEATURES]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=120, max_depth=9, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    iso = IsolationForest(contamination=0.06, random_state=42)
    iso.fit(X_train[y_train == 0][FEATURES])
    metrics = {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "cm": confusion_matrix(y_test, pred)
    }
    return clf, iso, metrics

def analyse(df, clf, iso):
    out = df.copy()
    X = out[FEATURES]
    out["fraud_probability"] = clf.predict_proba(X)[:, 1]
    anomaly_raw = iso.decision_function(X)
    # Convert lower decision-function values into a higher 0..1 anomaly score
    lo, hi = np.percentile(anomaly_raw, 2), np.percentile(anomaly_raw, 98)
    out["anomaly_score"] = np.clip((hi - anomaly_raw) / max(hi-lo, 1e-8), 0, 1)
    out["aml_score"] = out["aml_flag"].astype(float)
    out["risk_score"] = (0.45*out["fraud_probability"] +
                         0.20*out["anomaly_score"] +
                         0.35*out["aml_score"]) * 100
    out["risk_score"] = out["risk_score"].round().astype(int)
    out["risk_level"] = pd.cut(out["risk_score"], [-1,24,49,74,100],
                               labels=["Low","Medium","High","Critical"])
    out["reasons"] = out.apply(explain_row, axis=1)
    return out

def explain_row(r):
    reasons = []
    if r["fraud_probability"] >= 0.60: reasons.append("ML model found a fraud-like pattern")
    if r["anomaly_score"] >= 0.75: reasons.append("Behaviour differs from normal activity")
    if r["tx_count_1h"] >= 8: reasons.append("High transaction frequency in one hour")
    if r["amount_ratio"] >= 5: reasons.append("Amount is over 5x account average")
    if r["unique_receivers_1h"] >= 6: reasons.append("Transfers to many distinct receivers")
    if r["international"] and r["new_device"]: reasons.append("New device with international transfer")
    if not reasons: reasons.append("No strong suspicious indicator in this prototype")
    return "; ".join(reasons)

st.title("🛡️ FinGuard AI")
st.caption("Intelligent Financial Fraud Detection & Anti-Money Laundering Prototype")
st.warning("Hackathon demo: uses synthetic data only. Scores are experimental decision-support signals, not proof of fraud or money laundering.")

with st.sidebar:
    st.header("Demo controls")
    n_rows = st.slider("Synthetic transactions", 1000, 8000, 3000, 500)
    seed = st.number_input("Dataset seed", min_value=1, max_value=9999, value=42)
    st.caption("Change the seed to generate a different synthetic scenario.")

@st.cache_data
def data_for(n, seed_value):
    return make_data(n, int(seed_value))

with st.spinner("Preparing demo data and ML models..."):
    clf, iso, metrics = train_models()
    raw = data_for(n_rows, seed)
    analysed = analyse(raw, clf, iso)

total = len(analysed)
flagged = int((analysed["risk_score"] >= 50).sum())
critical = int((analysed["risk_level"] == "Critical").sum())
fraud_rate = float(analysed["is_fraud"].mean()*100)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Transactions analysed", f"{total:,}")
c2.metric("Needs review (risk ≥ 50)", f"{flagged:,}")
c3.metric("Critical alerts", f"{critical:,}")
c4.metric("Synthetic fraud share", f"{fraud_rate:.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Transaction screening", "AML monitoring", "Model evaluation", "About"]
)

with tab1:
    left, right = st.columns(2)
    with left:
        risk_counts = analysed["risk_level"].value_counts().reindex(
            ["Low","Medium","High","Critical"], fill_value=0
        ).rename_axis("Risk level").reset_index(name="Transactions")
        st.subheader("Risk distribution")
        st.plotly_chart(px.bar(risk_counts, x="Risk level", y="Transactions",
                               color="Risk level"), use_container_width=True)
    with right:
        st.subheader("Transaction amount distribution")
        st.plotly_chart(px.histogram(analysed, x="amount", color="risk_level",
                                     nbins=35, title="Amount by risk level"),
                        use_container_width=True)
    st.subheader("Latest high-risk alerts")
    alerts = analysed[analysed["risk_score"] >= 50].sort_values("risk_score", ascending=False)
    st.dataframe(alerts[["transaction_id","sender_id","receiver_id","amount",
                         "risk_score","risk_level","reasons"]].head(12),
                 use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Screen a transaction")
    chosen = st.selectbox("Select synthetic transaction", analysed["transaction_id"].tolist())
    row = analysed.loc[analysed["transaction_id"] == chosen].iloc[0]
    a,b,c = st.columns(3)
    a.metric("Amount", f"₹{row['amount']:,.0f}")
    b.metric("Risk score", f"{row['risk_score']}/100")
    c.metric("Risk level", str(row["risk_level"]))
    st.write("**Detection explanation**")
    for reason in row["reasons"].split("; "):
        st.write("• " + reason)
    st.write("**Model signals**")
    st.json({
        "fraud_model_score": round(float(row["fraud_probability"]), 3),
        "anomaly_score": round(float(row["anomaly_score"]), 3),
        "aml_pattern_flag": bool(row["aml_flag"]),
        "risk_score": int(row["risk_score"])
    })
    st.caption("A flagged transaction should be reviewed by a human analyst.")

with tab3:
    st.subheader("AML pattern screening")
    aml = analysed[analysed["aml_flag"] == 1].copy()
    st.metric("Transactions matching prototype AML rules", len(aml))
    st.dataframe(aml[["transaction_id","sender_id","receiver_id","amount",
                      "tx_count_1h","unique_receivers_1h","international","new_device"]],
                 use_container_width=True, hide_index=True)
    st.info("Current prototype AML checks: rapid transaction splitting indicators, many receivers, and new-device/international-transfer combinations. These are risk indicators, not regulatory conclusions.")
    if len(aml):
        G = nx.DiGraph()
        for _, r in aml.head(150).iterrows():
            G.add_edge(r["sender_id"], r["receiver_id"], amount=float(r["amount"]))
        cycles = list(nx.simple_cycles(G))
        st.write(f"Directed account network: **{G.number_of_nodes()} accounts**, **{G.number_of_edges()} transfers**.")
        st.write(f"Short circular paths found in this sample: **{sum(1 for x in cycles if 2 <= len(x) <= 5)}**")
    else:
        st.write("No AML-rule matches in this sample. Try another dataset seed.")

with tab4:
    st.subheader("Random Forest test-set performance")
    m1,m2,m3 = st.columns(3)
    m1.metric("Precision", f"{metrics['precision']:.3f}")
    m2.metric("Recall", f"{metrics['recall']:.3f}")
    m3.metric("F1 score", f"{metrics['f1']:.3f}")
    cm = metrics["cm"]
    cm_df = pd.DataFrame(cm, index=["Actual normal","Actual fraud"],
                         columns=["Predicted normal","Predicted fraud"])
    st.write("Confusion matrix")
    st.dataframe(cm_df, use_container_width=True)
    imp = pd.DataFrame({"Feature": FEATURES, "Importance": clf.feature_importances_}).sort_values(
        "Importance", ascending=False)
    st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                           title="Random Forest feature importance"), use_container_width=True)
    st.caption("Metrics are measured on a held-out split of synthetic data. They do not establish real-world banking performance.")

with tab5:
    st.subheader("Prototype methodology")
    st.markdown("""
    - **Random Forest:** supervised classification trained on synthetic labels.
    - **Isolation Forest:** unsupervised anomaly screening.
    - **AML rules + NetworkX:** simple transaction-pattern and directed-network analysis.
    - **Risk score:** weighted combination of fraud-model, anomaly and AML signals.
    - **Explainability:** human-readable indicators attached to each alert.

    **Limitations:** Synthetic data is simplified and may encode artificial patterns. This prototype is not connected to banks, UPI rails, KYC systems or regulatory reporting. Do not use it to make real customer decisions.
    """)
    csv = analysed.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download analysed transactions (CSV)", csv,
                       "finguard_ai_demo_results.csv", "text/csv")

st.divider()
st.caption("FinGuard AI | Hackathon prototype | Synthetic data only")
