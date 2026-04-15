"""
Climate Risk & Sovereign Bond Dashboard — Flask Backend
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Adjust this to your actual data folder ──────────────────────────────────
DATA_PATH = os.path.dirname(os.path.abspath(__file__))

BOND_FILES = {
    "India":         "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\India 10-Year Bond Yield Historical Data.csv",
    "Nigeria":       "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Nigeria 10-Year Bond Yield Historical Data.csv",
    "Uganda":        "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Uganda 10-Year Bond Yield Historical Data.csv",
    "Kenya":         "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Kenya 10-Year Bond Yield Historical Data.csv",
    "Pakistan":      "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Pakistan 10-Year Bond Yield Historical Data.csv",
    "Russia":        "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Russia 10-Year Bond Yield Historical Data.csv",
    "Philippines":   "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Philippines 10-Year Bond Yield Historical Data.csv",
    "Brazil":        "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Brazil 10-Year Bond Yield Historical Data.csv",
    "United States": "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Nigeria 10-Year Bond Yield Historical Data.csv",
    "Turkey":        "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Turkey 10-Year Bond Yield Historical Data.csv",
    "Zambia":        "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Zambia 10-Year Bond Yield Historical Data.csv",
    "Egypt":         "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Egypt 10-Year Bond Yield Historical Data.csv",
    "Kazakhstan":    "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Kazakhstan 10-Year Bond Yield Historical Data.csv",
    "Bangladesh":    "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Bangladesh 10-Year Bond Yield Historical Data.csv",
    "Vietnam":       "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Vietnam 10-Year Bond Yield Historical Data.csv",
    "Indonesia":     "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\bonds\\Indonesia 10-Year Bond Yield Historical Data.csv",
}

GAIN_FILES = {
    "gain":           "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\ndgain_countryindex_2026\\resources\\gain.csv",
    "vulnerability":  "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\ndgain_countryindex_2026\\resources\\vulnerability.csv",
    "ecosystems":     "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\ndgain_countryindex_2026\\resources\\ecosystems.csv",
    "health":         "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\ndgain_countryindex_2026\\resources\\health.csv",
    "infrastructure": "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\ndgain_countryindex_2026\\resources\\infrastructure.csv",
    "water":          "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\ndgain_countryindex_2026\\resources\\water.csv",
    "habitat":        "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\ndgain_countryindex_2026\\resources\\habitat.csv",
    "food":           "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\ndgain_countryindex_2026\\resources\\food.csv",
}

COUNTRY_ISO = {
    "India": "IND", "Nigeria": "NGA", "Uganda": "UGA", "Kenya": "KEN",
    "Pakistan": "PAK", "Russia": "RUS", "Philippines": "PHL", "Brazil": "BRA",
    "United States": "USA", "Turkey": "TUR", "Zambia": "ZMB", "Egypt": "EGY",
    "Kazakhstan": "KAZ", "Bangladesh": "BGD", "Vietnam": "VNM", "Indonesia": "IDN",
}
ISO_COUNTRY = {v: k for k, v in COUNTRY_ISO.items()}


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_bonds():
    dfs = []
    for country, fname in BOND_FILES.items():
        path = os.path.join(DATA_PATH, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df["Country"] = country
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Change %"] = df["Change %"].str.replace("%", "").astype(float, errors="ignore")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).dropna(subset=["Date", "Price"])

def load_gain():
    result = {}
    for key, fname in GAIN_FILES.items():
        path = os.path.join(DATA_PATH, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            result[key] = df
    return result

def load_emdat():
    path = os.path.join(DATA_PATH,
        "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\P_Data_Extract_From_World_Development_Indicators\\climate_public_emdat_custom_request_2026-03-11_76792794-efa5-49e5-8eee-c1aed771995f.xlsx")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df

def load_econ():
    path = os.path.join(DATA_PATH, "C:\\Users\\anuja\\OneDrive\\Desktop\\capstone project 2\\ndgain_countryindex_2026\\data_of_economic data.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["Country Name"].notna() &
            ~df["Country Name"].str.contains("database|Updated", case=False, na=True)]
    return df


# ── Load once at startup ──────────────────────────────────────────────────────
print("Loading data…")
BONDS   = load_bonds()
GAIN    = load_gain()
EMDAT   = load_emdat()
ECON    = load_econ()
print("Data ready.")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/bond_timeseries")
def bond_timeseries():
    """Monthly bond yield series for every country."""
    result = {}
    for country in BOND_FILES:
        sub = BONDS[BONDS["Country"] == country].sort_values("Date")
        result[country] = {
            "dates":  sub["Date"].dt.strftime("%Y-%m").tolist(),
            "yields": sub["Price"].round(3).tolist(),
        }
    return jsonify(result)


@app.route("/api/bond_latest")
def bond_latest():
    """Latest yield for every country + basic stats."""
    rows = []
    for country in BOND_FILES:
        sub = BONDS[BONDS["Country"] == country].sort_values("Date")
        if sub.empty:
            continue
        latest = sub.tail(1).iloc[0]
        rows.append({
            "country":   country,
            "yield":     round(float(latest["Price"]), 3),
            "mean":      round(float(sub["Price"].mean()), 3),
            "std":       round(float(sub["Price"].std()), 3),
            "min":       round(float(sub["Price"].min()), 3),
            "max":       round(float(sub["Price"].max()), 3),
        })
    rows.sort(key=lambda x: x["yield"], reverse=True)
    return jsonify(rows)


@app.route("/api/gain/<metric>")
def gain_metric(metric):
    """ND-GAIN time series for a given metric (gain, vulnerability, etc.)."""
    df = GAIN.get(metric)
    if df is None:
        return jsonify({"error": "metric not found"}), 404
    isos = list(COUNTRY_ISO.values())
    sub  = df[df["ISO3"].isin(isos)].copy()
    years = [str(y) for y in range(1995, 2024) if str(y) in sub.columns]
    out = []
    for _, row in sub.iterrows():
        iso     = row["ISO3"]
        country = ISO_COUNTRY.get(iso, row["Name"])
        vals    = [round(float(row[y]), 4) if pd.notna(row[y]) else None for y in years]
        out.append({"country": country, "iso": iso, "years": years, "values": vals})
    return jsonify(out)


@app.route("/api/gain_snapshot/<int:year>")
def gain_snapshot(year):
    """All ND-GAIN metrics for study countries at a single year."""
    yr = str(year)
    dims = ["vulnerability", "ecosystems", "health", "infrastructure", "water", "habitat", "food", "gain"]
    result = {c: {"country": c} for c in COUNTRY_ISO}
    for dim in dims:
        df = GAIN.get(dim)
        if df is None or yr not in df.columns:
            continue
        for country, iso in COUNTRY_ISO.items():
            row = df[df["ISO3"] == iso][yr].values
            if len(row) > 0 and not np.isnan(row[0]):
                result[country][dim] = round(float(row[0]), 4)
    return jsonify(list(result.values()))


@app.route("/api/emdat_summary")
def emdat_summary():
    if EMDAT.empty:
        return jsonify({})
    out = {}

    # By disaster type
    if "Disaster Type" in EMDAT.columns:
        out["by_type"] = EMDAT["Disaster Type"].value_counts().to_dict()

    # Deaths by country
    if "Total Deaths" in EMDAT.columns and "Country" in EMDAT.columns:
        deaths = (EMDAT.groupby("Country")["Total Deaths"]
                  .sum().sort_values(ascending=False).head(15))
        out["deaths_by_country"] = deaths.to_dict()

    # Events per year
    if "Start Year" in EMDAT.columns:
        yearly = (EMDAT.dropna(subset=["Start Year"])
                  .groupby("Start Year").size()
                  .reset_index(name="Events"))
        yearly["Start Year"] = yearly["Start Year"].astype(int)
        out["events_per_year"] = {
            "years":  yearly["Start Year"].tolist(),
            "events": yearly["Events"].tolist(),
        }

    # Damage
    dmg_col = "Total Damage ('000 US$)"
    if dmg_col in EMDAT.columns:
        dmg = EMDAT.copy()
        dmg[dmg_col] = pd.to_numeric(dmg[dmg_col], errors="coerce")
        dmg = dmg.dropna(subset=[dmg_col])
        top = (dmg.groupby("Country")[dmg_col]
               .sum().sort_values(ascending=False).head(10))
        out["damage_by_country"] = {k: round(v / 1e6, 2) for k, v in top.items()}

    return jsonify(out)


@app.route("/api/econ_indicators")
def econ_indicators():
    """List available World Bank series."""
    if ECON.empty:
        return jsonify([])
    return jsonify(sorted(ECON["Series Name"].dropna().unique().tolist()))


@app.route("/api/econ/<path:series_name>")
def econ_series(series_name):
    """Time series for one WB indicator across study countries."""
    if ECON.empty:
        return jsonify([])
    sub = ECON[ECON["Series Name"] == series_name]
    year_cols = [c for c in ECON.columns if "YR" in str(c)]
    out = []
    for _, row in sub.iterrows():
        country = row["Country Name"]
        vals = {}
        for yc in year_cols:
            y = int(str(yc).split("[")[0].strip()[-4:])
            v = pd.to_numeric(row[yc], errors="coerce")
            if not np.isnan(v):
                vals[y] = round(float(v), 4)
        if vals:
            out.append({"country": country, "data": vals})
    return jsonify(out)


@app.route("/api/clustering/<int:year>/<int:n_clusters>")
def clustering(year, n_clusters):
    """K-Means clustering + PCA projection."""
    yr = str(year)
    n_clusters = max(2, min(n_clusters, 6))
    dims = ["vulnerability", "ecosystems", "health", "infrastructure", "water", "habitat", "food"]

    feature_rows = {}
    for dim in dims:
        df = GAIN.get(dim)
        if df is None or yr not in df.columns:
            continue
        for _, row in df[["ISO3", yr]].dropna().iterrows():
            iso = row["ISO3"]
            if iso not in feature_rows:
                feature_rows[iso] = {}
            feature_rows[iso][dim] = row[yr]

    feat_df = pd.DataFrame(feature_rows).T.dropna()
    if len(feat_df) < n_clusters:
        return jsonify({"error": "not enough data"}), 400

    scaler  = StandardScaler()
    X       = scaler.fit_transform(feat_df)
    kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels  = kmeans.fit_predict(X)
    pca     = PCA(n_components=2)
    coords  = pca.fit_transform(X)
    var     = pca.explained_variance_ratio_

    result = []
    for i, iso in enumerate(feat_df.index):
        country = ISO_COUNTRY.get(iso, iso)
        result.append({
            "iso":     iso,
            "country": country,
            "cluster": int(labels[i]) + 1,
            "pc1":     round(float(coords[i, 0]), 4),
            "pc2":     round(float(coords[i, 1]), 4),
            "features": {d: round(float(feat_df.loc[iso, d]), 4) for d in dims if d in feat_df.columns},
        })

    return jsonify({
        "points": result,
        "variance_explained": [round(float(v), 4) for v in var],
        "n_clusters": n_clusters,
    })


@app.route("/api/cross_analysis/<int:year>")
def cross_analysis(year):
    """Vulnerability + ND-GAIN score vs latest bond yield."""
    yr  = str(year)
    g   = GAIN.get("gain")
    v   = GAIN.get("vulnerability")
    out = []
    for country, iso in COUNTRY_ISO.items():
        row = {"country": country, "iso": iso}

        # Bond yield
        sub = BONDS[BONDS["Country"] == country].sort_values("Date").tail(1)
        row["yield"] = round(float(sub["Price"].values[0]), 3) if not sub.empty else None

        # ND-GAIN
        if g is not None and yr in g.columns:
            r = g[g["ISO3"] == iso][yr].values
            row["gain_score"] = round(float(r[0]), 4) if len(r) and not np.isnan(r[0]) else None

        # Vulnerability
        if v is not None and yr in v.columns:
            r = v[v["ISO3"] == iso][yr].values
            row["vulnerability"] = round(float(r[0]), 4) if len(r) and not np.isnan(r[0]) else None

        out.append(row)
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
