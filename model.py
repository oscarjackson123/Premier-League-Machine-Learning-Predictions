import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Laster inn data fra alle sesongene
filer = [
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2026.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2025.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2024.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2023.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2022.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2021.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2020.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2019.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2018.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2017.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2016.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2015.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2014.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2013.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2012.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2011.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2010.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2009.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2008.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2007.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2006.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2005.csv",
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2004.csv",
]

df = pd.concat([pd.read_csv(f, encoding="latin-1", on_bad_lines="skip") for f in filer], ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
df = df.sort_values("Date").reset_index(drop=True)

# Noen sesonger bruker forskjellige lagnavn, standardiserer til fixture-filens navn
df["HomeTeam"] = df["HomeTeam"].replace({"Man United": "Man Utd", "Tottenham": "Spurs"})
df["AwayTeam"] = df["AwayTeam"].replace({"Man United": "Man Utd", "Tottenham": "Spurs"})

df["HomePoints"] = df["FTR"].map({"H": 3, "D": 1, "A": 0})
df["AwayPoints"] = df["FTR"].map({"H": 0, "D": 1, "A": 3})
df["weight"] = 1.0

# Laster inn FDR (fixture difficulty rating) for neste sesong fra Premier League
fdr_df = pd.read_csv("/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/fdr_2027.csv")
fdr_lookup = {
    (row["HomeTeam"], row["AwayTeam"]): (row["HomeFDR"], row["AwayFDR"])
    for _, row in fdr_df.iterrows()
}

# Bruker 2026-tabellen til å lage FDR for historiske kamper
sesong_2026 = df[df["Date"].dt.year == 2026].copy()
poeng_2026 = {}
for _, row in sesong_2026.iterrows():
    home, away = row["HomeTeam"], row["AwayTeam"]
    if home not in poeng_2026: poeng_2026[home] = 0
    if away not in poeng_2026: poeng_2026[away] = 0
    if row["FTR"] == "H": poeng_2026[home] += 3
    elif row["FTR"] == "A": poeng_2026[away] += 3
    else: poeng_2026[home] += 1; poeng_2026[away] += 1

sortert = sorted(poeng_2026, key=poeng_2026.get, reverse=True)
fdr_fallback = {}
for i, team in enumerate(sortert):
    if i < 4: fdr_fallback[team] = 5
    elif i < 8: fdr_fallback[team] = 4
    elif i < 14: fdr_fallback[team] = 3
    elif i < 17: fdr_fallback[team] = 2
    else: fdr_fallback[team] = 1

# ELO-rating
# Startpunktet er basert på totale poeng i hele datasettet, slik at lag som
# Arsenal og Man Utd starter høyere enn nyopprykkede lag.
# K=2 gjør at ratingen endrer seg sakte, så historisk prestisje veier tungt.
def beregn_elo(df, k=2):
    home_pts = df.groupby("HomeTeam")["HomePoints"].sum()
    away_pts = df.groupby("AwayTeam")["AwayPoints"].sum()
    alle_lag = set(home_pts.index) | set(away_pts.index)

    totale_poeng = {team: home_pts.get(team, 0) + away_pts.get(team, 0) for team in alle_lag}
    min_p = min(totale_poeng.values())
    max_p = max(totale_poeng.values())

    elo = {}
    for team in alle_lag:
        normalized = (totale_poeng[team] - min_p) / (max_p - min_p)
        elo[team] = 1300 + normalized * 500

    home_elo_list = []
    away_elo_list = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        if home not in elo: elo[home] = 1500
        if away not in elo: elo[away] = 1500

        home_elo_list.append(elo[home])
        away_elo_list.append(elo[away])

        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        ftr = row["FTR"]
        score_home = 1 if ftr == "H" else (0 if ftr == "A" else 0.5)
        score_away = 1 - score_home

        elo[home] += k * (score_home - exp_home)
        elo[away] += k * (score_away - (1 - exp_home))

    return home_elo_list, away_elo_list, elo


# Beregner features for hver kamp
# For form, mål scoret og mål sluppet inn brukes gjennomsnittet av siste 5 kamper
def calc_features(df):
    home_form, away_form = [], []
    home_scored, home_conceded = [], []
    away_scored, away_conceded = [], []
    home_fdr_list, away_fdr_list = [], []
    h2h_home_wins, h2h_away_wins, h2h_draws = [], [], []
    team_stats = {}
    h2h_stats = {}

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        def get_stat(team, key):
            games = team_stats.get(team, {}).get(key, [])
            if not games:
                return 0
            return sum(games[-5:]) / len(games[-5:])

        home_form.append(get_stat(home, "points"))
        away_form.append(get_stat(away, "points"))
        home_scored.append(get_stat(home, "scored"))
        home_conceded.append(get_stat(home, "conceded"))
        away_scored.append(get_stat(away, "scored"))
        away_conceded.append(get_stat(away, "conceded"))
        home_fdr_list.append(fdr_fallback.get(home, 3))
        away_fdr_list.append(fdr_fallback.get(away, 3))

        # Head-to-head historikk
        h2h_key = tuple(sorted([home, away]))
        h2h = h2h_stats.get(h2h_key, {"H": 0, "A": 0, "D": 0})
        total = sum(h2h.values()) or 1
        h2h_home_wins.append(h2h.get(home, 0) / total)
        h2h_away_wins.append(h2h.get(away, 0) / total)
        h2h_draws.append(h2h["D"] / total)

        # Oppdater statistikk etter kampen
        for team, pts, scored, conceded in [
            (home, row["HomePoints"], row["FTHG"], row["FTAG"]),
            (away, row["AwayPoints"], row["FTAG"], row["FTHG"])
        ]:
            if team not in team_stats:
                team_stats[team] = {"points": [], "scored": [], "conceded": []}
            team_stats[team]["points"].append(pts)
            team_stats[team]["scored"].append(scored)
            team_stats[team]["conceded"].append(conceded)

        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {"D": 0}
        if row["FTR"] == "D":
            h2h_stats[h2h_key]["D"] += 1
        else:
            winner = home if row["FTR"] == "H" else away
            h2h_stats[h2h_key][winner] = h2h_stats[h2h_key].get(winner, 0) + 1

    df = df.copy()
    df["HomeForm"] = home_form
    df["AwayForm"] = away_form
    df["HomeScored"] = home_scored
    df["HomeConceded"] = home_conceded
    df["AwayScored"] = away_scored
    df["AwayConceded"] = away_conceded
    df["HomeFDR"] = home_fdr_list
    df["AwayFDR"] = away_fdr_list
    df["H2H_Home"] = h2h_home_wins
    df["H2H_Away"] = h2h_away_wins
    df["H2H_Draw"] = h2h_draws

    return df, team_stats, h2h_stats


df, team_stats, h2h_stats = calc_features(df)
home_elo, away_elo, siste_elo = beregn_elo(df)
df["HomeElo"] = home_elo
df["AwayElo"] = away_elo

features = [
    "HomeElo", "AwayElo",
    "HomeForm", "AwayForm",
    "HomeScored", "HomeConceded",
    "AwayScored", "AwayConceded",
    "HomeFDR", "AwayFDR",
    "H2H_Home", "H2H_Away", "H2H_Draw"
]

# Trener modellen på alle sesonger før 2026, tester på 2026
train_df = df[df["Date"].dt.year < 2026]
test_df = df[df["Date"].dt.year >= 2026]

X_train, y_train = train_df[features], train_df["FTR"]
X_val, y_val = test_df[features], test_df["FTR"]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

xgb_model = XGBClassifier(random_state=1, eval_metric="mlogloss", n_estimators=200)
xgb_model.fit(X_train, y_train_enc, sample_weight=train_df["weight"])

y_pred = le.inverse_transform(xgb_model.predict(X_val))
print("Accuracy:", accuracy_score(y_val, y_pred))

# Prediker alle kampene i 2026/27-sesongen
fixtures = pd.read_csv("/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/2027.csv")
fixtures = fixtures.rename(columns={"Home Team": "HomeTeam", "Away Team": "AwayTeam"})

def get_stat(team, key):
    games = team_stats.get(team, {}).get(key, [])
    if not games:
        return 0
    return sum(games[-5:]) / len(games[-5:])

def get_h2h(home, away, stat):
    key = tuple(sorted([home, away]))
    h2h = h2h_stats.get(key, {"D": 0})
    total = sum(h2h.values()) or 1
    if stat == "home": return h2h.get(home, 0) / total
    elif stat == "away": return h2h.get(away, 0) / total
    else: return h2h["D"] / total

fixture_features = []
for _, row in fixtures.iterrows():
    home, away = row["HomeTeam"], row["AwayTeam"]
    home_fdr, away_fdr = fdr_lookup.get((home, away), (3, 3))
    fixture_features.append([
        siste_elo.get(home, 1500), siste_elo.get(away, 1500),
        get_stat(home, "points"), get_stat(away, "points"),
        get_stat(home, "scored"), get_stat(home, "conceded"),
        get_stat(away, "scored"), get_stat(away, "conceded"),
        home_fdr, away_fdr,
        get_h2h(home, away, "home"),
        get_h2h(home, away, "away"),
        get_h2h(home, away, "draw"),
    ])

fixture_X = pd.DataFrame(fixture_features, columns=features)
probs = xgb_model.predict_proba(fixture_X)
classes = le.classes_

kamp_probs = []
for i, row in fixtures.iterrows():
    prob_dict = dict(zip(classes, probs[i]))
    kamp_probs.append({
        "GW": row.get("Round Number", ""),
        "HomeTeam": row["HomeTeam"],
        "AwayTeam": row["AwayTeam"],
        "P_H": prob_dict.get("H", 0),
        "P_D": prob_dict.get("D", 0),
        "P_A": prob_dict.get("A", 0),
    })

kamp_df = pd.DataFrame(kamp_probs)

results = []
for _, row in kamp_df.iterrows():
    predicted = max(["H", "D", "A"], key=lambda x: row[f"P_{x}"])
    max_prob = max(row["P_H"], row["P_D"], row["P_A"])
    results.append({
        "GW": row["GW"],
        "HomeTeam": row["HomeTeam"],
        "AwayTeam": row["AwayTeam"],
        "Predicted": predicted,
        "P(H)": f"{row['P_H']:.2f}",
        "P(D)": f"{row['P_D']:.2f}",
        "P(A)": f"{row['P_A']:.2f}",
        "Confident": "✓" if max_prob >= 0.80 else "",
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/prediksjoner_2027.csv",
    index=False
)

print(f"\n{'GW':<5} {'Hjemme':<22} {'Borte':<22} {'Pred':<6} {'P(H)':<6} {'P(D)':<6} {'P(A)':<6} {'OK'}")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"{row['GW']:<5} {row['HomeTeam']:<22} {row['AwayTeam']:<22} {row['Predicted']:<6} {row['P(H)']:<6} {row['P(D)']:<6} {row['P(A)']:<6} {row['Confident']}")

# Monte Carlo-simulering: kjører sesongen 1000 ganger
# I hver simulering trekkes kampresultatene tilfeldig basert på modellens sannsynligheter
# Dette gir et mer realistisk bilde av usikkerheten enn å alltid velge mest sannsynlige utfall
print("\nSimulerer 1000 sesonger...")
N = 1000
lag = list(set(kamp_df["HomeTeam"].tolist() + kamp_df["AwayTeam"].tolist()))

poeng_sim = {team: [] for team in lag}
topp4_sim = {team: 0 for team in lag}
nedrykk_sim = {team: 0 for team in lag}
gull_sim = {team: 0 for team in lag}

for _ in range(N):
    sesong_poeng = {team: 0 for team in lag}
    for _, kamp in kamp_df.iterrows():
        home, away = kamp["HomeTeam"], kamp["AwayTeam"]
        p = np.array([kamp["P_H"], kamp["P_D"], kamp["P_A"]])
        p = p / p.sum()
        utfall = np.random.choice(["H", "D", "A"], p=p)
        if utfall == "H":
            sesong_poeng[home] += 3
        elif utfall == "A":
            sesong_poeng[away] += 3
        else:
            sesong_poeng[home] += 1
            sesong_poeng[away] += 1

    sortert = sorted(sesong_poeng, key=sesong_poeng.get, reverse=True)
    for team in lag:
        poeng_sim[team].append(sesong_poeng[team])
    for team in sortert[:4]:
        topp4_sim[team] += 1
    for team in sortert[17:]:
        nedrykk_sim[team] += 1
    gull_sim[sortert[0]] += 1

sim_rows = []
for team in lag:
    sim_rows.append({
        "Lag": team,
        "Snitt poeng": round(np.mean(poeng_sim[team]), 1),
        "80% intervall": f"{int(np.percentile(poeng_sim[team], 10))}-{int(np.percentile(poeng_sim[team], 90))}",
        "Gull %": round(gull_sim[team] / N * 100, 1),
        "Topp 4 %": round(topp4_sim[team] / N * 100, 1),
        "Nedrykk %": round(nedrykk_sim[team] / N * 100, 1),
    })

sim_df = pd.DataFrame(sim_rows).sort_values("Snitt poeng", ascending=False).reset_index(drop=True)
sim_df.index += 1

print("\n── Predikert tabell 2026/27 (1000 simulerte sesonger) ──")
print(sim_df.to_string())

sim_df.to_csv(
    "/Users/oscarjackson/Documents/Maskinlæring prosjekt/data/simulert_tabell_2027.csv",
    index=True
)