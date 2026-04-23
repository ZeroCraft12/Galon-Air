"""
Football Score Prediction - DEFINITIVE BEST SOLUTION
AW-MAE Competition

Key insight: Use organizer's ELO+form features for training,
compute them from history for test set.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── Load Data ────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
sub   = pd.read_csv('sample submission.csv')

train['date'] = pd.to_datetime(train['date'])
test['date']  = pd.to_datetime(test['date'])

print(f"Train: {train.shape}  Test: {test.shape}")
print(f"Train: {train['date'].min().date()} -> {train['date'].max().date()}")
print(f"Test:  {test['date'].min().date()} -> {test['date'].max().date()}")

# ─── Compute ELO from Train History ──────────────────────────────────────────
print("\nComputing ELO from train history...")
elo = {}
INIT_ELO = 1500
K = 30

for _, row in train.sort_values('date').iterrows():
    t, o = row['team'], row['opponent']
    et = elo.get(t, INIT_ELO)
    eo = elo.get(o, INIT_ELO)
    gf, ga = row['team_goals'], row['opp_goals']
    we = 1 / (1 + 10**((eo - et) / 400))
    s = 1 if gf > ga else (0.5 if gf == ga else 0)
    elo[t] = et + K * (s - we)
    elo[o] = eo + K * ((1-s) - (1-we))

print(f"  Final ELO for {len(elo)} teams ready for test inference.")

# ─── Rolling Form from Train History ──────────────────────────────────────────
print("Computing rolling form for test teams...")
team_hist = {}  # team → list of (date, gf, ga, pts)

for _, row in train.sort_values('date').iterrows():
    t, o = row['team'], row['opponent']
    gf, ga = row['team_goals'], row['opp_goals']
    pts_t = 3 if gf > ga else (1 if gf == ga else 0)
    pts_o = 3 - pts_t if pts_t != 1 else 1
    if t not in team_hist: team_hist[t] = []
    if o not in team_hist: team_hist[o] = []
    team_hist[t].append((row['date'], gf, ga, pts_t))
    team_hist[o].append((row['date'], ga, gf, pts_o))

def form_stats(team):
    h = team_hist.get(team, [])
    if not h: return [np.nan]*5
    l5 = h[-5:]; l10 = h[-10:]
    return [
        sum(x[3] for x in l5),
        sum(x[1]-x[2] for x in l5),
        np.mean([x[1] for x in l5]),
        np.mean([x[2] for x in l5]),
        sum(x[3]==3 for x in l10)/len(l10),
    ]

# ─── Feature Engineering ──────────────────────────────────────────────────────
CONF_STR = {'UEFA':5,'CONMEBOL':5,'CONCACAF':3,'AFC':3,'CAF':3,'OFC':2,'Unknown':2}

def add_features(df, is_test=False):
    d = df.copy()
    d['year']  = d['date'].dt.year
    d['month'] = d['date'].dt.month
    d['is_women'] = (d['gender'] == 'W').astype(int)
    d['home_not_neutral'] = d['is_home'] * (1 - d['neutral'])

    d['conf_t_str'] = d['confederation_team'].map(CONF_STR).fillna(2)
    d['conf_o_str'] = d['confederation_opp'].map(CONF_STR).fillna(2)
    d['conf_diff']  = d['conf_t_str'] - d['conf_o_str']

    d['gdp_diff']  = np.log1p(d['gdp_per_capita_team'].fillna(0)) \
                   - np.log1p(d['gdp_per_capita_opp'].fillna(0))
    d['pop_log_t'] = np.log1p(d['population_team'].fillna(0))
    d['pop_log_o'] = np.log1p(d['population_opp'].fillna(0))
    d['dist_diff'] = d['distance_travel_opp'].fillna(0) - d['distance_travel_team'].fillna(0)
    d['alt_log']   = np.log1p(d['altitude_venue'].fillna(0))

    def tier(t):
        t = str(t).lower()
        if 'world cup' in t and 'qual' not in t: return 5
        if any(x in t for x in ['copa','euro','african cup','asian cup','gold cup']
               ) and 'qual' not in t: return 4
        if 'qual' in t: return 3
        if 'friendly' in t: return 1
        return 2
    d['tourn_tier'] = d['tournament'].apply(tier)

    if is_test:
        # Inject computed ELO + form into test
        d['elo_team']     = d['team'].map(elo).fillna(INIT_ELO)
        d['elo_opponent'] = d['opponent'].map(elo).fillna(INIT_ELO)

        tf = d['team'].apply(form_stats)
        of = d['opponent'].apply(form_stats)
        for i, col in enumerate(['team_points_last5','team_gd_last5',
                                  'team_avg_goals_last5','team_avg_conceded_last5',
                                  'team_win_rate_last10']):
            d[col] = tf.apply(lambda x: x[i])
        for i, col in enumerate(['opp_points_last5','opp_gd_last5',
                                  'opp_avg_goals_last5','opp_avg_conceded_last5',
                                  'opp_win_rate_last10']):
            d[col] = of.apply(lambda x: x[i])

        d['h2h_points_last5'] = np.nan
        d['h2h_gd_last5']     = np.nan

    # Derived form features
    d['elo_diff']     = d['elo_team'] - d['elo_opponent']
    d['elo_win_prob'] = 1 / (1 + 10**((d['elo_opponent'] - d['elo_team']) / 400))
    d['elo_diff_sq']  = d['elo_diff'] ** 2

    if 'team_points_last5' in d.columns:
        d['form_diff_5'] = d['team_points_last5'] - d['opp_points_last5']
        d['gd_diff_5']   = d['team_gd_last5']     - d['opp_gd_last5']
        d['atk_vs_def']  = d['team_avg_goals_last5']   - d['opp_avg_conceded_last5']
        d['opp_atk_def'] = d['opp_avg_goals_last5']    - d['team_avg_conceded_last5']
        d['wr_diff']     = d['team_win_rate_last10']   - d['opp_win_rate_last10']
        d['goal_ratio']  = (d['team_avg_goals_last5'] + 0.1) / \
                           (d['opp_avg_conceded_last5'] + 0.1)

    return d

train_fe = add_features(train, is_test=False)
test_fe  = add_features(test, is_test=True)

# ─── Feature List ─────────────────────────────────────────────────────────────
FEATS = [
    'elo_team','elo_opponent','elo_diff','elo_diff_sq','elo_win_prob',
    'team_points_last5','opp_points_last5','form_diff_5',
    'team_gd_last5','opp_gd_last5','gd_diff_5',
    'team_avg_goals_last5','opp_avg_goals_last5',
    'team_avg_conceded_last5','opp_avg_conceded_last5',
    'atk_vs_def','opp_atk_def','wr_diff','goal_ratio',
    'team_win_rate_last10','opp_win_rate_last10',
    'h2h_points_last5','h2h_gd_last5',
    'is_home','neutral','home_not_neutral','is_women',
    'tourn_tier',
    'conf_t_str','conf_o_str','conf_diff',
    'gdp_diff','pop_log_t','pop_log_o',
    'alt_log','dist_diff','temperature_venue',
    'year','month',
]
FEATS = [f for f in FEATS if f in train_fe.columns and f in test_fe.columns]
print(f"\nTotal features: {len(FEATS)}")

X      = train_fe[FEATS].fillna(-999)
X_test = test_fe[FEATS].fillna(-999)
y_team = train_fe['team_goals']
y_opp  = train_fe['opp_goals']

X_tr, X_val, yt_tr, yt_val, yo_tr, yo_val = train_test_split(
    X, y_team, y_opp, test_size=0.2, random_state=42, shuffle=True)

# ─── Training ─────────────────────────────────────────────────────────────────
params = dict(
    objective='poisson', metric='mae', learning_rate=0.04,
    num_leaves=127, min_child_samples=30,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l1=0.1, lambda_l2=0.1, verbose=-1, n_jobs=-1, seed=42,
)

print("\n-- Training team_goals --")
m_team = lgb.LGBMRegressor(**params, n_estimators=3000)
m_team.fit(X_tr, yt_tr, eval_set=[(X_val, yt_val)],
           callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(400)])
p_team_val  = m_team.predict(X_val)
p_team_test = m_team.predict(X_test)
print(f"  VAL MAE = {mean_absolute_error(yt_val, p_team_val):.4f}  iter={m_team.best_iteration_}")

print("\n-- Training opp_goals --")
m_opp = lgb.LGBMRegressor(**params, n_estimators=3000)
m_opp.fit(X_tr, yo_tr, eval_set=[(X_val, yo_val)],
          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(400)])
p_opp_val  = m_opp.predict(X_val)
p_opp_test = m_opp.predict(X_test)
print(f"  VAL MAE = {mean_absolute_error(yo_val, p_opp_val):.4f}  iter={m_opp.best_iteration_}")

# ─── Evaluation ───────────────────────────────────────────────────────────────
def aw_mae(yt, yo, pt, po):
    pt_r = np.round(pt).clip(0).astype(int)
    po_r = np.round(po).clip(0).astype(int)
    exact = (pt_r == yt.astype(int)) & (po_r == yo.astype(int))
    base  = (np.abs(yt - pt) + np.abs(yo - po)) / 2
    return np.where(exact, 0, base).mean(), exact.mean()

aw_s, ex_r = aw_mae(yt_val.values, yo_val.values, p_team_val, p_opp_val)
print(f"\n{'='*50}")
print(f"  Validation AW-MAE ~= {aw_s:.4f}")
print(f"  Exact score rate  = {ex_r*100:.2f}%")
print(f"{'='*50}")

# ─── Build Submission (correct ordering) ──────────────────────────────────────
pred_df = test_fe[['Id']].copy()
pred_df['team_goals'] = np.round(p_team_test).clip(0).astype(int)
pred_df['opp_goals']  = np.round(p_opp_test).clip(0).astype(int)

final_sub = sub[['Id']].merge(pred_df, on='Id', how='left')
final_sub['team_goals'] = final_sub['team_goals'].fillna(1).astype(int)
final_sub['opp_goals']  = final_sub['opp_goals'].fillna(1).astype(int)

assert len(final_sub) == len(sub)
assert (final_sub['Id'] == sub['Id']).all()

print(f"\nSubmission shape: {final_sub.shape}")
print("team_goals:", dict(final_sub['team_goals'].value_counts().sort_index().head(8)))
print("opp_goals :", dict(final_sub['opp_goals'].value_counts().sort_index().head(8)))
print("\nFirst 5 predictions:")
print(final_sub.head(5).to_string(index=False))

final_sub.to_csv('submission_best.csv', index=False)
print("\n[OK] Saved: submission_best.csv")

# ─── Feature Importances ─────────────────────────────────────────────────────
print("\n=== Top Feature Importances ===")
fi = pd.Series(m_team.feature_importances_, index=FEATS).sort_values(ascending=False)
print(fi.head(20).to_string())