import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("=== START V5: THE GRANDMASTER ENSEMBLE ===")

print("1. Membaca Data & Ekstraksi Fitur Historis...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_cols = test.columns.tolist()

# Bikin target baru di train
def get_outcome(t, o):
    if t > o: return 2   # Home Win
    elif t == o: return 1 # Draw
    else: return 0       # Away Win

train['Outcome'] = train.apply(lambda r: get_outcome(r['team_goals'], r['opp_goals']), axis=1)
train['GD'] = train['team_goals'] - train['opp_goals']
target_cols = ['Outcome', 'GD']

train_base = train[test_cols + target_cols].copy()

# Ekstrak Last Known Elo
train['date'] = pd.to_datetime(train['date'])
train_sorted = train.sort_values('date')
last_elo_team = train_sorted.dropna(subset=['elo_team']).groupby('team')['elo_team'].last().reset_index()
last_form = train_sorted.dropna(subset=['team_points_last10']).groupby('team')[['team_points_last10', 'team_win_rate_last10', 'rank_team']].last().reset_index()
team_strength = pd.merge(last_elo_team, last_form, on='team', how='outer')

all_team_features = team_strength.copy()

def inject_features(df):
    df = df.merge(all_team_features, on='team', how='left')
    opp_features = all_team_features.copy()
    opp_features.columns = ['opponent' if col == 'team' else col + '_opp' for col in opp_features.columns]
    df = df.merge(opp_features, on='opponent', how='left')
    return df

train_injected = inject_features(train_base)
test_injected = inject_features(test)

print("2. Preprocessing & Imputasi NaN...")
drop_cols = ['Id', 'match_id', 'date', 'team', 'opponent', 'venue_country']
features = [c for c in test_injected.columns if c not in drop_cols + target_cols]

for df in [train_injected, test_injected]:
    for col in features:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype('category').cat.codes
    df[features] = df[features].fillna(-999)

X_train = train_injected[features]
y_out = train_injected['Outcome']
y_gd = train_injected['GD']
X_test = test_injected[features]

print("3. Melatih Tiga Raksasa Classifier (Soft Voting)...")
# 1. XGBoost
xgb_c = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, objective='multi:softprob', num_class=3, random_state=42)
xgb_c.fit(X_train, y_out)
prob_xgb = xgb_c.predict_proba(X_test)

# 2. LightGBM
lgb_c = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
lgb_c.fit(X_train, y_out)
prob_lgb = lgb_c.predict_proba(X_test)

# 3. CatBoost
cat_c = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, random_state=42, verbose=False)
cat_c.fit(X_train, y_out)
prob_cat = cat_c.predict_proba(X_test)

# ENSEMBLE: Rata-rata probabilitas
prob_ensemble = (prob_xgb + prob_lgb + prob_cat) / 3.0
pred_out = np.argmax(prob_ensemble, axis=1)

print("4. Melatih Tiga Raksasa Regressor (GD Averaging)...")
xgb_r = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
xgb_r.fit(X_train, y_gd)

lgb_r = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
lgb_r.fit(X_train, y_gd)

cat_r = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5, random_state=42, verbose=False)
cat_r.fit(X_train, y_gd)

# ENSEMBLE: Rata-rata tebakan selisih gol
gd_xgb = xgb_r.predict(X_test)
gd_lgb = lgb_r.predict(X_test)
gd_cat = cat_r.predict(X_test)
pred_gd = np.round((gd_xgb + gd_lgb + gd_cat) / 3.0).astype(int)

print("5. Heuristic Goal Mapping...")
team_goals = np.zeros(len(X_test), dtype=int)
opp_goals = np.zeros(len(X_test), dtype=int)

for i in range(len(X_test)):
    outcome = pred_out[i]
    gd = pred_gd[i]
    
    if outcome == 2: # Home Win
        gd = max(1, gd) # Harus positif
        team_goals[i] = gd
        opp_goals[i] = 0
    elif outcome == 1: # Draw
        team_goals[i] = 1 
        opp_goals[i] = 1
    elif outcome == 0: # Away Win
        gd = min(-1, gd) # Harus negatif
        opp_goals[i] = abs(gd)
        team_goals[i] = 0

submission_ml_v5 = pd.DataFrame({
    'Id': test['Id'],
    'team_goals': team_goals,
    'opp_goals': opp_goals
})
submission_ml_v5.to_csv('submission_ml_v5.csv', index=False)
print("File 'submission_ml_v5.csv' (Grandmaster Ensemble) berhasil dibuat.")
