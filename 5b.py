import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from collections import deque
import warnings
warnings.filterwarnings('ignore')

print("Tahap 1: Memuat Dataset")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample submission.csv')

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

print("Tahap 2: Prapemrosesan dan Pemisahan Gender")
all_df = pd.concat([train_df.drop(['team_goals', 'opp_goals'], axis=1), test_df], ignore_index=True)
cat_cols = ['team', 'opponent', 'tournament', 'venue_country', 'confederation_team', 'confederation_opp']

for c in cat_cols:
    le = LabelEncoder()
    all_df[c] = le.fit_transform(all_df[c].astype(str))
    train_df[c] = le.transform(train_df[c].astype(str))
    test_df[c] = le.transform(test_df[c].astype(str))

def get_weight(t):
    t_lower = str(t).lower()
    if 'afc' in t_lower: return 2.0
    elif 'world cup' in t_lower and 'qualification' not in t_lower: return 1.8
    elif 'friendly' in t_lower: return 0.96
    else: return 1.20

train_df['weight'] = train_df['tournament'].apply(get_weight)
test_df['weight'] = test_df['tournament'].apply(get_weight)

train_df['year'] = train_df['date'].dt.year
train_df_modern = train_df[train_df['year'] >= 2000].sort_values('date').copy()

train_M = train_df_modern[train_df_modern['gender'] == 'M'].copy()
train_W = train_df_modern[train_df_modern['gender'] == 'W'].copy()

print("Tahap 3: Inisialisasi State Tracker Terisolasi")
missing_cols = [c for c in train_df.columns if c not in test_df.columns and c not in ['team_goals', 'opp_goals']]

def init_tracker(df_subset):
    tracker = {}
    history = df_subset.groupby('team').last()[missing_cols].reset_index()
    for _, row in history.iterrows():
        tracker[row['team']] = {
            'elo': row['elo_team'] if not pd.isna(row['elo_team']) else 1500.0,
            'last5_points': deque([row['team_points_last5']/5]*5 if not pd.isna(row['team_points_last5']) else [0]*5, maxlen=5),
            'last5_gd': deque([row['team_gd_last5']/5]*5 if not pd.isna(row['team_gd_last5']) else [0]*5, maxlen=5),
            'rank': row['rank_team'] if not pd.isna(row['rank_team']) else 150.0,
        }
    return tracker

state_tracker_M = init_tracker(train_M)
state_tracker_W = init_tracker(train_W)

print("Tahap 4: Pelatihan Model Ansambel (XGBoost Poisson)")
features = [c for c in train_df_modern.columns if c not in ['Id', 'match_id', 'date', 'gender', 'team_goals', 'opp_goals', 'weight', 'year']]

# Parameter Optuna terverifikasi
xgb_params_base = {
    'objective': 'count:poisson',
    'learning_rate': 0.012729,
    'n_estimators': 560,
    'max_depth': 3,
    'subsample': 0.6078,
    'colsample_bytree': 0.9561,
    'min_child_weight': 5,
    'verbosity': 0
}

seeds = [42, 123, 2026]
models_M_team, models_M_opp = [], []
models_W_team, models_W_opp = [], []

# Pelatihan Model Pria
for s in seeds:
    p = xgb_params_base.copy()
    p['random_state'] = s
    models_M_team.append(xgb.XGBRegressor(**p).fit(train_M[features].fillna(0), train_M['team_goals'], sample_weight=train_M['weight']))
    models_M_opp.append(xgb.XGBRegressor(**p).fit(train_M[features].fillna(0), train_M['opp_goals'], sample_weight=train_M['weight']))

# Pelatihan Model Wanita
for s in seeds:
    p = xgb_params_base.copy()
    p['random_state'] = s
    models_W_team.append(xgb.XGBRegressor(**p).fit(train_W[features].fillna(0), train_W['team_goals'], sample_weight=train_W['weight']))
    models_W_opp.append(xgb.XGBRegressor(**p).fit(train_W[features].fillna(0), train_W['opp_goals'], sample_weight=train_W['weight']))

print("Tahap 5: Eksekusi Prediksi Dinamis pada Data Uji")
test_df = test_df.sort_values('date')

def update_elo_advanced(elo_a, elo_b, goals_a, goals_b, is_home_a):
    home_adv = 100 if is_home_a == 1 else 0
    expected_a = 1 / (1 + 10 ** (((elo_b) - (elo_a + home_adv)) / 400))
    expected_b = 1 / (1 + 10 ** (((elo_a + home_adv) - (elo_b)) / 400))
    
    score_a = 1 if goals_a > goals_b else (0.5 if goals_a == goals_b else 0)
    score_b = 1 if goals_b > goals_a else (0.5 if goals_b == goals_a else 0)
    
    k_factor = 20 + (abs(goals_a - goals_b) * 2) 
    return elo_a + k_factor * (score_a - expected_a), elo_b + k_factor * (score_b - expected_b)

def custom_round(pred, threshold=0.55):
    floor_val = np.floor(pred)
    return np.where((pred - floor_val) > threshold, np.ceil(pred), floor_val).astype(int)

pred_team_list, pred_opp_list = [], []
weights_decay = [0.1, 0.15, 0.2, 0.25, 0.3]

for idx, row in test_df.iterrows():
    gender = row['gender']
    team, opp, is_home = row['team'], row['opponent'], row['is_home']
    
    # Seleksi tracker berdasarkan gender
    tracker = state_tracker_M if gender == 'M' else state_tracker_W
    
    if team not in tracker: tracker[team] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150.0}
    if opp not in tracker: tracker[opp] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150.0}
    
    state_t, state_o = tracker[team], tracker[opp]
    row_dict = row.to_dict()
    
    row_dict['elo_team'] = state_t['elo']
    row_dict['team_points_last5'] = np.dot(list(state_t['last5_points']), weights_decay) * 5
    row_dict['team_gd_last5'] = np.dot(list(state_t['last5_gd']), weights_decay) * 5
    row_dict['rank_team'] = state_t['rank']
    
    row_dict['elo_opponent'] = state_o['elo']
    row_dict['opp_points_last5'] = np.dot(list(state_o['last5_points']), weights_decay) * 5
    row_dict['opp_gd_last5'] = np.dot(list(state_o['last5_gd']), weights_decay) * 5
    row_dict['rank_opponent'] = state_o['rank']
    
    row_dict['rank_diff'] = state_t['rank'] - state_o['rank']
    row_dict['points_last5_diff'] = row_dict['team_points_last5'] - row_dict['opp_points_last5']
    row_dict['gd_last5_diff'] = row_dict['team_gd_last5'] - row_dict['opp_gd_last5']
    
    for col in missing_cols:
        if col not in row_dict: row_dict[col] = 0.0
            
    X_pred = pd.DataFrame([row_dict])[features].fillna(0)
    
    # Seleksi ansambel model berdasarkan gender
    if gender == 'M':
        raw_pred_t = np.mean([m.predict(X_pred)[0] for m in models_M_team])
        raw_pred_o = np.mean([m.predict(X_pred)[0] for m in models_M_opp])
    else:
        raw_pred_t = np.mean([m.predict(X_pred)[0] for m in models_W_team])
        raw_pred_o = np.mean([m.predict(X_pred)[0] for m in models_W_opp])
    
    # Penyesuaian ambang batas dan clipping
    pred_t = np.clip(custom_round(raw_pred_t, threshold=0.55), 0, 10)
    pred_o = np.clip(custom_round(raw_pred_o, threshold=0.55), 0, 10)
    
    pred_team_list.append(pred_t)
    pred_opp_list.append(pred_o)
    
    # Pembaruan status historis
    new_elo_t, new_elo_o = update_elo_advanced(state_t['elo'], state_o['elo'], pred_t, pred_o, is_home)
    tracker[team]['elo'], tracker[opp]['elo'] = new_elo_t, new_elo_o
    
    pts_t = 3 if pred_t > pred_o else (1 if pred_t == pred_o else 0)
    pts_o = 3 if pred_o > pred_t else (1 if pred_t == pred_o else 0)
    
    tracker[team]['last5_points'].append(pts_t)
    tracker[team]['last5_gd'].append(pred_t - pred_o)
    tracker[opp]['last5_points'].append(pts_o)
    tracker[opp]['last5_gd'].append(pred_o - pred_t)

print("Tahap 6: Ekstraksi Format Submisi Akhir")
test_df['team_goals'] = pred_team_list
test_df['opp_goals'] = pred_opp_list

submission = test_df[['Id', 'team_goals', 'opp_goals']]
final_sub = pd.merge(sample_sub[['Id']], submission, on='Id', how='left').fillna(0).astype({'team_goals': int, 'opp_goals': int})

final_sub.to_csv('submission_gimana.csv', index=False)
print("Selesai. File submission.csv berhasil dihasilkan.")