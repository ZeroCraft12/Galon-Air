import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from collections import deque
import warnings
warnings.filterwarnings('ignore')

print("1. Membaca Dataset Resmi Kompetisi...")
# Sesuaikan path ini dengan path input Kaggle Anda
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample submission.csv')

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

print("2. Preprocessing & Encoding...")
all_df = pd.concat([train_df.drop(['team_goals', 'opp_goals'], axis=1), test_df], ignore_index=True)
cat_cols = ['gender', 'team', 'opponent', 'tournament', 'venue_country', 'confederation_team', 'confederation_opp']

for c in cat_cols:
    le = LabelEncoder()
    all_df[c] = le.fit_transform(all_df[c].astype(str))
    train_df[c] = le.transform(train_df[c].astype(str))
    test_df[c] = le.transform(test_df[c].astype(str))

print("3. Membangun Mesin State Tracker Historis...")
train_df = train_df.sort_values('date')
missing_cols = [c for c in train_df.columns if c not in test_df.columns and c not in ['team_goals', 'opp_goals']]
team_history_df = train_df.groupby('team').last()[missing_cols].reset_index()

state_tracker = {}
for _, row in team_history_df.iterrows():
    state_tracker[row['team']] = {
        'elo': row['elo_team'] if not pd.isna(row['elo_team']) else 1500.0,
        'last5_points': deque([row['team_points_last5']/5]*5 if not pd.isna(row['team_points_last5']) else [0]*5, maxlen=5),
        'last5_gd': deque([row['team_gd_last5']/5]*5 if not pd.isna(row['team_gd_last5']) else [0]*5, maxlen=5),
        'rank': row['rank_team'] if not pd.isna(row['rank_team']) else 150,
    }

print("4. Persiapan Data Latih Era Modern...")
train_df['year'] = train_df['date'].dt.year
train_df_modern = train_df[train_df['year'] >= 2000].copy()

def get_weight(t):
    t = str(t).lower()
    if 'afc' in t: return 2.0
    elif 'world cup' in t and 'qualification' not in t: return 1.8
    elif 'friendly' in t: return 0.96
    else: return 1.20

train_df_modern['weight'] = train_df_modern['tournament'].apply(get_weight)
features = [c for c in train_df_modern.columns if c not in ['Id', 'match_id', 'date', 'team_goals', 'opp_goals', 'weight']]

X_train = train_df_modern[features].fillna(0)
y_team = train_df_modern['team_goals']
y_opp = train_df_modern['opp_goals']
w_train = train_df_modern['weight']

print("5. Melatih Model XGBoost dengan Hyperparameter Optimal...")
# =====================================================================
# MASUKKAN ANGKA HASIL OPTUNA (DARI KOMPUTER LOKAL) ANDA DI SINI
# =====================================================================
xgb_params = {
    'objective': 'count:poisson',
    'learning_rate': 0.015100487356777,       
    'n_estimators': 165,           
    'max_depth': 3,               
    'subsample': 0.788825888593326,            
    'colsample_bytree': 0.8425328449531392,     
    'min_child_weight': 3,         
    'random_state': 42
}

model_team = xgb.XGBRegressor(**xgb_params).fit(X_train, y_team, sample_weight=w_train)
model_opp = xgb.XGBRegressor(**xgb_params).fit(X_train, y_opp, sample_weight=w_train)

print("6. Memulai Simulasi Prediksi Masa Depan (Dynamic Updating)...")
test_df = test_df.sort_values('date')
test_df['year'] = test_df['date'].dt.year

def update_elo_advanced(elo_a, elo_b, goals_a, goals_b, is_home_a):
    home_adv = 100 if is_home_a == 1 else 0
    expected_a = 1 / (1 + 10 ** (((elo_b) - (elo_a + home_adv)) / 400))
    expected_b = 1 / (1 + 10 ** (((elo_a + home_adv) - (elo_b)) / 400))
    
    score_a = 1 if goals_a > goals_b else (0.5 if goals_a == goals_b else 0)
    score_b = 1 if goals_b > goals_a else (0.5 if goals_b == goals_a else 0)
    
    k_factor = 20 + (abs(goals_a - goals_b) * 2) 
    return elo_a + k_factor * (score_a - expected_a), elo_b + k_factor * (score_b - expected_b)

pred_team_list, pred_opp_list = [], []
weights = [0.1, 0.15, 0.2, 0.25, 0.3] # Bobot eksponensial pertandingan terbaru

for idx, row in test_df.iterrows():
    team, opp, is_home = row['team'], row['opponent'], row['is_home']
    
    if team not in state_tracker: state_tracker[team] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150}
    if opp not in state_tracker: state_tracker[opp] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150}
    
    state_t, state_o = state_tracker[team], state_tracker[opp]
    row_dict = row.to_dict()
    
    row_dict['elo_team'] = state_t['elo']
    row_dict['team_points_last5'] = np.dot(list(state_t['last5_points']), weights) * 5
    row_dict['team_gd_last5'] = np.dot(list(state_t['last5_gd']), weights) * 5
    row_dict['rank_team'] = state_t['rank']
    
    row_dict['elo_opponent'] = state_o['elo']
    row_dict['opp_points_last5'] = np.dot(list(state_o['last5_points']), weights) * 5
    row_dict['opp_gd_last5'] = np.dot(list(state_o['last5_gd']), weights) * 5
    row_dict['rank_opponent'] = state_o['rank']
    
    row_dict['rank_diff'] = state_t['rank'] - state_o['rank']
    row_dict['points_last5_diff'] = row_dict['team_points_last5'] - row_dict['opp_points_last5']
    row_dict['gd_last5_diff'] = row_dict['team_gd_last5'] - row_dict['opp_gd_last5']
    
    for col in missing_cols:
        if col not in row_dict: row_dict[col] = 0.0
            
    X_pred = pd.DataFrame([row_dict])[features].fillna(0)
    
    pred_t = np.round(model_team.predict(X_pred)[0]).astype(int)
    pred_o = np.round(model_opp.predict(X_pred)[0]).astype(int)
    
    pred_team_list.append(pred_t)
    pred_opp_list.append(pred_o)
    
    # Update State dari tebakan masa depan
    new_elo_t, new_elo_o = update_elo_advanced(state_t['elo'], state_o['elo'], pred_t, pred_o, is_home)
    state_tracker[team]['elo'], state_tracker[opp]['elo'] = new_elo_t, new_elo_o
    
    pts_t = 3 if pred_t > pred_o else (1 if pred_t == pred_o else 0)
    pts_o = 3 if pred_o > pred_t else (1 if pred_t == pred_o else 0)
    
    state_tracker[team]['last5_points'].append(pts_t)
    state_tracker[team]['last5_gd'].append(pred_t - pred_o)
    state_tracker[opp]['last5_points'].append(pts_o)
    state_tracker[opp]['last5_gd'].append(pred_o - pred_t)

print("7. Menyimpan File Final...")
test_df['team_goals'] = pred_team_list
test_df['opp_goals'] = pred_opp_list

submission = test_df[['Id', 'team_goals', 'opp_goals']]
final_sub = pd.merge(sample_sub[['Id']], submission, on='Id', how='left').fillna(0).astype({'team_goals': int, 'opp_goals': int})

final_sub.to_csv('submission.csv', index=False)
print("Selesai! File submission.csv 100% legal dan siap dikumpulkan.")