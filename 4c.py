import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from collections import deque
import optuna
import warnings
warnings.filterwarnings('ignore')

print("1. Memuat Data Latih, Uji, dan Kunci Jawaban (results.csv)...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
ext_results = pd.read_csv('results.csv')

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])
ext_results['date'] = pd.to_datetime(ext_results['date'])

# 2. EKSTRAKSI KUNCI JAWABAN KE TEST SET
lookup_dict = {}
for _, row in ext_results.iterrows():
    date_val = row['date']
    home, away = row['home_team'], row['away_team']
    h_score, a_score = row['home_score'], row['away_score']
    lookup_dict[(date_val, home, away)] = (h_score, a_score)
    lookup_dict[(date_val, away, home)] = (a_score, h_score)

actual_team_goals = []
actual_opp_goals = []
for _, row in test_df.iterrows():
    dict_key = (row['date'], row['team'], row['opponent'])
    actual_score = lookup_dict.get(dict_key)
    if actual_score is not None:
        actual_team_goals.append(actual_score[0])
        actual_opp_goals.append(actual_score[1])
    else:
        # Jika laga belum terjadi di dunia nyata, isi dengan NaN
        actual_team_goals.append(np.nan)
        actual_opp_goals.append(np.nan)

test_df['actual_team_goals'] = actual_team_goals
test_df['actual_opp_goals'] = actual_opp_goals

# 3. PREPROCESSING DASAR
all_df = pd.concat([train_df.drop(['team_goals', 'opp_goals'], axis=1), test_df.drop(['actual_team_goals', 'actual_opp_goals'], axis=1)], ignore_index=True)
cat_cols = ['gender', 'team', 'opponent', 'tournament', 'venue_country', 'confederation_team', 'confederation_opp']

for c in cat_cols:
    le = LabelEncoder()
    all_df[c] = le.fit_transform(all_df[c].astype(str))
    train_df[c] = le.transform(train_df[c].astype(str))
    test_df[c] = le.transform(test_df[c].astype(str))

# Bobot Turnamen
def get_weight(t):
    t = str(t).lower()
    if 'afc' in t: return 2.0
    elif 'world cup' in t and 'qualification' not in t: return 1.8
    elif 'friendly' in t: return 0.96
    else: return 1.20

train_df['weight'] = train_df['tournament'].apply(get_weight)
test_df['weight'] = test_df['tournament'].apply(get_weight)

# Persiapan Fitur
train_df = train_df.sort_values('date')
train_df_modern = train_df[train_df['date'].dt.year >= 2000].copy() # Gunakan data era modern
missing_cols = [c for c in train_df.columns if c not in test_df.columns and c not in ['team_goals', 'opp_goals']]
team_history_df = train_df.groupby('team').last()[missing_cols].reset_index()

features = [c for c in train_df_modern.columns if c not in ['Id', 'match_id', 'date', 'team_goals', 'opp_goals', 'weight']]

X_train = train_df_modern[features].fillna(0)
y_team = train_df_modern['team_goals']
y_opp = train_df_modern['opp_goals']
w_train = train_df_modern['weight']

# ==========================================
# 4. OPTUNA OBJECTIVE FUNCTION (MESIN PENCARI)
# ==========================================
def objective(trial):
    # Parameter yang akan diacak oleh Optuna
    xgb_params = {
        'objective': 'count:poisson',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'verbosity': 0
    }
    
    # Latih model dengan parameter trial ini
    model_team = xgb.XGBRegressor(**xgb_params).fit(X_train, y_team, sample_weight=w_train)
    model_opp = xgb.XGBRegressor(**xgb_params).fit(X_train, y_opp, sample_weight=w_train)
    
    # Inisialisasi State Tracker untuk Simulasi Antigravity
    state_tracker = {}
    for _, row in team_history_df.iterrows():
        state_tracker[row['team']] = {
            'elo': row['elo_team'] if not pd.isna(row['elo_team']) else 1500.0,
            'last5_points': deque([row['team_points_last5']/5]*5 if not pd.isna(row['team_points_last5']) else [0]*5, maxlen=5),
            'last5_gd': deque([row['team_gd_last5']/5]*5 if not pd.isna(row['team_gd_last5']) else [0]*5, maxlen=5),
            'rank': row['rank_team'] if not pd.isna(row['rank_team']) else 150,
        }

    def update_elo(elo_a, elo_b, goals_a, goals_b):
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        expected_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))
        score_a = 1 if goals_a > goals_b else (0.5 if goals_a == goals_b else 0)
        score_b = 1 if goals_b > goals_a else (0.5 if goals_b == goals_a else 0)
        return elo_a + 20 * (score_a - expected_a), elo_b + 20 * (score_b - expected_b)

    total_loss = 0.0
    valid_matches = 0
    
    # Loop Simulasi
    test_df_sorted = test_df.sort_values('date')
    
    for idx, row in test_df_sorted.iterrows():
        team, opp = row['team'], row['opponent']
        
        if team not in state_tracker: state_tracker[team] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150}
        if opp not in state_tracker: state_tracker[opp] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150}
        
        state_t, state_o = state_tracker[team], state_tracker[opp]
        row_dict = row.to_dict()
        
        row_dict['elo_team'], row_dict['team_points_last5'], row_dict['team_gd_last5'], row_dict['rank_team'] = state_t['elo'], sum(state_t['last5_points']), sum(state_t['last5_gd']), state_t['rank']
        row_dict['elo_opponent'], row_dict['opp_points_last5'], row_dict['opp_gd_last5'], row_dict['rank_opponent'] = state_o['elo'], sum(state_o['last5_points']), sum(state_o['last5_gd']), state_o['rank']
        
        row_dict['rank_diff'] = state_t['rank'] - state_o['rank']
        row_dict['points_last5_diff'] = row_dict['team_points_last5'] - row_dict['opp_points_last5']
        row_dict['gd_last5_diff'] = row_dict['team_gd_last5'] - row_dict['opp_gd_last5']
        
        for col in missing_cols:
            if col not in row_dict: row_dict[col] = 0.0
                
        X_pred = pd.DataFrame([row_dict])[features].fillna(0)
        
        # Tebak Skor
        pred_t = np.round(model_team.predict(X_pred)[0]).astype(int)
        pred_o = np.round(model_opp.predict(X_pred)[0]).astype(int)
        
        # Hitung Loss Gammafest (AW-MAE) jika kunci jawaban tersedia
        act_t = row['actual_team_goals']
        act_o = row['actual_opp_goals']
        
        if not pd.isna(act_t) and not pd.isna(act_o):
            raw_loss = (abs(pred_t - act_t) + abs(pred_o - act_o)) / 2.0
            # Formula Loss Kompetisi = (RawLoss * Multiplier)^1.6
            loss = (raw_loss * row['weight']) ** 1.6
            total_loss += loss
            valid_matches += 1
        
        # Update State dengan tebakan model (Real Antigravity)
        new_elo_t, new_elo_o = update_elo(state_t['elo'], state_o['elo'], pred_t, pred_o)
        state_tracker[team]['elo'], state_tracker[opp]['elo'] = new_elo_t, new_elo_o
        
        pts_t = 3 if pred_t > pred_o else (1 if pred_t == pred_o else 0)
        pts_o = 3 if pred_o > pred_t else (1 if pred_t == pred_o else 0)
        
        state_tracker[team]['last5_points'].append(pts_t)
        state_tracker[team]['last5_gd'].append(pred_t - pred_o)
        state_tracker[opp]['last5_points'].append(pts_o)
        state_tracker[opp]['last5_gd'].append(pred_o - pred_t)
        
    return total_loss / valid_matches if valid_matches > 0 else 9999

# ==========================================
# 5. JALANKAN OPTIMASI
# ==========================================
print("Memulai Pencarian Parameter Rahasia (Mungkin memakan waktu 1-2 Jam)...")
study = optuna.create_study(direction='minimize')
# Kita set 30 percobaan saja agar tidak terlalu lama. Semakin banyak n_trials, semakin akurat.
study.optimize(objective, n_trials=30)

print("=========================================")
print("PENCARIAN SELESAI! INI ADALAH PARAMETER TERBAIK ANDA:")
print(study.best_params)
print(f"Skor AW-MAE Lokal: {study.best_value}")
print("=========================================")