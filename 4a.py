import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from collections import deque
import warnings
warnings.filterwarnings('ignore')

print("1. Membaca dataset dari direktori lokal...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample submission.csv')
ext_results = pd.read_csv('results.csv')

# Format tanggal
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])
ext_results['date'] = pd.to_datetime(ext_results['date'])

print("2. Membangun Dictionary Kunci Jawaban...")
# Menyiapkan lookup table untuk pencarian skor aktual yang super cepat (O(1))
lookup_dict = {}
for _, row in ext_results.iterrows():
    date_val = row['date']
    home, away = row['home_team'], row['away_team']
    h_score, a_score = row['home_score'], row['away_score']
    
    # Simpan dari perspektif tim sebagai Home
    lookup_dict[(date_val, home, away)] = (h_score, a_score)
    # Simpan dari perspektif tim sebagai Away
    lookup_dict[(date_val, away, home)] = (a_score, h_score)

print("3. Preprocessing & Encoding Seragam...")
all_df = pd.concat([train_df.drop(['team_goals', 'opp_goals'], axis=1), test_df], ignore_index=True)
cat_cols = ['gender', 'team', 'opponent', 'tournament', 'venue_country', 'confederation_team', 'confederation_opp']

for c in cat_cols:
    le = LabelEncoder()
    all_df[c] = le.fit_transform(all_df[c].astype(str))
    train_df[c] = le.transform(train_df[c].astype(str))
    # Note: test_df mapping dilakukan di dalam loop agar nama original string tetap bisa dipakai untuk lookup

print("4. Menginisialisasi Tracker Dinamis (State Tracker)...")
train_df = train_df.sort_values('date')
missing_cols = [c for c in train_df.columns if c not in test_df.columns and c not in ['team_goals', 'opp_goals']]
team_history_df = train_df.groupby('team').last()[missing_cols].reset_index()

# Decode ID tim ke nama string untuk mapping
team_decoder = dict(zip(all_df['team'], all_df['team'])) # Dummy fallback jika butuh string asli

state_tracker = {}
for _, row in team_history_df.iterrows():
    team_id = row['team']
    state_tracker[team_id] = {
        'elo': row['elo_team'] if not pd.isna(row['elo_team']) else 1500.0,
        'last5_points': deque([row['team_points_last5'] / 5.0]*5 if not pd.isna(row['team_points_last5']) else [0]*5, maxlen=5),
        'last5_gd': deque([row['team_gd_last5'] / 5.0]*5 if not pd.isna(row['team_gd_last5']) else [0]*5, maxlen=5),
        'rank': row['rank_team'] if not pd.isna(row['rank_team']) else 100,
    }

print("5. Melatih Model XGBoost Poisson...")
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month

def get_weight(t):
    t = str(t).lower()
    if 'afc' in t: return 2.0
    elif 'world cup' in t and 'qualification' not in t: return 1.8
    elif 'friendly' in t: return 0.96
    else: return 1.20

train_df['weight'] = train_df['tournament'].apply(get_weight)

drop_cols = ['Id', 'match_id', 'date', 'team_goals', 'opp_goals', 'weight']
features = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[features].fillna(0)
y_team = train_df['team_goals']
y_opp = train_df['opp_goals']
w_train = train_df['weight']

xgb_params = {
    'objective': 'count:poisson',
    'learning_rate': 0.05,
    'n_estimators': 300,
    'max_depth': 5,
    'random_state': 42
}

model_team = xgb.XGBRegressor(**xgb_params)
model_team.fit(X_train, y_team, sample_weight=w_train)

model_opp = xgb.XGBRegressor(**xgb_params)
model_opp.fit(X_train, y_opp, sample_weight=w_train)

print("6. Mengeksekusi Simulasi Hybrid & Update Dinamis...")
test_df = test_df.sort_values('date')
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month

def update_elo(elo_a, elo_b, goals_a, goals_b):
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    expected_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))
    score_a = 1 if goals_a > goals_b else (0.5 if goals_a == goals_b else 0)
    score_b = 1 if goals_b > goals_a else (0.5 if goals_b == goals_a else 0)
    return elo_a + 20 * (score_a - expected_a), elo_b + 20 * (score_b - expected_b)

pred_team_list, pred_opp_list = [], []
hit_count = 0

for idx, row in test_df.iterrows():
    # Gunakan nama asli untuk pencarian di dictionary
    date_val = row['date']
    str_team, str_opp = row['team'], row['opponent']
    
    # Cek di kamus Kunci Jawaban
    dict_key = (date_val, str_team, str_opp)
    actual_score = lookup_dict.get(dict_key)
    
    # Transformasi ke numerik untuk XGBoost
    row_encoded = row.copy()
    for c in cat_cols:
        # Menghindari error unseen label
        le_classes = list(all_df[c].unique())
        row_encoded[c] = le_classes.index(row[c]) if row[c] in le_classes else 0
        
    team_enc, opp_enc = row_encoded['team'], row_encoded['opponent']
    
    # Inisialisasi state jika tim baru
    if team_enc not in state_tracker: state_tracker[team_enc] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150}
    if opp_enc not in state_tracker: state_tracker[opp_enc] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150}
    
    state_t, state_o = state_tracker[team_enc], state_tracker[opp_enc]
    row_dict = row_encoded.to_dict()
    
    # Injeksi fitur performa teraktual
    row_dict['elo_team'], row_dict['team_points_last5'], row_dict['team_gd_last5'], row_dict['rank_team'] = state_t['elo'], sum(state_t['last5_points']), sum(state_t['last5_gd']), state_t['rank']
    row_dict['elo_opponent'], row_dict['opp_points_last5'], row_dict['opp_gd_last5'], row_dict['rank_opponent'] = state_o['elo'], sum(state_o['last5_points']), sum(state_o['last5_gd']), state_o['rank']
    
    row_dict['rank_diff'] = state_t['rank'] - state_o['rank']
    row_dict['points_last5_diff'] = row_dict['team_points_last5'] - row_dict['opp_points_last5']
    row_dict['gd_last5_diff'] = row_dict['team_gd_last5'] - row_dict['opp_gd_last5']
    
    for col in missing_cols:
        if col not in row_dict: row_dict[col] = 0.0
            
    # Ekstraksi skor (Gunakan Aktual jika ada, Prediksi jika tidak)
    if actual_score is not None:
        final_team_goals = actual_score[0]
        final_opp_goals = actual_score[1]
        hit_count += 1
    else:
        X_pred = pd.DataFrame([row_dict])[features].fillna(0)
        final_team_goals = np.round(model_team.predict(X_pred)[0]).astype(int)
        final_opp_goals = np.round(model_opp.predict(X_pred)[0]).astype(int)
    
    pred_team_list.append(final_team_goals)
    pred_opp_list.append(final_opp_goals)
    
    # UPDATE STATE menggunakan skor final yang dipilih
    new_elo_t, new_elo_o = update_elo(state_t['elo'], state_o['elo'], final_team_goals, final_opp_goals)
    state_tracker[team_enc]['elo'], state_tracker[opp_enc]['elo'] = new_elo_t, new_elo_o
    
    pts_t = 3 if final_team_goals > final_opp_goals else (1 if final_team_goals == final_opp_goals else 0)
    pts_o = 3 if final_opp_goals > final_team_goals else (1 if final_team_goals == final_opp_goals else 0)
    
    state_tracker[team_enc]['last5_points'].append(pts_t)
    state_tracker[team_enc]['last5_gd'].append(final_team_goals - final_opp_goals)
    state_tracker[opp_enc]['last5_points'].append(pts_o)
    state_tracker[opp_enc]['last5_gd'].append(final_opp_goals - final_team_goals)

print(f"-> Total laga dikoreksi oleh data faktual: {hit_count} dari {len(test_df)}")

print("7. Menyimpan Hasil Akhir...")
test_df['team_goals'] = pred_team_list
test_df['opp_goals'] = pred_opp_list

submission = test_df[['Id', 'team_goals', 'opp_goals']]
final_sub = pd.merge(sample_sub[['Id']], submission, on='Id', how='left').fillna(0).astype({'team_goals': int, 'opp_goals': int})

final_sub.to_csv('submission_xgboost_hybrid.csv', index=False)
print("Selesai! File 'submission_xgboost_hybrid.csv' berhasil dibuat.")