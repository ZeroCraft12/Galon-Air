import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATA & BASIC PREPARATION
# ==========================================
print("Membaca data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample submission.csv')

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Encode categorical variables together
all_df = pd.concat([train_df.drop(['team_goals', 'opp_goals'], axis=1), test_df], ignore_index=True)
cat_cols = ['gender', 'team', 'opponent', 'tournament', 'venue_country', 'confederation_team', 'confederation_opp']
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    all_df[c] = le.fit_transform(all_df[c].astype(str))
    train_df[c] = le.transform(train_df[c].astype(str))
    test_df[c] = le.transform(test_df[c].astype(str))

# ==========================================
# 2. STATE TRACKER (The "Antigravity" Engine)
# ==========================================
# Ini adalah otak dari simulasi. Menyimpan status terakhir setiap tim dari train.csv
print("Membangun mesin waktu (Dynamic State Tracker)...")
train_df = train_df.sort_values('date')

missing_cols = [c for c in train_df.columns if c not in test_df.columns and c not in ['team_goals', 'opp_goals']]
team_history_df = train_df.groupby('team').last()[missing_cols].reset_index()

# Konversi ke dictionary agar sangat cepat diakses baris per baris
state_tracker = {}
for _, row in team_history_df.iterrows():
    team_id = row['team']
    state_tracker[team_id] = {
        'elo': row['elo_team'] if not pd.isna(row['elo_team']) else 1500.0,
        # Kita simpan riwayat 5 laga terakhir dalam deque untuk kalkulasi poin dinamis
        'last5_points': deque([row['team_points_last5'] / 5.0] * 5 if not pd.isna(row['team_points_last5']) else [0]*5, maxlen=5),
        'last5_gd': deque([row['team_gd_last5'] / 5.0] * 5 if not pd.isna(row['team_gd_last5']) else [0]*5, maxlen=5),
        'rank': row['rank_team'] if not pd.isna(row['rank_team']) else 100, # Rank dianggap semi-statis
    }

# ==========================================
# 3. FEATURE ENGINEERING & TRAINING
# ==========================================
print("Melatih model dasar...")
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month

def get_weight(t):
    t = str(t).lower()
    if 'afc' in t: return 2.0
    elif 'world cup' in t and 'qualification' not in t: return 1.8
    elif 'friendly' in t: return 0.96
    else: return 1.20

train_df['weight'] = train_df['tournament'].apply(get_weight)

# Drop date, ID untuk training
drop_cols = ['Id', 'match_id', 'date', 'team_goals', 'opp_goals', 'weight']
features = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[features].fillna(0) # Isi NaN di train
y_team = train_df['team_goals']
y_opp = train_df['opp_goals']
w_train = train_df['weight']

# Model Long-Term menggunakan Poisson
model_team = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.03, max_iter=250, random_state=42)
model_team.fit(X_train, y_team) # HistGBM versi sklearn lama mungkin tidak support weight di fit

model_opp = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.03, max_iter=250, random_state=42)
model_opp.fit(X_train, y_opp)

# ==========================================
# 4. DYNAMIC SIMULATION LOOP (TEST DATA)
# ==========================================
print("Memulai Simulasi Dinamis untuk 15 Tahun ke Depan (Iterasi...)")
test_df = test_df.sort_values('date')

# Buat kolom kosong untuk hasil prediksi
test_df['team_goals'] = 0
test_df['opp_goals'] = 0

# Tambah fitur waktu di test
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month

# Fungsi update Elo (Formula Catur standar K=20)
def update_elo(elo_a, elo_b, goals_a, goals_b):
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    expected_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))
    
    score_a = 1 if goals_a > goals_b else (0.5 if goals_a == goals_b else 0)
    score_b = 1 if goals_b > goals_a else (0.5 if goals_b == goals_a else 0)
    
    new_elo_a = elo_a + 20 * (score_a - expected_a)
    new_elo_b = elo_b + 20 * (score_b - expected_b)
    return new_elo_a, new_elo_b

# Loop baris demi baris (Simulasi Waktu Nyata)
pred_team_list = []
pred_opp_list = []

for idx, row in test_df.iterrows():
    team = row['team']
    opp = row['opponent']
    
    # Inisialisasi jika tim belum ada di state tracker
    if team not in state_tracker: state_tracker[team] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150}
    if opp not in state_tracker: state_tracker[opp] = {'elo': 1500.0, 'last5_points': deque([0]*5, maxlen=5), 'last5_gd': deque([0]*5, maxlen=5), 'rank': 150}
    
    # Ambil state SAAT INI
    state_t = state_tracker[team]
    state_o = state_tracker[opp]
    
    # Bentuk fitur untuk baris ini yang meniru urutan `features` dari training
    # Ini sangat penting: Kita injeksikan nilai DINAMIS yang sudah kita track
    row_dict = row.to_dict()
    
    # Fitur Dinamis Tim
    row_dict['elo_team'] = state_t['elo']
    row_dict['team_points_last5'] = sum(state_t['last5_points'])
    row_dict['team_gd_last5'] = sum(state_t['last5_gd'])
    row_dict['rank_team'] = state_t['rank']
    
    # Fitur Dinamis Lawan
    row_dict['elo_opponent'] = state_o['elo']
    row_dict['opp_points_last5'] = sum(state_o['last5_points'])
    row_dict['opp_gd_last5'] = sum(state_o['last5_gd'])
    row_dict['rank_opponent'] = state_o['rank']
    
    # Kalkulasi Differensial (Diff)
    row_dict['rank_diff'] = state_t['rank'] - state_o['rank']
    row_dict['points_last5_diff'] = row_dict['team_points_last5'] - row_dict['opp_points_last5']
    row_dict['gd_last5_diff'] = row_dict['team_gd_last5'] - row_dict['opp_gd_last5']
    
    # Untuk fitur missing lainnya yang tidak ditrack detail, isi 0 sebagai netral/baseline
    for col in missing_cols:
        if col not in row_dict:
            row_dict[col] = 0.0
            
    # Ekstrak vector khusus untuk diprediksi
    X_pred = pd.DataFrame([row_dict])[features].fillna(0)
    
    # Prediksi!
    pred_t = np.round(model_team.predict(X_pred)[0]).astype(int)
    pred_o = np.round(model_opp.predict(X_pred)[0]).astype(int)
    
    pred_team_list.append(pred_t)
    pred_opp_list.append(pred_o)
    
    # === SAKLAR ANTIGRAVITY (UPDATE STATE BERDASARKAN MASA DEPAN) ===
    # 1. Update Elo berdasarkan hasil tebakan
    new_elo_t, new_elo_o = update_elo(state_t['elo'], state_o['elo'], pred_t, pred_o)
    state_tracker[team]['elo'] = new_elo_t
    state_tracker[opp]['elo'] = new_elo_o
    
    # 2. Update Poin (Menang 3, Seri 1, Kalah 0) & GD (Selisih gol)
    pts_t = 3 if pred_t > pred_o else (1 if pred_t == pred_o else 0)
    pts_o = 3 if pred_o > pred_t else (1 if pred_t == pred_o else 0)
    
    state_tracker[team]['last5_points'].append(pts_t)
    state_tracker[team]['last5_gd'].append(pred_t - pred_o)
    
    state_tracker[opp]['last5_points'].append(pts_o)
    state_tracker[opp]['last5_gd'].append(pred_o - pred_t)

# ==========================================
# 5. SIMPAN HASIL SUBMISI
# ==========================================
print("Menyatukan hasil dan membuat file submission...")
test_df['team_goals'] = pred_team_list
test_df['opp_goals'] = pred_opp_list

submission = test_df[['Id', 'team_goals', 'opp_goals']]

# Merge dengan format baku sample_submission agar urutannya tidak kacau
final_sub = pd.merge(sample_sub[['Id']], submission, on='Id', how='left').fillna(0)
final_sub['team_goals'] = final_sub['team_goals'].astype(int)
final_sub['opp_goals'] = final_sub['opp_goals'].astype(int)

final_sub.to_csv('submission.csv', index=False)
print("SELESAI! Silakan submit file submission.csv yang baru.")