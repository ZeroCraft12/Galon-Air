import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PERSIAPAN DATA & IMPUTASI HISTORIS
# ==========================================
print("Memuat dataset...")
train_df = pd.read_csv('/kaggle/input/nama-dataset-kompetisi/train.csv')
test_df = pd.read_csv('/kaggle/input/nama-dataset-kompetisi/test.csv')
sample_sub = pd.read_csv('/kaggle/input/nama-dataset-kompetisi/sample submission.csv')

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Menyelamatkan fitur yang dihilangkan panitia di test.csv
train_df = train_df.sort_values('date')
missing_cols = [c for c in train_df.columns if c not in test_df.columns and c not in ['team_goals', 'opp_goals']]

team_history = train_df.groupby('team').last()[missing_cols].reset_index()
opponent_history = train_df.groupby('opponent').last()[missing_cols].reset_index()

team_cols = [c for c in missing_cols if 'team' in c or 'h2h' in c or 'diff' in c]
test_df = pd.merge(test_df, team_history[['team'] + team_cols], on='team', how='left')

opp_cols = [c for c in missing_cols if 'opp' in c] 
test_df = pd.merge(test_df, opponent_history[['opponent'] + opp_cols], on='opponent', how='left')

# ==========================================
# 2. FEATURE ENGINEERING & AW-MAE WEIGHTING
# ==========================================
train_df['is_train'] = 1
test_df['is_train'] = 0
all_df = pd.concat([train_df, test_df], ignore_index=True)

all_df['year'] = all_df['date'].dt.year
all_df['month'] = all_df['date'].dt.month

cat_cols = ['gender', 'team', 'opponent', 'tournament', 'venue_country', 'confederation_team', 'confederation_opp']
for c in cat_cols:
    le = LabelEncoder()
    all_df[c] = le.fit_transform(all_df[c].astype(str))

# Kalkulasi bobot (Multiplier) untuk menekan Loss AW-MAE
all_df['tournament_str'] = pd.concat([train_df['tournament'], test_df['tournament']], ignore_index=True)

def get_weight(t):
    t = str(t).lower()
    if 'afc' in t: return 2.0
    elif 'world cup' in t and 'qualification' not in t: return 1.8
    elif 'friendly' in t: return 0.96
    else: return 1.20

all_df['weight'] = all_df['tournament_str'].apply(get_weight)

drop_cols = ['Id', 'match_id', 'date', 'tournament_str', 'is_train', 'team_goals', 'opp_goals']
features = [c for c in all_df.columns if c not in drop_cols]

# ==========================================
# 3. TRAINING MODEL LONG-TERM POISSON
# ==========================================
print("Melatih model dengan bobot Turnamen...")
train_data = all_df[all_df['is_train'] == 1].copy()
X_train = train_data[features]
y_team = train_data['team_goals']
y_opp = train_data['opp_goals']
w_train = train_data['weight']

# Model untuk Prediksi Gol Tim
model_team = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.03, max_iter=300, random_state=42)
try: 
    model_team.fit(X_train, y_team, sample_weight=w_train)
except TypeError: 
    model_team.fit(X_train, y_team)

# Model untuk Prediksi Gol Lawan
model_opp = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.03, max_iter=300, random_state=42)
try: 
    model_opp.fit(X_train, y_opp, sample_weight=w_train)
except TypeError: 
    model_opp.fit(X_train, y_opp)

# ==========================================
# 4. SIMPAN MODEL UNTUK PANITIA
# ==========================================
print("Menyimpan model ke direktori Kaggle...")
joblib.dump(model_team, '/kaggle/working/model_team_gammafest.pkl')
joblib.dump(model_opp, '/kaggle/working/model_opp_gammafest.pkl')
print("Model berhasil disimpan (.pkl)!")

# ==========================================
# 5. PREDIKSI DATA TEST & FORMAT SUBMISI
# ==========================================
print("Membuat prediksi simulasi akhir...")
test_data = all_df[all_df['is_train'] == 0].copy()

# Pembulatan matematis ketat untuk incaran skor pas (Integer)
test_data['team_goals'] = np.round(model_team.predict(test_data[features])).astype(int)
test_data['opp_goals'] = np.round(model_opp.predict(test_data[features])).astype(int)

submission = test_data[['Id', 'team_goals', 'opp_goals']]
final_sub = pd.merge(sample_sub[['Id']], submission, on='Id', how='left').fillna(0).astype({'team_goals': int, 'opp_goals': int})

# Simpan CSV untuk dikirim ke Leaderboard
final_sub.to_csv('/kaggle/working/submission.csv', index=False)
print("Selesai! File submission.csv siap disubmit ke Kaggle.")